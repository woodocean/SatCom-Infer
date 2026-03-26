import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from algorithms.pmp_solver import PMPSolver

class Scheduler:
    def __init__(self, net_config_path="config/network_config.json", models_config_path="config/model_profiles.json", sizes_fit_path="config/model_profiles_sizes.json"):
        # 加载网络拓扑配置
        if not os.path.exists(net_config_path):
            raise FileNotFoundError(f"找不到网络配置文件: {net_config_path}")
        with open(net_config_path, 'r', encoding='utf-8') as f:
            self.net_config = json.load(f)
            
        # 加载模型性能配置
        if not os.path.exists(models_config_path):
            raise FileNotFoundError(f"找不到模型配置文件: {models_config_path}")
        with open(models_config_path, 'r', encoding='utf-8') as f:
            self.models_config = json.load(f)

        # 【改动】：新增对逐层变尺寸回归公式参数的读取
        if os.path.exists(sizes_fit_path):
            with open(sizes_fit_path, 'r', encoding='utf-8') as f:
                self.sizes_fit_data = json.load(f)
        else:
            self.sizes_fit_data = {}

    # 【改动】：额外增加 target_h 和 target_w 这两项输入来进行长宽估算
    def generate_task_and_schedule(self, task_id="task_001", model_name="yolov5s", input_size_mb=3.0, target_h=224, target_w=224):
        """
        任务生成与调度，自动适配 network_config.json 中的 pipeline 路径
        """
        print(f"\n[{task_id}] 接收到新任务, 模型: {model_name}, 分辨率: {target_h}x{target_w}")
        
        # ================== 1. 获取流水线路径 ==================
        # 从拓扑中读取预定义的 Pipeline 路径: ["RS", "SAT-03", "SAT-04", "SAT-05", "GS"]
        pipeline_path = self.net_config["simulation_paths"]["pipeline"]
        print(f"[{task_id}] 分配路由路径: {' -> '.join(pipeline_path)}")
        
        # 提取计算节点（去掉源节点 RS）-> ["SAT-03", "SAT-04", "SAT-05", "GS"]
        compute_nodes_ids = pipeline_path[1:]
        
        # ================== 2. 构建环境状态 (env_status) ==================
        env_status = {
            "nodes": [],
            "bandwidths": []
        }
        
        # 2.1 提取节点硬件信息
        for node_id in compute_nodes_ids:
            hardware = self.net_config["nodes"][node_id]["hardware"]
            env_status["nodes"].append({
                "id": node_id,
                "compute_speed_gflops_per_ms": hardware.get("compute_speed_gflops_per_ms", 10.0),
                "memory_mb": hardware.get("memory_mb", 1024)
            })
            
        # 2.2 提取链路带宽信息
        for i in range(len(pipeline_path) - 1):
            u = pipeline_path[i]
            v = pipeline_path[i+1]
            link_key = f"{u}_to_{v}"
            
            bandwidth_mbps = self.net_config["links"].get(link_key, {}).get("bandwidth_mbps")
            if not bandwidth_mbps:
                print(f"[警告] 缺少链路 {link_key} 的带宽配置，将默认使用 100 Mbps")
                bandwidth_mbps = 100.0
            
            # 带宽单位转换: Mbps -> MB/s -> MB/ms 
            # (因为算力是 GFLOPs/ms，把通信也转成 /ms 能直接得出 ms 级别的延迟)
            bandwidth_mb_per_ms = (bandwidth_mbps / 8.0) / 1000.0 
            env_status["bandwidths"].append(bandwidth_mb_per_ms) 
            
        # 【改动】：写入标定的本机算力参数，以便PMPSolver使用进行比例投影
        env_status["reference_compute_speed"] = self.net_config.get("global_settings", {}).get("base_gpu_speed_gflops_per_ms", 100.0)

        # ================== 3. 构建模型特征 (model_profile) ==================
        if model_name not in self.models_config:
            raise ValueError(f"models.json 中找不到模型 {model_name} 的配置!")
            
        raw_model_data = self.models_config[model_name]
        num_layers = len(raw_model_data["flops_g"])
        
        # 【改动】：计算面积比与提取回归参数以缩放通信开销
        area = target_h * target_w
        base_dim = 640 if "yolo" in model_name.lower() else 224
        area_ratio = area / (base_dim * base_dim)
        
        fit_data = self.sizes_fit_data.get(model_name, {})
        layer_fits = fit_data.get("layer_fits", [])

        model_profile = {
            "input_size_raw": input_size_mb,
            "layers": []
        }
        
        # 防止部分模型配置没写 params_mb
        params_list = raw_model_data.get("params_mb", [0] * num_layers)
        
        for i in range(num_layers):
            # 【改动核心】：估算纯层级推理时延 (代入 y = slope*X + intercept 公式)
            if layer_fits and i < len(layer_fits):
                base_lat = layer_fits[i]["slope"] * area + layer_fits[i]["intercept"]
            else:
                base_lat = raw_model_data.get("actual_latency_ms", raw_model_data["flops_g"])[i]
            base_lat = max(base_lat, 0.0001)

            # 【改动】：使用缩放比例同步膨胀/缩减特征张量的通信体积
            c_mb = raw_model_data["comm_size_mb"][i] * area_ratio if area_ratio > 0 else 0

            model_profile["layers"].append({
                "flops_g": raw_model_data["flops_g"][i],
                "comm_size_mb": c_mb,  
                "params_mb": params_list[i],
                "base_latency_ms": base_lat # 新增底层延时供求解器使用
            })

        # ================== 4. 调用各种算法求解器 ==================
        solver = PMPSolver(model_profile, env_status)

        print(f"\n[{task_id}] 开始执行调度算法...")

        # 1. LA-DP (本文提出)
        la_lat, la_plan = solver.solve_la_dp()
        print(f"[LA-DP]    预估时延: {la_lat:.2f} ms | 切分方案: {la_plan}")

        # 2. Greedy (贪婪基线)
        greedy_lat, greedy_plan = solver.solve_communication_greedy()
        print(f"[Greedy]   预估时延: {greedy_lat:.2f} ms | 切分方案: {greedy_plan}")

        # 2. uniform_partition (均分基线)
        uniform_lat, uniform_plan = solver.solve_uniform_partition()
        print(f"[uniform]   预估时延: {uniform_lat:.2f} ms | 切分方案: {uniform_plan}")

        # 3. Bent-Pipe (弯管基线)
        bp_lat, bp_plan = solver.solve_bent_pipe()
        print(f"[BentPipe] 预估时延: {bp_lat:.2f} ms | 切分方案: {bp_plan}")

        # 4. Random Split (随机基线)
        rand_lat, rand_plan = solver.solve_random_split(n_trials=50)
        print(f"[Random]   预估时延: {rand_lat:.2f} ms | 切分方案: {rand_plan}")

        # 5. Genetic Algorithm (GA)
        ga_lat, ga_plan = solver.solve_ga(pop_size=30, generations=100, mutation_rate=0.2)
        print(f"[GA]       预估时延: {ga_lat:.2f} ms | 切分方案: {ga_plan}")

        theoretical_csv = "theoretical_results.csv"
        with open(theoretical_csv, 'a', encoding='utf-8') as f:
            f.write(f"{task_id},LA-DP,{la_lat:.2f}\n")
            f.write(f"{task_id},Greedy,{greedy_lat:.2f}\n")
            f.write(f"{task_id},Greedy,{uniform_lat:.2f}\n")
            f.write(f"{task_id},BentPipe,{bp_lat:.2f}\n")
            f.write(f"{task_id},Random,{rand_lat:.2f}\n")
            f.write(f"{task_id},GA,{ga_lat:.2f}\n")
            
        # === 统一返回：所有算法的 plan（不含 latency）===
        return {
            "LA-DP": la_plan,
            "Greedy": greedy_plan,
            "uniform":uniform_plan,
            "BentPipe": bp_plan,
            "Random": rand_plan,
            "GA": ga_plan
        }

# ================= 测试代码 =================
if __name__ == "__main__":
    scheduler = Scheduler(net_config_path="config/network_config.json", models_config_path="config/model_profiles.json")
    best_plan = scheduler.generate_task_and_schedule(task_id="Task_001", model_name="yolov5")