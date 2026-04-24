import json
import os
import csv
from datetime import datetime
from algorithms.pmp_solver import PMPSolver

class Scheduler:
    def __init__(self, net_config_path="config/network_config.json", 
                 pc_profiles_path="config/dnn_profiles_database_pc.json", 
                 jetson_profiles_path="config/dnn_profiles_database_jetson.json"):
        # 1. 加载卫星网络配置
        if not os.path.exists(net_config_path):
            raise FileNotFoundError(f"找不到网络配置文件: {net_config_path}")
        with open(net_config_path, 'r', encoding='utf-8') as f:
            self.net_config = json.load(f)
            
        # 2. 根据设备类型加载模型物理测绘数据库
        self.dnn_profiles = {}
        
        if not os.path.exists(pc_profiles_path):
            raise FileNotFoundError(f"找不到 PC 配置文件: {pc_profiles_path}")
        with open(pc_profiles_path, 'r', encoding='utf-8') as f:
            self.dnn_profiles["pc"] = json.load(f)
            
        if not os.path.exists(jetson_profiles_path):
            raise FileNotFoundError(f"找不到 Jetson 配置文件: {jetson_profiles_path}")
        with open(jetson_profiles_path, 'r', encoding='utf-8') as f:
            self.dnn_profiles["jetson"] = json.load(f)

    def _extract_bw_metrics(self, raw_links):
        """从链路配置中提取星间/星地带宽统计，用于标准化结果表。"""
        isl_bws = []
        gsl_bws = []

        for link_name, info in raw_links.items():
            bw = float(info.get("bandwidth_mbps", 0.0))
            if "GS" in link_name:
                gsl_bws.append(bw)
            else:
                isl_bws.append(bw)

        isl_avg = sum(isl_bws) / len(isl_bws) if isl_bws else 0.0
        gsl_avg = sum(gsl_bws) / len(gsl_bws) if gsl_bws else 0.0
        return isl_avg, gsl_avg

    def _append_standardized_theory_rows(
        self,
        task_id,
        model_name,
        batch_size,
        target_h,
        target_w,
        plans,
        isl_avg_bw,
        gsl_avg_bw,
        run_id,
        exp_type,
        mode,
        output_csv,
    ):
        """将理论调度结果写入标准化长表，便于统一分析与绘图。"""
        file_exists = os.path.isfile(output_csv)
        timestamp = datetime.now().isoformat(timespec="seconds")

        # 归一化基线：同任务下 GS-Only 的时延
        gs_only_latency = plans.get("GS-Only", {}).get("latency", float("inf"))
        use_norm = gs_only_latency not in (None, float("inf"), 0.0)

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "run_id",
                    "exp_type",
                    "mode",
                    "task_id",
                    "algorithm",
                    "model_name",
                    "batch_size",
                    "input_h",
                    "input_w",
                    "isl_avg_bw_mbps",
                    "gsl_avg_bw_mbps",
                    "latency_ms",
                    "norm_latency_vs_gs",
                    "timestamp",
                ])

            for alg_name, data in plans.items():
                latency = data.get("latency", float("inf"))
                if use_norm and latency != float("inf"):
                    norm_latency = latency / gs_only_latency
                else:
                    norm_latency = ""

                writer.writerow([
                    run_id,
                    exp_type,
                    mode,
                    task_id,
                    alg_name,
                    model_name,
                    batch_size,
                    target_h,
                    target_w,
                    f"{isl_avg_bw:.4f}",
                    f"{gsl_avg_bw:.4f}",
                    latency,
                    norm_latency,
                    timestamp,
                ])

    def generate_task_and_schedule(
        self,
        task_id="task_001",
        model_name="yolov5",
        batch_size=32,
        target_h=640,
        target_w=640,
        run_id="default",
        exp_type="algo_effectiveness",
        mode="theory",
        standardized_csv_file="results_long.csv",
        persist_theory=True,
    ):
        # print(f"\n[{task_id}] 接收任务: {model_name} | 规格: b{batch_size}_{target_h}x{target_w}")
        
        # ================= 1. 物理档案查表提取 =================
        config_key = f"b{batch_size}_{target_h}x{target_w}"
        
        layers_dict = {"pc": [], "jetson": []}
        for device in ["pc", "jetson"]:
            if model_name not in self.dnn_profiles[device] or config_key not in self.dnn_profiles[device][model_name]:
                raise KeyError(f"数据库({device})中缺少配置: {model_name} -> {config_key}，请先运行 validator 进行测绘！")
            raw_profile = self.dnn_profiles[device][model_name][config_key]
            
            for i in range(len(raw_profile)):
                if str(i) in raw_profile:
                    layers_dict[device].append(raw_profile[str(i)])
            
        # 计算原始输入体积 (MB): Batch * C(3) * H * W * 4Bytes / 1024^2
        input_mb = (batch_size * 3 * target_h * target_w * 4) / (1024 ** 2)

        model_profile = {
            "layers": dict(layers_dict), # 将不同设备的layers字典传给算法
            "input_size_raw": input_mb
        }

        # ================= 2. 环境状态组装 (修复 KeyError 的核心逻辑) =================
        raw_nodes = self.net_config.get("nodes", {})
        raw_links = self.net_config.get("links", {})
        
        # [核心] 从 JSON 字典中按顺序提取 "参与计算" 的节点，必须过滤掉 RS 任务节点
        if "simulation_paths" in self.net_config and "pipeline" in self.net_config["simulation_paths"]:
            compute_node_ids = self.net_config["simulation_paths"]["pipeline"][1:] 
        else:
            compute_node_ids = [nid for nid in raw_nodes.keys() if "RS" not in nid]

        # 组装算法能够直接吃的标准列表结构 List[Dict]
        nodes = []
        for nid in compute_node_ids:
            node_info = raw_nodes[nid].copy()
            node_info["id"] = nid
            nodes.append(node_info)
        
        # 构建逐跳带宽向量 B (Mbps)
        bandwidth_list = []
        propagation_delay_list = []
        # 获取源节点 ID (通常是 RS)
        current_source = self.net_config["simulation_paths"]["pipeline"][0] if "simulation_paths" in self.net_config else "RS"

        for i in range(len(nodes)):
            target_node = nodes[i]["id"]
            bw = 100.0  # 默认保底带宽
            prop_ms = 0.0  # 默认保底传播时延
            
            # 使用字符串匹配 JSON 中的 Key
            forward_key = f"{current_source}_to_{target_node}"
            backward_key = f"{target_node}_to_{current_source}"
            
            if forward_key in raw_links:
                bw = float(raw_links[forward_key].get("bandwidth_mbps", 100.0))
                prop_ms = float(raw_links[forward_key].get("propagation_delay_ms", 0.0))
            elif backward_key in raw_links:
                bw = float(raw_links[backward_key].get("bandwidth_mbps", 100.0))
                prop_ms = float(raw_links[backward_key].get("propagation_delay_ms", 0.0))
            
            bandwidth_list.append(bw)
            propagation_delay_list.append(prop_ms)
            current_source = target_node # 移动指针，下一跳的源是当前节点

        env_status = {
            "nodes": nodes,
            "bandwidths": bandwidth_list,
            "propagation_delays_ms": propagation_delay_list,
            "reference_compute_speed": self.net_config.get("reference_compute_speed", 100.0)
        }

        # 为标准化结果表提取带宽变量
        isl_avg_bw, gsl_avg_bw = self._extract_bw_metrics(raw_links)

        # ================= 3. 执行六大算法 (对标实验核心) =================
        solver = PMPSolver(model_profile, env_status)
        plans = {}

        try:
            la_lat, la_plan = solver.solve_la_dp()
            plans["LA-DP"] = {"plan": la_plan, "latency": la_lat}
        except Exception as e: plans["LA-DP"] = {"plan": None, "latency": float('inf')}

        try:
            gr_lat, gr_plan = solver.solve_communication_greedy()
            plans["Greedy"] = {"plan": gr_plan, "latency": gr_lat}
        except Exception as e: plans["Greedy"] = {"plan": None, "latency": float('inf')}

        try:
            un_lat, un_plan = solver.solve_uniform_partition()
            plans["Uniform"] = {"plan": un_plan, "latency": un_lat}
        except Exception as e: plans["Uniform"] = {"plan": None, "latency": float('inf')}

        try:
            rd_lat, rd_plan = solver.solve_random_split(n_trials=1)
            plans["Random"] = {"plan": rd_plan, "latency": rd_lat}
        except Exception as e: plans["Random"] = {"plan": None, "latency": float('inf')}

        try:
            bp_lat, bp_plan = solver.solve_bent_pipe()
            plans["GS-Only"] = {"plan": bp_plan, "latency": bp_lat}
        except Exception as e: plans["GS-Only"] = {"plan": None, "latency": float('inf')}

        try:
            ga_lat, ga_plan = solver.solve_ga(pop_size=30, generations=500, mutation_rate=0.2)
            plans["GA"] = {"plan": ga_plan, "latency": ga_lat}
        except Exception as e: plans["GA"] = {"plan": None, "latency": float('inf')}

        # ================= 4. 终端打印与日志记录 =================
        print(f"\n--- 调度结果汇总 ({task_id}) ---")
        for name, data in plans.items():
            lat_str = f"{data['latency']:.2f} ms" if data['latency'] != float('inf') else "无解/越界"
            print(f"| {name.ljust(15)} | 预计时延: {lat_str.ljust(12)} | 层级决策: {data['plan']}")

        if persist_theory:
            # 记录到 CSV 中
            csv_file = "theoretical_results.csv"
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # if not file_exists:
                #     writer.writerow(["TaskID", "Model", "Algorithm", "Latency_ms"])
                for name, data in plans.items():
                    writer.writerow([task_id,name, data['latency']])

            # 双写：新增标准化长表输出，不影响旧流程读取。
            self._append_standardized_theory_rows(
                task_id=task_id,
                model_name=model_name,
                batch_size=batch_size,
                target_h=target_h,
                target_w=target_w,
                plans=plans,
                isl_avg_bw=isl_avg_bw,
                gsl_avg_bw=gsl_avg_bw,
                run_id=run_id,
                exp_type=exp_type,
                mode=mode,
                output_csv=standardized_csv_file,
            )

        return {k: v['plan'] for k, v in plans.items()}