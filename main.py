import argparse
import time
import json
import logging
from core.node import ComputeNode
from core.router_qos import RouterQoSClient
import sys
import os
import torch
import random  # 用于生成多样化任务

sys.path.append(os.getcwd())

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def update_network_topology(config_path):
    try:
        config = load_config(config_path)
        
        # 路由器登录凭测（如果要真实下发硬件的话请修改为真实 IP 和 root 密码）
        router_ip = "192.168.10.1"
        router_user = "root"
        router_password = "wslhy110"
        
        # 打开 QoS 客户端连接实例
        qos_client = RouterQoSClient(router_ip, router_user, router_password, ssh_timeout=3)
        
        for link_name, info in config['links'].items():
            # 1. 产生新的（或来自 STK 的）带宽与时延
            if "GS" in link_name: 
                new_bw = random.randint(100, 500)
                new_delay = random.uniform(30.0, 60.0) # 地面站高延迟
            else:
                new_bw = random.randint(100, 20000)
                new_delay = random.uniform(2.0, 15.0)  # 星间低延迟
                
            info['bandwidth_mbps'] = new_bw
            info['propagation_delay_ms'] = round(new_delay, 2)
            
            # 2. 从 JSON 里解析出你要操作的设备 IP
            dst_node_id = link_name.split("_to_")[-1]
            if dst_node_id in config['nodes']:
                dst_ip = config['nodes'][dst_node_id]['ip']
                
                # 只有非本机的远端路由才去下发硬件操作
                if not dst_ip.startswith("127.0.0.1") and not dst_ip.startswith("localhost"):
                    # 3. 硬件层面直接去刷入 netem 的传播延迟，不限带宽
                    qos_client.set_root_delay(dst_ip, delay_ms=int(new_delay))

        qos_client.close()
            
        tmp_path = config_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # =================【加入写冲突重试机制】=================
        for _ in range(10):  # 最多重试10次
            try:
                os.replace(tmp_path, config_path)
                break  # 替换成功，跳出循环
            except PermissionError:
                # 在Windows上，如果恰好别的节点在读，会报 PermissionError，稍微等几毫秒
                time.sleep(0.02)
        # ========================================================
        
    except Exception as e:
        print(f"更新带宽配置失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="Distributed Satellite Node")
    parser.add_argument('--id', type=str, required=True, help="节点ID，如 Sat_1, RS, GS")
    args = parser.parse_args()

    # 1. 直接固定加载网络拓扑配置
    net_config_path = 'config/network_config.json'
    try:
        net_config = load_config(net_config_path)
    except FileNotFoundError:
        print(f"Error: 找不到网络配置文件 {net_config_path}")
        return
    
    if args.id not in net_config['nodes']:
        print(f"Error: 节点 {args.id} 不在网络配置中！")
        return

    my_net_info = net_config['nodes'][args.id]

    print(f"--- 正在启动节点: {args.id} ---")

    # ==========================================================
    # 2. 一键实例化 ComputeNode 
    # ==========================================================
    node = ComputeNode(
        node_id=args.id, 
        ip=my_net_info['ip'],               
        port=my_net_info['port'],           
        role=my_net_info.get('role', 'Sat')
    )

    # 3. 加载计算引擎 (RS 节点本身不需要真跑模型)
    if "RS" not in args.id:
        print(f"[{args.id}] 检测为工作节点/地面站，初始化推理引擎...")
        node.load_model()                                        
    else:
        print(f"[{args.id}] 检测为遥感卫星RS，仅作为任务源。")

    # 4. 查表组网
    print(f"[{args.id}] 正在构建邻居路由表...")
    neighbors_parsed = []
    if "neighbors" in my_net_info:
        for neighbor_id in my_net_info["neighbors"]:
            if neighbor_id in net_config["nodes"]:
                n_info = net_config["nodes"][neighbor_id]
                neighbors_parsed.append((neighbor_id, n_info["ip"], n_info["port"]))
            else:
                print(f"  [警告] 邻居 {neighbor_id} 未在配置中定义！")
    
    node.join_network(neighbors_parsed)  # 批量注册邻居

    # 5. 启动网络监听
    node.start()

    # ==========================================================
    # 6. 任务触发逻辑 (仅遥感卫星 RS 执行)
    # ==========================================================
    if args.id == "RS":
        time.sleep(3) # 等待其他节点的 TCP 服务和心跳通信兵稳定建立
        print("\n" + "="*50)
        print("--- 发起 PMP 动态带宽 六大算法 对比实验 (100个任务) ---")
        print("="*50)
        
        from core.scheduler import Scheduler
        
        # 异构任务池：通过控制分辨率和 batch 模拟真实的复杂网络负荷
        model_pool = [ "swin_base"]
        batch_pool = [16,32]
        res_pool = {"yolov5": [(640, 640)],
                    "resnet101": [(224, 224)],
                    "vgg19": [(224, 224)],
                    "swin_base": [(224, 224)]}
        
        for i in range(0, 151):  # 运行150个任务演示
            task_id = f"Task_{i:03d}"
            
            # --- 步骤 A: 环境动态演进，并添加短暂的睡眠防止所有节点同时读写 ---
            update_network_topology(net_config_path)
            time.sleep(0.5) 
            
            # --- 步骤 B: 调度器感知更新 (注入真实的深度学习测绘档案) ---
            scheduler = Scheduler(
                net_config_path=net_config_path, 
                pc_profiles_path="config/dnn_profiles_database_pc.json",
                jetson_profiles_path="config/dnn_profiles_database_jetson.json"
            )

            # 每隔 50 个任务换一个模型，循环获取
            model_idx = (i // 50) % len(model_pool)
            chosen_model = model_pool[model_idx]
            
            chosen_bs = random.choice(batch_pool)
            chosen_res = random.choice(res_pool[chosen_model])
            
            print(f"\n[任务生成] ==> {task_id} | 模型: {chosen_model} | Batch: {chosen_bs} | 尺寸: {chosen_res[0]}x{chosen_res[1]}")
            
            # 调度器在内部测算所有 6 种算法，并写入实验 CSV 日志
            plans = scheduler.generate_task_and_schedule(
                task_id=task_id, 
                model_name=chosen_model,
                batch_size=chosen_bs,
                target_h=chosen_res[0],
                target_w=chosen_res[1]
            )
            
            # === [关键新增] 按照选择的排产分辨率，构建对应的物理尺寸随机矩阵 ===
            # 不要固定 [1,3,224,224] 了，否则体现不出异构网络下的通信瓶颈！
            fake_img = torch.randn(chosen_bs, 3, chosen_res[0], chosen_res[1])
            
            # --- 步骤 C: 顺序分发给管道，一次执行一个算法进行公平仿真 ---
            for alg, plan in plans.items():
                if plan is None:
                    print(f"  [RS] 🛑 算法 [{alg}] 当前由于硬件约束无解，直接跳过管道仿真。")
                    continue

                # 健壮路由获取：有的节点可能没有写 simulation_paths，如果没有，就按照物理串联取出来
                if "simulation_paths" in scheduler.net_config and "pipeline" in scheduler.net_config["simulation_paths"]:
                    ordered_route = scheduler.net_config["simulation_paths"]["pipeline"][1:] 
                else:
                    # 把排除了 RS 自己之外的其他节点均作为路由候选
                    ordered_route = [n["id"] for n in scheduler.net_config["nodes"] if "RS" not in n["id"]]

                # ==== 控制仿真节奏流控阻塞 ====
                # 检查 node.py 中定义的 task_ack_event，确保管道内只有一个任务在跑，避免堵车
                if hasattr(node, "task_ack_event"):
                    if not node.task_ack_event.is_set():
                        print(f"  [RS] ⏳ 管道占用中: 正在等待前序算法应答...")
                    
                    node.task_ack_event.wait()   # 等待收到地面的 ACK 后置位释放
                    node.task_ack_event.clear()  # 手动关闭闸门，准备装弹
                
                time.sleep(0.5) # 微调睡眠确保底层 Socket 缓冲区准备就绪
                
                print(f"  [RS] 🚀 正在向网络下发任务: [{task_id}] | 驱动策略: {alg}")
                
                # 重新打包你的任务载荷，把精准制导的层级策略一并塞进去
                rs_payload = {
                    'mode': 'PMP',
                    'task_id': task_id,
                    'algorithm': alg,
                    'model_name': chosen_model,
                    'accumulated_latency': 0.0,
                    'tensor': fake_img,    # 会通过 Pickle 被压进网络传输
                    'batch': chosen_bs,   
                    'route': ordered_route,   
                    'layer_plan': plan     
                }

                # 模拟系统自身触发，正式倒进网络首个计算节点
                node.handle_message({
                    'type': 'NEW_TASK',
                    'src': 'system_trigger',
                    'payload': rs_payload
                })

        print("\n🎉 [RS] 第一阶段：100组对比实验所有任务已经全部轰炸分发完毕。")
        print("🎉 [RS] 你可以前往目录查看自动生成的 `experiment_results.csv`。")

    # ========== 7. 保持进程循环以维持监听 ==========
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{args.id}] 正在安全终止节点...")
        node.stop()
        
if __name__ == "__main__":
    # 为了避免输出过于眼花缭乱，这里将日志限制在 INFO 级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    main()