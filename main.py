import argparse
import time
import json
import logging
from core.node import ComputeNode
import sys
import os
import torch
import random  # <--- 新增随机库，用于生成多样化任务

sys.path.append(os.getcwd())

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def update_network_topology(config_path):
    try:
        config = load_config(config_path)
        
        for link_name, info in config['links'].items():
            if "GS" in link_name: 
                new_bw = random.randint(100, 500)
            else:
                new_bw = random.randint(1000, 20000)
            info['bandwidth_mbps'] = new_bw
            
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
    # 删除了冗余的 net_cfg, dev_cfg, task_cfg 参数
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
        # 移除了单调写死的 model_name=model_name，交给后续按任务动态处理
    )

    # 3. 加载计算引擎 (RS 节点本身不需要真跑模型)
    if "RS" not in args.id:
        print(f"[{args.id}] 检测为工作节点/地面站，初始化推理引擎...")
        node.load_model() # 如果你的引擎支持动态换模型，这里可能只需要预热                                        
    else:
        print(f"[{args.id}] 检测为遥感卫星RS，仅作为任务源。")

    # 4. 查表组网 (调用 node 内置的组网函数)
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
        time.sleep(3) # 等待网络环境稳定
        print("\n" + "="*50)
        print("--- 发起 PMP 动态带宽对比实验 (100个任务) ---")
        print("="*50)
        
        from core.scheduler import Scheduler
        
        model_pool = ["vgg19"] # 这里可以扩展模型
        batch_pool = [1]       # 锁死 Batch=1 观察带宽影响
        
        for i in range(1, 101):
            task_id = f"Task_{i:03d}"
            
            # --- 步骤 A: 环境动态演进 ---
            # 每一轮任务前随机修改带宽配置文件
            update_network_topology(net_config_path)
            
            # --- 步骤 B: 调度器感知更新 ---
            # 必须重新实例化，Scheduler 内部才会读取刚才被修改的 JSON
            scheduler = Scheduler(
                net_config_path=net_config_path, 
                models_config_path="config/model_profiles.json",
                sizes_fit_path="config/model_profiles_sizes.json"
            )

            chosen_model = random.choice(model_pool)
            chosen_bs = random.choice(batch_pool)
            
            print(f"\n[任务生成] {task_id} | 模型: {chosen_model} | 尺寸: 224×224")
            
            # 计算方案
            fake_img = torch.randn(chosen_bs, 3, 224, 224)
            plans = scheduler.generate_task_and_schedule(task_id=task_id, model_name=chosen_model)
            
        #     # 依次执行三种算法进行对比
        #     for alg, plan in plans.items():
        #         ordered_route = scheduler.net_config["simulation_paths"]["pipeline"][1:] 

        #         # --- 步骤 C: 流控阻塞 ---
        #         # 检查 node.py 中定义的 task_ack_event，确保管道清空
        #         if not node.task_ack_event.is_set():
        #             print(f"[RS] 🛑 阻塞排队: 等待前序算法 [{alg}] 的 ACK 应答...")
                
        #         node.task_ack_event.wait()   # 等待置位
        #         node.task_ack_event.clear()  # 手动复位（关门）
                
        #         # 微调睡眠确保 Socket 缓冲区就绪
        #         # time.sleep(0.3)
                
        #         print(f"[RS] 🚀 下发 {task_id} | 算法: {alg}")
        #         # print(f"     -> [带宽路由]: {ordered_route}")
                
        #         rs_payload = {
        #             'mode': 'PMP',
        #             'task_id': task_id,
        #             'algorithm': alg,
        #             'model_name': chosen_model,
        #             'accumulated_latency': 0.0,
        #             'tensor': fake_img, 
        #             'batch': chosen_bs,   
        #             'route': ordered_route,   
        #             'layer_plan': plan     
        #         }

        #         # 模拟系统触发，正式流进计算管道
        #         node.handle_message({
        #             'type': 'NEW_TASK',
        #             'src': 'system_trigger',
        #             'payload': rs_payload
        #         })

        # print("\n[RS] 所有对比实验任务已配给完毕。")

    # ========== 7. 保持运行 ==========
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{args.id}] 正在安全关闭...")
        node.stop()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    main()