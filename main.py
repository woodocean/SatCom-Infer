import argparse
import time
import json
import logging
from core.node import ComputeNode
import sys
import os

sys.path.append(os.getcwd())

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Distributed Satellite Node")
    parser.add_argument('--id', type=str, required=True, help="节点ID，如 Sat_1, RS, GS")
    parser.add_argument('--model-name', type=str, default='swin_base', help="Initial model loaded by compute nodes")
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
        role=my_net_info.get('role', 'Sat'),
        model_name=args.model_name,
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

    if args.id == "RS":
        print("[RS] 已启动为纯节点进程。请使用 experiments_runner.py 执行实验编排。")

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
