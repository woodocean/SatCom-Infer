import argparse
import time
from core.node import SatelliteNode
import torch
from utils.data_utils import get_cifar10_batch
import os
import sys
import json

# 导入当前文件路径
sys.path.append(os.getcwd())

# 导入配置文件函数
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():

    # 创建参数读取对象
    parser = argparse.ArgumentParser(description='卫星协同推理仿真')
    # 只需要读取卫星的id，其余参数查表
    parser.add_argument('--id', required=True, help='节点ID')
    parser.add_argument('--net_cfg', default='config/network_config.json', help="网络拓扑配置文件")
    parser.add_argument('--task_cfg', default='config/task_config.json', help="任务配置文件")
    # 可选配置，遥感卫星才需要
    parser.add_argument('--run_task', required=False, help="要运行的任务名称 (如 parallel_task_demo)")

    args = parser.parse_args()

    # 1.加载网络配置
    try:
        net_config = load_config(args.net_cfg)
        my_config = net_config['nodes'][args.id]
    except KeyError:
        print(f"错误: 节点 [{args.id}] 未在配置文件 {args.net_cfg} 中定义！")
        return
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {args.net_cfg}")
        return

    # 2. 创建节点
    node = SatelliteNode(args.id, my_config["ip"], my_config["port"], my_config["role"])

    # 3. 查表组网
    if my_config["neighbors"]:
        neighbors_parsed = []
        for n_id in my_config["neighbors"]:
            if not net_config['nodes'][n_id]:
                print(f"警告: 邻居 [{n_id}] 未在配置中定义，跳过。")
                break
            else:
                n_info = net_config['nodes'][n_id]
                neighbors_parsed.append((n_id, n_info["ip"], n_info["port"]))
        node.join_network(neighbors_parsed)

    # 4. 启动
    node.start()

    # 5.任务触发逻辑 (仅遥感卫星)
    if my_config['role'] == 'remote_sensing' and args.run_task:
        time.sleep(3)  # 等待其他节点就绪

        # 加载任务配置
        task_config = load_config(args.task_cfg)
        if args.run_task not in task_config:
            print(f"错误！任务{args.run_task}未定义")
            return

        task = task_config[args.run_task]
        print(f"任务{args.run_task}已触发，任务为：{task['desc']}")

        data_path = './data/cifar-10-batches-py'
        try:
            images, labels = get_cifar10_batch(data_path, batch_size=8)
            print(f"   数据加载成功: {labels}")
        except Exception as e:
            print(f" 数据加载失败: {e}")
            return

        # -------流水线推理--------
        if task['type'] == 'pipeline':
            route = task['route']
            node.start_pip_task(route,images)
        # --------并行推理---------
        elif task['type'] == 'parallel':
            dist_map = {}
            dist_cfg = task['distribution']

            # 根据配置切分数据
            for target_node, range_idx in dist_cfg.items():
                start, end = range_idx
                if end > len(images): end = len(images)
                dist_map[target_node] = images[start:end]

            node.start_para_task(dist_map, agg_dest=task['agg_node'])
    # 6. 阻塞防止退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("停止节点")


if __name__ == "__main__":
    main()