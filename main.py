import argparse
import time
from core.node import SatelliteNode
import torch

def main():
    parser = argparse.ArgumentParser(description='卫星协同推理仿真')
    parser.add_argument('--id', required=True, help='节点ID')
    parser.add_argument('--port', type=int, required=True, help='端口')
    parser.add_argument('--role', default='leo_computing', help='角色')
    parser.add_argument('--neighbors', nargs='+', help='邻居列表 ID:IP:PORT')

    args = parser.parse_args()

    # 1. 创建节点
    node = SatelliteNode(args.id, "127.0.0.1", args.port, args.role)

    # 2. 添加邻居
    if args.neighbors:
        neighbors_parsed = []
        for n in args.neighbors:
            nid, nip, nport = n.split(':')
            neighbors_parsed.append((nid, nip, int(nport)))
        node.join_network(neighbors_parsed)

    # 3. 启动
    node.start()

    # 4. 仅遥感卫星触发测试逻辑
    if args.role == 'remote_sensing':
        time.sleep(3)  # 等待全网就绪

        print("\n=== 启动在轨聚合(On-Orbit Aggregation)测试 ===")

        # 生成两份数据，各算8张图
        fake_image_1 = torch.randn(8, 3, 224, 224)
        fake_image_2 = torch.randn(8, 3, 224, 224)
        # 任务分配：把数据分给 01 和 02
        dist_map = {
            "SAT-01": fake_image_1,
            "SAT-02": fake_image_2
        }

        # 关键点：指定聚合节点是 'SAT-AGG'，而不是直接发给 'GS'
        # 流程：RS -> 01/02 -> SAT-AGG (聚合) -> GS (落盘)
        node.start_para_task(distribute_map=dist_map, agg_dest="SAT-AGG")

    # 5. 阻塞防止退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("停止节点")


if __name__ == "__main__":
    main()