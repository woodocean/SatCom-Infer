import argparse
import time
import torch
from satellite_node import SatelliteNode, SatelliteType


def main():
    parser = argparse.ArgumentParser(description='卫星节点API')
    parser.add_argument('--node_id', type=str, required=True, help='卫星节点ID')
    parser.add_argument('--satellite_type', type=str, required=True, choices=['remote_sensing', 'leo_computing'],
                        help='卫星类型')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='本节点IP')
    parser.add_argument('--port', type=int, required=True, help='本节点端口')
    parser.add_argument('--compute_capacity', type=float, required=True, help='计算能力')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')

    # 地面站参数
    parser.add_argument('--ground_station_id', type=str, help='地面站ID')
    parser.add_argument('--ground_station_ip', type=str, help='地面站IP')
    parser.add_argument('--ground_station_port', type=int, help='地面站端口')

    # 邻居节点参数
    parser.add_argument('--neighbors', type=str, nargs='+', help='邻居节点列表 ip:port')

    args = parser.parse_args()

    # 创建卫星节点
    sat_type = SatelliteType.REMOTE_SENSING if args.satellite_type == 'remote_sensing' else SatelliteType.LEO_COMPUTING
    satellite = SatelliteNode(
        node_id=args.node_id,
        satellite_type=sat_type,
        ip=args.ip,
        port=args.port,
        compute_capacity=args.compute_capacity,
        device=args.device
    )

    # 添加地面站
    if args.ground_station_id:
        satellite.add_ground_station(args.ground_station_id, {
            'ip': args.ground_station_ip,
            'port': args.ground_station_port,
            'bandwidth': 200.0,
            'latency': 50.0
        })

    # 启动发现服务
    satellite.start_discovery_service()

    # 启动任务服务
    satellite.start_task_service()

    print(
        f"卫星节点 {args.node_id} 启动完成，监听 {args.ip}:{args.port}(发现服务) 和 {args.ip}:{args.port + 1000}(任务服务)")

    # 连接邻居节点（只保留一次）
    if args.neighbors:
        print(f"🔗 开始连接邻居节点...")
        for neighbor in args.neighbors:
            neighbor_ip, neighbor_port = neighbor.split(':')
            success = satellite.discover_neighbor(neighbor_ip, int(neighbor_port))
            if success:
                print(f"   ✅ 成功连接到 {neighbor_ip}:{neighbor_port}")
            else:
                print(f"   ❌ 连接失败 {neighbor_ip}:{neighbor_port}")
            time.sleep(0.5)

        # 打印网络拓扑
        time.sleep(1)
        satellite.print_network_info()

    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"卫星节点 {args.node_id} 关闭")


if __name__ == "__main__":
    main()