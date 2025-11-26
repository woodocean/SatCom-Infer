import argparse
import socket
import pickle
import threading
import time
from satellite_node import SatelliteNode, SatelliteType


class GroundStationNode(SatelliteNode):
    def __init__(self, station_id, ip, port, compute_capacity=100.0, device="cuda"):
        # 地面站类型可以新定义一个，或者使用现有的
        super().__init__(station_id, SatelliteType.LEO_COMPUTING, ip, port, compute_capacity, device)
        self.station_id = station_id
        self.received_results = []

    def get_info(self):
        """重写get_info，标识为地面站"""
        info = super().get_info()
        info['type'] = 'ground_station'  # 特殊标识
        return info

    def start_service(self):
        """启动地面站服务（包含发现服务和任务服务）"""
        # 启动卫星节点的标准服务
        self.start_discovery_service()
        self.start_task_service()

        # 同时启动专门的结果接收服务
        def result_handler():
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.ip, self.port))
            server_socket.listen(5)

            print(f"地面站 {self.station_id} 启动在 {self.ip}:{self.port}")

            while True:
                conn, addr = server_socket.accept()
                threading.Thread(target=self.handle_result_connection, args=(conn, addr)).start()

        thread = threading.Thread(target=result_handler, daemon=True)
        thread.start()
        return thread

    def handle_result_connection(self, conn, addr):
        """处理最终结果接收"""
        try:
            data = conn.recv(4096)
            if data:
                result = pickle.loads(data)
                self.received_results.append(result)
                print(f"地面站收到最终推理结果: {result}")
                conn.send(b"ACK")
        except Exception as e:
            print(f"地面站处理结果连接错误: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description='地面站API')
    parser.add_argument('--station_id', type=str, required=True, help='地面站ID')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='地面站IP')
    parser.add_argument('--port', type=int, required=True, help='地面站端口')
    parser.add_argument('--compute_capacity', type=float, default=100.0, help='计算能力')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='计算设备')

    args = parser.parse_args()

    # 创建地面站节点（高算力）
    ground_station = GroundStationNode(
        args.station_id,
        args.ip,
        args.port,
        compute_capacity=args.compute_capacity,
        device=args.device
    )

    # 启动服务
    ground_station.start_service()

    print(f"地面站 {args.station_id} 启动完成，算力: {args.compute_capacity}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"地面站 {args.station_id} 关闭")


if __name__ == "__main__":
    main()