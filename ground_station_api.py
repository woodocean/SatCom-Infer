import argparse
import socket
import pickle
import threading
import time


class GroundStation:
    def __init__(self, station_id, ip, port):
        self.station_id = station_id
        self.ip = ip
        self.port = port
        self.received_results = []

    def start_service(self):
        def handler():
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.ip, self.port))
            server_socket.listen(5)

            print(f"地面站 {self.station_id} 启动在 {self.ip}:{self.port}")

            while True:
                conn, addr = server_socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr)).start()

        thread = threading.Thread(target=handler, daemon=True)
        thread.start()
        return thread

    def handle_connection(self, conn, addr):
        try:
            data = conn.recv(4096)
            if data:
                result = pickle.loads(data)
                self.received_results.append(result)
                print(f"地面站收到推理结果: {result}")
                conn.send(b"ACK")
        except Exception as e:
            print(f"地面站处理连接错误: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description='地面站API')
    parser.add_argument('--station_id', type=str, required=True, help='地面站ID')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='地面站IP')
    parser.add_argument('--port', type=int, required=True, help='地面站端口')

    args = parser.parse_args()

    ground_station = GroundStation(args.station_id, args.ip, args.port)
    ground_station.start_service()

    print(f"地面站 {args.station_id} 启动完成")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"地面站 {args.station_id} 关闭")


if __name__ == "__main__":
    main()