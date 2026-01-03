import socket
import struct
import pickle
import threading
import time


class CommManager:
    def __init__(self, host, port, node_id):
        self.host = host                # 主机ip
        self.port = port                # 进程所在端口
        self.node_id = node_id          # 节点名
        self.running = False            # 是否在忙
        self.socket = None
        # 消息回调函数字典： { 'msg_type': callback_function }
        self.handlers = {}
        self.neighbors = {}  # {node_id: (ip, port)} 相邻节点字典

    def start_server(self):
        """启动监听服务"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        '''表示创建socket对象，socket.AF_INET表示用IPv4地址，
            # socket.SOCK_STREAM表示用TCP协议的套接字
            # 其中self.socket指的是CommManager类的一个成员变量
            # socket.socket前一个socket指的是socket模块，后一个则是创建socket的函数'''
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 这里是说可端口复用？
        self.socket.bind((self.host, self.port))                            # 绑定ip 端口到该socket对象
        self.socket.listen(5)                                               # 这里是说排队区（Backlog）只能容纳 5
        self.running = True

        print(f"[{self.node_id}] 通信服务启动于 {self.host}:{self.port}")

        # 开启接受线程
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def register_handler(self, msg_type, handler):
        """注册消息类型的处理函数"""
        self.handlers[msg_type] = handler

    def add_neighbor(self, neighbor_id, ip, port):
        self.neighbors[neighbor_id] = (ip, int(port))

    def send_message(self, target_node_id, msg_type, payload=None):
        """主动发送消息"""
        if target_node_id not in self.neighbors:
            print(f"未知邻居: {target_node_id}")
            return False

        ip, port = self.neighbors[target_node_id]       # 目的ip端口

        # 这里的message就相当于TCP数据报,因为这个数据包只包含需要传递的信息，而不需要理会IP层的封装
        # 而这里的数据报包含数据发送方-节点的id、消息的类型-告诉对方应该如何处理该消息（调用什么函数）、以及该消息的负载信息、以及时间戳
        message = {
            "source": self.node_id,
            "type": msg_type,
            "payload": payload,
            "timestamp": time.time()
        }

        try:
            # 序列化
            data = pickle.dumps(message)
            # 封装：4字节长度头 + 数据体
            packet = struct.pack('>I', len(data)) + data    # '>I' 大端序> 无符号整数Int 即把data长度固定放在数据包的前4字节

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)  # 连接超时
                s.connect((ip, port))   # 连接上这个目的进程
                s.sendall(packet)       # 发包
            return True
        except Exception as e:
            print(f"发送给 {target_node_id} 失败: {e}")
            return False

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.socket.accept()
                # 为每个连接启动一个线程处理，处理完即关闭（短连接模式，适合简单信令）
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
            except Exception as e:
                if self.running:
                    print(f"监听错误: {e}")

    def _handle_connection(self, conn):
        try:
            # 1. 读取4字节头部（消息长度）
            raw_len = self._recv_all(conn, 4)
            if not raw_len: return
            msg_len = struct.unpack('>I', raw_len)[0]

            # 2. 读取消息体
            raw_data = self._recv_all(conn, msg_len)
            if not raw_data: return

            # 3. 反序列化
            message = pickle.loads(raw_data)
            msg_type = message.get("type")

            # 4. 路由到对应的处理函数
            if msg_type in self.handlers:
                # 调用回调函数
                self.handlers[msg_type](message)
            else:
                print(f"未知消息类型: {msg_type}")

        except Exception as e:
            print(f"连接处理错误: {e}")
        finally:
            conn.close()

    def _recv_all(self, conn, n):
        """辅助函数：确保接收指定字节数"""
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet: return None
            data += packet
        return data