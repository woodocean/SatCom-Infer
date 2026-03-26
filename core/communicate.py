import socket
import struct
import pickle
import threading
import time
import torch

class Communicator:
    """
    通信模块 - 支持PC/Jetson跨公网真实通信
    协议: [4字节长度头][pickle序列化数据]
    """

    def __init__(self, node_id, listen_ip, listen_port, simulate_bw=False):
        self.node_id = node_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.simulate_bw = simulate_bw          # 这里是模拟的与邻居的带宽值，硬件的带宽是可以改，但目前是只能在路由器改，没有必要，
                                                # 直接在这里按比例缩放传输时延即可

        self.peers = {}            # {peer_id: (ip, port)}  邻居的套接字
        self.simulated_bw = {}     # {peer_id: Mbps}        与邻居的信道带宽
        self.server_socket = None                           
        self.running = False                                # 该节点的通信模块是否运行中
        self.handler = None        # 外部注册的消息回调函数    

    # ===================== 组网 =====================
    def register_peer(self, peer_id, ip, port):
        self.peers[peer_id] = (ip, port)
        # 公网带宽模型，这个模型一定要调研具体带宽并修改的，目前ISL其实可以定为1Gbps - 20Gbps 看到文献是这么大 应该没问题 GSL还需要调研
        if self.simulate_bw:
            self.simulated_bw[peer_id] = self._estimate_link_bw(peer_id)

    def _estimate_link_bw(self, peer_id):
        """基于节点ID推断链路带宽 (Mbps)"""
        # SAT之间: 星间链路 ~50Mbps (公网模拟)
        # SAT-GS/RS: 星地链路 ~80Mbps (公网模拟)
        if 'SAT' in self.node_id and 'SAT' in peer_id:
            return 50.0
        return 80.0

    # ===================== 发送 =====================
    def send_message(self, target_id, msg_type, payload):
        """
        兼容旧接口: send_message(target_id, msg_type, payload_dict)
        """
        message = {
            'type': msg_type,
            'src': self.node_id,
            'payload': payload,
            'timestamp': time.time()
        }
        return self._send_raw(target_id, message)

    def send_tensor(self, target_id, msg_type, tensor, metadata=None):
        """
        发送Tensor专用 (大数据)
        """
        message = {
            'type': msg_type,
            'src': self.node_id,
            'tensor': tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        return self._send_raw(target_id, message)

    def _send_raw(self, target_id, message):
        if target_id not in self.peers:
            print(f"  [COMM] 未知目标: {target_id}")
            return False

        ip, port = self.peers[target_id]
        payload = pickle.dumps(message)
        data_mb = len(payload) / (1024 * 1024)

        # 带宽模拟
        if self.simulate_bw and target_id in self.simulated_bw:
            bw = self.simulated_bw[target_id]
            delay = (data_mb * 8) / bw
            print(f"  [BW-SIM] {self.node_id}->{target_id}: "
                  f"{data_mb:.3f}MB @ {bw}Mbps, delay={delay:.3f}s")
            time.sleep(delay)

        # TCP发送 (带重试)
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.connect((ip, port))

                # 发送: [4字节长度][数据]
                header = struct.pack('!I', len(payload))
                sock.sendall(header + payload)

                # 等ACK
                ack = sock.recv(3)
                sock.close()

                if ack == b'ACK':
                    print(f"  [COMM] {self.node_id}->{target_id} 发送成功 ({data_mb:.3f}MB)")
                    return True
                else:
                    print(f"  [COMM] {self.node_id}->{target_id} ACK异常")
                    return False

            except Exception as e:
                print(f"  [COMM] {self.node_id}->{target_id} 第{attempt+1}次失败: {e}")
                time.sleep(2)

        print(f"  [COMM] {self.node_id}->{target_id} 发送彻底失败!")
        return False

    # ===================== 监听 =====================
    def start_listening(self, handler_callback):
        self.handler = handler_callback
        self.running = True

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         #创建套接字对象
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)       
        self.server_socket.bind(('0.0.0.0', self.listen_port))                          #监听本机的接收端口
        self.server_socket.listen(10)

        thread = threading.Thread(target=self._listen_loop, daemon=True)                
        thread.start()                                                                 #创建持续监听的线程，
        print(f"[{self.node_id}] 监听启动 @ 0.0.0.0:{self.listen_port}")

    def _listen_loop(self):
        while self.running:
            try:
                self.server_socket.settimeout(2.0)
                conn, addr = self.server_socket.accept()                                # 一旦建立TCP连接并接收到信息，就返回tcp连接对象 以及 连接对象的地址（套接字）
                t = threading.Thread(target=self._handle_conn, args=(conn, addr), daemon=True) # 同时创建并启动解读信息的线程，该函数内有接收消息的回调函数
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_conn(self, conn, addr):
        try:
            # 读4字节长度头
            raw_len = self._recv_exact(conn, 4)
            if not raw_len:
                return
            msg_len = struct.unpack('!I', raw_len)[0]

            # 读数据体
            raw_data = self._recv_exact(conn, msg_len)
            if not raw_data:
                return

            # 发ACK
            conn.sendall(b'ACK')

            # 反序列化并回调
            message = pickle.loads(raw_data)
            if self.handler:
                self.handler(message)       

        except Exception as e:
            print(f"  [COMM] 接收处理异常: {e}")
        finally:
            conn.close()

    def _recv_exact(self, conn, num_bytes):
        buf = bytearray()
        while len(buf) < num_bytes:
            try:
                chunk = conn.recv(min(num_bytes - len(buf), 65536))
                if not chunk:
                    return None
                buf.extend(chunk)
            except socket.timeout:
                return None
        return bytes(buf)

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass