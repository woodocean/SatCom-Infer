import socket
import struct
import pickle
import threading
import time
import torch
import random
import os

CHUNK_SIZE = 60000  # UDP MTU 限制下的安全分块大小 (不要超过 65507)

class Communicator:
    """
    通信模块 - 基于极速可靠 UDP 重构版本
    弃用 TCP 及 TC 层面的带宽限制，最大化吞吐，仅由算法测算分离【传播时延】与【传输时延】。
    """

    def __init__(self, node_id, listen_ip, listen_port, simulate_bw=False):
        self.node_id = node_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.simulate_bw = simulate_bw          

        self.peers = {}            # {peer_id: (ip, port)}
        self.simulated_bw = {}     
        self.server_socket = None                           
        self.running = False                                
        self.handler = None        
        
        # ===== UDP 分包重组缓存区 =====
        self.recv_buffers = {}
        self.buf_lock = threading.Lock()

    # ===================== 组网 =====================
    def register_peer(self, peer_id, ip, port):
        self.peers[peer_id] = (ip, port)
        if self.simulate_bw:
            self.simulated_bw[peer_id] = self._estimate_link_bw(peer_id)

    def _estimate_link_bw(self, peer_id):
        if 'SAT' in self.node_id and 'SAT' in peer_id:
            return 50.0
        return 80.0

    # ===================== 发送 =====================
    def send_message(self, target_id, msg_type, payload):
        message = {
            'type': msg_type,
            'src': self.node_id,
            'payload': payload,
            'timestamp': time.time()
        }
        return self._send_raw(target_id, message)

    def send_tensor(self, target_id, msg_type, tensor, metadata=None):
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
            return False, 0.0

        ip, port = self.peers[target_id]
        payload = pickle.dumps(message)
        total_len = len(payload)
        data_mb = total_len / (1024 * 1024)

        # 1. 对数据打成碎片
        chunks = [payload[i:i + CHUNK_SIZE] for i in range(0, total_len, CHUNK_SIZE)]
        num_chunks = len(chunks)

        # 2. 建立发送端 UDP 套接字并增加发送信道缓存
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Windows/Linux 通杀的大发包缓冲配置
        try:
            udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
        except OSError:
            pass  # 有的系统不允许设这么大，忽略即可
            
        udp_sock.settimeout(1.5)  # 单次超时上限 (包含往返时延)

        msg_id = random.randint(0, 65535)
        
        # 获取物理测速硬件的峰值带宽基准 (您那边 Iperf UDP 能跑到 450Mbps)
        # 以此用于分离测算“传输”与“传播”时延
        hw_bandwidth_mbps = getattr(self, 'hardware_max_bw', 450.0) 

        start_t = time.perf_counter()

        # UDP 发送轮次 (带超时重发整个包块保障)
        MAX_RETRIES = 10
        for attempt in range(MAX_RETRIES):
            # A. 高速突发注入并进行防丢包节奏控制
            for i, chunk in enumerate(chunks):
                # 头部协议 -> msg_id: (2 bytes), 总数: (2 bytes), 序号: (2 bytes) = 6 Bytes
                header = struct.pack("!HHH", msg_id, num_chunks, i)
                udp_sock.sendto(header + chunk, (ip, port))
                
                # 减缓路由器交换队列溢出导致的高丢包率 (1.2% 是可接受底线)
                if i % 20 == 0:
                    time.sleep(0.0005)

            # B. 阻塞等待接收端所有块重组完成后的 DONE 应答
            try:
                ack_data, _ = udp_sock.recvfrom(1024)
                if ack_data == b'DONE':
                    end_t = time.perf_counter()
                    real_time = end_t - start_t
                    
                    # =============== 时延分离与测算 ===============
                    # 1. 计算理论传输时延 (仅因为数据量和光猫硬件极限而产生的时延)
                    transmission_delay = (data_mb * 8) / hw_bandwidth_mbps
                    # 2. 推断传播时延 (扣除掉纯数据位移动时间后，剩下的就是物理接线的长距离和协议处理时间，近似取一半为单向传播时延)
                    propagation_delay = max(0.0, (real_time - transmission_delay) / 2)
                    
                    print(f"  [COMM-UDP] 🟢 {self.node_id}->{target_id}  ({data_mb:.3f}MB) 发送彻底成功 | "
                          f"总物理耗时: {real_time*1000:.1f} ms\n"
                          f"             --> [记录分析] 分析传输时延: {transmission_delay*1000:.1f} ms | 单向传播时延估值: {propagation_delay*1000:.1f} ms")
                    
                    udp_sock.close()
                    return True, real_time
            except socket.timeout:
                print(f"  [COMM-UDP] 🟡 {self.node_id}->{target_id} 等待最终 ACK 超时 (物理延迟过大或严重丢包)，触发第 {attempt+1} 次强制重载重传...")
                continue
            except ConnectionResetError:
                print(f"  [COMM-UDP] 🔴 {self.node_id}->{target_id} 端口不可达，对端可能未启动...")
                time.sleep(1)
                continue

        print(f"  [COMM-UDP] ❌ {self.node_id}->{target_id} 达到最大重试次数，数据丢失!")
        udp_sock.close()
        return False, 0.0

    # ===================== 监听 =====================
    def start_listening(self, handler_callback):
        self.handler = handler_callback
        self.running = True

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        except OSError:
            pass
            
        self.server_socket.bind(('0.0.0.0', self.listen_port))

        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
        
        # 垃圾回收机制线程 (清理死包漏包的残余碎片)
        clean_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        clean_thread.start()
        
        print(f"[{self.node_id}] UDP 守护监听全速运转中 @ 0.0.0.0:{self.listen_port} 🌊")

    def _cleanup_loop(self):
        while self.running:
            time.sleep(5)
            now = time.time()
            with self.buf_lock:
                to_delete = []
                for mid, info in self.recv_buffers.items():
                    if now - info['ts'] > 10.0:  # 超过 10 秒没收齐拼装包就被判定流产
                        to_delete.append(mid)
                for mid in to_delete:
                    del self.recv_buffers[mid]

    def _listen_loop(self):
        while self.running:
            try:
                self.server_socket.settimeout(2.0)
                data, addr = self.server_socket.recvfrom(65536)
                
                # 头不完整视为乱码直接扔
                if len(data) < 6: continue
                msg_id, num_chunks, chunk_idx = struct.unpack('!HHH', data[:6])
                chunk_data = data[6:]

                with self.buf_lock:
                    if msg_id not in self.recv_buffers:
                        self.recv_buffers[msg_id] = {
                            'chunks': {}, 
                            'num': num_chunks, 
                            'addr': addr, 
                            'ts': time.time()
                        }

                    # 更新存片和时间
                    self.recv_buffers[msg_id]['chunks'][chunk_idx] = chunk_data
                    self.recv_buffers[msg_id]['ts'] = time.time()

                    # 当字典里的存储数量与总数量相等时，数据全了！
                    if len(self.recv_buffers[msg_id]['chunks']) == num_chunks:
                        # [关键] 必须在这里火速给发送方回应 DONE！解开发送方的超时阻塞！
                        self.server_socket.sendto(b'DONE', addr)
                        
                        # 把拼积木拆分到另一边，别挡住其他高频包的进来
                        full_data = bytearray()
                        for i in range(num_chunks):
                            full_data.extend(self.recv_buffers[msg_id]['chunks'][i])
                        
                        del self.recv_buffers[msg_id]
                        
                        # 丢去处理业务
                        threading.Thread(
                            target=self._process_async, 
                            args=(full_data,), 
                            daemon=True
                        ).start()

            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                print(f"  [COMM] UDP 接收总线严重异常: {e}")

    def _process_async(self, raw_data):
        try:
            message = pickle.loads(raw_data)
            if self.handler:
                self.handler(message)
        except Exception as e:
            print(f"  [COMM] 致命异常 - 数据流反序列化被损坏，可能是由于 MTU 切片不齐: {e}")

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass