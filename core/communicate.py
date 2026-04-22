import socket
import struct
import pickle
import threading
import time
import torch
import random
import os
import json

# 恢复为大分包方案，降低分片数量压力，并通过增加 Sleep 放缓速率
CHUNK_SIZE = 60000
SEND_PACING_EVERY = 10
SEND_PACING_SLEEP = 0.001
MIN_ACK_TIMEOUT = 3.0
RECV_BUFFER_TTL = 45.0

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

    def _get_link_profile(self, target_id):
        """读取链路带宽与传播时延，用于动态设置 ACK 超时。"""
        theoretical_bw_mbps = 580.0
        hardware_baseline_mbps = 580.0
        propagation_delay_s = 0.0

        try:
            cfg_path = os.path.join("config", "network_config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                net_cfg = json.load(f)

            global_settings = net_cfg.get("global_settings", {})
            hardware_baseline_mbps = global_settings.get("hardware_baseline_mbps", 580.0)

            links = net_cfg.get("links", {})
            link_key1 = f"{self.node_id}_to_{target_id}"
            link_key2 = f"{target_id}_to_{self.node_id}"
            link_info = links.get(link_key1) or links.get(link_key2) or {}

            theoretical_bw_mbps = link_info.get("bandwidth_mbps", hardware_baseline_mbps)
            propagation_delay_ms = link_info.get("propagation_delay_ms", 0.0)
            propagation_delay_s = max(0.0, float(propagation_delay_ms) / 1000.0)
        except Exception:
            pass

        return theoretical_bw_mbps, propagation_delay_s, hardware_baseline_mbps

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
        theoretical_bw_mbps, propagation_delay_s, hardware_baseline_mbps = self._get_link_profile(target_id)

        # ACK 超时应至少覆盖：发送耗时 + 往返传播 + 处理裕量，避免高传播链路上的假超时重传
        est_tx_s = (data_mb * 8.0 / max(1.0, theoretical_bw_mbps)) if data_mb > 0 else 0.001
        base_ack_timeout = max(MIN_ACK_TIMEOUT, est_tx_s * 1.5 + propagation_delay_s * 2.0 + 1.0)

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

        # 针对大报文适当增大发送缓冲，减少内核拥塞导致的丢包
        if data_mb >= 20:
            try:
                udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024 * 1024)
            except OSError:
                pass

        udp_sock.settimeout(base_ack_timeout)

        msg_id = random.randint(0, 65535)

        start_t = time.perf_counter()

        # UDP 发送轮次 (带超时重发整个包块保障)
        MAX_RETRIES = 4

        if data_mb >= 20:
            print(
                f"  [COMM-UDP] {self.node_id}->{target_id} 大报文发送参数: "
                f"chunks={num_chunks}, chunk={CHUNK_SIZE}B, ack_timeout={base_ack_timeout:.2f}s"
            )

        for attempt in range(MAX_RETRIES):
            current_timeout = min(base_ack_timeout * (1.0 + 0.4 * attempt), base_ack_timeout + 20.0)
            udp_sock.settimeout(current_timeout)

            if data_mb >= 100:
                pacing_every = max(16, SEND_PACING_EVERY // 2)
                pacing_sleep = SEND_PACING_SLEEP * 2
            elif data_mb >= 20:
                pacing_every = max(24, SEND_PACING_EVERY)
                pacing_sleep = SEND_PACING_SLEEP
            else:
                pacing_every = 64
                pacing_sleep = 0.00008

            # A. 高速突发注入并进行防丢包节奏控制
            for i, chunk in enumerate(chunks):
                # 恢复原生 HHH：msg_id: (2 bytes), 总数: (2 bytes), 序号: (2 bytes) = 6 Bytes
                header = struct.pack("!HHH", msg_id, num_chunks, i)
                udp_sock.sendto(header + chunk, (ip, port))
                
                # 减缓路由器交换队列溢出导致的高丢包率
                # 适当减缓发送速率：增加 sleep 时间和频次
                if (i + 1) % 10 == 0:
                    time.sleep(0.002)
                else:
                    time.sleep(0.0001)

            # B. 阻塞等待接收端所有块重组完成后的 DONE 应答
            try:
                ack_data, _ = udp_sock.recvfrom(1024)
                if ack_data.startswith(b'DONE'):
                    end_t = time.perf_counter()
                    total_real_time = end_t - start_t
                    
                    # =============== 时延测算核心点 ===============
                    # 1. 传输时延: 接收端实打实在收这些数据块所花费的总时间 (最后一块落下的时间 - 收到第一块落下的时间)
                    if len(ack_data) >= 12:
                        rx_time = struct.unpack('!d', ack_data[4:12])[0]
                    else:
                        rx_time = 0.001
                        
                    transmission_delay = rx_time if rx_time > 0.001 else 0.001
                    throughput_mbps = (data_mb * 8) / transmission_delay if data_mb > 0 else 0.0
                    
                    # 2. 传播时延 (单向): [(发送端发出起~等到回应的往返时间) - 接收的耗时] / 2
                    # 无论你在 bw.py 中怎么设定 (50ms, 10ms..)，这完美等价于这趟物理线缆的传播时延
                    propagation_delay = max(0.0, (total_real_time - transmission_delay) / 2)
                    
                    # 3. 本次通信总时延
                    comm_latency = propagation_delay + transmission_delay

                    # 按照等比比例进行目标传输时延折算
                    # 比如受硬件局限，实测带宽上限为580M，但想要模拟星群间高速链路5800M，即可将实测时延除以10
                    scale_ratio = hardware_baseline_mbps / theoretical_bw_mbps if theoretical_bw_mbps > 0 else 1.0
                    simulated_transmission_time = transmission_delay * scale_ratio
                    simulated_comm_latency = propagation_delay + simulated_transmission_time
                    # =========================================================
                    
                    # 🌟 [更新] 将实测数据与理论折算数据一并暴露给返回值
                    metrics = {
                        'success': True,
                        'total_time': total_real_time,
                        'transmission_time': transmission_delay,
                        'propagation_time': propagation_delay,
                        'throughput_mbps': throughput_mbps,
                        'data_mb': data_mb,
                        'simulated_transmission_time': simulated_transmission_time,
                        'simulated_comm_latency': simulated_comm_latency,
                        'theoretical_bandwidth': theoretical_bw_mbps
                    }
                    
                    print(f"  [COMM-UDP] 🟢 {self.node_id}->{target_id}  ({data_mb:.3f}MB) 发送成功，时延统筹:")
                    print(f"             --> | 真实传输耗时: {transmission_delay*1000:6.1f} ms  (接收推算吞吐: {throughput_mbps:6.1f} Mbps)")
                    print(f"             --> | 物理单程传播: {propagation_delay*1000:6.1f} ms")
                    print(f"             --> | 【等效缩放】理论带宽: {theoretical_bw_mbps} Mbps (系数 {scale_ratio:.3f})")
                    print(f"             --> | 【等效缩放】传输耗时: {simulated_transmission_time*1000:6.1f} ms")
                    print(f"             --> | 【等效缩放】整体时延: {simulated_comm_latency*1000:6.1f} ms")
                    
                    udp_sock.close()
                    return True, metrics  # 🌟 修改返回值为详细指标字典
            except socket.timeout:
                print(
                    f"  [COMM-UDP] 🟡 {self.node_id}->{target_id} 等待最终 ACK 超时 "
                    f"(timeout={current_timeout:.2f}s)，触发第 {attempt+1} 次重传..."
                )
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
                    if now - info['ts'] > RECV_BUFFER_TTL:  # 大报文传输期更长，避免误清理导致反复重传
                        to_delete.append(mid)
                for mid in to_delete:
                    del self.recv_buffers[mid]

    def _listen_loop(self):
        while self.running:
            try:
                self.server_socket.settimeout(2.0)
                data, addr = self.server_socket.recvfrom(65536)
                
                # 头不完整视为乱码直接扔 (恢复为 6 Bytes: HHH)
                if len(data) < 6: continue
                msg_id, num_chunks, chunk_idx = struct.unpack('!HHH', data[:6])
                chunk_data = data[6:]

                with self.buf_lock:
                    if msg_id not in self.recv_buffers:
                        self.recv_buffers[msg_id] = {
                            'chunks': {}, 
                            'num': num_chunks, 
                            'addr': addr, 
                            'first_ts': time.perf_counter(),
                            'ts': time.time()
                        }

                    # 更新存片和时间
                    self.recv_buffers[msg_id]['chunks'][chunk_idx] = chunk_data
                    self.recv_buffers[msg_id]['ts'] = time.time()

                    # 当字典里的存储数量与总数量相等时，数据全了！
                    if len(self.recv_buffers[msg_id]['chunks']) == num_chunks:
                        # 记录接收侧的纯数据块拼装耗时 (第一片落下 到 最后一片落下的时间) 也就是【传输发送时长】
                        rx_time = time.perf_counter() - self.recv_buffers[msg_id]['first_ts']
                        
                        # [关键] 必须在这里火速给发送方回应 DONE！同时将我们精准侧算到的 rx_时间 发回去
                        ack_msg = b'DONE' + struct.pack('!d', rx_time)
                        self.server_socket.sendto(ack_msg, addr)
                        
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