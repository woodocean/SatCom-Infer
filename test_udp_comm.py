import time
import torch
import threading
import socket
from core.communicate import Communicator

def run_receiver():
    print("[Jetson/Receiver 端] 正在启动接受服务...")
    # 这里模拟接收端（比如 Jetson 设备）
    comm_rx = Communicator("Jetson", "0.0.0.0", 5050)
    
    def handler(msg):
        print(f"\n[接收回调] 📩 成功收到来自 {msg['src']} 的消息！")
        if 'tensor' in msg:
            print(f"         包含特征图 Tensor 形状: {msg['tensor'].shape}")
            
    comm_rx.start_listening(handler)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        comm_rx.stop()

def run_sender(target_ip="127.0.0.1"):
    print(f"\n[PC/Sender 端] 准备向 {target_ip}:5050 发送数据...")
    # 这里模拟发送端（比如 PC 设备）
    comm_tx = Communicator("PC", "0.0.0.0", 6060)
    # 注册对端的 IP 路由
    comm_tx.register_peer("Jetson", target_ip, 5050)
    
    # 稍微等下接收端准备
    time.sleep(1)
    
    print("\n🚀 [1/2] 测试发送轻量级控制指令...")
    comm_tx.send_message("Jetson", "CTRL_TEST", {"hello": "world", "status": "ok"})
    
    time.sleep(2)
    
    print("\n🚀 [2/2] 测试发送大体积 Tensor (重活)...")
    # 构建一只大概等于 3-5MB 数据量的特征图（模拟 swin_base 等截断切割产物）
    fake_tensor = torch.randn(1, 128, 224, 224) 
    
    print(f"   准备发送的 Tensor 大小约为: {fake_tensor.element_size() * fake_tensor.nelement() / (1024*1024):.2f} MB")
    succ, lat = comm_tx.send_tensor("Jetson", "DATA_TEST", fake_tensor)
    
    if succ:
        print(f"\n✅ [测试成功] Tensor 已发送并得到确认，本次总耗时: {lat*1000:.2f} ms")
    else:
        print(f"\n❌ [测试失败] 发送异常或确认超时！")
        
    time.sleep(1)
    comm_tx.stop()

if __name__ == "__main__":
    import sys
    # 简单的运行指令分离：
    # python test_udp_comm.py server  (在 Jetson 端跑)
    # python test_udp_comm.py client 192.168.3.181 (在 PC 端跑)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            run_receiver()
        elif sys.argv[1] == "client":
            target = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
            run_sender(target)
    else:
        print("未指定启动模式，采用本地自发自收测试模式...")
        t_rx = threading.Thread(target=run_receiver, daemon=True)
        t_rx.start()
        time.sleep(1)
        run_sender("127.0.0.1")
