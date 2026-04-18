import time
import torch
import numpy as np
from core.communicate import Communicator

def run_benchmark(target_ip, num_trials=30):
    print(f"🚀 开始吞吐基准测试: 目标 {target_ip}, 轮次 {num_trials}")
    
    comm_tx = Communicator("PC_Benchmark", "0.0.0.0", 6061)
    comm_tx.register_peer("Jetson", target_ip, 5050)
    
    # 存储测得的吞吐量 (Mbps)
    throughputs = []
    
    # 构造不同大小的张量进行测试
    # 选取的范围从 2MB 到 40MB，模拟不同的特征图大小
    sizes = np.linspace(2, 40, num_trials)
    
    time.sleep(2)  # 等待接收端就绪
    
    for i, size_mb in enumerate(sizes):
        # 计算对应大小的 Tensor (1 float = 4 bytes)
        num_elements = int((size_mb * 1024 * 1024) / 4)
        fake_tensor = torch.randn(num_elements)
        
        print(f"\n[轮次 {i+1}/{num_trials}] 准备发送 {size_mb:.2f} MB Tensor...")
        
        succ, metrics = comm_tx.send_tensor("Jetson", "BENCHMARK", fake_tensor)
        
        if succ:
            tp = metrics['throughput_mbps']
            throughputs.append(tp)
            print(f"  🏁 测得速率: {tp:.2f} Mbps")
        else:
            print(f"  ❌ 轮次 {i+1} 发送失败")
            
        time.sleep(0.5) 

    comm_tx.stop()
    
    if throughputs:
        avg_tp = np.mean(throughputs)
        std_tp = np.std(throughputs)
        print("\n" + "="*40)
        print(f"📊 基准测试完成 (n={len(throughputs)})")
        print(f"📈 平均接收速率: {avg_tp:.2f} Mbps")
        print(f"📉 速率标准差:   {std_tp:.2f} Mbps")
        print(f"📋 建议基准值:   {avg_tp - std_tp:.2f} Mbps (平均值-1σ, 保守估计)")
        print("="*40)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python benchmark_udp.py <Target_IP>")
    else:
        run_benchmark(sys.argv[1])
