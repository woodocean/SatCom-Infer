import torch
import time

def measure_device_flops(device='cuda', num_runs=100, size=4096):
    """
    通过大矩阵乘法测算设备的真实 FP32 算力 (GFLOPS)
    原理: C(MxN) = A(MxK) * B(KxN) 的计算量约为 2*M*N*K
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，切换回 CPU")
        device = 'cpu'
        
    device = torch.device(device)
    
    # 准备数据 (使用大矩阵保证 GPU 满载，忽略 Python 开销)
    # 4096 x 4096 的矩阵乘法
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    
    # 理论计算量 (GFLOPs)
    # 乘法+加法 = 2 * size^3
    flops_per_run = 2 * (size ** 3) / 1e9
    
    print(f"[{device}] 正在基准测试 (矩阵大小 {size}x{size})...")
    
    # 预热
    for _ in range(10):
        torch.matmul(A, B)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    # 正式测试
    start = time.time()
    for _ in range(num_runs):
        torch.matmul(A, B)
    if device.type == 'cuda': torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    avg_time = total_time / num_runs
    
    real_flops = flops_per_run / avg_time
    
    print(f"  平均耗时: {avg_time*1000:.2f} ms")
    print(f"  真实算力: {real_flops:.2f} GFLOPS")
    
    return real_flops

if __name__ == "__main__":
    # 测 GPU
    gpu_flops = measure_device_flops('cuda', size=4096)
    # 测 CPU (可选，矩阵改小点否则太慢)
    cpu_flops = measure_device_flops('cpu', size=2048)