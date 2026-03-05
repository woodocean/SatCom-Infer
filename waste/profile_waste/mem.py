import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())

from models.dag_wrappers import (
    YOLOv5_DAG_Wrapper, ResNet_DAG_Wrapper, ViT_Huge_DAG_Wrapper,
    UNet_DAG_Wrapper, Swin_Base_DAG_Wrapper
)
from torchvision.models import vgg19, VGG19_Weights, resnet101, ResNet101_Weights

# 同样的 VGG Fix 和 ResNet Wrapper ... (为了节省篇幅，假设你已经有了或者直接复制上面的)
# 这里简写，实际运行请复制上面的 Wrapper 定义
class VGG19_DAG_Wrapper_Fix(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.raw_model = vgg19(weights=VGG19_Weights.DEFAULT).to(device)
        self.raw_model.eval()
    def parameters(self): return self.raw_model.parameters()

class ResNet101_Wrapper(ResNet_DAG_Wrapper):
    def __init__(self, device): super().__init__(version='101', device=device)

# --- 绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_model_memory_profile(name, builder, base_size=640):
    """
    获取模型的静态参数量和基准动态内存
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Profiling {name} at base size {base_size}...")
    
    try:
        try: model = builder(device, base_size)
        except: model = builder(device)
        
        # 1. 静态参数 (Weights)
        params_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
        
        # 2. 动态内存 (Activations)
        # 这里我们需要跑一次前向，估算最大的中间特征图
        # 简单估算：Max(Layer Input + Layer Output + Cache)
        # 为了精确，我们利用 torch.cuda.max_memory_allocated (如果是cuda)
        # 或者 模拟计算
        
        dynamic_peak_mb = 0
        x = torch.randn(1, 3, base_size, base_size).to(device)
        
        if hasattr(model, 'reset_cache'): model.reset_cache()
        
        # 模拟运行
        # 我们假设峰值出现在某一层，不仅仅是输入+输出，还包含 pytorch 上下文
        # 这里用一种通用的估算公式：对于 inference，通常是 max_layer(Input+Output)
        
        if hasattr(model, 'forward_slice'):
            curr = x
            for i in range(len(model)):
                # 计算输入大小
                in_size = curr.numel() * 4 / (1024**2) if isinstance(curr, torch.Tensor) else 0 # 简化
                with torch.no_grad():
                    out_pack = model.forward_slice(curr, i, i+1)
                
                # 计算输出+Cache大小
                out = out_pack['main']
                cache = out_pack.get('cache', {})
                
                out_size = 0
                if isinstance(out, torch.Tensor): out_size = out.numel() * 4 / (1024**2)
                elif isinstance(out, (list,tuple)): out_size = sum([t.numel()*4 for t in out if isinstance(t, torch.Tensor)])/(1024**2)
                
                cache_size = sum([t.numel()*4 for t in cache.values()])/(1024**2)
                
                # 这一层的峰值需求 ≈ 权重 + 输入 + 输出 + 缓存
                # 但我们这里只算动态部分
                layer_dynamic = in_size + out_size + cache_size
                if layer_dynamic > dynamic_peak_mb:
                    dynamic_peak_mb = layer_dynamic
                    
                curr = out_pack
                
        # 加上一点系统开销余量 (PyTorch overhead)
        dynamic_peak_mb *= 1.2 
        
        return params_mb, dynamic_peak_mb
        
    except Exception as e:
        print(f"Error profiling {name}: {e}")
        return 0, 0

def plot_memory_scaling():
    # 1. 准备数据
    BASE_SIZE = 640
    # 目标尺寸列表
    target_sizes = np.arange(224, 4200, 100) # 从 224 到 4096
    
    configs = [
        ('YOLOv5n', lambda d, s: YOLOv5_DAG_Wrapper('checkpoints/yolov5nu.pt', d)),
        ('ResNet-101', lambda d, s: ResNet101_Wrapper(d)),
        ('VGG-19', lambda d, s: VGG19_DAG_Wrapper_Fix(device=d)),
        ('U-Net', lambda d, s: UNet_DAG_Wrapper(device=d)),
        ('ViT-Huge', lambda d, s: ViT_Huge_DAG_Wrapper(device=d, img_size=s)),
        ('Swin-Base', lambda d, s: Swin_Base_DAG_Wrapper(device=d, img_size=s if s%32==0 else 640))
    ]
    
    plt.figure(figsize=(12, 8))
    
    # 调色板
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, (name, builder) in enumerate(configs):
        # 这里的 base_size 对于 ViT/Swin 最好用 224，因为 640 它们可能跑得慢
        base = 224 if 'ViT' in name or 'Swin' in name else 640
        
        static_mem, dynamic_base = get_model_memory_profile(name, builder, base)
        
        if static_mem == 0: continue
        
        # 推演公式
        # Mem = Static + Dynamic_Base * (Target / Base)^2
        mem_curve = [static_mem + dynamic_base * ((s / base)**2) for s in target_sizes]
        
        plt.plot(target_sizes, mem_curve, label=f"{name} (W:{static_mem:.0f}MB)", 
                 linewidth=2.5, color=colors[i])
        
        # 标注 4096 处的数值
        plt.text(target_sizes[-1], mem_curve[-1], f"{mem_curve[-1]:.0f}", color=colors[i], fontweight='bold')

    # 绘制内存红线
    plt.axhline(y=4096, color='red', linestyle='--', linewidth=2, label='4GB Memory Limit')
    plt.axhline(y=8192, color='darkred', linestyle=':', linewidth=2, label='8GB Memory Limit')
    
    plt.title("不同模型峰值内存占用随输入尺寸变化趋势 (Inference Memory Scaling)", fontsize=16)
    plt.xlabel("Input Image Size (Width=Height)", fontsize=14)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 标注区域
    plt.text(3500, 2000, "Safe Zone", fontsize=15, color='green', alpha=0.3)
    plt.text(1000, 10000, "OOM Zone", fontsize=15, color='red', alpha=0.3)
    
    plt.savefig("fig_memory_scaling_trend.png", dpi=300)
    print("✅ 生成: fig_memory_scaling_trend.png")

if __name__ == "__main__":
    plot_memory_scaling()