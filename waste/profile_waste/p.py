import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
sys.path.append(os.getcwd())

# 引入所有 Wrapper
from models.dag_wrappers import (
    YOLOv5_DAG_Wrapper, ResNet_DAG_Wrapper, ViT_Huge_DAG_Wrapper,
    UNet_DAG_Wrapper, Swin_Base_DAG_Wrapper
)
from torchvision.models import vgg19, VGG19_Weights, resnet101, ResNet101_Weights

# --- 必要的 Wrapper 定义 (防止 import 缺失) ---
class VGG19_DAG_Wrapper_Fix(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Wrapper] Loading VGG-19...")
        self.raw_model = vgg19(weights=VGG19_Weights.DEFAULT).to(device)
        self.raw_model.eval()
        self.layers = nn.ModuleList()
        for layer in self.raw_model.features: self.layers.append(layer)
        self.layers.append(self.raw_model.avgpool)
        self.layers.append(nn.Flatten())
        for layer in self.raw_model.classifier: self.layers.append(layer)
        self.len = len(self.layers)
        self.save_indices = []
    def __len__(self): return self.len
    def forward_slice(self, input_pack, start_idx, end_idx):
        x = input_pack['main'] if isinstance(input_pack, dict) else input_pack
        for i in range(start_idx, end_idx): x = self.layers[i](x)
        return {'main': x, 'cache': {}}

class ResNet101_Wrapper(ResNet_DAG_Wrapper):
    def __init__(self, device):
        super().__init__(version='101', device=device)

# --- 绘图设置 ---
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
except: pass

def calculate_mb(tensor):
    if tensor is None: return 0
    return tensor.numel() * 4 / (1024**2)

def analyze_main_output(name, builder, input_size=672):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"分析 {name} 主干输出 (Input: {input_size})...")
    
    try:
        if 'ViT' in name: model = builder(device, input_size)
        else:
            try: model = builder(device, input_size)
            except: model = builder(device)
    except: return [], [], 0
    
    # 原始输入
    input_mb = (1 * 3 * input_size * input_size * 4) / (1024**2)
    x_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    layer_indices = [0]
    # 只记录 Main Output
    main_sizes = [input_mb]
    
    current_pack = x_input
    if hasattr(model, 'reset_cache'): model.reset_cache()
    
    for i in range(len(model)):
        try:
            with torch.no_grad():
                output_pack = model.forward_slice(current_pack, i, i+1)
            
            # 【核心差异】只取 main，不加 cache
            main_out = output_pack['main']
            
            size = 0
            if isinstance(main_out, torch.Tensor):
                size = calculate_mb(main_out)
            elif isinstance(main_out, (list, tuple)):
                size = sum([calculate_mb(t) for t in main_out if isinstance(t, torch.Tensor)])
            
            layer_indices.append(i + 1)
            main_sizes.append(size)
            current_pack = output_pack
        except: break
        
    return layer_indices, main_sizes, input_mb

def plot_main_outputs():
    TEST_SIZE = 672
    
    configs = [
        ('VGG-19 (Linear)', lambda d, s: VGG19_DAG_Wrapper_Fix(device=d)),
        ('ResNet-101 (Linear)', lambda d, s: ResNet101_Wrapper(d)),
        ('YOLOv5n (DAG)', lambda d, s: YOLOv5_DAG_Wrapper('checkpoints/yolov5nu.pt', d)),
        ('U-Net (U-Shape)', lambda d, s: UNet_DAG_Wrapper(device=d)),
        ('ViT-Huge (Transformer)', lambda d, s: ViT_Huge_DAG_Wrapper(device=d, img_size=s)),
        ('Swin-Base (Hierarchical)', lambda d, s: Swin_Base_DAG_Wrapper(device=d, img_size=s))
    ]
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 24), constrained_layout=True)
    
    for i, (name, builder) in enumerate(configs):
        ax = axes[i]
        x, y, input_base = analyze_main_output(name, builder, TEST_SIZE)
        
        if not x: continue
        
        # 颜色：主干输出通常应该小于输入（压缩），如果大于则是膨胀
        colors = ['#2ca02c' if idx==0 else '#1f77b4' for idx in range(len(y))]
        
        bars = ax.bar(x, y, color=colors, alpha=0.85)
        
        # 画基准线
        ax.axhline(y=input_base, color='green', linestyle='--', label='原始输入')
        ax.text(0, y[0], "Input", ha='center', va='bottom', fontweight='bold')

        # 标注最大值
        if len(y) > 1:
            max_val = max(y[1:])
            if max_val > 0.1:
                max_idx = y.index(max_val, 1)
                ax.text(x[max_idx], max_val, f"{max_val:.1f}", ha='center', va='bottom', fontsize=8)

        ax.set_title(f"{name} 主干层输出数据量 (纯特征图，不含Cache)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Size (MB)", fontsize=12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        if i==0: ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    axes[-1].set_xlabel("Layer Index", fontsize=14)
    plt.savefig("fig_main_output_only.png", dpi=300)
    print("✅ 生成: fig_main_output_only.png")

if __name__ == "__main__":
    plot_main_outputs()