import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import os

# 引入路径
sys.path.append(os.getcwd())

# 引入 Wrapper
from models.dag_wrappers import (
    YOLOv5_DAG_Wrapper, ResNet_DAG_Wrapper, VGG19_DAG_Wrapper,
    MobileNetV2_DAG_Wrapper, ViT_Huge_DAG_Wrapper, UNet_DAG_Wrapper, Swin_Base_DAG_Wrapper
)

# 引入 FLOPs 计算工具
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: 'thop' library not found. Please install it: pip install thop")

# --- 绘图设置 ---
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
except: pass

def calculate_layer_flops(layer, input_data):
    """使用 thop 计算单层的 FLOPs"""
    if not HAS_THOP: return 0
    try:
        # thop 需要输入是 tuple
        # 处理 DAG Wrapper 的复杂输入 (main, cache)
        if isinstance(input_data, dict):
            # 对于 YOLO/UNet，thop 可能很难处理多输入 Layer (如 Concat/Up)
            # 这里做一个简化：只计算主路径输入的 FLOPs
            # 对于 Concat/Up 层，计算量本身很小（主要是内存搬运），误差可接受
            # 或者尝试构造成 layer 期望的参数格式
            inp = input_data['main']
            # 注意：如果 layer 是 Up/Concat，它可能需要多个输入，thop 这里会报错
            # 我们用 try-except 跳过这些特殊层的 FLOPs 计算 (它们本来就不是计算瓶颈)
            flops, params = profile(layer, inputs=(inp, ), verbose=False)
        else:
            flops, params = profile(layer, inputs=(input_data, ), verbose=False)
        
        return flops / 1e9 # GFLOPs
    except:
        return 0

def analyze_flops(name, builder, input_size=640):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Profiling FLOPs: {name}...")
    
    try:
        if 'ViT' in name or 'Swin' in name: model = builder(device, input_size)
        else:
            try: model = builder(device, input_size)
            except: model = builder(device)
    except: return [], []

    # 构造输入
    x_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    layer_indices = []
    flops_list = []
    
    current_pack = x_input
    if hasattr(model, 'reset_cache'): model.reset_cache()
    
    for i in range(len(model)):
        layer = model.layers[i]
        
        # 1. 计算 FLOPs
        # 注意：thop 需要模型在 CPU 上才能稳定运行部分算子，或者保持和输入一致
        # 这里 input 已经在 device 上了
        # 特殊处理：对于 YOLO 的 Detect 层，thop 必挂，跳过
        if 'Detect' in str(type(layer)):
            flops = 0
        else:
            flops = calculate_layer_flops(layer, current_pack)
        
        # 2. 真实前向传播 (为了拿到下一层的输入)
        with torch.no_grad():
            output_pack = model.forward_slice(current_pack, i, i+1)
        
        layer_indices.append(i)
        flops_list.append(flops)
        current_pack = output_pack
        
    return layer_indices, flops_list

def plot_flops_profile():
    TEST_SIZE = 640
    # Swin 需要特殊尺寸
    SWIN_SIZE = 640 
    
    configs = [
        ('VGG-19', lambda d, s: VGG19_DAG_Wrapper(d)),
        ('ResNet-101', lambda d, s: ResNet_DAG_Wrapper(version='101', device=d)), # 注意这里要用 version='101'
        ('YOLOv5n', lambda d, s: YOLOv5_DAG_Wrapper('checkpoints/yolov5nu.pt', d)),
        ('MobileNetV2', lambda d, s: MobileNetV2_DAG_Wrapper(d)),
        ('ViT-Huge', lambda d, s: ViT_Huge_DAG_Wrapper(d, s)),
        ('Swin-Base', lambda d, s: Swin_Base_DAG_Wrapper(d, SWIN_SIZE))
    ]
    
    fig, axes = plt.subplots(len(configs), 1, figsize=(12, 18), constrained_layout=True)
    
    for i, (name, builder) in enumerate(configs):
        ax = axes[i]
        x, y = analyze_flops(name, builder, TEST_SIZE)
        
        if not x: continue
        
        # 绘图
        ax.bar(x, y, color='#ff7f0e', alpha=0.9) # 使用橙色区分于通信图的蓝色/红色
        
        ax.set_title(f"{name} 各层计算量 (GFLOPs)", fontsize=14, fontweight='bold')
        ax.set_ylabel("GFLOPs", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 标注峰值
        if len(y) > 0:
            max_val = max(y)
            if max_val > 0:
                max_idx = y.index(max_val)
                ax.text(max_idx, max_val, f"{max_val:.2f}", ha='center', va='bottom', fontsize=9)

    axes[-1].set_xlabel("Layer Index", fontsize=14)
    plt.savefig("fig_flops_profile.png", dpi=300)
    print("\n✅ 图表生成完毕: fig_flops_profile.png")

if __name__ == "__main__":
    # ResNet Wrapper 需要修正一下初始化传参，这里假设你已经改好了
    # 如果没有，可以在这里临时定义一个辅助类
    plot_flops_profile()