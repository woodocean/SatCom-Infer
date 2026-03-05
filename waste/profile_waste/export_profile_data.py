import torch
import pandas as pd
import os
import sys
import platform
import time
import gc
import warnings

# 忽略 thop 的一些无关紧要的 warning
warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

# 引入所有 Wrapper
from models.dag_wrappers import (
    YOLOv5_DAG_Wrapper, ResNet_DAG_Wrapper, 
    VGG19_DAG_Wrapper, 
    MobileNetV2_DAG_Wrapper,
    # 如果你还没有加 ViT/UNet/Swin，请注释掉下面这行，或者确保 dag_wrappers.py 里有它们
     ViT_Huge_DAG_Wrapper, UNet_DAG_Wrapper, Swin_Base_DAG_Wrapper
)

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: 'thop' not found. FLOPs will be 0.")

def get_data_size_mb(data):
    total = 0
    if isinstance(data, torch.Tensor):
        total += data.numel() * 4
    elif isinstance(data, dict):
        for v in data.values():
            total += get_data_size_mb(v)
    elif isinstance(data, (list, tuple)):
        for v in data:
            total += get_data_size_mb(v)
    return total / (1024**2)

def profile_single_model(name, builder, input_size, device):
    print(f"   ... Profiling {name} @ {input_size}x{input_size}")
    
    # 【修正】：直接调用 builder，不搞 try-except 掩盖错误
    # 所有的 lambda 必须写成 lambda d, s: ... 的形式
    try:
        model = builder(device, input_size)
    except Exception as e:
        print(f"   ❌ 模型初始化致命错误 ({name}): {e}")
        import traceback
        traceback.print_exc()
        return []

    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
    total_layers = len(model)
    
    if hasattr(model, 'reset_cache'): model.reset_cache()
    
    layer_records = []
    cumulative_params = 0.0
    current_pack = input_tensor

    for i in range(total_layers):
        layer = model.layers[i]
        layer_type = layer.__class__.__name__
        
        # 1. Params
        try:
            params_cnt = sum(p.numel() for p in layer.parameters())
            params_mb = params_cnt * 4 / (1024**2)
        except: params_mb = 0.0
        cumulative_params += params_mb
        
        # 2. FLOPs
        flops = 0.0
        thop_input = None
        if isinstance(current_pack, torch.Tensor):
            thop_input = current_pack
        elif isinstance(current_pack, dict) and 'main' in current_pack:
            thop_input = current_pack['main']
            
        if HAS_THOP and thop_input is not None and isinstance(thop_input, torch.Tensor):
            try:
                # 避开 Detect 层等无法计算 FLOPs 的层
                if 'Detect' not in layer_type and 'Head' not in layer_type:
                    macs, _ = profile(layer, inputs=(thop_input, ), verbose=False)
                    flops = (macs * 2) / 1e9
            except: pass

        # 3. Forward
        try:
            with torch.no_grad():
                output_pack = model.forward_slice(current_pack, i, i+1)
        except Exception as e:
            print(f"   ❌ Layer {i} ({layer_type}) Forward Error: {e}")
            break

        # 4. Output + Cache
        comm_mb = get_data_size_mb(output_pack)
        mem_mb = cumulative_params + comm_mb

        layer_records.append({
            'model': name,
            'input_size': input_size,
            'layer_idx': i,
            'layer_type': layer_type,
            'flops_g': flops,
            'comm_mb': comm_mb,
            'mem_mb': mem_mb,
            'params_mb': params_mb
        })
        
        current_pack = output_pack

    del model, input_tensor, current_pack
    if device == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    
    return layer_records

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Starting Rigorous Profiling on {device.upper()} ===")
    
    sizes = [224, 640, 1024] 
    
    # 【核心修正】：所有 lambda 统一接收 (d, s) 两个参数
    # 即使模型不需要 s (如 VGG)，也要写成 lambda d, s: Wrapper(d)
    configs = [
        ('YOLOv5n', lambda d, s: YOLOv5_DAG_Wrapper('checkpoints/yolov5nu.pt', d)),
        ('ResNet101', lambda d, s: ResNet_DAG_Wrapper('101', d)),
        ('VGG19', lambda d, s: VGG19_DAG_Wrapper(d)),
        ('MobileNetV2', lambda d, s: MobileNetV2_DAG_Wrapper(d)),
        ('ViT_Huge', lambda d, s: ViT_Huge_DAG_Wrapper(d, img_size=s)),
        ('UNet', lambda d, s: UNet_DAG_Wrapper(d)), # UNet 对尺寸不敏感，全卷积
        ('Swin_Base', lambda d, s: Swin_Base_DAG_Wrapper(d, img_size=s if s%32==0 else 640)) 
    ]

    all_data = []
    
    for name, builder in configs:
        print(f"\n>> Model: {name}")
        for size in sizes:
            # VGG 保护
            if 'VGG' in name and size > 640: 
                print(f"   [Skip] {name} {size}x{size} to avoid OOM")
                continue
            
            data = profile_single_model(name, builder, size, device)
            all_data.extend(data)
            
    df = pd.DataFrame(all_data)
    filename = 'profile_database.csv'
    df.to_csv(filename, index=False)
    print(f"\n✅ Profiling Complete. Data saved to {filename}")

if __name__ == "__main__":
    main()