import os
import time
import json
import copy
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from dag_wrappers import (
    ResNet_DAG_Wrapper, 
    YOLOv5_DAG_Wrapper,
    VGG19_DAG_Wrapper,
    Swin_Base_DAG_Wrapper,
    ViT_Huge_DAG_Wrapper
)

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

def parse_pack_size(pack):
    """加入基于内存地址去重的智能包裹尺寸解析 (防止 Cache 和 Main 双重计算)"""
    visited_ids = set()
    total_bytes = 0
    pure_bytes = 0
    
    def recurse(v, is_main=False):
        nonlocal total_bytes, pure_bytes
        if isinstance(v, torch.Tensor):
            if id(v) not in visited_ids:
                visited_ids.add(id(v))
                sz = v.nelement() * v.element_size()
                total_bytes += sz
            if is_main and pure_bytes == 0:
                pure_bytes = v.nelement() * v.element_size()
        elif isinstance(v, dict):
            main_key_hints = ['x', 'out', 'output', 'hidden_states', 'main', 'features']
            for k, item in v.items():
                recurse(item, is_main=(k in main_key_hints))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                recurse(item, is_main=(is_main and i == 0))

    recurse(pack, is_main=True)
    return total_bytes / (1024**2), pure_bytes / (1024**2)

class SliceWrapper(torch.nn.Module):
    def __init__(self, model, start_idx, end_idx):
        super().__init__()
        self.wrapper_model = model
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, *args):
        x = args[0] if len(args) == 1 else args
        return self.wrapper_model.forward_slice(x, self.start_idx, self.end_idx)

def generate_hardware_profiles(config_grid, output_json, num_runs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] 启动全景硬件档案生成的任务。设备: {device.upper()}")
    
    # 最后保存的大字典
    profile_db = {}
    
    for model_name, cfg in config_grid.items():
        profile_db[model_name] = {}
        
        for batch in cfg['batches']:
            for res in cfg['resolutions']:
                config_key = f"b{batch}_{res[0]}x{res[1]}"
                print(f"\n>> 正在构建档案: {model_name} | {config_key}")
                
                # 1. 动态实例化模型
                if model_name == 'vgg19':
                    model = VGG19_DAG_Wrapper(device=device)
                elif model_name == 'resnet101':
                    model = ResNet_DAG_Wrapper(version='101', device=device)
                elif model_name == 'yolov5':
                    model = YOLOv5_DAG_Wrapper(device=device)
                elif model_name == 'swin_base':
                    model = Swin_Base_DAG_Wrapper(device=device, img_size=res[0])
                elif model_name == 'vit_huge':
                    model = ViT_Huge_DAG_Wrapper(device=device)
                else:
                    print(f"未找到 {model_name} 的包装器，跳过！")
                    continue
                
                model.eval()
                configured_end_idx = int(cfg.get('end_idx', len(model)))
                if model_name == 'yolov5' and configured_end_idx == len(model) - 1:
                    print(f"   - [修正] YOLOv5 profiling 上界由 {configured_end_idx} 自动补到最后一层 {len(model)}")
                    end_idx = len(model)
                else:
                    end_idx = min(configured_end_idx, len(model))
                # 针对分辨率的不同通道数，大部分是3通道
                dummy_in = torch.randn(batch, 3, *res).to(device)
                
                layer_data = {}
                current_pack = dummy_in
                
                # 第一阶段：静态物理测算 (GFLOPs & 通信体积)
                print("   - 测算网络静态物理属性 (FLOPs与通信量)...")
                for i in range(end_idx):
                    # 备份原始的 Cache 状态，防止 fvcore 计算图追踪器污染内部字典！
                    backup_cache = None
                    if hasattr(model, 'feature_cache'):
                        backup_cache = copy.copy(model.feature_cache)
                        
                    flops_g = 0.0
                    if HAS_FVCORE:
                        try:
                            flops_ana = FlopCountAnalysis(SliceWrapper(model, i, i+1), (current_pack,))
                            flops_ana.unsupported_ops_warnings(False)
                            flops_ana.uncalled_modules_warnings(False)
                            flops_g = (flops_ana.total() * 2) / 1e9
                        except Exception as e:
                            flops_g = 0.0
                            
                    # 恢复干净的 Cache 状态供真实前向传递使用
                    if backup_cache is not None:
                        model.feature_cache = backup_cache

                    with torch.no_grad():
                        current_pack = model.forward_slice(current_pack, i, i+1)
                        
                    c_tot, c_pur = parse_pack_size(current_pack)

                    # === 新增：层级物理参数占用（权重 + BN缓冲层）===
                    w_size = 0.0
                    if hasattr(model, 'layers') and i < len(model.layers):
                        layer_module = model.layers[i]
                        # 汇总该层的所有网络参数与缓存占用的元素数量
                        tensors = list(layer_module.parameters()) + list(layer_module.buffers())
                        w_elements = sum(t.nelement() for t in tensors)
                        # 每个 float32 参数占用 4 Bytes，转换为 MB
                        w_size = (w_elements * 4) / (1024**2) 

                    layer_data[i] = {
                        "flops_gf": round(flops_g, 3),
                        "comm_total_mb": round(c_tot, 3),
                        "comm_pure_mb": round(c_pur, 3),
                        "weight_size_mb": round(w_size, 3)
                    }

                # 第二阶段：真机高吞吐时延测绘
                print("   - 测绘流水线实际推理时延...")
                with torch.no_grad():
                    for _ in range(3): # 预热
                        model.forward_slice(dummy_in, 0, end_idx)
                torch.cuda.synchronize()

                latencies = {i: [] for i in range(end_idx)}
                for _ in range(num_runs):
                    pack_run = dummy_in
                    with torch.no_grad():
                        for i in range(end_idx):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            pack_run = model.forward_slice(pack_run, i, i+1)
                            torch.cuda.synchronize()
                            t1 = time.perf_counter()
                            latencies[i].append((t1 - t0) * 1000)

                # 数据整合与统计计算
                for i in range(end_idx):
                    arr = latencies[i]
                    mean_ms = np.mean(arr)
                    cv_percent = (np.std(arr) / mean_ms * 100) if mean_ms > 0 else 0
                    tflops = (layer_data[i]["flops_gf"] / mean_ms) if mean_ms > 0 else 0
                    
                    layer_data[i].update({
                        "latency_mean_ms": round(mean_ms, 3),
                        "latency_cv_pct": round(cv_percent, 2),
                        "tflops_efficiency": round(tflops, 3)
                    })
                
                profile_db[model_name][config_key] = layer_data
                
                # 释放显存，防止遍历下一个配置时爆掉 OOM
                del model
                torch.cuda.empty_cache()

    # 序列化为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(profile_db, f, indent=4, ensure_ascii=False)
    
    print(f"\n[成 功] 档案已全面生成并保存至 -> '{output_json}'")

if __name__ == "__main__":
    # 在这里自由配置你论文实验涉及的所有测试网格！
    EXPERIMENT_GRID = {
        'yolov5': {
            'end_idx': 24, # 确认 Wrapper 是 0 -> 24
            'batches': [16, 32,64,128],
            'resolutions': [(640, 640)]  
        },
        'resnet101': {
            'end_idx': 33,
            'batches': [16, 32,64,128],
            'resolutions': [(224, 224)]
        },
        'vgg19': {
            'end_idx': 45,
            'batches': [16, 32,64,128],
            'resolutions': [(224, 224)]
        },
        'swin_base': {
            'end_idx': 6,
            'batches': [16, 32,64],
            'resolutions': [(224, 224)]
        },
        'vit_huge':{
            'end_idx': 33,
            'batches': [16, 32,64],
            'resolutions': [(224, 224)]
        }
    }

    OUTPUT_FILE = "dnn_profiles_database.json"
    
    # 推荐跑 10 轮时延，可以熨平 GPU 电压跳动带来的轻微干扰
    generate_hardware_profiles(EXPERIMENT_GRID, OUTPUT_FILE, num_runs=10)
