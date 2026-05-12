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
    """鍔犲叆鍩轰簬鍐呭瓨鍦板潃鍘婚噸鐨勬櫤鑳藉寘瑁瑰昂瀵歌В鏋?(闃叉 Cache 鍜?Main 鍙岄噸璁＄畻)"""
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
    print(f"[*] 鍚姩鍏ㄦ櫙纭欢妗ｆ鐢熸垚鐨勪换鍔°€傝澶? {device.upper()}")
    
    # 鏈€鍚庝繚瀛樼殑澶у瓧鍏?
    profile_db = {}
    
    for model_name, cfg in config_grid.items():
        profile_db[model_name] = {}
        
        for batch in cfg['batches']:
            for res in cfg['resolutions']:
                config_key = f"b{batch}_{res[0]}x{res[1]}"
                print(f"\n>> 姝ｅ湪鏋勫缓妗ｆ: {model_name} | {config_key}")
                
                # 1. 鍔ㄦ€佸疄渚嬪寲妯″瀷
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
                    print(f"鏈壘鍒?{model_name} 鐨勫寘瑁呭櫒锛岃烦杩囷紒")
                    continue
                
                model.eval()
                configured_end_idx = int(cfg.get('end_idx', len(model)))
                if model_name == 'yolov5' and configured_end_idx == len(model) - 1:
                    print(f"   - [淇] YOLOv5 profiling 涓婄晫鐢?{configured_end_idx} 鑷姩琛ュ埌鏈€鍚庝竴灞?{len(model)}")
                    end_idx = len(model)
                else:
                    end_idx = min(configured_end_idx, len(model))
                # 閽堝鍒嗚鲸鐜囩殑涓嶅悓閫氶亾鏁帮紝澶ч儴鍒嗘槸3閫氶亾
                dummy_in = torch.randn(batch, 3, *res).to(device)
                
                layer_data = {}
                current_pack = dummy_in
                
                # 绗竴闃舵锛氶潤鎬佺墿鐞嗘祴绠?(GFLOPs & 閫氫俊浣撶Н)
                print("   - 娴嬬畻缃戠粶闈欐€佺墿鐞嗗睘鎬?(FLOPs涓庨€氫俊閲?...")
                for i in range(end_idx):
                    # 澶囦唤鍘熷鐨?Cache 鐘舵€侊紝闃叉 fvcore 璁＄畻鍥捐拷韪櫒姹℃煋鍐呴儴瀛楀吀锛?
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
                            
                    # 鎭㈠骞插噣鐨?Cache 鐘舵€佷緵鐪熷疄鍓嶅悜浼犻€掍娇鐢?
                    if backup_cache is not None:
                        model.feature_cache = backup_cache

                    with torch.no_grad():
                        current_pack = model.forward_slice(current_pack, i, i+1)
                        
                    c_tot, c_pur = parse_pack_size(current_pack)

                    # === 鏂板锛氬眰绾х墿鐞嗗弬鏁板崰鐢紙鏉冮噸 + BN缂撳啿灞傦級===
                    w_size = 0.0
                    if hasattr(model, 'layers') and i < len(model.layers):
                        layer_module = model.layers[i]
                        # 姹囨€昏灞傜殑鎵€鏈夌綉缁滃弬鏁颁笌缂撳瓨鍗犵敤鐨勫厓绱犳暟閲?
                        tensors = list(layer_module.parameters()) + list(layer_module.buffers())
                        w_elements = sum(t.nelement() for t in tensors)
                        # 姣忎釜 float32 鍙傛暟鍗犵敤 4 Bytes锛岃浆鎹负 MB
                        w_size = (w_elements * 4) / (1024**2) 

                    layer_data[i] = {
                        "flops_gf": round(flops_g, 3),
                        "comm_total_mb": round(c_tot, 3),
                        "comm_pure_mb": round(c_pur, 3),
                        "weight_size_mb": round(w_size, 3)
                    }

                # 绗簩闃舵锛氱湡鏈洪珮鍚炲悙鏃跺欢娴嬬粯
                print("   - 娴嬬粯娴佹按绾垮疄闄呮帹鐞嗘椂寤?..")
                with torch.no_grad():
                    for _ in range(3): # 棰勭儹
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

                # 鏁版嵁鏁村悎涓庣粺璁¤绠?
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
                
                # 閲婃斁鏄惧瓨锛岄槻姝㈤亶鍘嗕笅涓€涓厤缃椂鐖嗘帀 OOM
                del model
                torch.cuda.empty_cache()

    # 搴忓垪鍖栦负 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(profile_db, f, indent=4, ensure_ascii=False)
    
    print(f"\n[鎴?鍔焆 妗ｆ宸插叏闈㈢敓鎴愬苟淇濆瓨鑷?-> '{output_json}'")

if __name__ == "__main__":
    # 鍦ㄨ繖閲岃嚜鐢遍厤缃綘璁烘枃瀹為獙娑夊強鐨勬墍鏈夋祴璇曠綉鏍硷紒
    EXPERIMENT_GRID = {
        'yolov5': {
            'end_idx': 24, # 纭 Wrapper 鏄?0 -> 24
            'batches': [16, 32,64,128],
            'resolutions': [(640, 640)]  
        },
        'resnet101': {
            'batches': [16, 32,64,128],
            'resolutions': [(224, 224)]
        },
        'vgg19': {
            'batches': [16, 32,64,128],
            'resolutions': [(224, 224)]
        },
        'swin_base': {
            'batches': [16, 32,64],
            'resolutions': [(224, 224)]
        },
        'vit_huge':{
            'batches': [16, 32,64],
            'resolutions': [(224, 224)]
        }
    }

    OUTPUT_FILE = "config/dnn_profiles_database_pc.json"
    
    # 鎺ㄨ崘璺?10 杞椂寤讹紝鍙互鐔ㄥ钩 GPU 鐢靛帇璺冲姩甯︽潵鐨勮交寰共鎵?
    generate_hardware_profiles(EXPERIMENT_GRID, OUTPUT_FILE, num_runs=10)


