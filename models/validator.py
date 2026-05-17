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


def _persist_outputs(profile_db, output_json, memory_rows=None):
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(profile_db, f, indent=4, ensure_ascii=False)
    if memory_rows is not None:
        try:
            import csv
            report_path = os.path.splitext(output_json)[0] + "_memory_report.csv"
            if memory_rows:
                fieldnames = list(memory_rows[0].keys())
                with open(report_path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(memory_rows)
        except Exception as exc:
            print(f"[warn] memory report save failed: {exc}")


def _estimate_full_model_peak_mb(layer_data, input_size_raw, end_idx):
    total_weight_mb = 0.0
    peak_input_plus_output_mb = 0.0
    peak_layer_idx = -1
    for i in range(end_idx):
        layer = layer_data.get(i, {})
        total_weight_mb += float(layer.get("weight_size_mb", 0.0))
        input_mb = float(input_size_raw) if i == 0 else float(layer_data.get(i - 1, {}).get("comm_pure_mb", 0.0))
        output_mb = float(layer.get("comm_pure_mb", 0.0))
        io_mb = input_mb + output_mb
        if io_mb > peak_input_plus_output_mb:
            peak_input_plus_output_mb = io_mb
            peak_layer_idx = i
    return {
        "estimated_total_weight_mb": round(total_weight_mb, 3),
        "estimated_peak_input_output_mb": round(peak_input_plus_output_mb, 3),
        "estimated_peak_memory_mb": round(total_weight_mb + peak_input_plus_output_mb, 3),
        "estimated_peak_layer_idx": peak_layer_idx,
    }

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
    memory_report_rows = []
    
    for model_name, cfg in config_grid.items():
        profile_db[model_name] = {}
        
        for batch in cfg['batches']:
            for res in cfg['resolutions']:
                config_key = f"b{batch}_{res[0]}x{res[1]}"
                print(f"\n>> 姝ｅ湪鏋勫缓妗ｆ: {model_name} | {config_key}")
                model = None
                layer_data = {}
                memory_row = {
                    "model_name": model_name,
                    "config_key": config_key,
                    "batch_size": batch,
                    "resolution": f"{res[0]}x{res[1]}",
                    "device": device,
                    "status": "started",
                }
                try:
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
                        memory_row["status"] = "missing_wrapper"
                        memory_report_rows.append(memory_row)
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
                    input_size_raw_mb = (dummy_in.nelement() * dummy_in.element_size()) / (1024**2)
                    memory_row["input_size_raw_mb"] = round(input_size_raw_mb, 3)
                    
                    current_pack = dummy_in
                    
                    # 绗竴闃舵锛氶潤鎬佺墿鐞嗘祴绠?(GFLOPs & 閫氫俊浣撶Н)
                    print("   - 娴嬬畻缃戠粶闈欐€佺墿鐞嗗睘鎬?(FLOPs涓庨€氫俊閲?...")
                    for i in range(end_idx):
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
                            except Exception:
                                flops_g = 0.0
                                
                        if backup_cache is not None:
                            model.feature_cache = backup_cache

                        with torch.no_grad():
                            current_pack = model.forward_slice(current_pack, i, i+1)
                            
                        c_tot, c_pur = parse_pack_size(current_pack)

                        w_size = 0.0
                        if hasattr(model, 'layers') and i < len(model.layers):
                            layer_module = model.layers[i]
                            tensors = list(layer_module.parameters()) + list(layer_module.buffers())
                            w_elements = sum(t.nelement() for t in tensors)
                            w_size = (w_elements * 4) / (1024**2) 

                        layer_data[i] = {
                            "flops_gf": round(flops_g, 3),
                            "comm_total_mb": round(c_tot, 3),
                            "comm_pure_mb": round(c_pur, 3),
                            "weight_size_mb": round(w_size, 3)
                        }

                    estimate = _estimate_full_model_peak_mb(layer_data, input_size_raw_mb, end_idx)
                    memory_row.update(estimate)

                    if device == 'cuda':
                        print("   - 娴嬬粯鏁存ā鍨嬫樉瀛樺嘲鍊?..")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        base_allocated = torch.cuda.memory_allocated()
                        base_reserved = torch.cuda.memory_reserved()
                        torch.cuda.reset_peak_memory_stats()
                        with torch.no_grad():
                            model.forward_slice(dummy_in, 0, end_idx)
                        torch.cuda.synchronize()
                        peak_allocated = torch.cuda.max_memory_allocated()
                        peak_reserved = torch.cuda.max_memory_reserved()
                        memory_row["measured_peak_allocated_mb"] = round(max(0, peak_allocated - base_allocated) / (1024**2), 3)
                        memory_row["measured_peak_reserved_mb"] = round(max(0, peak_reserved - base_reserved) / (1024**2), 3)
                    else:
                        memory_row["measured_peak_allocated_mb"] = None
                        memory_row["measured_peak_reserved_mb"] = None

                    print(
                        "   - [memory] estimate={:.1f} MB, peak_alloc={}".format(
                            memory_row["estimated_peak_memory_mb"],
                            "N/A" if memory_row["measured_peak_allocated_mb"] is None else f"{memory_row['measured_peak_allocated_mb']:.1f} MB",
                        )
                    )

                    # 绗簩闃舵锛氱湡鏈洪珮鍚炲悙鏃跺欢娴嬬粯
                    print("   - 娴嬬粯娴佹按绾垮疄闄呮帹鐞嗘椂寤?..")
                    with torch.no_grad():
                        for _ in range(3): # 棰勭儹
                            model.forward_slice(dummy_in, 0, end_idx)
                    if device == 'cuda':
                        torch.cuda.synchronize()

                    latencies = {i: [] for i in range(end_idx)}
                    for _ in range(num_runs):
                        pack_run = dummy_in
                        with torch.no_grad():
                            for i in range(end_idx):
                                if device == 'cuda':
                                    torch.cuda.synchronize()
                                t0 = time.perf_counter()
                                pack_run = model.forward_slice(pack_run, i, i+1)
                                if device == 'cuda':
                                    torch.cuda.synchronize()
                                t1 = time.perf_counter()
                                latencies[i].append((t1 - t0) * 1000)

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
                    memory_row["status"] = "ok"
                    memory_report_rows.append(memory_row)
                    _persist_outputs(profile_db, output_json, memory_report_rows)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"   - [OOM] {model_name} | {config_key}: {exc}")
                        memory_row["status"] = "oom"
                        memory_row["error"] = str(exc).replace("\n", " ")
                        memory_report_rows.append(memory_row)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        _persist_outputs(profile_db, output_json, memory_report_rows)
                        continue
                    raise
                finally:
                    if model is not None:
                        del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    _persist_outputs(profile_db, output_json, memory_report_rows)
    
    print(f"\n[鎴?鍔焆 妗ｆ宸插叏闈㈢敓鎴愬苟淇濆瓨鑷?-> '{output_json}'")

if __name__ == "__main__":
    # 鍦ㄨ繖閲岃嚜鐢遍厤缃綘璁烘枃瀹為獙娑夊強鐨勬墍鏈夋祴璇曠綉鏍硷紒
    EXPERIMENT_GRID = {
        'yolov5': {
            'end_idx': 24, # 纭 Wrapper 鏄?0 -> 24
            'batches': [16, 32, 64, 128, 256, 512],
            'resolutions': [(640, 640)]  
        },
        'resnet101': {
            'batches': [16, 32, 64, 128, 256, 512],
            'resolutions': [(224, 224)]
        },
        'vgg19': {
            'batches': [16, 32, 64, 128, 256, 512],
            'resolutions': [(224, 224)]
        },
        'swin_base': {
            'batches': [16, 32, 64, 128, 256, 512],
            'resolutions': [(224, 224)]
        },
        'vit_huge':{
            'batches': [16, 32, 64],
            'resolutions': [(224, 224)]
        }
    }

    OUTPUT_FILE = "config/dnn_profiles_database_pc.json"
    
    # 鎺ㄨ崘璺?10 杞椂寤讹紝鍙互鐔ㄥ钩 GPU 鐢靛帇璺冲姩甯︽潵鐨勮交寰共鎵?
    generate_hardware_profiles(EXPERIMENT_GRID, OUTPUT_FILE, num_runs=10)


