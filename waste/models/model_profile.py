import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dag_wrappers import (
    VGG19_DAG_Wrapper, ResNet_DAG_Wrapper, YOLOv5_DAG_Wrapper, 
    Swin_Base_DAG_Wrapper
)

class ModelProfiler:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def profile_model_with_sizes(self, name, wrapper_class, sizes, **kwargs):
        print(f"\n[Profiling] 正在为模型 {name} 测量多尺寸【逐层】时延...")
        model = wrapper_class(device=self.device, **kwargs)

        total_layers = len(model.layers)
        size_layer_latencies = {}  # {(H,W): [layer0_ms, layer1_ms, ...]}
        valid_sizes = []

        for h, w in sizes:
            input_size = (1, 3, h, w)
            print(f"  → 尝试测量尺寸 {h}x{w}...")
            
            try:
                # 校验是否会维度异常报错（如果不行直接触发异常跳过）
                dummy = torch.randn(*input_size).to(self.device).requires_grad_(False)
                with torch.no_grad():
                    for i in range(total_layers):
                        dummy = model.forward_slice(dummy, i, i + 1)
                
                # 开始逐层精准测速 (为消除误差，每层跑5次求平均)
                layer_times = [0.0] * total_layers
                runs = 5
                with torch.no_grad():
                    for _ in range(runs):
                        dummy = torch.randn(*input_size).to(self.device)
                        for i in range(total_layers):
                            starter = torch.cuda.Event(enable_timing=True)
                            ender = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            
                            starter.record()
                            dummy = model.forward_slice(dummy, i, i + 1)
                            ender.record()
                            
                            torch.cuda.synchronize()
                            layer_times[i] += starter.elapsed_time(ender)
                            
                # 求平均值
                avg_layer_times = [t / runs for t in layer_times]
                size_layer_latencies[(h, w)] = avg_layer_times
                valid_sizes.append((h, w))
                
                total_t = sum(avg_layer_times)
                print(f"    ✅ 测量完成! 总耗时 {total_t:.2f} ms")

            except (RuntimeError, AssertionError, ValueError) as e:
                print(f"    ⚠️  跳过: 无法处理尺寸 {h}x{w} | 架构限制原因: {type(e).__name__}")
                continue

        if len(valid_sizes) < 2:
            print(f"❌ 模型 {name} 成功执行的尺寸不足2个，无法进行【面积—时延】拟合！")
            return None

        # ================= 为“每一层”独立进行线性拟合 =================
        areas = np.array([h * w for h, w in valid_sizes])
        layer_fits = []
        
        # 终端表格打印表头
        print(f"\n📊 [{name}] 逐层线性拟合结果分析 (y = Slope * Area + Intercept):")
        print(f"| {'Layer':<6} | {'Slope (ms/px)':<15} | {'Intercept (ms)':<15} | {'R²':<8} | {'MSE (ms²)':<10} |")
        print("-" * 69)

        for i in range(total_layers):
            y_latencies = np.array([size_layer_latencies[(h, w)][i] for h, w in valid_sizes])
            coeffs = np.polyfit(areas, y_latencies, 1)  # 一元一次线性回归
            
            # 由于可能出现测量极小误差，斜率若为微小负数，则拉平为0
            slope = float(coeffs[0]) if coeffs[0] > 0 else 0.0
            intercept = float(coeffs[1]) if coeffs[1] > 0 else 0.0
            
            # 依据裁剪后的合法方程重新测算预测值，来评估此方程的误差
            y_pred = slope * areas + intercept
            
            # 计算 均方误差 (MSE) 和 决定系数 (R²)
            mse = np.mean((y_latencies - y_pred) ** 2)
            ss_tot = np.sum((y_latencies - np.mean(y_latencies)) ** 2)
            
            # 如果某层是不受面积影响的固定耗时开销层 (如 FC)，ss_tot 会趋近于0
            if ss_tot > 1e-10:
                r_squared = 1.0 - (np.sum((y_latencies - y_pred) ** 2) / ss_tot)
            else:
                r_squared = 1.0 if mse < 1e-5 else 0.0

            layer_fits.append({
                "layer_idx": i,
                "slope": slope,
                "intercept": intercept,
                "r_squared": float(r_squared),
                "mse": float(mse)
            })
            
            # 将该层结果输出在终端表格里
            print(f"| Layer {i:<2} | {slope:>14.3e}  | {intercept:>14.4f}  | {r_squared:>8.4f} | {mse:>10.4e} |")

        print("-" * 69)
        
        return {
            "model_name": name,
            "anchor_sizes": valid_sizes,
            "layer_fits": layer_fits     # 精确的逐层公式
        }

def run_size_scaling_profile():
    profiler = ModelProfiler()
    results = {}
    anchor_sizes = [(224, 224), (384, 384), (640, 640), (1024, 1024)]

    configs = [
        ("vgg19", VGG19_DAG_Wrapper, {}),
        ("resnet101", ResNet_DAG_Wrapper, {"version": "101"}),
        ("yolov5", YOLOv5_DAG_Wrapper, {"model_path": "../checkpoints/yolov5nu.pt"})
    ]

    for name, cls, args in configs:
        res = profiler.profile_model_with_sizes(name, cls, anchor_sizes, **args)
        if res is not None:
            results[name] = res

    with open("model_profiles_sizes.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[Done] ✅ 模型的【逐层变尺寸】时延回归参数已存入 model_profiles_sizes.json")

if __name__ == "__main__":
    run_size_scaling_profile()