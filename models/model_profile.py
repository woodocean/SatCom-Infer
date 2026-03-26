import torch
import time
import json
import numpy as np
from thop import profile
from dag_wrappers import (
    ResNet_DAG_Wrapper, VGG19_DAG_Wrapper, YOLOv5_DAG_Wrapper, 
    MobileNetV2_DAG_Wrapper, UNet_DAG_Wrapper, AlexNet_DAG_Wrapper
)

class ModelProfiler:
    def __init__(self, device='cuda'):
        self.device = device

    def get_model_size_mb(self, module):
        """计算模块权重大小 (MB)"""
        param_size = 0
        for param in module.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)

    def get_tensor_size_mb(self, obj):
        """计算张量或字典中所有张量的大小 (MB)"""
        if isinstance(obj, torch.Tensor):
            return obj.nelement() * obj.element_size() / (1024 * 1024)
        elif isinstance(obj, dict):
            return sum(self.get_tensor_size_mb(v) for v in obj.values())
        return 0

    def profile_model(self, wrapper, input_shape=(1, 3, 224, 224), iterations=50):
        print(f"\n>>> 正在剖析模型: {wrapper.__class__.__name__} | 输入: {input_shape}")
        wrapper.eval().to(self.device)
        
        # 准备初始输入
        dummy_input = torch.randn(*input_shape).to(self.device)
        layer_results = []
        
        # 初始 Pack (模拟 forward_slice 的逻辑)
        current_pack = dummy_input 
        
        total_layers = len(wrapper)
        
        with torch.no_grad():
            for i in range(total_layers):
                # 1. 计算当前层的计算量 (FLOPs)
                # 由于 thop 无法直接处理 forward_slice 复杂的字典逻辑，我们需要单独提取层对象
                # 注意：这里只针对非容器层计算 FLOPs
                layer_obj = wrapper.layers[i]
                try:
                    # 模拟一层的输入来算 FLOPs
                    sample_input = current_pack if isinstance(current_pack, torch.Tensor) else current_pack['main']
                    # 简化处理：如果不涉及Skip-connection，直接算；否则记为0或特殊处理
                    flops, _ = profile(layer_obj, inputs=(sample_input,), verbose=False)
                except:
                    flops = 0

                # 2. 测量推理时延 (GPU Synchronized)
                # 预热
                for _ in range(10):
                    _ = wrapper.forward_slice(current_pack, i, i + 1)
                
                torch.cuda.synchronize()
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                
                latencies = []
                for _ in range(iterations):
                    starter.record()
                    _ = wrapper.forward_slice(current_pack, i, i + 1)
                    ender.record()
                    torch.cuda.synchronize()
                    latencies.append(starter.elapsed_time(ender))
                
                avg_latency = np.mean(latencies)

                # 3. 运行一次拿到真实的输出包，用于计算大小和作为下一层输入
                output_pack = wrapper.forward_slice(current_pack, i, i + 1)
                
                # 4. 统计数据大小
                main_output_size = self.get_tensor_size_mb(output_pack['main'])
                # 通信量 = main + cache
                total_comm_size = main_output_size + self.get_tensor_size_mb(output_pack['cache'])
                weight_size = self.get_model_size_mb(layer_obj)
                
                # 记录
                res = {
                    "layer_idx": i,
                    "layer_type": str(type(layer_obj)).split('.')[-1].replace("'>", ""),
                    "latency_ms": avg_latency,
                    "flops": flops,
                    "params_mb": weight_size,
                    "output_main_mb": main_output_size,
                    "output_total_comm_mb": total_comm_size,
                    "input_shape": list(sample_input.shape)
                }
                layer_results.append(res)
                
                # 更新输入，准备测下一层
                current_pack = output_pack
                print(f"  Layer {i:02d} | Latency: {avg_latency:.3f}ms | Comm: {total_comm_size:.2f}MB")

        return layer_results

# === 运行测试 ===
if __name__ == "__main__":
    profiler = ModelProfiler()
    
    # 定义测试任务
    # 为了完成你的第5点，我们先测 ResNet50 和 YOLOv5
    test_configs = [
        {"name": "ResNet101", "model": ResNet_DAG_Wrapper(version='101')},
        # {"name": "YOLOv5", "model": YOLOv5_DAG_Wrapper(model_path='yolov5s.pt')}, # 确保路径正确
    ]
    
    db = {}
    for config in test_configs:
        results = profiler.profile_model(config['model'], input_shape=(1, 3, 224, 224))
        db[config['name']] = results
        
    # 保存为查找表
    with open("inference_lookup_table.json", "w") as f:
        json.dump(db, f, indent=4)
        
    print("\n[Done] 查找表已生成至 inference_lookup_table.json")