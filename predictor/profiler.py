import torch
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class DeviceProfiler:
    """设备性能自动探测"""

    def __init__(self, node_id):
        self.node_id = node_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_hardware(self):
        info = {
            "node_id": self.node_id,
            "device": self.device,
            "cuda": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
        try:
            import psutil
            info["cpu_count"] = psutil.cpu_count()
            info["ram_mb"] = psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            info["cpu_count"] = os.cpu_count()
        return info

    def benchmark_model(self, model_name='alexnet', warmup=3, repeat=10):
        from core.inference import InferenceEngine
        engine = InferenceEngine(self.node_id, model_name)
        engine.load_model()

        if model_name in ['alexnet', 'lenet']:
            x = torch.randn(1, 3, 32, 32)
        else:
            x = torch.randn(1, 3, 224, 224)

        # Warmup
        for _ in range(warmup):
            engine.run_full(x)

        # Benchmark
        times = []
        for _ in range(repeat):
            _, ms = engine.run_full(x)
            times.append(ms)

        return {
            "model": model_name,
            "device": self.device,
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
        }

    def profile_layers(self, model_name='alexnet'):
        from core.inference import InferenceEngine
        engine = InferenceEngine(self.node_id, model_name)
        engine.load_model()

        if model_name in ['alexnet', 'lenet']:
            x = torch.randn(1, 3, 32, 32)
        else:
            x = torch.randn(1, 3, 224, 224)

        # Warmup
        for _ in range(warmup):
            engine.run_full(x)

        # Benchmark
        times = []
        for _ in range(repeat):
            _, ms = engine.run_full(x)
            times.append(ms)

        return {
            "model": model_name,
            "device": self.device,
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
        }

    def profile_layers(self, model_name='alexnet'):
        from core.inference import InferenceEngine
        engine = InferenceEngine(self.node_id, model_name)
        engine.load_model()

        if model_name in ['alexnet', 'lenet']:
            x = torch.randn(1, 3, 32, 32)
        else:
            x = torch.randn(1, 3, 224, 224)

        _, _, layer_details = engine.run_layers(x, 0, engine.num_layers - 1)
        
        profile_data = []
        for d in layer_details:
            profile_data.append({
                'layer_idx': d['idx'],
                'layer_type': d['name'],
                'flops': d['time_ms'] * 1e6,          # 用实测时间近似FLOPs（论文建模够用）
                'output_mb': d['output_mb'],
                'mem_mb': d['output_mb'] * 2,
                'compute_time_ms': d['time_ms']
            })
        return profile_data