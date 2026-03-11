# config.py
import torch

# 模型物理特征库 (基于 FP32 实测)
MODEL_CONFIGS = {
    "VGG19": {"weights_mb": 548.0, "gamma": 4.5, "layers": 16, "ops_gflops": 19.6},
    "ResNet101": {"weights_mb": 170.0, "gamma": 1.2, "layers": 101, "ops_gflops": 7.6},
    "YOLOv5n": {"weights_mb": 7.5, "gamma": 0.4, "layers": 50, "ops_gflops": 4.5},
    "ViT": {"weights_mb": 340.0, "gamma": 1.0, "layers": 12, "ops_gflops": 18.2}
}

# 节点拓扑与能力配置 (Heterogeneity Factor)
# Factor < 1 代表性能更强, Factor > 1 代表时延增加
NODES = {
    "RS": {"type": "PC", "factor": 1.0, "mem_limit": 32000},
    "GS": {"type": "PC", "factor": 1.0, "mem_limit": 32000},
    "SAT-01": {"type": "Jetson_1", "factor": 1.2, "mem_limit": 8000}, # 邻居1
    "SAT-02": {"type": "Jetson_2", "factor": 1.2, "mem_limit": 8000}, # 邻居2
    "SAT-03": {"type": "Jetson_1", "factor": 1.5, "mem_limit": 8000}, # 邻居3 & 起点 & 聚合
    "SAT-04": {"type": "Jetson_2", "factor": 1.2, "mem_limit": 8000}, # 流水线节点
    "SAT-05": {"type": "Jetson_1", "factor": 1.5, "mem_limit": 8000}  # 流水线节点
}

# 带宽配置 (Mbps)
STAR_BANDWIDTH = 1000.0  # 星间激光 1Gbps
GROUND_BANDWIDTH = 200.0 # 弯管回传/星地 200Mbps
PROPAGATION_DELAY = 0.01 # 10ms 传输延迟