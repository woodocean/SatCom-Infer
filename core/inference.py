import torch
import time
import importlib
import sys
import os
import torch.nn as nn

# 确保能导入 models 和 dag_wrappers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 1. 尝试导入 dag_wrappers
try:
    from models.dag_wrappers import (
        YOLOv5_DAG_Wrapper,
        ResNet_DAG_Wrapper,
        VGG19_DAG_Wrapper,
        MobileNetV2_DAG_Wrapper,
        ViT_Huge_DAG_Wrapper,
        AlexNet_DAG_Wrapper,
        Swin_Base_DAG_Wrapper
    )
    HAS_DAG_WRAPPERS = True
except ImportError as e:
    print(f"[WARNING] dag_wrappers.py not found: {e}")
    HAS_DAG_WRAPPERS = False

# from models.AlexNet import AlexNet
# from models.VggNet import vgg16_bn
# from models.MobileNet import MobileNet
# from models.LeNet import LeNet

class InferenceEngine:
    """
    推理引擎 - 支持全模型推理 & 分层推理
    兼容项目中 models/ 目录下的 AlexNet, VggNet, MobileNet, LeNet
    以及通过 dag_wrappers.py 支持的 YOLOv5, ResNet, ViT 等 DAG 结构模型
    """

    def __init__(self, node_id, model_name='alexnet'):
        self.node_id = node_id
        self.model_name = model_name.lower()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.layers = []          # 展平后的层列表 [(name, module), ...] 或直接为 list[nn.Module]
        self.num_layers = 0
        self.is_dag_wrapper = False  # 标志：是否使用了 dag_wrappers 的包装类

    def load_model(self, checkpoint_path=None):
        """加载模型，支持原生模型与 DAG Wrapper 模型"""
        print(f"[{self.node_id}] 加载模型: {self.model_name} -> {self.device}")

        if HAS_DAG_WRAPPERS:
            # ==============================
            # 优先尝试 DAG Wrapper 模型
            # ==============================
            if self.model_name == 'yolov5':
                self.model = YOLOv5_DAG_Wrapper(model_path='checkpoints/yolov5nu.pt', device=self.device)
                self.is_dag_wrapper = True
            elif 'resnet' in self.model_name:
                version = self.model_name.replace('resnet', '').strip()
                self.model = ResNet_DAG_Wrapper(version=version, device=self.device)
                self.is_dag_wrapper = True
            elif self.model_name == 'vgg19':
                self.model = VGG19_DAG_Wrapper(device=self.device)
                self.is_dag_wrapper = True
            elif self.model_name == 'mobilenet':
                self.model = MobileNetV2_DAG_Wrapper(device=self.device)
                self.is_dag_wrapper = True
            elif self.model_name == 'vit_huge':
                self.model = ViT_Huge_DAG_Wrapper(device=self.device)
                self.is_dag_wrapper = True
            elif self.model_name == 'alexnet':
                self.model = AlexNet_DAG_Wrapper(device=self.device)
                self.is_dag_wrapper = True
            elif self.model_name == 'swin_base':
                self.model = Swin_Base_DAG_Wrapper(device=self.device, img_size=224)
                self.is_dag_wrapper = True

        # ==============================
        # 否则走原生线性模型路径
        # ==============================
        # if not self.is_dag_wrapper:
        #     if self.model_name == 'alexnet':
        #         self.model = AlexNet().to(self.device)
        #     elif self.model_name == 'vgg16_bn':
        #         self.model = vgg16_bn().to(self.device)
        #     elif self.model_name == 'mobilenetv2':
        #         self.model = MobileNet().to(self.device)
        #     elif self.model_name == 'lenet':
        #         self.model = LeNet().to(self.device)
        #     else:
        #         raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.eval()

        # 统一构建 layers 列表（兼容两种路径）
        if self.is_dag_wrapper:
            # Wrapper 类已预构建 self.layers（是 nn.Module 列表）
            self.layers = getattr(self.model, 'layers', [])
        else:
            # 原生模型：手动展平（仅适用于 Sequential-like）
            self._extract_sequential_layers()
        self.num_layers = len(self.layers)

        print(f"[{self.node_id}] 模型加载完成，总层数: {self.num_layers}，DAG模式: {self.is_dag_wrapper}")

    def _extract_sequential_layers(self):
        """为原生模型手动提取层（仅适用于无跳连模型）"""
        self.layers.clear()
        if hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            for layer in self.model.features:
                self.layers.append(('features', layer))
            self.layers.append(('flatten', nn.Flatten()))
            for layer in self.model.classifier:
                self.layers.append(('classifier', layer))
        else:
            # fallback: 递归遍历所有子模块（谨慎，可能含重复）
            for name, module in self.model.named_children():
                if isinstance(module, nn.Sequential):
                    for sub_module in module:
                        self.layers.append((name, sub_module))
                else:
                    self.layers.append((name, module))

    def exec_layers(self, input_data, start_layer, end_layer):
        """
        执行从 start_layer 到 end_layer（含）的层
        input_data: torch.Tensor 或 dict (DAG Wrapper 可能需要 dict 输入)
        """
        if not self.is_dag_wrapper and len(self.layers) == 0:
            print(f"[{self.node_id}] [WARNING] 引擎未加载模型或无计算层，直接透传数据！")
            return input_data, 0.0
        
        
        if isinstance(input_data, torch.Tensor):
            x = input_data.to(self.device)
        elif isinstance(input_data, dict):
            # 如果是 DAG 含有缓存的字典，把里面的张量都转移到对应设备
            x = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in input_data.items()}
        else:
            x = input_data

        start_time = time.time()
        with torch.no_grad():
            if self.is_dag_wrapper:
                # 关键：DAG Wrapper 必须用其 own forward_slice
                # 注意：wrapper.forward_slice 返回 dict({'main': ..., 'cache': ...})
                result = self.model.forward_slice(x, start_layer, end_layer + 1)
                x = result['main']
            else:
                # 原逻辑：直接遍历 self.layers[i][1]（即 module）
                for i in range(start_layer, end_layer + 1):
                    if isinstance(self.layers[i], tuple):
                        _, layer = self.layers[i]
                    else:
                        layer = self.layers[i]
                    x = layer(x)

        end_time = time.time()
        cost_ms = (end_time - start_time) * 1000
        return x, cost_ms

    def run_full(self, input_data):
        """运行完整模型"""
        return self.exec_layers(input_data, 0, self.num_layers - 1)