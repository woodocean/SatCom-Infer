import torch
import torch.nn as nn
import time
import sys
import os

# 确保能导入 models 目录下的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.AlexNet import AlexNet
from models.VggNet import vgg16_bn


class InferenceManager:
    def __init__(self, model_name="alexnet", device="cpu"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = None  # 这里的 model 本身就是支持切片的

        self._load_model()

    def _load_model(self):
        print(f"[Inference] 加载模型: {self.model_name} on {self.device}...")

        if self.model_name == "alexnet":
            # AlexNet 自带 num_classes=1000，如果你做 CIFAR10 可以改为 10
            self.model = AlexNet(input_channels=3, num_classes=10)

            # 尝试加载权重 (可选)
            checkpoint_path = os.path.join(parent_dir, 'checkpoints/alexnet_cifar10_epoch_115.pth')
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  成功加载权重: {checkpoint_path}")
                except Exception as e:
                    print(f"   权重加载失败: {e}")
            else:
                print("   未找到权重文件，使用随机初始化")

        elif self.model_name == "vgg16":
            self.model = vgg16_bn(input_channels=3, num_classes=10)

        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        self.model.to(self.device)
        self.model.eval()  # 推理模式

        # 预热
        self._warm_up()

    def _warm_up(self):
        """简单的预热"""
        try:
            dummy = torch.randn(1, 3, 32, 32).to(self.device)  # CIFAR10 是 32x32
            self.exec_layers(dummy, 0, 1)
        except Exception as e:
            print(f"预热失败(忽略): {e}")

    # ================= 核心接口 =================

    def get_layer_count(self):
        """利用模型自带的 __len__"""
        return len(self.model)

    def exec_layers(self, input_data, start_idx, end_idx):
        """
        执行模型切片 [start_idx, end_idx)
        利用模型自带的 __getitem__ 特性
        """
        # 1. 确保数据是 Tensor 并移至设备
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)
        input_data = input_data.to(self.device)

        # 2. 提取层序列
        # 原作者的模型类支持 model[i] 获取第 i 层
        # 我们把它们串起来变成一个临时的小模型
        layers = []
        for i in range(start_idx, end_idx):
            layers.append(self.model[i])

        sub_model = nn.Sequential(*layers)
        sub_model.to(self.device)  # 确保子层也在设备上

        # 3. 执行推理
        t_start = time.time()
        with torch.no_grad():
            output = sub_model(input_data)
        t_end = time.time()

        cost_ms = (t_end - t_start) * 1000
        return output, cost_ms

    def exec_full(self, input_data):
        return self.exec_layers(input_data, 0, len(self.model))