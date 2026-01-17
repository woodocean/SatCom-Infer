import torch
import torch.nn as nn
import time
import sys
import os
from models.dag_wrappers import YOLOv5_DAG_Wrapper, ResNet_DAG_Wrapper


# 确保能导入 models 目录下的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.AlexNet import AlexNet
from models.VggNet import vgg16_bn


class InferenceManager:
    def __init__(self, model_name="alexnet", device="cuda"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = None 
        self._load_model()

    def _load_model(self):
        print(f"[Inference] 初始化模型: {self.model_name} on {self.device}...")

        if self.model_name == "alexnet":
            # --- AlexNet 逻辑 (保持不变) ---
            self.model = AlexNet(input_channels=3, num_classes=10)
            checkpoint_path = 'checkpoints/alexnet_cifar10_epoch_115.pth'
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print(f"   [AlexNet] 权重加载成功")
                except Exception as e:
                    print(f"   [AlexNet] 权重加载错误: {e}")
            else:
                print(f"   [AlexNet] 警告: 未找到权重文件 {checkpoint_path}")
            
            self.model.to(self.device)
            self.model.eval()

        elif self.model_name == "yolov5":
            # --- YOLOv5 逻辑 (新增) ---
            # 假设你把权重放在 checkpoints/yolov5n.pt
            # YOLO(...) 会自动加载权重，不需要手写 load_state_dict
            weight_path = 'checkpoints/yolov5n.pt'
            
            # 如果本地没有，代码会尝试下载，或者报错
            # 这里的 self.model 实际上是 YOLOv5_DAG_Wrapper 的实例
            self.model = YOLOv5_DAG_Wrapper(model_path=weight_path, device=self.device)
            # 注意：Wrapper 内部已经做过 .to(device) 和 .eval() 了

        elif self.model_name == "resnet18":
            # ResNet 不需要手动指定路径，torchvision 会自动处理缓存
            self.model = ResNet_DAG_Wrapper(device=self.device)

        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        self._warm_up()

    def _warm_up(self):
        """预热逻辑更新"""
        try:
            # YOLO 和 ResNet 都可以吃大图，AlexNet 吃小图
            if self.model_name in ["yolov5", "resnet18"]:
                # 模拟 224x224 (ResNet标准) 或 640x640 (YOLO标准)
                dummy = torch.randn(1, 3, 640, 640).to(self.device)
            else:
                dummy = torch.randn(1, 3, 32, 32).to(self.device)
            
            # 调用一次 exec_layers 进行预热
            self.exec_layers(dummy, 0, 1)
            print("[Inference] 预热完成")
        except Exception as e:
            print(f"[Inference] 预热跳过: {e}")

    # ================= 核心接口 =================

    def get_layer_count(self):
        """利用模型自带的 __len__"""
        return len(self.model)

    def exec_layers(self, input_data, start_idx, end_idx):
        """
        执行切片推理
        兼容 Tensor 输入 (AlexNet/YOLO第一跳) 和 Dict 输入 (YOLO中间跳)
        """
        # 1. 数据搬运 (如果是 Dict，要把里面的 Tensor 也搬到 GPU)
        if isinstance(input_data, dict):
            # 这是一个包含缓存的数据包
            input_data['main'] = input_data['main'].to(self.device)
            for k, v in input_data['cache'].items():
                input_data['cache'][k] = v.to(self.device)
        else:
            # 这是一个纯 Tensor
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data)
            input_data = input_data.to(self.device)

        t_start = time.time()

        # 2. 推理分支
        if self.model_name in ["yolov5", "resnet18"]:
            # YOLO 走 Wrapper 的专用接口
            with torch.no_grad():
                # output 是一个字典 {'main': ..., 'cache': ...}
                output = self.model.forward_slice(input_data, start_idx, end_idx)
        else:
            # AlexNet 走原有逻辑 (线性)
            layers = []
            for i in range(start_idx, end_idx):
                layers.append(self.model[i])
            sub_model = nn.Sequential(*layers).to(self.device)
            
            with torch.no_grad():
                output = sub_model(input_data)
        
        t_end = time.time()
        cost_ms = (t_end - t_start) * 1000
        
        return output, cost_ms

    def exec_full(self, input_data):
        # 并行推理用
        if self.model_name == "yolov5":
            # YOLO 返回的是字典，我们需要取出主结果
            out_pack, cost = self.exec_layers(input_data, 0, len(self.model))
            return out_pack['main'], cost
        else:
            return self.exec_layers(input_data, 0, len(self.model))