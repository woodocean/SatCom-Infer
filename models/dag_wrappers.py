import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import resnet18, ResNet18_Weights
import os

class YOLOv5_DAG_Wrapper(nn.Module):
    
    def __init__(self, model_path, device='cuda'):
        """
        model_path: 权重文件的路径，例如 'checkpoints/yolov5n.pt'
        """
        super().__init__()
        self.device = device
        
        # --- 核心解释 ---
        # 这一行代码 YOLO(model_path) 做了三件事：
        # 1. 解析 .pt 文件里的 yaml 配置，构建网络结构 (Structure)
        # 2. 读取 .pt 文件里的 state_dict，加载权重 (Weights)
        # 3. 如果文件不存在，它会自动尝试去官网下载 (但在卫星环境最好手动提供)
        print(f"[Wrapper] 正在加载 YOLOv5 及其权重: {model_path} ...")
        try:
            self.yolo_wrapper = YOLO(model_path)
        except Exception as e:
            print(f"错误: 加载权重失败! 请检查路径 {model_path}")
            raise e

        # 提取核心模型并放入设备 (Jetson GPU)
        self.model = self.yolo_wrapper.model.to(device)
        self.model.eval() # 设为推理模式，冻结 BatchNorm 等
        
        # 展平层列表，方便通过 index 访问
        self.layers = list(self.model.model.children())
        self.len = len(self.layers)
        
        # 获取需要保存中间结果的层索引 (处理 DAG 跳跃连接的关键)
        self.save_indices = self.model.save 
        
        # 本地缓存
        self.feature_cache = {}

    def __len__(self):
        return self.len

    def reset_cache(self):
        self.feature_cache = {}

    def forward_slice(self, input_data, start_idx, end_idx):
        """
        执行模型切片推理
        input_data: 
            - 若为 Tensor: 说明是第一跳
            - 若为 Dict: {'main': tensor, 'cache': {...}} 说明是中间跳
        """
        # 1. 解析输入
        if isinstance(input_data, torch.Tensor):
            current_input = input_data
            if start_idx == 0: self.reset_cache()
        elif isinstance(input_data, dict):
            current_input = input_data['main']
            # 合并上游传来的缓存到本地
            self.feature_cache.update(input_data['cache'])
        else:
            raise ValueError(f"输入类型错误: {type(input_data)}")

        # 2. 逐层计算
        for i, m in enumerate(self.layers):
            if i < start_idx: continue
            if i >= end_idx: break
            
            # --- 处理 DAG 输入 ---
            if m.f != -1: 
                if isinstance(m.f, int):
                    required_idx = m.f if m.f >= 0 else i + m.f
                    if required_idx == i - 1:
                        x = current_input
                    else:
                        x = self.feature_cache.get(required_idx, current_input)
                else:
                    # Concat 层
                    input_tensors = []
                    for idx in m.f:
                        real_idx = idx if idx >= 0 else i + idx
                        if real_idx == i - 1:
                            input_tensors.append(current_input)
                        else:
                            # 如果缓存里没有，说明切分点选的有问题，但在仿真中可以用 current_input 兜底防崩
                            cached = self.feature_cache.get(real_idx, current_input)
                            input_tensors.append(cached)
                    x = input_tensors
            else:
                x = current_input

            # --- 执行计算 ---
            try:
                current_output = m(x)
            except:
                # 兼容 Head 层或其他特殊层
                current_output = m(x)

            # --- 缓存结果 ---
            if i in self.save_indices:
                self.feature_cache[i] = current_output
            
            current_input = current_output

        # 3. 打包输出
        return {
            'main': current_output,
            'cache': self.feature_cache
        }
class ResNet_DAG_Wrapper(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Wrapper] 正在加载 ResNet18 (Torchvision官方权重) ...")
        
        # 1. 加载官方预训练模型
        # weights='DEFAULT' 会自动下载 ImageNet 的预训练权重
        try:
            self.raw_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        except:
            # 兼容旧版本 torchvision
            self.raw_model = resnet18(pretrained=True).to(device)
            
        self.raw_model.eval()
        
        # 2. 【关键】手动将 ResNet 拆解为 6 个大的 Stage
        # ResNet = Stem(前处理) + 4个Layer(残差块组) + Head(分类头)
        self.layers = nn.ModuleList([
            # Layer 0: Stem (Conv1 + BN + ReLU + MaxPool)
            nn.Sequential(
                self.raw_model.conv1,
                self.raw_model.bn1,
                self.raw_model.relu,
                self.raw_model.maxpool
            ),
            # Layer 1: Stage 1
            self.raw_model.layer1,
            # Layer 2: Stage 2
            self.raw_model.layer2,
            # Layer 3: Stage 3
            self.raw_model.layer3,
            # Layer 4: Stage 4
            self.raw_model.layer4,
            # Layer 5: Head (AvgPool + Flatten + FC)
            # 注意：原始 forward 里的 flatten 是函数调用，这里我们需要用 nn.Flatten 模块代替
            nn.Sequential(
                self.raw_model.avgpool,
                nn.Flatten(),
                self.raw_model.fc
            )
        ])
        
        self.len = len(self.layers)
        
        # ResNet 在 Stage 级别没有跨层依赖，不需要 save_indices
        self.save_indices = [] 

    def __len__(self):
        return self.len

    def forward_slice(self, input_pack, start_idx, end_idx):
        """
        统一接口：支持切片推理
        """
        # 1. 解析输入 (兼容 YOLO 的接口格式)
        if isinstance(input_pack, dict):
            x = input_pack['main']
            # ResNet 不需要读取 cache，忽略 input_pack['cache']
        else:
            x = input_pack # 第一跳传入的是 Tensor
            
        # 2. 逐层计算
        for i, layer in enumerate(self.layers):
            if i < start_idx: continue
            if i >= end_idx: break
            
            x = layer(x)
        
        # 3. 打包输出
        # ResNet 不需要缓存中间层，所以 cache 为空字典
        # 这样设计是为了和 YOLO 的通信协议保持一致
        output_pack = {
            'main': x,
            'cache': {} 
        }
        
        return output_pack