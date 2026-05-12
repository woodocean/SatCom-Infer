import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from ultralytics import YOLO
from torchvision.models import (
    resnet18, resnet50, resnet101, 
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    vgg19, VGG19_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from torchvision.models import alexnet, AlexNet_Weights

# === 1. YOLOv5 ===
# === 1. YOLOv5 ===
class YOLOv5_DAG_Wrapper(nn.Module):
    def __init__(self, model_path='models/checkpoints/yolov5nu.pt', device='cuda'):
        super().__init__()
        self.device = device
        self.max_det = 100
        model_path = self._resolve_local_model_path(model_path)
        print(f"[Wrapper] YOLOv5 ({model_path})...")
        
        # 1. 使用临时变量加载，避免污染 self 命名空间，防止被框架的任何 hook 误触！
        temp_yolo = YOLO(model_path)
        
        # 2. 提取最纯净的 PyTorch 底层计算模型 (DetectionModel)
        self.model = temp_yolo.model.to(device)
        self.model.eval()                                   # 设置为推理模式
        
        # 3. 彻底锁死所有参数梯度，节约大量显存并斩断训练可能
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.layers = list(self.model.model.children())     # 将YOLO模型的子层转为可索引的list
        
        self.len = len(self.layers)
        self.save_indices = getattr(self.model, 'save', []) # 获取需要缓存输出的层索引
        
        # 4. 彻底删除临时的高级包装器对象，切断后续触发下载数据集的隐患
        del temp_yolo
        
        self.feature_cache = {}             # 用于缓存需要保存的层的输出张量

    def __len__(self): 
        return self.len

    @staticmethod
    def _resolve_local_model_path(model_path):
        """只使用项目内本地权重，避免 Ultralytics 在 Jetson 上自动联网下载。"""
        raw = Path(model_path)
        candidates = [
            raw,
            Path('models/checkpoints') / raw.name,
            Path('checkpoints') / raw.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        checked = ', '.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"YOLOv5 local checkpoint not found. Checked: {checked}. "
            "Please copy yolov5nu.pt to models/checkpoints/ before profiling."
        )
    
    def reset_cache(self): 
        self.feature_cache = {}             # 清空缓存张量

    def _finalize_detection_output(self, output_pack):
        """将 Detect 末端输出压成紧凑任务结果，避免把检测头特征当作最终通信结果。"""
        if not isinstance(output_pack, tuple) or len(output_pack) == 0:
            return output_pack

        pred = output_pack[0]
        if not isinstance(pred, torch.Tensor) or pred.ndim != 3 or pred.shape[1] < 6:
            return output_pack

        pred = pred.transpose(1, 2).contiguous()  # [B, N, C]
        boxes = pred[..., :4]
        class_logits = pred[..., 4:]
        if class_logits.shape[-1] == 0:
            return output_pack

        scores, class_ids = torch.max(class_logits, dim=-1)
        topk = min(self.max_det, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k=topk, dim=1)

        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, 4)
        top_boxes = torch.gather(boxes, 1, gather_idx)
        top_classes = torch.gather(class_ids, 1, top_indices).to(top_boxes.dtype).unsqueeze(-1)
        top_scores = top_scores.unsqueeze(-1)
        return torch.cat([top_boxes, top_scores, top_classes], dim=-1)

    # 基于yolo模型的层级切分 
    # input_pack 输入数据 start_idx, end_idx 始末层序号
    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, torch.Tensor):    # 输入数据只是张量的话 说明不需要处理缓存数据
            current_input = input_pack
            if start_idx == 0: 
                self.reset_cache()
        elif isinstance(input_pack, dict):          # 输入数据是字典 说明有前级模块需要传递的缓存数据
            current_input = input_pack['main']      # 提取‘main’部分作为层级输入
            self.feature_cache.update(input_pack.get('cache', {}))  # 提取cache部分作为这个yolo包装器对象的缓存
        else:
            raise ValueError(f"Type error: {type(input_pack)}")

        # 遍历从start_idx到end_idx的层 i为索引层序号 m为层对象
        for i, m in enumerate(self.layers):
            if i < start_idx: 
                continue
            if i >= end_idx: 
                break
            
            if hasattr(m, 'f') and m.f != -1:       # 判断该层对象是否需要合并之前的特征图
                if isinstance(m.f, int):
                    required_idx = m.f if m.f >= 0 else i + m.f
                    x = self.feature_cache.get(required_idx, current_input) if required_idx != i - 1 else current_input
                else:
                    x = [self.feature_cache.get(idx if idx >= 0 else i + idx, current_input) 
                         if (idx if idx >= 0 else i + idx) != i - 1 else current_input for idx in m.f]
            else:
                x = current_input

            try:
                current_output = m(x)
            except Exception as e:
                # 对于特殊层（如Detect），可能需要特殊处理
                if 'Detect' in str(type(m)):
                    # Detect层通常需要模型参数，这里简化处理
                    current_output = current_input
                else:
                    print(f"Warning: Layer {i} ({type(m)}) failed: {e}")
                    current_output = current_input

            if i in self.save_indices:
                self.feature_cache[i] = current_output
            current_input = current_output

        active_cache = {}
        # 扫描在切分点 (end_idx) 之后的所有层，看看它们还需要前面产生的哪些特征图
        for future_i in range(end_idx, self.len):
            future_m = self.layers[future_i]
            if hasattr(future_m, 'f') and future_m.f != -1:
                refs = future_m.f if isinstance(future_m.f, list) else [future_m.f]
                for ref in refs:
                    req_idx = ref if ref >= 0 else future_i + ref
                    
                    # 规则 A: 依赖项必须在切分点及之前产生（即小于等于我们刚执行完的那一层）
                    # 修正：将 < end_idx - 1 改为 < end_idx，救活刚刚产生的 Cache！
                    if req_idx < end_idx and req_idx in self.feature_cache:
                        active_cache[req_idx] = self.feature_cache[req_idx]

        if end_idx >= self.len:
            current_output = self._finalize_detection_output(current_output)
            active_cache = {}

        # 更新自身的缓存为“已净化”版本
        self.feature_cache = active_cache


        return {'main': current_output, 'cache': self.feature_cache}
    
# === 2. ResNet ===
class ResNet_DAG_Wrapper(nn.Module):
    def __init__(self, version='50', device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Wrapper] ResNet-{version} (使用随机权重，避免下载)...")
        
        if str(version) == '18':
            raw = resnet18(weights=None).to(device)
        elif str(version) == '50':
            raw = resnet50(weights=None).to(device)
        elif str(version) == '101':
            raw = resnet101(weights=None).to(device)
        else:
            raw = resnet50(weights=None).to(device)
        raw.eval()      
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(raw.conv1, raw.bn1, raw.relu, raw.maxpool))
        for block in raw.layer1: 
            self.layers.append(block)
        for block in raw.layer2: 
            self.layers.append(block)
        for block in raw.layer3: 
            self.layers.append(block)
        for block in raw.layer4: 
            self.layers.append(block)
        self.layers.append(nn.Sequential(raw.avgpool, nn.Flatten(), raw.fc, nn.Softmax(dim=1)))
        
        self.len = len(self.layers)
        self.save_indices = []

    def __len__(self): 
        return self.len

    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
        else: 
            x = input_pack
        
        for i in range(start_idx, end_idx): 
            # print(f"[层 {i}] 输入形状: {x.shape}")
            x = self.layers[i](x)
            # print(f"[层 {i}] 输出形状: {x.shape}")

        return {'main': x, 'cache': {}}

# === 3. VGG-19 ===
class VGG19_DAG_Wrapper(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Wrapper] 正在加载 VGG-19 (随机权重模式)...")
        self.raw_model = vgg19(weights=None).to(device)
        self.raw_model.eval()
        self.layers = nn.ModuleList()
        for layer in self.raw_model.features: 
            self.layers.append(layer)
        self.layers.append(self.raw_model.avgpool)
        self.layers.append(nn.Flatten())
        for layer in self.raw_model.classifier: 
            self.layers.append(layer)
        self.layers.append(nn.Softmax(dim=1))
        self.len = len(self.layers)
        self.save_indices = []

    def __len__(self): 
        return self.len
    
    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
        else: 
            x = input_pack
        
        for i in range(start_idx, end_idx): 
            x = self.layers[i](x)
        
        return {'main': x, 'cache': {}}

# === 4. MobileNet V2 ===
class MobileNetV2_DAG_Wrapper(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Wrapper] 正在加载 MobileNet V2 (随机权重模式)...")
        self.raw_model = mobilenet_v2(weights=None).to(device)
        self.raw_model.eval()
        self.layers = nn.ModuleList()
        for layer in self.raw_model.features: 
            self.layers.append(layer)
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers.append(nn.Flatten())
        self.layers.append(self.raw_model.classifier)
        self.layers.append(nn.Softmax(dim=1))
        self.len = len(self.layers)
        self.save_indices = []

    def __len__(self): 
        return self.len
    
    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
        else: 
            x = input_pack
        
        for i in range(start_idx, end_idx): 
            x = self.layers[i](x)
        
        return {'main': x, 'cache': {}}

# === 5. ViT Huge ===
class ViT_Huge_DAG_Wrapper(nn.Module):
    def __init__(self, device='cuda', img_size=224):
        super().__init__()
        self.device = device
        print(f"[Wrapper] 正在加载 ViT-Huge (NO Pretrain, Size={img_size})...")
        self.raw_model = timm.create_model(
            'vit_huge_patch14_224',
            pretrained=False,
            img_size=img_size,
            num_classes=1000,
        )
        if getattr(self.raw_model, 'num_classes', 0) != 1000 or isinstance(getattr(self.raw_model, 'head', None), nn.Identity):
            if hasattr(self.raw_model, 'reset_classifier'):
                self.raw_model.reset_classifier(num_classes=1000)
            if isinstance(getattr(self.raw_model, 'head', None), nn.Identity):
                self.raw_model.head = nn.Linear(getattr(self.raw_model, 'num_features', 1280), 1000)
        self.raw_model = self.raw_model.to(device)
        self.raw_model.eval()
        self.layers = nn.ModuleList()
        self.layers.append(self.raw_model.patch_embed)
        for block in self.raw_model.blocks: 
            self.layers.append(block)
        self.layers.append(self.raw_model.norm)
        
        # 自定义 Head 处理 CLS token
        class CLSHead(nn.Module):
            def __init__(self, original_head):
                super().__init__()
                self.original_head = original_head
            def forward(self, x):
                return self.original_head(x[:, 0])  # 取 CLS token
        
        self.layers.append(CLSHead(self.raw_model.head))
        self.layers.append(nn.Softmax(dim=1))
        self.len = len(self.layers)
        self.save_indices = []

    def __len__(self): 
        return self.len
    
    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
        else: 
            x = input_pack
        
        for i in range(start_idx, end_idx): 
            x = self.layers[i](x)
        
        return {'main': x, 'cache': {}}

# === 6. Swin Transformer ===
class Swin_Base_DAG_Wrapper(nn.Module):
    def __init__(self, device='cuda', img_size=224):
        super().__init__()
        self.device = device
        print(f"[Wrapper] 正在加载 Swin-Base (NO Pretrain, Size={img_size})...")
        self.raw_model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, img_size=img_size).to(device)
        self.raw_model.eval()
        self.layers = nn.ModuleList()
        self.layers.append(self.raw_model.patch_embed)
        for layer in self.raw_model.layers: 
            self.layers.append(layer)
        self.layers.append(self.raw_model.norm)
        self.layers.append(self.raw_model.head)
        self.layers.append(nn.Softmax(dim=1))
        self.len = len(self.layers)
        self.save_indices = []

    def __len__(self): 
        return self.len
    
    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
        else: 
            x = input_pack
        
        for i in range(start_idx, end_idx): 
            x = self.layers[i](x)
        
        return {'main': x, 'cache': {}}

# === 7. U-Net ===
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: 
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x): 
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x): 
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x): 
        return self.conv(x)

class UNet_DAG_Wrapper(nn.Module):

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print("[Wrapper] 正在加载 U-Net ...")
        self.layers = nn.ModuleList([
            DoubleConv(3, 64),       # 0
            Down(64, 128),           # 1
            Down(128, 256),          # 2
            Down(256, 512),          # 3
            Down(512, 1024),         # 4
            Up(1024 + 512, 512),     # 5
            Up(512 + 256, 256),      # 6
            Up(256 + 128, 128),      # 7
            Up(128 + 64, 64),        # 8
            OutConv(64, 2)           # 9
        ])
        self.to(device)
        self.eval()
        self.len = len(self.layers)
        self.skip_map = {5: 3, 6: 2, 7: 1, 8: 0}
        self.save_indices = list(self.skip_map.values())

    def __len__(self): 
        return self.len

    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
            cache = input_pack.get('cache', {})
        else: 
            x = input_pack
            cache = {}
        
        for i in range(start_idx, end_idx):
            layer = self.layers[i]
            if i in self.skip_map:
                required_idx = self.skip_map[i]
                skip_tensor = cache.get(required_idx)
                if skip_tensor is None: 
                    skip_tensor = x  # 容错
                x = layer(x, skip_tensor)
            else:
                x = layer(x)
            
            if i in self.save_indices:
                cache[i] = x
        
        return {'main': x, 'cache': cache}
    

# === 8. AlexNet ===
class AlexNet_DAG_Wrapper(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Wrapper] 正在加载 AlexNet (随机权重模式)...")
        self.raw_model = alexnet(weights=None).to(device)
        self.raw_model.eval()
        self.layers = nn.ModuleList()
        for layer in self.raw_model.features: 
            self.layers.append(layer)
        self.layers.append(self.raw_model.avgpool)
        self.layers.append(nn.Flatten())
        for layer in self.raw_model.classifier: 
            self.layers.append(layer)
        self.layers.append(nn.Softmax(dim=1))
        self.len = len(self.layers)
        self.save_indices = []

    def __len__(self): 
        return self.len
    
    def forward_slice(self, input_pack, start_idx, end_idx):
        if isinstance(input_pack, dict): 
            x = input_pack['main']
        else: 
            x = input_pack
        
        for i in range(start_idx, end_idx): 
            x = self.layers[i](x)
        
        return {'main': x, 'cache': {}}
