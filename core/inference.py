import torch
import time
import importlib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class InferenceEngine:
    """
    推理引擎 - 支持全模型推理 & 分层推理
    兼容项目中 models/ 目录下的 AlexNet, VggNet, MobileNet, LeNet
    """

    def __init__(self, node_id, model_name='alexnet'):
        self.node_id = node_id
        self.model_name = model_name.lower()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.layers = []       # 展平后的层列表 [(name, module), ...]
        self.num_layers = 0

    def load_model(self, checkpoint_path=None):
        """加载模型, 支持项目中已有的模型定义"""
        print(f"[{self.node_id}] 加载模型: {self.model_name} -> {self.device}")

        if self.model_name == 'alexnet':
            from models.AlexNet import AlexNet
            self.model = AlexNet(num_classes=10)
            if checkpoint_path and os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                print(f"  已加载权重: {checkpoint_path}")

        elif self.model_name == 'vgg' or self.model_name == 'vggnet':
            from models.VggNet import VggNet
            self.model = VggNet()

        elif self.model_name == 'mobilenet':
            from models.MobileNet import MobileNetV2
            self.model = MobileNetV2()

        elif self.model_name == 'lenet':
            from models.LeNet import LeNet
            self.model = LeNet()

        else:
            # 兜底: 用 torchvision
            import torchvision.models as tv_models
            factory = {
                'resnet18': tv_models.resnet18,
                'resnet50': tv_models.resnet50,
                'mobilenet_v2': tv_models.mobilenet_v2,
            }
            if self.model_name in factory:
                self.model = factory[self.model_name](pretrained=False)
            else:
                raise ValueError(f"不支持的模型: {self.model_name}")

        self.model = self.model.to(self.device).eval()
        self.layers = self._flatten_model(self.model)
        self.num_layers = len(self.layers)
        print(f"  模型就绪, 共 {self.num_layers} 层")
        return self.num_layers

    def _flatten_model(self, model):
        """将模型展平为有序层列表"""
        layers = []

        # 优先检查是否有 features + classifier 结构 (AlexNet, VGG 等)
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            for i, layer in enumerate(model.features):
                layers.append((f"features.{i}", layer))
            # 添加一个标记用于 flatten
            layers.append(("__flatten__", None))
            for i, layer in enumerate(model.classifier):
                layers.append((f"classifier.{i}", layer))
        else:
            # 通用展平
            for name, module in model.named_children():
                if len(list(module.children())) > 0 and not isinstance(module, torch.nn.Linear):
                    for sub_name, sub_module in module.named_children():
                        layers.append((f"{name}.{sub_name}", sub_module))
                else:
                    layers.append((name, module))

        return layers

    def run_full(self, input_tensor):
        """完整推理 (baseline)"""
        x = input_tensor.to(self.device)
        t0 = time.perf_counter()

        with torch.no_grad():
            x = self._forward_layers(x, 0, self.num_layers - 1)

        if self.device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        return x.cpu(), (t1 - t0) * 1000  # output, ms

    def run_layers(self, input_tensor, start_layer, end_layer):
        """分层推理: 执行 [start_layer, end_layer]"""
        x = input_tensor.to(self.device)
        layer_times = []
        t_total_start = time.perf_counter()

        with torch.no_grad():
            for i in range(start_layer, min(end_layer + 1, self.num_layers)):
                name, layer = self.layers[i]

                t0 = time.perf_counter()

                if name == "__flatten__":
                    x = x.view(x.size(0), -1)
                else:
                    x = layer(x)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                layer_times.append({
                    'idx': i,
                    'name': name,
                    'time_ms': (t1 - t0) * 1000,
                    'output_shape': list(x.shape),
                    'output_mb': x.nelement() * x.element_size() / (1024 * 1024)
                })

        t_total_end = time.perf_counter()
        total_ms = (t_total_end - t_total_start) * 1000

        return x.cpu(), total_ms, layer_times

    def _forward_layers(self, x, start, end):
        """内部前向传播"""
        for i in range(start, min(end + 1, self.num_layers)):
            name, layer = self.layers[i]
            if name == "__flatten__":
                x = x.view(x.size(0), -1)
            else:
                x = layer(x)
        return x

    def get_layer_info(self):
        """获取所有层的元信息"""
        info = []
        for i, (name, layer) in enumerate(self.layers):
            info.append({
                'idx': i,
                'name': name,
                'type': type(layer).__name__ if layer else 'Flatten'
            })
        return info