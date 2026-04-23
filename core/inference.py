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
        self.stable_timing = True
        self.warmup_runs = 3
        self.warmed_signatures = set()

        # 为了降低不同任务间的测时抖动，关闭 cudnn benchmark 的动态算法搜索。
        if self.device == 'cuda' and self.stable_timing:
            torch.backends.cudnn.benchmark = False

    def _sync_if_cuda(self):
        """仅在 CUDA 场景下同步，避免 CPU 模式调用 cuda.synchronize 报错。"""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _describe_input(self, x):
        """返回输入张量的诊断字符串，便于排查在线时延波动。"""
        if isinstance(x, torch.Tensor):
            return (
                f"shape={tuple(x.shape)}, dtype={x.dtype}, "
                f"device={x.device}, contiguous={x.is_contiguous()}"
            )

        if isinstance(x, dict):
            main = x.get('main')
            cache = x.get('cache', {})
            if isinstance(main, torch.Tensor):
                main_desc = (
                    f"shape={tuple(main.shape)}, dtype={main.dtype}, "
                    f"device={main.device}, contiguous={main.is_contiguous()}"
                )
            else:
                main_desc = f"type={type(main).__name__}"
            cache_count = len(cache) if isinstance(cache, dict) else 0
            return f"main({main_desc}), cache_count={cache_count}"

        return f"type={type(x).__name__}"

    def _make_warmup_signature(self, x, start_layer, end_layer):
        """按层段和输入主张量形状归一化 warmup 签名。"""
        if isinstance(x, torch.Tensor):
            main = x
        elif isinstance(x, dict) and isinstance(x.get('main'), torch.Tensor):
            main = x['main']
        else:
            main = None

        if main is None:
            return (self.model_name, start_layer, end_layer, 'non_tensor')

        return (
            self.model_name,
            start_layer,
            end_layer,
            tuple(main.shape),
            str(main.dtype),
        )

    def _run_slice_once(self, x, start_layer, end_layer):
        """执行一次切片推理，统一 DAG 与线性模型路径。"""
        if self.is_dag_wrapper:
            # 若输入带 cache，先重置本地 cache，再从输入包恢复，避免任务间串扰。
            if isinstance(x, dict) and hasattr(self.model, 'reset_cache'):
                self.model.reset_cache()
            result = self.model.forward_slice(x, start_layer, end_layer + 1)
            return result['main']

        out = x
        for i in range(start_layer, end_layer + 1):
            if isinstance(self.layers[i], tuple):
                _, layer = self.layers[i]
            else:
                layer = self.layers[i]
            out = layer(out)
        return out

    def load_model(self, checkpoint_path=None):
        """加载模型，支持原生模型与 DAG Wrapper 模型"""
        print(f"[{self.node_id}] 加载模型: {self.model_name} -> {self.device}")

        if HAS_DAG_WRAPPERS:
            # ==============================
            # 优先尝试 DAG Wrapper 模型
            # ==============================
            if self.model_name == 'yolov5':
                self.model = YOLOv5_DAG_Wrapper(model_path='models/checkpoints/yolov5nu.pt', device=self.device)
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
        
        
        # 输入迁移（CPU->GPU）的耗时与纯计算分离，避免通信路径上的拷贝抖动污染算子测时。
        transfer_start = time.perf_counter()
        if isinstance(input_data, torch.Tensor):
            x = input_data.to(self.device)
        elif isinstance(input_data, dict):
            # 如果是 DAG 含有缓存的字典，把里面的张量都转移到对应设备
            x = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in input_data.items()}
        else:
            x = input_data
        self._sync_if_cuda()
        transfer_ms = (time.perf_counter() - transfer_start) * 1000

        focus_layers = {(33, 44), (40, 44)}
        focus_tag = "[TIMING-FOCUS]" if (start_layer, end_layer) in focus_layers else "[TIMING-IN]"
        print(
            f"[{self.node_id}] {focus_tag} layers=[{start_layer}->{end_layer}] "
            f"input={self._describe_input(x)}"
        )

        warmup_sig = self._make_warmup_signature(x, start_layer, end_layer)
        if warmup_sig not in self.warmed_signatures:
            print(
                f"[{self.node_id}] [TIMING] 首次命中层段签名，执行 {self.warmup_runs} 次 warmup: "
                f"layers=[{start_layer}->{end_layer}]"
            )
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    _ = self._run_slice_once(x, start_layer, end_layer)
                    self._sync_if_cuda()
            self.warmed_signatures.add(warmup_sig)

        self._sync_if_cuda()
        start_time = time.perf_counter()
        with torch.no_grad():
            x = self._run_slice_once(x, start_layer, end_layer)

        self._sync_if_cuda()
        end_time = time.perf_counter()
        cost_ms = (end_time - start_time) * 1000

        # 只在迁移时间明显偏大时打印一次提示，帮助定位“同 batch 不同耗时”的来源。
        if transfer_ms > max(10.0, cost_ms * 0.5):
            print(
                f"[{self.node_id}] [TIMING] 输入迁移耗时偏高: transfer={transfer_ms:.2f}ms, "
                f"compute={cost_ms:.2f}ms, layers=[{start_layer}->{end_layer}]"
            )
        return x, cost_ms

    def run_full(self, input_data):
        """运行完整模型"""
        return self.exec_layers(input_data, 0, self.num_layers - 1)