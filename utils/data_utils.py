import pickle
import numpy as np
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定义 CIFAR-10 的类别名称 (硬编码，或者从 meta 文件读均可)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_cifar10_batch(root_dir, batch_size=8):
    """
    从本地加载 CIFAR-10 数据
    :param root_dir: 数据集根目录 (e.g., './data/cifar-10-batches-py')
    :param batch_size: 需要提取多少张图片
    :return: (images_tensor, label_names)
    """
    # 拼接测试集文件路径
    test_file = os.path.join(root_dir, 'test_batch')

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"找不到数据文件: {test_file}\n请检查路径是否正确！")

    # 读取二进制文件
    with open(test_file, 'rb') as fo:
        # encoding='bytes' 是必须的，因为它是 Python2 生成的 pickle
        data_dict = pickle.load(fo, encoding='bytes')

    # 获取原始数据
    raw_images = data_dict[b'data'][:batch_size]
    raw_labels = data_dict[b'labels'][:batch_size]

    # --- 数据转换 (关键步骤) ---
    # 原始 shape 是 (N, 3072)，需要 reshape 成 (N, 3, 32, 32)
    images = raw_images.reshape((batch_size, 3, 32, 32))

    # 归一化: 像素值 0-255 转为 0.0-1.0，并转为 FloatTensor
    images_tensor = torch.from_numpy(images).float() / 255.0

    # 将 32x32 的图片放大到 224x224 以适配标准 AlexNet
    print(f" 正在将图片从 32x32 Resize 到 224x224 以适配模型...")
    images_tensor = F.interpolate(images_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    # 获取对应的中文/英文标签名
    label_names = [CIFAR10_CLASSES[label_id] for label_id in raw_labels]

    return images_tensor, label_names

def decode_prediction(output_tensor):
    """
    将模型输出的 Tensor 转换为类别名称
    :param output_tensor: shape [N, 10]
    :return: list of strings
    """
    # 取概率最大的索引 (Argmax)
    pred_indices = output_tensor.argmax(dim=1).tolist()
    pred_names = [CIFAR10_CLASSES[idx] for idx in pred_indices]
    return pred_names


def save_result_image(tensor_data, pred_labels, filename="result.png"):
    """
    可视化结果：保存一张包含多张小图的拼接图
    :param tensor_data: [N, 3, 224, 224] 的归一化 Tensor
    :param pred_labels: ['cat', 'dog', ...] 对应的标签列表
    """
    # 1. 如果 Tensor 在 GPU 上，先移回 CPU
    if tensor_data.device.type != 'cpu':
        tensor_data = tensor_data.cpu()

    # 2. 转换维度: (N, 3, 224, 224) -> (N, 224, 224, 3) 以适配 Matplotlib
    images = tensor_data.numpy().transpose(0, 2, 3, 1)

    count = len(images)
    # 动态计算行列数（每行放4张）
    cols = 4
    rows = (count + cols - 1) // cols

    plt.figure(figsize=(12, 3 * rows))  # 设置画布大小

    for i in range(count):
        plt.subplot(rows, cols, i + 1)
        # 反归一化显示 (简单处理，假设之前是除以255的)
        img = images[i]
        plt.imshow(img)
        plt.title(f"Pred: {pred_labels[i]}", color='blue', fontsize=12)
        plt.axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()  # 关闭画布释放内存
    print(f"[Visualization] 结果图片已保存为: {filename}")