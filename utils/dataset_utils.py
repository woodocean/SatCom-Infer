import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


def load_test_dataset(dataset_name='cifar10', batch_size=32):
    """加载测试数据集"""
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),  # 转为3通道
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader


def evaluate_model_accuracy(model, testloader, device='cuda'):
    """评估模型精度"""
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i >= 10:  # 只测试前10个批次，加快速度
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"测试样本数: {total}, 正确数: {correct}")
    return accuracy


def get_sample_batch(testloader, batch_size=1):
    """获取一个样本批次用于推理测试"""
    for data in testloader:
        images, labels = data
        return images, labels
    return None, None