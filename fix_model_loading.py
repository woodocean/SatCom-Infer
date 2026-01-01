# quick_test.py
import torch
from utils.inference_utils import get_dnn_model
import torchvision.transforms as transforms
import torchvision


def quick_model_test():
    """快速测试模型加载和推理"""
    print("🚀 快速模型测试...")

    # 加载模型
    model = get_dnn_model('alex_net')

    # 测试数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

    # 测试几个样本
    images, labels = next(iter(testloader))

    # 在CPU上测试
    model.eval()
    model.cpu()

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, 1)

        correct = (predictions == labels).sum().item()
        accuracy = 100 * correct / len(labels)

        print(f"🔍 快速测试结果:")
        print(f"   样本数: {len(labels)}")
        print(f"   正确数: {correct}")
        print(f"   准确率: {accuracy:.2f}%")
        print(f"   预测: {predictions.tolist()}")
        print(f"   真实: {labels.tolist()}")


if __name__ == "__main__":
    quick_model_test()