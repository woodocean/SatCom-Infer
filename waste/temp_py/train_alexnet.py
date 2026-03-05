import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from models.AlexNet import AlexNet  # 导入你的AlexNet模型


def train_alexnet_cifar10(resume_epoch=None, total_epochs=80):
    """训练AlexNet在CIFAR-10数据集上 - 支持继续训练到指定总epoch数"""

    # 训练参数
    batch_size = 256
    num_classes = 10

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    print("加载CIFAR-10数据集...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    if len(trainset) > 25000:
        from torch.utils.data import Subset
        trainset = Subset(trainset, range(25000))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    model = AlexNet(input_channels=3, num_classes=num_classes)
    model = model.to(device)

    # 使用DataParallel如果有多GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)

    # 初始化训练变量
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 学习率设置
    if resume_epoch is not None:
        learning_rate = 0.005  # 继续训练时用更小的学习率
    else:
        learning_rate = 0.01

    # 如果从检查点恢复
    if resume_epoch is not None:
        checkpoint_path = f'../../checkpoints/alexnet_cifar10_epoch_{resume_epoch}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            # 加载模型状态
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # 加载训练历史
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            train_accuracies = checkpoint.get('train_accuracies', [])
            test_accuracies = checkpoint.get('test_accuracies', [])

            print(f"✅ 从epoch {resume_epoch}恢复训练，目标epoch: {total_epochs}")
            print(f"   之前训练精度: {checkpoint['train_accuracy']:.2f}%")
            print(f"   之前测试精度: {checkpoint['test_accuracy']:.2f}%")
            print(f"   使用学习率: {learning_rate}")
        else:
            print(f"❌ 检查点 {checkpoint_path} 不存在，从头开始训练")
            resume_epoch = None

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # 调整学习率衰减步长

    print("开始训练...")
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Train Acc':^10} | {'Test Acc':^10} | {'Time':^8}")
    print("-" * 60)

    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                current_acc = 100. * correct / total
                print(
                    f'Epoch: {epoch + 1}/{total_epochs} | Batch: {batch_idx}/{len(trainloader)} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%')

        # 计算训练精度
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total

        # 测试阶段 - 每个epoch都测试
        test_acc = evaluate_model(model, testloader, device)

        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 打印进度
        epoch_time = time.time() - start_time
        print(f"{epoch + 1:^6} | {train_loss:^12.4f} | {train_acc:^10.2f}% | {test_acc:^10.2f}% | {epoch_time:^8.2f}s")
        print(f"      当前学习率: {current_lr:.6f}")

        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            save_checkpoint_with_history(model, epoch, train_acc, test_acc, train_losses, train_accuracies,
                                         test_accuracies)

    print("训练完成!")

    # 保存最终模型
    save_final_model(model, test_acc)

    # 绘制训练曲线
    plot_training_curve(train_losses, train_accuracies, test_accuracies)

    return model, train_losses, train_accuracies, test_accuracies

def save_checkpoint_with_history(model, epoch, train_acc, test_acc, train_losses, train_accuracies, test_accuracies):
    """保存检查点（包含训练历史）"""
    # 如果是DataParallel，保存原始模型
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

    os.makedirs('../../checkpoints', exist_ok=True)
    filename = f'../../checkpoints/alexnet_cifar10_epoch_{epoch + 1}.pth'
    torch.save(checkpoint, filename)
    print(f"检查点已保存: {filename}")


def evaluate_model(model, testloader, device):
    """快速评估模型精度"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # 只评估部分测试数据以节省时间
        for i, (inputs, targets) in enumerate(testloader):
            if i >= 20:  # 只评估20个批次
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def save_checkpoint(model, epoch, train_acc, test_acc):
    """保存检查点"""
    # 如果是DataParallel，保存原始模型
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }

    os.makedirs('../../checkpoints', exist_ok=True)
    filename = f'../../checkpoints/alexnet_cifar10_epoch_{epoch + 1}.pth'
    torch.save(checkpoint, filename)
    print(f"检查点已保存: {filename}")


def save_final_model(model, test_acc):
    """保存最终模型"""
    os.makedirs('../../trained_models', exist_ok=True)

    # 如果是DataParallel，保存原始模型
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # 保存完整模型
    model_path = '../../trained_models/alexnet_cifar10_final.pth'
    torch.save(model_state_dict, model_path)
    print(f"最终模型已保存: {model_path}, 测试精度: {test_acc:.2f}%")

    # 保存用于卫星系统的模型
    satellite_model_path = '../../trained_models/alexnet_cifar10_satellite.pth'
    torch.save({
        'model_state_dict': model_state_dict,
        'num_classes': 10,
        'input_channels': 3,
        'test_accuracy': test_acc
    }, satellite_model_path)
    print(f"卫星系统模型已保存: {satellite_model_path}")


def plot_training_curve(train_losses, train_accuracies, test_accuracies):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # 绘制精度曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("训练曲线已保存: training_curve.png")

    except ImportError:
        print("Matplotlib未安装，跳过绘制训练曲线")


def test_trained_model():
    """测试训练好的模型"""
    print("\n测试训练好的模型...")

    # 加载模型
    model = AlexNet(input_channels=3, num_classes=10)
    model_path = '../../trained_models/alexnet_cifar10_final.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("模型加载成功!")

        # 完整测试
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 创建测试数据加载器
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = DataLoader(testset, batch_size=100, shuffle=False)

        accuracy = evaluate_model_full(model, testloader, device)
        print(f"完整测试精度: {accuracy:.2f}%")
    else:
        print("未找到训练好的模型，请先运行训练")


def evaluate_model_full(model, testloader, device):
    """完整评估模型精度"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练AlexNet在CIFAR-10上')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--resume', type=int, help='从哪个epoch恢复训练')
    parser.add_argument('--total_epochs', type=int, default=80, help='总训练epoch数')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.resume:
            print(f"🚀 从epoch {args.resume}继续训练AlexNet...")
            print(f"   目标总epoch数: {args.total_epochs}")
            print(f"   将使用更小的学习率(0.001)继续优化")
        else:
            print("🚀 开始训练AlexNet...")
            print(f"   目标总epoch数: {args.total_epochs}")
        print("-" * 50)

        model, train_losses, train_accuracies, test_accuracies = train_alexnet_cifar10(args.resume, args.total_epochs)
    else:
        test_trained_model()