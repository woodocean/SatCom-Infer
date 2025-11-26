import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上级目录
parent_dir = os.path.dirname(current_dir)
# 添加到Python路径
sys.path.insert(0, parent_dir)

print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")

# 现在可以导入AlexNet
from models.AlexNet import AlexNet


def measure_alexnet_layer_timings():
    """测量AlexNet每层推理时间和输出数据量"""

    # 创建模型
    model = AlexNet(input_channels=3, num_classes=1000)
    model.eval()

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 输入数据
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.rand(input_shape).to(device)

    # 计算输入数据大小（字节）
    # float32占4字节，所以：元素个数 × 4 字节
    input_elements = dummy_input.numel()
    input_bytes = input_elements * 4
    input_mb = input_bytes / (1024 * 1024)  # 转换为MB

    # 预热
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 测量每层时间和输出数据量
    layer_timings = {}
    layer_output_bytes = {}  # 存储字节数
    layer_output_mb = {}  # 存储MB数
    current_input = dummy_input.clone()

    print("Measuring layer timings and output sizes...")
    print(f"Input data: {input_elements:,} elements = {input_mb:.2f} MB")

    for idx, layer in enumerate(model.layers):
        layer_name = f"{idx + 1}-{layer.__class__.__name__}"

        # 先获取该层的正确输入
        with torch.no_grad():
            # 使用当前正确的输入形状
            correct_input = current_input.clone()
            output = layer(correct_input)

            # 计算输出数据大小（字节）
            output_elements = output.numel()
            output_bytes = output_elements * 4  # float32占4字节
            output_mb = output_bytes / (1024 * 1024)  # 转换为MB

            layer_output_bytes[layer_name] = output_bytes
            layer_output_mb[layer_name] = output_mb

        # 跳过激活函数等简单层的详细时间测量
        if isinstance(layer, (nn.ReLU, nn.Dropout, nn.Flatten)):
            layer_timings[layer_name] = 0.001
            current_input = output  # 更新输入为当前输出
            print(f"Layer {idx + 1}: {layer_name} - 0.001 ms, Output: {output_mb:.2f} MB")
            continue

        # 测量该层推理时间（使用正确的输入形状）
        epoch = 100 if device == "cuda" else 50
        total_time = 0.0

        for i in range(epoch):
            with torch.no_grad():
                test_input = correct_input.clone()  # 使用正确的输入

                if device == "cuda":
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    _ = layer(test_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    layer_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    start_time = time.perf_counter()
                    _ = layer(test_input)
                    end_time = time.perf_counter()
                    layer_time = end_time - start_time

                total_time += layer_time

        avg_time = total_time / epoch * 1000
        layer_timings[layer_name] = avg_time
        current_input = output  # 更新输入为当前输出

        print(f"Layer {idx + 1}: {layer_name} - {avg_time:.3f} ms, Output: {output_mb:.2f} MB")

    # 绘制双柱状图（包含输入数据）
    plot_combined_chart_with_input(layer_timings, layer_output_mb, input_mb)

    return layer_timings, layer_output_mb, input_mb


def plot_combined_chart_with_input(layer_timings, layer_output_mb, input_mb):
    """在同一图中绘制时延和输出数据量的双柱状图（包含输入数据）"""

    # 添加输入数据到数据中
    all_layers = ['input'] + list(layer_timings.keys())
    all_times = [0] + [layer_timings[layer] for layer in layer_timings.keys()]  # 输入没有时延
    all_sizes_mb = [input_mb] + [layer_output_mb[layer] for layer in layer_timings.keys()]

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(20, 10))

    x = np.arange(len(all_layers))
    width = 0.35

    # 绘制时延柱状图
    bars1 = ax1.bar(x - width / 2, all_times, width,
                    color='lightcoral', alpha=0.8,
                    label='Inference Time (ms)', edgecolor='darkred', linewidth=1)

    # 创建第二个y轴用于输出数据量
    ax2 = ax1.twinx()

    # 绘制输出数据量柱状图
    bars2 = ax2.bar(x + width / 2, all_sizes_mb, width,
                    color='lightseagreen', alpha=0.8,
                    label='Data Size (MB)', edgecolor='darkgreen', linewidth=1)

    # 设置第一个y轴（时延）
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Time (ms)', color='darkred', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_layers, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 设置第二个y轴（数据量）
    ax2.set_ylabel('Data Size (MB)', color='darkgreen', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    # 设置标题
    plt.title('AlexNet Layer Inference Time and Data Size Comparison (Including Input)',
              fontsize=14, fontweight='bold', pad=20)

    # 在时延柱子上显示数值（输入没有时延，不显示）
    for i, (bar, time_val) in enumerate(zip(bars1, all_times)):
        height = bar.get_height()
        if i > 0 and time_val > 0.1:  # 跳过输入，只显示有意义的数值
            ax1.text(bar.get_x() + bar.get_width() / 2., height + max(all_times) * 0.02,
                     f'{time_val:.2f}ms',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        elif i == 0:  # 输入层显示"N/A"
            ax1.text(bar.get_x() + bar.get_width() / 2., height + max(all_times) * 0.02,
                     'N/A',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # 在数据量柱子上显示数值（MB）
    for i, (bar, size_val) in enumerate(zip(bars2, all_sizes_mb)):
        height = bar.get_height()
        display_text = f'{size_val:.2f}MB'

        ax2.text(bar.get_x() + bar.get_width() / 2., height + max(all_sizes_mb) * 0.02,
                 display_text,
                 ha='center', va='bottom', fontsize=8, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # 添加分隔线区分输入和网络层
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 打印详细信息（包含输入）
    total_time = sum(all_times[1:])  # 不包括输入的时延
    print(f"\n{'=' * 90}")
    print(f"{'AlexNet Layer Details (Including Input)':^90}")
    print(f"{'=' * 90}")
    print(f"{'Type':^8} {'Name':<25} {'Time(ms)':<12} {'Size(MB)':<12} {'Elements':<15}")
    print(f"{'-' * 90}")

    # 输入数据
    input_elements = int(input_mb * 1024 * 1024 / 4)  # 反向计算元素个数
    print(f"{'Input':^8} {'input':<25} {'N/A':<12} {input_mb:<12.2f} {input_elements:<15,}")

    # 各层数据
    for i, layer_name in enumerate(layer_timings.keys()):
        time_val = layer_timings[layer_name]
        size_mb = layer_output_mb[layer_name]
        elements = int(size_mb * 1024 * 1024 / 4)  # 反向计算元素个数

        print(f"{'Layer':^8} {layer_name:<25} {time_val:<12.3f} {size_mb:<12.2f} {elements:<15,}")

    print(f"{'-' * 90}")
    print(f"{'Total':^8} {'':<25} {total_time:<12.3f} {'':<12} {'':<15}")
    print(f"{'=' * 90}")


# 运行测试
if __name__ == "__main__":
    timings, output_sizes_mb, input_mb = measure_alexnet_layer_timings()