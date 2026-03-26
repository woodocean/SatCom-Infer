import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据，指定列名（因为原文件列名与实际数据不匹配）
df = pd.read_csv('simulation_results.csv', names=['task_id', 'model', 'algorithm', 'latency_ms'], skiprows=1)

# 查看数据是否正确
print(df.head())

# 模型列表（按给定顺序）
models = ["yolov5", "vgg19", "resnet101", "vit_huge", "swin_base"]
algorithms = ["LA-DP", "Greedy", "BentPipe"]

# 构建数据字典
data = {model: {} for model in models}
for _, row in df.iterrows():
    model = row['model']
    alg = row['algorithm']
    lat = row['latency_ms']
    if model in data:
        data[model][alg] = lat

# 计算相对于BentPipe的百分比
percent_data = {model: {} for model in models}
for model in models:
    base = data[model].get('BentPipe')
    if base is None:
        print(f"警告: 模型 {model} 缺少 BentPipe 数据")
        continue
    for alg in algorithms:
        if alg in data[model]:
            percent = (data[model][alg] / base) * 100
            percent_data[model][alg] = percent

# 绘图
x = np.arange(len(models))
width = 0.25
colors = {'LA-DP': '#2E86AB', 'Greedy': '#A23B72', 'BentPipe': '#F18F01'}

plt.figure(figsize=(10, 6))
for i, alg in enumerate(algorithms):
    heights = [percent_data[model].get(alg, 0) for model in models]
    plt.bar(x + (i - 1) * width, heights, width, label=alg, color=colors[alg])

plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='BentPipe 基线 (100%)')
plt.xlabel('模型')
plt.ylabel('相对于基线的时延百分比 (%)')
plt.title('各算法端到端时延对比 (以 BentPipe 为基准)')
plt.xticks(x, models)
plt.legend()

# 添加数值标签
for i, alg in enumerate(algorithms):
    for j, model in enumerate(models):
        val = percent_data[model].get(alg, 0)
        if val > 0:
            plt.text(x[j] + (i - 1) * width, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('latency_comparison_chinese.png', dpi=300)
plt.show()