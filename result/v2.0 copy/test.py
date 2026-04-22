import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

def process_data(csv_path):
    """通用的数据清洗和预处理函数"""
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到文件: {csv_path}，请检查路径。")
        return None
    
    # 1. 读取并清洗数据 
    df = pd.read_csv(csv_path, names=['Task', 'Algorithm', 'Latency'])
    # 去除因多次追加执行留下的重复脏数据
    df = df.drop_duplicates(subset=['Task', 'Algorithm'], keep='last')
    df_pivot = df.pivot(index='Task', columns='Algorithm', values='Latency').reset_index()
    
    # 提取 Task 编号（如 Task_000 → 0）
    df_pivot['Task_Num'] = df_pivot['Task'].apply(lambda x: int(x.split('_')[1]))
    df_pivot = df_pivot.sort_values('Task_Num').reset_index(drop=True)
    
    # 2. 按任务编号范围分类模型
    def assign_model(num):
        if 0 <= num <= 20:
            return 'VGG19'
        elif 21 <= num <= 40:
            return 'ResNet101'
        elif 41 <= num <= 60:
            return 'YOLOv5'
        elif 61 <= num <= 80:
            return 'Swin_Base'
        else:
            return 'Other'
            
    df_pivot['Model'] = df_pivot['Task_Num'].apply(assign_model)
    
    # 3. 计算归一化时延比值周（vs Bent-Pipe = 1.0）
    required_algs = ['LA-DP', 'Greedy', 'Bent-Pipe', 'Random', 'GA']
    for alg in required_algs:
        if alg not in df_pivot.columns:
            raise ValueError(f"文件 {csv_path} 缺失算法列: {alg}")
    
    for alg in required_algs:
        df_pivot[f'{alg}_Ratio'] = df_pivot[alg] / df_pivot['Bent-Pipe']
    
    return df_pivot

def draw_bar_chart(ax, df_pivot, title):
    """在指定的子图 (ax) 上绘制五簇柱状图"""
    if df_pivot is None or df_pivot.empty:
        ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return

    target_models = ['VGG19', 'ResNet101', 'YOLOv5', 'Swin_Base']
    algs_ratio = ['LA-DP_Ratio', 'Greedy_Ratio', 'Random_Ratio', 'GA_Ratio', 'Bent-Pipe_Ratio']
    labels = ['LA-DP', 'Greedy', 'Random', 'GA', 'GS-Only']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

    # 按模型分组求均值
    grouped_means = df_pivot.groupby('Model')[algs_ratio].mean()

    # 动态筛选出在这个 CSV 中真正有的模型（例如还没跑到 Swin_Base 就不会显示空柱形）
    active_models = [m for m in target_models if m in grouped_means.index]
    means_dict = {lbl: [] for lbl in labels}

    for m in active_models:
        for lbl, col in zip(labels, algs_ratio):
            means_dict[lbl].append(grouped_means.loc[m, col])

    x = np.arange(len(active_models))
    width = 0.16  # 宽度调为 0.16 恰好排下五根柱子

    bars_list = []
    # 画五簇柱状图，基于中心点(-2, -1, 0, 1, 2)对称偏移排布
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        offset = width * (i - 2)
        bars = ax.bar(x + offset, means_dict[lbl], width, label=lbl, color=c, edgecolor='black', linewidth=0.8)
        bars_list.append((bars, lbl))

    # 装饰坐标轴与背景
    ax.set_xlabel('任务所属推理模型', fontsize=10)
    ax.set_ylabel('平均归一化时延 (GS-Only=1.0)', fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(active_models, fontsize=9)
    ax.axhline(y=1.0, color='black', linewidth=1, linestyle=':')  # Bent-Pipe 基准线
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper left')

    # 文字打在柱形顶端中心
    for bars, lbl in bars_list:
        for bar in bars:
            height = bar.get_height()
            # 避免 GA 和 Random 因为数值相近导致文字重叠，错开文字高度
            offset_y = 0.02 if lbl in ['GA', 'Random'] else 0.015
            ax.text(bar.get_x() + bar.get_width() / 2., height + offset_y,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8, 
                     fontweight='bold', color='black')

def plot_comparison_results(theory_csv="theoretical_results.csv", exp_csv="experiment_results.csv"):
    # 全局字体配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    # 把两个 CSV 读进来
    df_theory = process_data(theory_csv)
    df_exp = process_data(exp_csv)
    
    # 开一个 1行2列 的画布（总宽度加长到 15）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制左图：纯理论推演结果
    draw_bar_chart(ax1, df_theory, '理论预估时延对比 ')
    
    # 绘制右图：物理设备实测结果
    draw_bar_chart(ax2, df_exp, '实际执行时延对比')
    
    plt.tight_layout()
    plt.savefig('theory_vs_experiment_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 制图完成！左右对比图已保存为 theory_vs_experiment_analysis.png")
    plt.show()

if __name__ == "__main__":
    # 执行绘图（请确保这两个CSV文件与此脚本同目录）
    plot_comparison_results("theoretical_results.csv", "experiment_results.csv")