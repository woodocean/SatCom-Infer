import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

def _remove_retrans_outliers(df, source_name):
    """剔除明显由超时重传导致的极端大时延点（仅处理高侧异常）"""
    cleaned_groups = []
    removed = 0

    for alg, grp in df.groupby('Algorithm', sort=False):
        g = grp.copy()
        if len(g) < 6:
            cleaned_groups.append(g)
            continue

        q1 = g['Latency'].quantile(0.25)
        q3 = g['Latency'].quantile(0.75)
        iqr = q3 - q1
        med = g['Latency'].median()

        # 仅清理“明显异常高值”：IQR 高门限 + 绝对门限 + 中位数倍率门限
        iqr_upper = q3 + (6.0 * iqr if iqr > 0 else 0.0)
        hard_upper = 5000.0
        ratio_upper = med * 8.0 if med > 0 else hard_upper
        upper = max(iqr_upper, hard_upper, ratio_upper)

        keep_mask = g['Latency'] <= upper
        removed += int((~keep_mask).sum())
        cleaned_groups.append(g[keep_mask])

    out = pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else df
    if removed > 0:
        print(f"[清洗] {source_name}: 剔除超时重传异常值 {removed} 条")
    return out


def process_data(csv_path):
    """通用的数据清洗和预处理函数"""
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到文件: {csv_path}，请检查路径。")
        return None
    
    # 1. 读取并清洗数据 
    df = pd.read_csv(csv_path, names=['Task', 'Algorithm', 'Latency'])
    df['Latency'] = pd.to_numeric(df['Latency'], errors='coerce')
    df = df.dropna(subset=['Task', 'Algorithm', 'Latency'])

    # 实验结果优先清理重传导致的离群大值
    # if 'experiment' in os.path.basename(csv_path).lower():
    #     df = _remove_retrans_outliers(df, os.path.basename(csv_path))

    # 去除因多次追加执行留下的重复脏数据
    df = df.drop_duplicates(subset=['Task', 'Algorithm'], keep='last')
    df_pivot = df.pivot(index='Task', columns='Algorithm', values='Latency').reset_index()
    
    # 提取 Task 编号（如 Task_000 → 0）
    df_pivot['Task_Num'] = df_pivot['Task'].apply(lambda x: int(x.split('_')[1]))
    df_pivot = df_pivot.sort_values('Task_Num').reset_index(drop=True)
    
    # 2. 改进模型分类逻辑：自动识别或统一归类
    def assign_model(num):
        if 0 <= num <= 9  :
            return 'ViT_Huge'
        elif 10 <= num <= 19:
            return 'VGG19'
        elif 20 <= num <= 29:
            return 'YOLOv5'
        elif 30 <= num <= 39:
            return 'Swin_Base'
        elif 40 <= num <= 49:
            return 'ResNet101'
        else:
            return 'Other'
            
    df_pivot['Model'] = df_pivot['Task_Num'].apply(assign_model)
    
    # 3. 计算归一化时延比值（vs GS-Only = 1.0）
    # 检查数据中实际存在的算法，增加容错
    all_algs = ['LA-DP', 'Greedy', 'GS-Only', 'Random', 'GA', 'Uniform']
    available_algs = [alg for alg in all_algs if alg in df_pivot.columns]
    
    if 'GS-Only' not in available_algs:
        # 如果没有 GS-Only，则不计算 Ratio，或者以第一个算法为基准
        print("⚠️ ⚠️ 缺失 GS-Only 算法，无法计算归一化比值")
        for alg in available_algs:
            df_pivot[f'{alg}_Ratio'] = 0.0
    else:
        for alg in available_algs:
            df_pivot[f'{alg}_Ratio'] = df_pivot[alg] / df_pivot['GS-Only']
    
    return df_pivot

def draw_bar_chart(ax, df_pivot, title):
    """在指定的子图 (ax) 上绘制柱状图"""
    if df_pivot is None or df_pivot.empty:
        ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return

    # 定义要显示的算法和颜色
    labels = ['LA-DP', 'Greedy', 'Random', 'GA', 'Uniform', 'GS-Only']
    # 过滤掉数据中不存在的算法
    labels = [l for l in labels if l in df_pivot.columns]
    algs_ratio = [f'{l}_Ratio' for l in labels]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
    colors = colors[:len(labels)]

    # 按模型分组求均值
    grouped_means = df_pivot.groupby('Model')[algs_ratio].mean()
    active_models = grouped_means.index.tolist()
    
    means_dict = {lbl: [] for lbl in labels}
    for m in active_models:
        for lbl, col in zip(labels, algs_ratio):
            means_dict[lbl].append(grouped_means.loc[m, col])

    x = np.arange(len(active_models))
    width = 0.12  # 调整宽度以容纳多根柱子

    bars_list = []
    # 居中对齐排布
    n_algs = len(labels)
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        offset = width * (i - (n_algs - 1) / 2)
        bars = ax.bar(x + offset, means_dict[lbl], width, label=lbl, color=c, edgecolor='black', linewidth=0.8)
        bars_list.append((bars, lbl))

    # 装饰坐标轴
    ax.set_xlabel('模型类型', fontsize=10)
    ax.set_ylabel('归一化时延 (GS-Only=1.0)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_models, fontsize=10)
    ax.axhline(y=1.0, color='black', linewidth=1, linestyle='--')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(fontsize=8, loc='best')

    # 在柱子顶端标注数值
    for bars, lbl in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=7)


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