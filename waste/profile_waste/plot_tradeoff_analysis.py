import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os  # 【修正：补上了这个】

# 学术字体设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
except: pass

def plot_dual_metric(df, model_name, target_size):
    # 筛选数据
    data = df[(df['model'] == model_name) & (df['input_size'] == target_size)]
    if data.empty:
        print(f"Skip: 无数据 {model_name} @ {target_size}")
        return

    data = data.sort_values('layer_idx')
    layers = data['layer_idx']
    flops = data['flops_g']
    comm = data['comm_mb']

    # --- 绘图 ---
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=150)
    
    x = np.arange(len(layers))
    width = 0.4
    
    # 左轴：计算量
    color_comp = '#ff7f0e'
    ax1.bar(x - width/2, flops, width, label='Computation (GFLOPs)', color=color_comp, alpha=0.9)
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Computation Load (GFLOPs)', color=color_comp, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_comp)
    ax1.set_ylim(0, max(flops.max(), 0.1) * 1.2)

    # 右轴：通信量
    ax2 = ax1.twinx()
    color_comm = '#1f77b4'
    ax2.bar(x + width/2, comm, width, label='Comm. Payload (MB)', color=color_comm, alpha=0.9)
    ax2.set_ylabel('Transmission Size (MB)', color=color_comm, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_comm)
    ax2.set_ylim(0, max(comm.max(), 1) * 1.2)
    
    plt.title(f"{model_name} (Input {target_size}x{target_size}): Computation vs Communication Profile", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.tight_layout()
    save_path = f"tradeoff_{model_name}_{target_size}.png"
    plt.savefig(save_path)
    print(f"生成: {save_path}")

if __name__ == "__main__":
    file_path = 'profile_database.csv'
    if not os.path.exists(file_path):
        print(f"错误: 找不到 {file_path}，请先运行 export_profile_data.py")
    else:
        df = pd.read_csv(file_path)
        models = df['model'].unique()
        print(f"找到模型: {models}")
        for m in models:
            # 找该模型最大的尺寸画图
            sizes = df[df['model']==m]['input_size']
            if not sizes.empty:
                max_s = sizes.max()
                plot_dual_metric(df, m, max_s)