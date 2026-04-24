import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

def _normalize_within_task_mode(df):
    """若缺失 norm_latency_vs_gs，则按同 run_id/task_id/mode 内 GS-Only 补齐。"""
    work = df.copy()
    work['latency_ms'] = pd.to_numeric(work['latency_ms'], errors='coerce')
    work['norm_latency_vs_gs'] = pd.to_numeric(work['norm_latency_vs_gs'], errors='coerce')

    gs_rows = work[work['algorithm'] == 'GS-Only'][['run_id', 'task_id', 'mode', 'latency_ms']].copy()
    gs_rows = gs_rows.rename(columns={'latency_ms': 'gs_latency'})
    work = work.merge(gs_rows, on=['run_id', 'task_id', 'mode'], how='left')

    missing_norm = work['norm_latency_vs_gs'].isna()
    valid_gs = work['gs_latency'].notna() & (work['gs_latency'] != 0)
    valid_latency = work['latency_ms'].notna()
    fill_mask = missing_norm & valid_gs & valid_latency
    work.loc[fill_mask, 'norm_latency_vs_gs'] = work.loc[fill_mask, 'latency_ms'] / work.loc[fill_mask, 'gs_latency']

    # GS-Only 自身归一化固定为 1.0
    gs_self = (work['algorithm'] == 'GS-Only') & work['norm_latency_vs_gs'].isna()
    work.loc[gs_self, 'norm_latency_vs_gs'] = 1.0

    return work.drop(columns=['gs_latency'])


def load_long_results(csv_path='results_long.csv', run_id=None, exp_type=None):
    """读取统一长表并按可选条件筛选。"""
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到文件: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    required_cols = {
        'run_id', 'exp_type', 'mode', 'task_id', 'algorithm',
        'model_name', 'latency_ms', 'norm_latency_vs_gs', 'timestamp'
    }
    if not required_cols.issubset(set(df.columns)):
        missing = sorted(list(required_cols - set(df.columns)))
        print(f"⚠️ 结果表缺少必要列: {missing}")
        return None

    if run_id:
        df = df[df['run_id'] == run_id]
    else:
        # 默认取最新一次 run_id，避免历史数据混入
        if not df.empty:
            latest_run_id = df.sort_values('timestamp').iloc[-1]['run_id']
            df = df[df['run_id'] == latest_run_id]
            print(f"[INFO] 未指定 run_id，自动使用最新批次: {latest_run_id}")

    if exp_type:
        df = df[df['exp_type'] == exp_type]

    if df.empty:
        print("⚠️ 过滤后无数据，无法绘图")
        return None

    df = _normalize_within_task_mode(df)
    # 去重：同 run/task/mode/alg 取最新记录
    df = df.sort_values('timestamp').drop_duplicates(
        subset=['run_id', 'task_id', 'mode', 'algorithm'],
        keep='last'
    )
    return df

def draw_bar_chart(ax, mode_df, title):
    """在指定的子图 (ax) 上绘制柱状图"""
    if mode_df is None or mode_df.empty:
        ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return

    # 定义要显示的算法顺序
    labels = ['LA-DP', 'Greedy', 'Random', 'GA', 'Uniform', 'GS-Only']
    labels = [l for l in labels if l in mode_df['algorithm'].unique()]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
    colors = colors[:len(labels)]

    # 按模型+算法对归一化时延求均值
    grouped = (
        mode_df
        .groupby(['model_name', 'algorithm'])['norm_latency_vs_gs']
        .mean()
        .reset_index()
    )
    grouped = grouped.dropna(subset=['norm_latency_vs_gs'])
    active_models = sorted(grouped['model_name'].unique().tolist())

    means_dict = {lbl: [] for lbl in labels}
    for m in active_models:
        gm = grouped[grouped['model_name'] == m]
        gm_map = {row['algorithm']: row['norm_latency_vs_gs'] for _, row in gm.iterrows()}
        for lbl in labels:
            means_dict[lbl].append(gm_map.get(lbl, np.nan))

    x = np.arange(len(active_models))
    width = 0.12

    bars_list = []
    n_algs = len(labels)
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        offset = width * (i - (n_algs - 1) / 2)
        bars = ax.bar(x + offset, means_dict[lbl], width, label=lbl, color=c, edgecolor='black', linewidth=0.8)
        bars_list.append((bars, lbl))

    ax.set_xlabel('模型类型', fontsize=10)
    ax.set_ylabel('归一化时延 (GS-Only=1.0)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_models, fontsize=10)
    ax.axhline(y=1.0, color='black', linewidth=1, linestyle='--')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(fontsize=8, loc='best')

    for bars, lbl in bars_list:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=7
            )


def plot_comparison_results(results_csv='results_long.csv', run_id=None, exp_type='algo_effectiveness'):
    """从统一长表绘制理论/实物对比图。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False

    df = load_long_results(results_csv, run_id=run_id, exp_type=exp_type)
    if df is None or df.empty:
        return

    theory_df = df[df['mode'] == 'theory']
    physical_df = df[df['mode'] == 'physical']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    draw_bar_chart(ax1, theory_df, '理论归一化时延对比')
    draw_bar_chart(ax2, physical_df, '实物归一化时延对比')

    if run_id:
        fig.suptitle(f'run_id: {run_id} | exp_type: {exp_type}', fontsize=11)
    else:
        fig.suptitle(f'最新 run_id | exp_type: {exp_type}', fontsize=11)

    plt.tight_layout()
    plt.savefig('theory_vs_experiment_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 制图完成：已从 results_long.csv 绘制理论/实物对比图 -> theory_vs_experiment_analysis.png")
    plt.show()


if __name__ == "__main__":
    # 默认读取统一长表，并自动选择最新 run_id
    plot_comparison_results('results_long.csv')