import os
import csv
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 全局字体配置 (用于正确显示中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================
# 1. CDP 数据并行模拟求解器 (含全量 6 种算法)
# ==========================================================
class CDPSolver:
    def __init__(self, model_profile: dict, env_status: dict):
        self.C_total = model_profile.get('compute_total_gflops', 0.0)
        self.D_in = model_profile.get('input_size_mb', 100.0)
        self.D_out = model_profile.get('output_size_mb', 5.0)

        # 环境网络资源抓取
        self.candidates = env_status.get('simulation_paths', {}).get('parallel_candidates', ["SAT-01", "SAT-02", "SAT-03"])
        self.aggregator = env_status.get('simulation_paths', {}).get('parallel_aggregator', "SAT-03")
        self.K = len(self.candidates)
        
        nodes = env_status.get('nodes', {})
        links = env_status.get('links', {})
        
        self.node_ids = self.candidates
        self.f = np.zeros(self.K)
        self.b_dist = np.zeros(self.K)
        self.b_agg = np.zeros(self.K)
        
        for i, node_id in enumerate(self.candidates):
            self.f[i] = nodes.get(node_id, {}).get('hardware', {}).get('compute_speed_gflops_per_ms', 1.0)
            bw_dist_mbps = self._get_bw(links, "RS", node_id)
            bw_agg_mbps = self._get_bw(links, node_id, self.aggregator)
            self.b_dist[i] = bw_dist_mbps / 8000.0  # 转化为 MB/ms
            self.b_agg[i] = bw_agg_mbps / 8000.0
            
        self.f = np.clip(self.f, a_min=1e-5, a_max=None)
        self.b_dist = np.clip(self.b_dist, a_min=1e-5, a_max=None)
        self.b_agg = np.clip(self.b_agg, a_min=1e-5, a_max=None)

    def _get_bw(self, links, src, dst):
        if src == dst: return float('inf')
        if f"{src}_to_{dst}" in links: return links[f"{src}_to_{dst}"]['bandwidth_mbps']
        if f"{dst}_to_{src}" in links: return links[f"{dst}_to_{src}"]['bandwidth_mbps']
        return 1e-1 

    def _evaluate_delay(self, D_alloc):
        t_div = D_alloc / self.b_dist
        t_comp = (self.C_total * (D_alloc / self.D_in)) / self.f
        t_agg = (D_alloc * (self.D_out / self.D_in)) / self.b_agg
        return float(np.max(t_div + t_comp + t_agg))

    def solve_lawa(self):
        gamma = 1.0 / (1.0 / self.b_dist + self.C_total / (self.D_in * self.f) + self.D_out / (self.D_in * self.b_agg))
        return self._evaluate_delay(self.D_in * (gamma / np.sum(gamma)))

    def solve_greedy(self):
        return self._evaluate_delay(self.D_in * (self.f / np.sum(self.f)))

    def solve_uniform(self):
        return self._evaluate_delay(np.full(self.K, self.D_in / self.K))

    def solve_random(self, n_trials=50):
        return min([self._evaluate_delay(self.D_in * np.random.dirichlet(np.ones(self.K))) for _ in range(n_trials)])

    def solve_pass_through(self):
        """对应原 Bent-Pipe：卫星透传模式"""
        D_alloc = np.zeros(self.K)
        idx = self.node_ids.index(self.aggregator) if self.aggregator in self.node_ids else 0
        D_alloc[idx] = self.D_in
        return self._evaluate_delay(D_alloc)

    def solve_ga(self, pop_size=20, gen=30):
        pop = [np.random.dirichlet(np.ones(self.K)) * self.D_in for _ in range(pop_size)]
        best = float('inf')
        for _ in range(gen):
            scores = [self._evaluate_delay(ind) for ind in pop]
            best = min(best, min(scores))
            # 简单精英与突变繁衍
            pop = [pop[i] for i in np.argsort(scores)[:pop_size//2]]
            while len(pop) < pop_size:
                p1, p2 = random.sample(pop[:pop_size//4], 2)
                alpha = random.random()
                child = alpha * p1 + (1 - alpha) * p2
                if random.random() < 0.2:
                    child += np.random.normal(0, self.D_in * 0.05, self.K)
                    child = np.clip(child, 0, None)
                    child = child / np.sum(child) * self.D_in
                pop.append(child)
        return best

# ==========================================================
# 2. 实验数据生成模块 (精确配合 4 款模型测试范围)
# ==========================================================
def run_simulation(csv_file="cdp_theoretical_results.csv"):
    env_mock = {
        "simulation_paths": {"parallel_candidates": ["SAT-01", "SAT-02", "SAT-03"], "parallel_aggregator": "SAT-03"},
        "nodes": {
            "SAT-01": {"hardware": {"compute_speed_gflops_per_ms": 100.0}},
            "SAT-02": {"hardware": {"compute_speed_gflops_per_ms": 200.0}},
            "SAT-03": {"hardware": {"compute_speed_gflops_per_ms": 80.0}},
        },
        "links": {"RS_to_SAT-01": {"bandwidth_mbps": 8250}, "RS_to_SAT-02": {"bandwidth_mbps": 9270},
                  "RS_to_SAT-03": {"bandwidth_mbps": 3670}, "SAT-01_to_SAT-03": {"bandwidth_mbps": 14120},
                  "SAT-02_to_SAT-03": {"bandwidth_mbps": 12300}}
    }
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Algorithm', 'Latency']) # 强制写入表头
        
        print("\n🚀 开始生成任务与执行算法推演...")
        # 严格按照您提供的分类区间赋予模型配置
        for i in range(1, 81):
            task_id = f"Task_{i:03d}"
            
            # 分区段分配模型特征
            if i <= 20:   model_conf = {"input_size_mb": 150.0, "output_size_mb": 5.0, "compute_total_gflops": 120.0} # VGG19
            elif i <= 40: model_conf = {"input_size_mb": 130.0, "output_size_mb": 2.0, "compute_total_gflops": 160.0} # ResNet101
            elif i <= 60: model_conf = {"input_size_mb": 80.0,  "output_size_mb": 3.0, "compute_total_gflops": 45.0}  # YOLOv5
            else:         model_conf = {"input_size_mb": 110.0, "output_size_mb": 1.0, "compute_total_gflops": 90.0}  # Swin_Base
            
            solver = CDPSolver(model_conf, env_mock)
            
            # 执行全量算法打表
            res = {
                "LAWA": solver.solve_lawa(),
                "Greedy": solver.solve_greedy(),
                "Uniform": solver.solve_uniform(),
                "Random": solver.solve_random(),
                "GA": solver.solve_ga(),
                "Pass-Through": solver.solve_pass_through()  # 更改名：透传
            }
            
            for alg, lat in res.items():
                writer.writerow([task_id, alg, lat])

# ==========================================================
# 3. 结果清算与绘图模块
# ==========================================================
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['Task', 'Algorithm'], keep='last')
    df_pivot = df.pivot(index='Task', columns='Algorithm', values='Latency').reset_index()
    
    # 提取编号并分类
    df_pivot['Task_Num'] = df_pivot['Task'].apply(lambda x: int(x.split('_')[1]))
    df_pivot = df_pivot.sort_values('Task_Num').reset_index(drop=True)
    
    def assign_model(num):
        if 1 <= num <= 20: return 'VGG19'
        elif 21 <= num <= 40: return 'ResNet101'
        elif 41 <= num <= 60: return 'YOLOv5'
        elif 61 <= num <= 80: return 'Swin_Base'
        return 'Other'
            
    df_pivot['Model'] = df_pivot['Task_Num'].apply(assign_model)
    
    # 全部归一化至 Pass-Through = 1.0 (基准线)
    required_algs = ['LAWA', 'Greedy', 'Uniform', 'Random', 'GA', 'Pass-Through']
    for alg in required_algs:
        df_pivot[f'{alg}_Ratio'] = df_pivot[alg] / df_pivot['Pass-Through']
    
    return df_pivot

def draw_bar_chart(df_pivot, title):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    target_models = ['VGG19', 'ResNet101', 'YOLOv5', 'Swin_Base']
    algs = ['LAWA', 'Greedy', 'Uniform', 'Random', 'GA', 'Pass-Through']
    algs_ratio = [f"{a}_Ratio" for a in algs]
    
    # 为 LAWA 选用显眼的深红色系，其他冷暖色搭配
    colors = ['#d62728', '#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e', '#8c564b']

    grouped_means = df_pivot.groupby('Model')[algs_ratio].mean()
    active_models = [m for m in target_models if m in grouped_means.index]
    
    x = np.arange(len(active_models))
    width = 0.13  # 6 根柱子调整宽度避免拥挤
    
    bars_list = []
    # 精准布置 6簇 排列偏移量
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    
    for i, (lbl, col_name, c) in enumerate(zip(algs, algs_ratio, colors)):
        offset = width * offsets[i]
        vals = [grouped_means.loc[m, col_name] for m in active_models]
        bars = ax.bar(x + offset, vals, width, label=lbl, color=c, edgecolor='black', linewidth=0.8, alpha=0.95)
        bars_list.append((bars, lbl))

    # 坐标周边装饰
    ax.set_xlabel('任务所属推理模型', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('平均归一化时延 (透传基准线 = 1.0)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(active_models, fontsize=10)
    
    # 绘制透传的 1.0 水平参考线
    ax.axhline(y=1.0, color='gray', linewidth=1.5, linestyle=':', zorder=0) 
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=3) # 分词三列防止挡住柱子

    # 标上中心字（高度避让交错）
    for bars, lbl in bars_list:
        for bar in bars:
            height = bar.get_height()
            off_y = 0.02 if lbl in ['Uniform', 'GA'] else 0.01 
            ax.text(bar.get_x() + bar.get_width() / 2., height + off_y,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8, 
                     fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig('cdp_theory_bar_chart.png', bbox_inches='tight')
    print("📈 画图完成！已保存为 'cdp_theory_bar_chart.png'")
    plt.show()

# ================= 实验主入口 =================
if __name__ == "__main__":
    csv_name = "cdp_theoretical_results.csv"
    
    # 1. 自动执行 80 个任务的计算生成
    run_simulation(csv_name)
    
    # 2. 清洗数据并执行画图
    df_data = process_data(csv_name)
    draw_bar_chart(df_data, "不同模型下各算法平均归一化时延对比 (协同数据并行)")