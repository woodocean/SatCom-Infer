# --- 绝对的第一和第二行，在这个前面不能有任何代码 ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import sys
import matplotlib.pyplot as plt

# 【新增】设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 (Windows可用SimHei或Microsoft YaHei)
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 🛠️ 1. 处理路径：将 ../models 加入 Python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models')
sys.path.append(models_dir)

try:
    from dag_wrappers import VGG19_DAG_Wrapper
    from pmp_solver import PMPSolver
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"请检查路径是否正确: {models_dir}")
    sys.exit(1)

# 🔥 2. 自动加载一次 VGG19 模型结构
print("[1/3] 正在从真实模型加载 VGG19 结构...")
model_wrapper = VGG19_DAG_Wrapper(device='cpu')
layers_count = len(model_wrapper.layers)
print(f"✅ 成功加载，模型共 {layers_count} 层\n")

# 转换格式
real_vgg_profile = []
for i in range(layers_count):
    real_vgg_profile.append({
        'base_latency_ms': 2.0,  
        'params_mb': 10.0,       
        'comm_size_mb': 5.0      
    })

model_profile = {
    'layers': real_vgg_profile,
    'input_size_raw': 3.0
}

# 用于记录数据的列表 (这里改成你想测的节点范围，例如 3 到 8)
nodes_list = list(range(3, 7))  
time_la_dp_list = []
time_ga_list = []
time_ex_list = []

print("[2/3] 开始进行多节点搜索时延测试...")
print("-" * 55)
print(f"{'节点数(K)':<10} | {'LA-DP (ms)':<12} | {'GA (ms)':<12} | {'穷举 (ms)':<12}")
print("-" * 55)

# 🚀 3. 开始循环测试不同的节点数
for k in nodes_list:
    # 动态构造当前 K 个节点的仿真环境
    nodes = []
    bandwidths = []
    for i in range(k-1):
        nodes.append({'id': f'SAT-{i+1:02d}', 'compute_speed_gflops_per_ms': 5.0 + i, 'memory_mb': 1000})
        bandwidths.append(10.0 + i)
    # 最后一个节点固定作为地面站 GS
    nodes.append({'id': 'GS', 'compute_speed_gflops_per_ms': 100.0, 'memory_mb': 8192})
    bandwidths.append(100.0)

    env_status = {
        'nodes': nodes,
        'bandwidths': bandwidths,
        'reference_compute_speed': 100.0
    }

    solver = PMPSolver(model_profile, env_status)

    # 1. 测 LA-DP
    t0 = time.perf_counter()
    solver.solve_la_dp()
    t_la = (time.perf_counter() - t0) * 1000
    time_la_dp_list.append(t_la)

    # 2. 测 GA
    t0 = time.perf_counter()
    solver.solve_ga(pop_size=30, generations=50) 
    t_ga = (time.perf_counter() - t0) * 1000
    time_ga_list.append(t_ga)

    # 3. 测 Exhaustive
    # ⚠️ 警告：因为穷举法复杂度是爆炸的，为了防止程序卡死，这里限制只测试到 K=6
    if k <= 6: 
        _, _, t_ex = solver.solve_exhaustive()
        time_ex_list.append(t_ex)
        ex_display = f"{t_ex:>10.2f}"
    else:
        # 节点大于 6 时由于耗时过长，我们直接标为 None 不再去傻等
        time_ex_list.append(None)
        ex_display = f"{'爆炸(放弃计算)':>10}"

    print(f"{k:<10} | {t_la:>10.2f} | {t_ga:>10.2f} | {ex_display}")

print("-" * 55)

# 📊 4. 绘制折线图
print("\n[3/3] 正在生成折线图...")

plt.figure(figsize=(10, 6))

# 使用对数坐标轴，因为穷举法的数据会呈指数级爆炸，不然 LA-DP 看着贴在 x 轴上
# 【修改处】图例标签改为中文
plt.plot(nodes_list, time_la_dp_list, marker='o', linewidth=2, color='blue', label='LA-DP (本文提出)')
plt.plot(nodes_list, time_ga_list, marker='s', linewidth=2, color='green', label='GA (遗传算法)')

# 过滤出有数据的部分画穷举法
valid_ex_indices = [i for i, v in enumerate(time_ex_list) if v is not None]
valid_ex_nodes = [nodes_list[i] for i in valid_ex_indices]
valid_ex_times = [time_ex_list[i] for i in valid_ex_indices]
plt.plot(valid_ex_nodes, valid_ex_times, marker='^', linewidth=2, color='red', linestyle='--', label='Exhaustive (穷举法)')

# 【修改处】标题和坐标轴改成中文
plt.title('计算节点数量与算法搜索时间的关系', fontsize=15, fontweight='bold')
plt.xlabel('计算节点数量 (K)', fontsize=13)
plt.ylabel('搜索时间 (ms) - 对数坐标 (Log Scale)', fontsize=13)

# 将 Y 轴设置为对数坐标，这样你可以十分清晰地看到三个算法复杂度的量级差异
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xticks(nodes_list)
plt.legend(fontsize=12)

# 保存并在窗口展示
plt.tight_layout()
plt.savefig("search_time_comparison.png", dpi=300)
print("✅ 图表已保存为 'search_time_comparison.png'")
plt.show()