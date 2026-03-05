import numpy as np
import matplotlib.pyplot as plt
from algorithm.sim_scheduler import SimScheduler
from algorithm.selector import ModeSelector
from algorithm.pmp_optimizer import PMPOptimizer
from algorithm.cdp_optimizer import CDPOptimizer

def generate_tasks(num_tasks, mix_ratio=(0.3,0.3,0.4)):
    # 生成混合任务，返回 tasks 列表
    pass

def run_experiment1():
    tasks_list = [10,20,30,40,50]
    results = {policy: [] for policy in ['adaptive','fixed_pmp','fixed_cdp','single_sat','bentpipe']}
    for n in tasks_list:
        tasks = generate_tasks(n)
        for policy in results.keys():
            scheduler = SimScheduler(satellites, selector)  # 需定义卫星和选择器
            avg_lat = scheduler.run(tasks, policy)
            results[policy].append(avg_lat)
    # 绘图
    for policy, vals in results.items():
        plt.plot(tasks_list, vals, label=policy)
    plt.xlabel('Number of Tasks')
    plt.ylabel('Average Latency (s)')
    plt.legend()
    plt.savefig('exp1.png')