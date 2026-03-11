import time
import random
import numpy as np
import pandas as pd
from config import MODEL_CONFIGS, NODES, STAR_BANDWIDTH, GROUND_BANDWIDTH, PROPAGATION_DELAY

class SpaceInferenceSimulator:
    def __init__(self):
        self.results = []

    def calc_comm_time(self, data_size_mb, bandwidth_mbps):
        return (data_size_mb * 8 / bandwidth_mbps) + PROPAGATION_DELAY

    def run_ground_only(self, task):
        """模式4：弯管回传 (RS -> GS 直接算)"""
        comm_time = self.calc_comm_time(task['input_size_mb'], GROUND_BANDWIDTH)
        comp_time = task['base_latency'] * NODES['GS']['factor']
        return comm_time + comp_time

    def run_fixed_pmp(self, task):
        """模式2：固定流水线 (RS -> SAT-03 -> SAT-04 -> SAT-05 -> GS)"""
        path = ["RS", "SAT-03", "SAT-04", "SAT-05", "GS"]
        total_time = 0
        current_data = task['input_size_mb']
        
        # 简化模拟：均分计算量到 3 个 SAT 节点
        for i in range(len(path)-1):
            src, dst = path[i], path[i+1]
            # 通信
            total_time += self.calc_comm_time(current_data, STAR_BANDWIDTH)
            # 膨胀更新 (由于是均分，逐级膨胀)
            current_data = current_data * (task['gamma'] ** (1/3)) 
            # 计算 (如果是SAT节点)
            if "SAT" in dst:
                total_time += (task['base_latency'] / 3) * NODES[dst]['factor']
        return total_time

    def run_fixed_cdp(self, task):
        """模式3：固定并行 (RS -> {SAT-01,02,03} -> SAT-03 -> SAT-04 -> SAT-05 -> GS)"""
        # 1. 分发阶段 (RS 到 01,02,03 并行)
        parallel_nodes = ["SAT-01", "SAT-02", "SAT-03"]
        data_per_node = task['input_size_mb'] / 3
        # 并行传输 + 计算，取最慢的木桶
        parallel_latencies = []
        for node in parallel_nodes:
            t_comm = self.calc_comm_time(data_per_node, STAR_BANDWIDTH)
            t_comp = task['base_latency'] * NODES[node]['factor']
            parallel_latencies.append(t_comm + t_comp)
        
        total_time = max(parallel_latencies)
        
        # 2. 聚合与后传 (SAT-03 -> 04 -> 05 -> GS) 结果级联
        result_size = 0.5 # 假设推理结果很小
        post_path = ["SAT-03", "SAT-04", "SAT-05", "GS"]
        for i in range(len(post_path)-1):
            total_time += self.calc_comm_time(result_size, STAR_BANDWIDTH)
            
        return total_time

    def algorithm_1_select(self, task):
        """模式1：智能模式选择算法 (Algorithm 1)"""
        # 简单的筛选与打分逻辑
        # 如果模型太大单星装不下，强制 PMP 分治参数
        if task['weights_mb'] > 8000: # 假设阈值
            return self.run_fixed_pmp(task)
        
        # 决策因子: 膨胀率 > 1.5 且 带宽不富裕时，倾向 CDP 避免中间数据爆炸
        if task['gamma'] > 1.5:
            return self.run_fixed_cdp(task)
        else:
            return self.run_fixed_pmp(task)

    def run_experiment(self, num_tasks=10):
        model_names = list(MODEL_CONFIGS.keys())
        
        for i in range(num_tasks):
            m_name = random.choice(model_names)
            m_info = MODEL_CONFIGS[m_name]
            
            task = {
                "id": i,
                "model": m_name,
                "input_size_mb": random.uniform(2.0, 10.0), # 模拟不同输入尺寸
                "base_latency": m_info['ops_gflops'] * 0.1,  # 模拟基础推理耗时
                "gamma": m_info['gamma'],
                "weights_mb": m_info['weights_mb']
            }
            
            # 记录四种模式的时延
            t_select = self.algorithm_1_select(task)
            t_pmp    = self.run_fixed_pmp(task)
            t_cdp    = self.run_fixed_cdp(task)
            t_ground = self.run_ground_only(task)
            
            self.results.append({
                "Task": i, "Model": m_name,
                "Algorithm_1": t_select,
                "Fixed_PMP": t_pmp,
                "Fixed_CDP": t_cdp,
                "Ground": t_ground
            })
            print(f"Task {i} [{m_name}] 完成模拟计算.")

    def summary(self):
        df = pd.DataFrame(self.results)
        print("\n" + "="*50)
        print("实验结果汇总 (总端到端时延 / s)")
        print("="*50)
        print(df.to_string(index=False))
        print("\n平均时延对比:")
        print(df[["Algorithm_1", "Fixed_PMP", "Fixed_CDP", "Ground"]].mean())
        df.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    sim = SpaceInferenceSimulator()
    sim.run_experiment(10)
    sim.summary()