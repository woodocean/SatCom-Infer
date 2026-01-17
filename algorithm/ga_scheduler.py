import numpy as np
from sko.GA import GA
import time

class GAScheduler:
    def __init__(self, neighbor_info, model_layers, memory_limit=4096):
        """
        初始化调度器
        :param neighbor_info: 邻居列表 [{'id': 'sat1', 'bw': 100, 'cpu': 50}, ...]
        :param model_layers: 模型总层数 (例如 YOLOv5=25, ResNet50=6)
        :param memory_limit: 单星内存限制 (MB)
        """
        self.neighbors = neighbor_info
        self.n_neighbors = len(neighbor_info)
        self.total_layers = model_layers
        self.mem_limit = memory_limit
        
        # 记录最优解
        self.best_latency = float('inf')
        self.best_plan = None

    def fitness_function(self, x):
        """
        核心：适应度函数 (输入染色体 x，输出时延)
        x 是一个数组，例如 [0.8, 0.4, 0.3]
        """
        # ===========================
        # 1. 解码 (Decoding)
        # ===========================
        # 基因1: 模式选择 (0~1) -> 0:Pipeline, 1:Parallel
        mode = 0 if x[0] < 0.5 else 1
        
        # 基因2: 目标卫星索引 (0~1) -> 映射到 neighbors 列表索引
        # x[1] * 邻居数量，向下取整
        neighbor_idx = int(x[1] * (self.n_neighbors - 0.001))
        target_node = self.neighbors[neighbor_idx]
        
        # 基因3: 切分参数 (0~1)
        # 如果是 Pipeline: 代表切分层 (Layer 0 ~ Total)
        # 如果是 Parallel: 代表数据切分比例 (Ratio 0.0 ~ 1.0)
        split_param = x[2]

        # ===========================
        # 2. 约束检查 (Constraints)
        # ===========================
        # 模拟内存约束
        # 假设：并行模式吃内存 = 基础内存 * (1 + ratio)
        # 假设：流水线模式吃内存 = 基础内存 * (layer / total)
        # 这里用简化的数学模型代替，后面接入真实 Profiling 数据
        
        required_mem = 0
        if mode == 1: # Parallel
            required_mem = 2000 * (1 + split_param) 
        else: # Pipeline
            layer = int(split_param * self.total_layers)
            required_mem = 2000 * (layer / self.total_layers) + 500

        # 如果内存超标，返回一个巨大的时延（惩罚）
        if required_mem > self.mem_limit:
            return 99999.0

        # ===========================
        # 3. 计算时延 (Cost Calculation)
        # ===========================
        # 这里填入你的 系统模型公式
        # T_total = T_comp + T_trans
        
        # 模拟数据：
        bandwidth = target_node['bw'] # Mbps
        cpu_power = target_node['cpu'] # GFLOPS
        
        if mode == 0: # Pipeline (切分层)
            split_layer = int(split_param * (self.total_layers - 1))
            
            # 假设：层数越深，数据量越小，计算量越大
            # 通信量 (MB)
            comm_data = 50 * (1 - split_layer / self.total_layers) 
            # 计算量 (GFLOPs)
            comp_load = 10 * (split_layer / self.total_layers)
            
            t_trans = comm_data * 8 / bandwidth * 1000 # ms
            t_comp = comp_load / cpu_power * 1000 # ms
            total_latency = t_trans + t_comp
            
        else: # Parallel (切分数据)
            ratio = split_param # 分给邻居的比例
            
            # 并行模式：取决于最慢的那个 (Max)
            # 本地计算 (1-ratio)
            t_local = (1 - ratio) * 100 # 假设本地算全图100ms
            
            # 邻居计算
            data_size = 50 * ratio # 原始图片50MB * 比例
            t_trans = data_size * 8 / bandwidth * 1000
            t_remote_comp = (ratio * 100) * (50 / cpu_power) # 根据算力折算
            
            total_latency = max(t_local, t_trans + t_remote_comp)

        return total_latency

    def run(self):
        """运行遗传算法"""
        # 定义变量范围：3个基因，范围都是 0~1
        lb = [0, 0, 0]
        ub = [1, 1, 1]
        
        # 实例化 GA
        # n_dim=3: 染色体长度
        # size_pop=50: 种群数量 (一次试50个方案)
        # max_iter=20: 进化代数 (迭代20次)
        ga = GA(func=self.fitness_function, n_dim=3, size_pop=50, max_iter=20, lb=lb, ub=ub, precision=1e-5)
        
        best_x, best_y = ga.run()
        
        self.best_latency = best_y[0]
        self.best_plan = self._decode_solution(best_x)
        
        return self.best_plan, self.best_latency

    def _decode_solution(self, x):
        """将最优解数字转回人类可读的配置"""
        mode_code = 0 if x[0] < 0.5 else 1
        neighbor_idx = int(x[1] * (self.n_neighbors - 0.001))
        target = self.neighbors[neighbor_idx]['id']
        
        result = {
            "mode": "pipeline" if mode_code == 0 else "parallel",
            "target_node": target
        }
        
        if mode_code == 0:
            result["split_layer"] = int(x[2] * (self.total_layers - 1))
        else:
            result["offload_ratio"] = round(x[2], 2)
            
        return result

# =======================
# 测试代码
# =======================
if __name__ == "__main__":
    # 1. 模拟邻居状态 (从通信模块获取)
    neighbors_mock = [
        {'id': 'Sat-B', 'bw': 50, 'cpu': 100},  # 带宽低，算力高
        {'id': 'Sat-C', 'bw': 500, 'cpu': 20},  # 带宽高，算力低 (适合并行)
        {'id': 'Sat-D', 'bw': 10, 'cpu': 10}    # 都很差
    ]
    
    print("正在运行遗传算法调度器...")
    start_time = time.time()
    
    # 2. 初始化调度器 (针对 YOLOv5, 25层)
    scheduler = GAScheduler(neighbors_mock, model_layers=25, memory_limit=4096)
    
    # 3. 求解
    plan, latency = scheduler.run()
    
    end_time = time.time()
    
    print("\n=== 🎯 最优调度方案 ===")
    print(f"模式: {plan['mode']}")
    print(f"目标卫星: {plan['target_node']}")
    if plan['mode'] == 'pipeline':
        print(f"切分层: 第 {plan['split_layer']} 层")
    else:
        print(f"卸载比例: {plan['offload_ratio']*100}%")
        
    print(f"预测时延: {latency:.2f} ms")
    print(f"算法耗时: {(end_time - start_time)*1000:.2f} ms")