import numpy as np
import random
import time
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

class CDPSolver:
    def __init__(self, model_profile: Dict, env_status: Dict):
        """
        :param model_profile: 任务及模型档案
            示例结构: {
                "input_size_mb": 100.0,    # D_in: 总输入数据量 (MB)
                "output_size_mb": 5.0,     # D_out: 总输出结果数据量 (MB)
                "compute_total_gflops": 50.0 # C_total: 模型处理整个 batch 所需的总算力 (GFLOPs)
            }
        :param env_status: 工作卫星集群环境配置
            示例结构: {
                "nodes": [
                    {"id": 0, "compute_speed_gflops_per_s": 10.0, "b_dist_mbps": 50.0, "b_agg_mbps": 20.0},
                    {"id": 1, "compute_speed_gflops_per_s": 25.0, "b_dist_mbps": 100.0, "b_agg_mbps": 40.0},
                    ...
                ]
            }
        """
        # ==================== 1. 解析模型与任务需求 ====================
        self.D_in = model_profile.get('input_size_mb', 100.0)
        self.D_out = model_profile.get('output_size_mb', 5.0)
        self.C_total = model_profile.get('compute_total_gflops', 50.0)
        
        # ==================== 2. 解析网络与计算资源 ====================
        self.nodes = env_status['nodes']
        self.K = len(self.nodes)
        
        # 提取异构资源并转化为 numpy 数组，方便矩阵运算加速
        self.node_ids = [n['id'] for n in self.nodes]
        # 算力 f_m
        self.f = np.array([n['compute_speed_gflops_per_s'] for n in self.nodes], dtype=float)
        # 注意：此处假设输入带宽单位为 MB/s (由于你公式中的单位一致性要求，如果是 Mbps 需除以 8)
        self.b_dist = np.array([n.get('b_dist_mbps', 10.0) for n in self.nodes], dtype=float)
        self.b_agg = np.array([n.get('b_agg_mbps', 10.0) for n in self.nodes], dtype=float)

        # 防止零除错误（设置一个极小值）
        self.f = np.clip(self.f, a_min=1e-5, a_max=None)
        self.b_dist = np.clip(self.b_dist, a_min=1e-5, a_max=None)
        self.b_agg = np.clip(self.b_agg, a_min=1e-5, a_max=None)

    # =================== 工具函数：评估数据分配方案的总时延 ===================
    def _evaluate_delay(self, D_alloc: np.ndarray) -> Tuple[float, Dict[int, float]]:
        """
        输入每颗卫星分配到的数据量数组 D_alloc，计算出最大阶段时延
        """
        # 1. 接收时延 t_div
        t_div = D_alloc / self.b_dist
        
        # 2. 计算时延 t_comp
        t_comp = (self.C_total * (D_alloc / self.D_in)) / self.f
        
        # 3. 结果聚合发送时延 t_agg
        t_agg = (D_alloc * (self.D_out / self.D_in)) / self.b_agg
        
        # 4. 单星总耗时 (接收 + 计算 + 发送)
        node_delays = t_div + t_comp + t_agg
        
        # 5. 木桶效应：寻找最长耗时的节点作为整体第一阶段时延
        max_delay = float(np.max(node_delays))
        
        # 生成分配方案字典 {node_id: allocate_data_mb}
        plan = {self.node_ids[i]: round(float(D_alloc[i]), 3) for i in range(self.K)}
        
        return max_delay, plan

    # =================== 1. LAWA (链路感知加权分配算法 - 本文提出) ===================
    def solve_lawa(self) -> Tuple[float, Dict[int, float]]:
        """
        基于综合吞吐率 $\gamma_m$ 的严谨解析解
        """
        # 计算综合负载系数的各个分母项
        term_div = 1.0 / self.b_dist
        term_comp = self.C_total / (self.D_in * self.f)
        term_agg = self.D_out / (self.D_in * self.b_agg)
        
        # 单星综合吞吐率 \gamma_m
        gamma = 1.0 / (term_div + term_comp + term_agg)
        
        # 总系数 \Gamma
        gamma_total = np.sum(gamma)
        
        # 自适应分配数据量 D_m
        D_alloc = self.D_in * (gamma / gamma_total)
        
        return self._evaluate_delay(D_alloc)

    # =================== 2. Compute-Greedy (纯算力加权分配基线) ===================
    def solve_compute_greedy(self) -> Tuple[float, Dict[int, float]]:
        """
        传统数据并行策略：只考虑节点算力强弱进行分配，无视通信链路差异
        """
        F_total = np.sum(self.f)
        D_alloc = self.D_in * (self.f / F_total)
        return self._evaluate_delay(D_alloc)

    def solve_greedy(self) -> Tuple[float, Dict[int, float]]:
        """
        与 solve_compute_greedy 保持兼容的贪心接口。
        """
        return self.solve_compute_greedy()

    # =================== 3. Uniform Partition (均匀分配基线) ===================
    def solve_uniform(self) -> Tuple[float, Dict[int, float]]:
        """
        基础均分策略：不考虑节点异构性（木桶效应极强）
        """
        D_alloc = np.full(self.K, self.D_in / self.K)
        return self._evaluate_delay(D_alloc)

    # =================== 4. Random Search (随机化搜索基线) ===================
    def solve_random_search(self, trials: int = 1000) -> Tuple[float, Dict[int, float]]:
        """
        蒙特卡洛随机分配：用于反映不合理的权重分配造成的时延恶化
        """
        best_lat = float('inf')
        best_D_alloc = None
        
        for _ in range(trials):
            # 生成 K 个随机权重
            weights = np.random.rand(self.K)
            weights /= np.sum(weights)  # 归一化
            
            D_alloc = self.D_in * weights
            lat, _ = self._evaluate_delay(D_alloc)
            
            if lat < best_lat:
                best_lat = lat
                best_D_alloc = D_alloc
                
        return self._evaluate_delay(best_D_alloc)

    # =================== 5. Genetic Algorithm ===================
    def solve_ga(self, pop_size: int = 50, generations: int = 150, mutation_rate: float = 0.3) -> Tuple[float, Dict[int, float]]:
        """
        遗传算法搜索分配方案。

        复杂度近似为 O(pop_size * generations * K)，其中 K 为节点数。
        """

        def normalize(individual: np.ndarray) -> np.ndarray:
            candidate = np.clip(individual.astype(float), 0.0, None)
            total = float(np.sum(candidate))
            if total <= 0.0:
                return np.full(self.K, self.D_in / self.K, dtype=float)
            return candidate / total * self.D_in

        def fitness(individual: np.ndarray) -> float:
            lat, _ = self._evaluate_delay(normalize(individual))
            return -lat

        if pop_size < 2:
            raise ValueError("pop_size 必须至少为 2")
        if generations < 1:
            raise ValueError("generations 必须至少为 1")

        population: List[np.ndarray] = [np.random.dirichlet(np.ones(self.K)) * self.D_in for _ in range(pop_size)]
        best_score = -float('inf')
        best_alloc: np.ndarray = np.full(self.K, self.D_in / self.K, dtype=float)

        for _ in range(generations):
            scores = [fitness(individual) for individual in population]
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_alloc = normalize(population[best_idx])

            ranked_indices = np.argsort(scores)[::-1]
            elite_count = max(2, pop_size // 2)
            selected = [population[index] for index in ranked_indices[:elite_count]]

            next_population: List[np.ndarray] = [normalize(individual.copy()) for individual in selected[:2]]
            while len(next_population) < pop_size:
                parent_1, parent_2 = random.sample(selected, 2)
                if self.K == 1:
                    child = parent_1.copy()
                else:
                    cut = random.randint(1, self.K - 1)
                    child = np.concatenate([parent_1[:cut], parent_2[cut:]])

                if random.random() < mutation_rate:
                    mut_idx = random.randint(0, self.K - 1)
                    child[mut_idx] += random.uniform(-0.2, 0.2) * self.D_in
                    if self.K > 1:
                        other_idx = random.randint(0, self.K - 1)
                        if other_idx != mut_idx:
                            child[other_idx] += random.uniform(-0.1, 0.1) * self.D_in

                next_population.append(normalize(child))

            population = next_population[:pop_size]

        return self._evaluate_delay(best_alloc)


def _build_mock_env(node_count: int) -> Dict:
    nodes = []
    for i in range(node_count):
        nodes.append({
            'id': i,
            'compute_speed_gflops_per_s': 10.0 + i * 5.0,
            'b_dist_mbps': 5.0 + i * 5.0,
            'b_agg_mbps': 2.0 + i * 3.0,
        })
    return {'nodes': nodes}


def _build_mock_model_profile() -> Dict:
    return {
        'input_size_mb': 150.0,
        'output_size_mb': 5.0,
        'compute_total_gflops': 80.0,
    }


def _benchmark(func, repeats: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        func()

    durations_ms: List[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        'mean_ms': statistics.mean(durations_ms),
        'std_ms': statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0,
        'min_ms': min(durations_ms),
        'max_ms': max(durations_ms),
        'last_latency_ms': float(last_result[0]) if last_result is not None else float('nan'),
    }


def _print_complexity_summary() -> None:
    print('理论复杂度对照')
    print('- LAWA: O(K)')
    print('- Greedy: O(K)')
    print('- GA: O(pop_size * generations * K)')
    print()


def run_benchmark(node_counts=None, repeats: int = 20, warmup: int = 3, pop_size: int = 20, generations: int = 30, output_csv: str = 'cdp_search_time_mean.csv') -> None:
    if node_counts is None:
        node_counts = [3, 4, 5, 6]

    random.seed(42)
    np.random.seed(42)

    _print_complexity_summary()
    print(f'重复次数: {repeats} | 预热次数: {warmup} | GA(pop={pop_size}, gen={generations})')
    print('-' * 96)
    print('{:<8} {:<12} {:>12} {:>12} {:>12} {:>12}'.format('K', 'Algorithm', 'Mean(ms)', 'Std(ms)', 'Min(ms)', 'Max(ms)'))
    print('-' * 96)

    csv_rows = []
    for node_count in node_counts:
        solver = CDPSolver(_build_mock_model_profile(), _build_mock_env(node_count))
        algorithm_map = {
            'LAWA': lambda: solver.solve_lawa(),
            'Greedy': lambda: solver.solve_greedy(),
            'GA': lambda: solver.solve_ga(pop_size=pop_size, generations=generations),
        }

        for algorithm_name in ['LAWA', 'Greedy', 'GA']:
            stat = _benchmark(algorithm_map[algorithm_name], repeats=repeats, warmup=warmup)
            csv_rows.append({
                'K': node_count,
                'Algorithm': algorithm_name,
                'Mean_ms': stat['mean_ms'],
                'Std_ms': stat['std_ms'],
                'Min_ms': stat['min_ms'],
                'Max_ms': stat['max_ms'],
                'Last_Latency_ms': stat['last_latency_ms'],
            })
            print(
                '{:<8} {:<12} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}'.format(
                    node_count,
                    algorithm_name,
                    stat['mean_ms'],
                    stat['std_ms'],
                    stat['min_ms'],
                    stat['max_ms'],
                )
            )

    csv_path = Path(__file__).resolve().parent.parent / output_csv
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['K', 'Algorithm', 'Mean_ms', 'Std_ms', 'Min_ms', 'Max_ms', 'Last_Latency_ms'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print('-' * 96)
    print(f'结果已保存到: {csv_path}')

# ==============================================================
# 5. 测试与比较主函数
# ==============================================================
if __name__ == "__main__":
    run_benchmark()