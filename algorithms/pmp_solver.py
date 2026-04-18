import numpy as np
import random
import itertools
import time
from typing import Dict, List, Tuple, Optional

class PMPSolver:
    def __init__(self, model_profile: Dict, env_status: Dict):
        """
        :param model_profile: 来自 dnn_profiles_database.json 的单个任务档案
            示例结构: {
                "layers": [
                    {"latency_mean_ms": 8.15, "comm_total_mb": 200.0, "comm_pure_mb": 200.0, "weight_size_mb": 1.2},
                    {"latency_mean_ms": 3.29, "comm_total_mb": 100.0, "comm_pure_mb": 100.0, "weight_size_mb": 0.8},
                    ...
                ],
                "input_size_raw": 5.0   # 单位: MB（即原始输入图像大小，如 640x640x3 => ~3.8MB）
            }
        :param env_status: 来自 network_config.json 的环境配置
            示例结构: {
                "nodes": [
                    {"id": 0, "memory_mb": 8192, "compute_speed_gflops_per_ms": 5.2},
                    {"id": 1, "memory_mb": 16384, "compute_speed_gflops_per_ms": 12.0},
                    ...
                ],
                "bandwidths": [100.0, 150.0, ..., 200.0],  # 单位: Mbps (每跳链路带宽)
                "reference_compute_speed": 100.0  # 本地标定算力基准（GFLOPs/ms）
            }
        """
        self.nodes: List[Dict] = env_status['nodes']
        self.K: int = len(self.nodes)
        self.B: List[float] = env_status['bandwidths']  # 每跳链路带宽（Mbps）
        self.input_size_raw: float = model_profile.get('input_size_raw', 3.0)  # MB

        # 根据传入的 dict 或者是 list 处理 layers (兼容旧版单 list)
        if isinstance(model_profile['layers'], dict):
            self.layers_dict = model_profile['layers']
            self.layers = self.layers_dict["pc"]  # 用 PC 的作为基准取 comm 和 weight
        else:
            self.layers = model_profile['layers']
            self.layers_dict = {"pc": self.layers, "jetson": self.layers}  # fallback
            
        self.L: int = len(self.layers)

        # 异构算力标定系数（保留，用于节点算力缩放）
        self.F_REF: float = env_status.get("reference_compute_speed", 100.0)

        # ================ 预计算前缀和（关键优化） ================
        # 注意：此处字段名已更新为新字段
        self.prefix_latency = {"pc": [0.0] * (self.L + 1), "jetson": [0.0] * (self.L + 1)}
        self.prefix_weight = [0.0] * (self.L + 1)
        self.prefix_pure = [0.0] * (self.L + 1)  # 用于求 max(comm_pure_mb) 的前缀最大值（后续用滑动窗口）
        
        for i in range(self.L):
            self.prefix_latency["pc"][i+1] = self.prefix_latency["pc"][i] + self.layers_dict["pc"][i].get('latency_mean_ms', 0.0)
            if "jetson" in self.layers_dict:
                self.prefix_latency["jetson"][i+1] = self.prefix_latency["jetson"][i] + self.layers_dict["jetson"][i].get('latency_mean_ms', 0.0)
            
            self.prefix_weight[i+1] = self.prefix_weight[i] + self.layers[i].get('weight_size_mb', 0.0)
            # 纯特征图大小：我们用前缀最大值数组（非前缀和）——但为简化，我们改用滑动窗口在 DP 中动态求 max
        # 说明：max(comm_pure_mb) 无法用前缀和直接表示，因此我们将在内存检查时用 O(1) 区间查询（需额外维护 RMQ 或直接遍历小段）。
        # 由于 L 一般 ≤ 50，此处可接受 O(L) 检查，或你可后续加 Sparse Table。此处为简洁，用遍历。

    # =================== 工具函数：查表计算通信时延（ms） ===================
    def _comm_delay_ms(self, comm_mb: float, bandwidth_mbps: float) -> float:
        """通信延时计算：(数据量(MB) * 8 比特/字节) / 带宽(Mbps) * 1000 => ms"""
        if bandwidth_mbps <= 0:
            return float('inf')
        return (comm_mb * 8.0 / bandwidth_mbps) * 1000.0

    # =================== 工具函数：检查内存约束（核心逻辑） ===================
    def _check_memory(self, node_idx: int, start: int, end: int) -> Tuple[bool, float]:
        """
        检查节点 node_idx 是否能承载 [start, end) 层
        内存约束 = 权重和 + max(comm_pure_mb) * 2
        """
        if start >= end:
            return True, 0.0  # 中继节点无内存占用

        weight_sum: float = self.prefix_weight[end] - self.prefix_weight[start]
        # 求区间 [start, end) 内最大的 comm_pure_mb
        max_pure = 0.0
        for i in range(start, end):
            pure = self.layers[i].get('comm_pure_mb', 0.0)
            if pure > max_pure:
                max_pure = pure
        mem_required = weight_sum + max_pure * 2.0

        node_mem = self.nodes[node_idx].get('memory_mb', 8192)
        return mem_required <= node_mem, mem_required

    # =================== 工具函数：计算某段层的计算时延（ms） ===================
    def _compute_delay(self, node_idx: int, start: int, end: int) -> float:
        if start >= end:
            return 0.0
        
        # 1. 判定设备类型 (默认 fallback 到 pc)
        # 这里需要网络配置里传进了 `device` 类型 (如 'PC', 'Jetson_1')
        device_str = str(self.nodes[node_idx].get('device', 'PC')).lower()
        if 'jetson' in device_str:
            device_type = 'jetson'
        else:
            device_type = 'pc'
            
        # 2. 直接取目标设备的 profile 耗时
        comp_latency = self.prefix_latency[device_type][end] - self.prefix_latency[device_type][start]
        
        return comp_latency

    # =================== 1. LA-DP (负载感知动态规划) ===================
    def solve_la_dp(self) -> Tuple[float, Dict]:
        dp: np.ndarray = np.full((self.K + 1, self.L + 1), float('inf'))
        parent: np.ndarray = np.full((self.K + 1, self.L + 1), -1, dtype=int)
        dp[0][0] = 0.0

        for k in range(1, self.K + 1):
            node_idx = k - 1
            for l in range(0, self.L + 1):  # l: 当前已计算到第 l 层
                for prev_l in range(l + 1):  # prev_l: 上一节点算到了哪层
                    if dp[k-1][prev_l] == float('inf'):
                        continue

                    # --- 情况1: 中继模式 (prev_l == l) ---
                    if prev_l == l:
                        # 传输输入：上一跳的输出（即第 prev_l-1 层的 comm_total_mb）
                        input_comm = self.layers[prev_l - 1]['comm_total_mb'] if prev_l > 0 else self.input_size_raw
                        t_trans = self._comm_delay_ms(input_comm, self.B[node_idx])
                        cost = dp[k-1][prev_l] + 0.0 + t_trans
                        if cost < dp[k][l]:
                            dp[k][l] = cost
                            parent[k][l] = prev_l
                        continue

                    # --- 情况2: 计算模式 (prev_l < l) ---
                    # 内存检查
                    mem_ok, _ = self._check_memory(node_idx, prev_l, l)
                    if not mem_ok:
                        continue

                    # 计算时延
                    t_comp = self._compute_delay(node_idx, prev_l, l)

                    # 传输时延：输入为 prev_l 层输出（即第 prev_l-1 层的 comm_total_mb）
                    input_comm = self.layers[prev_l - 1]['comm_total_mb'] if prev_l > 0 else self.input_size_raw
                    t_trans = self._comm_delay_ms(input_comm, self.B[node_idx])

                    cost = dp[k-1][prev_l] + t_comp + t_trans
                    if cost < dp[k][l]:
                        dp[k][l] = cost
                        parent[k][l] = prev_l

        total_lat = dp[self.K][self.L]
        if np.isinf(total_lat):
            raise RuntimeError("LA-DP: 无可行解！请检查内存/带宽约束")

        # 回溯生成 plan
        plan = {}
        curr_l = self.L
        for k in range(self.K, 0, -1):
            prev_l = int(parent[k][curr_l])
            if prev_l != curr_l:
                plan[self.nodes[k-1]['id']] = [prev_l, curr_l - 1]
            curr_l = prev_l
        plan = dict(reversed(list(plan.items())))
        return float(total_lat), plan

    # =================== 2. Communication-Greedy ===================
    def solve_communication_greedy(self) -> Tuple[float, Dict]:
        plan = {}
        curr_l = 0
        total_latency = 0.0

        for k in range(self.K):
            if curr_l >= self.L:
                # 全部已分完，剩余节点仅中继（传输最后一层输出）
                input_comm = self.layers[self.L - 1]['comm_total_mb']
                total_latency += self._comm_delay_ms(input_comm, self.B[k])
                continue

            node_idx = k
            best_next_l: int = curr_l
            min_comm_output: float = float('inf')

            # 尝试扩展切分点 [curr_l, next_l)
            for next_l in range(curr_l + 1, self.L + 1):
                mem_ok, _ = self._check_memory(node_idx, curr_l, next_l)
                if not mem_ok:
                    break  # 内存超限，后续更大区间必超，剪枝

                # 贪婪目标：最小化 next_l 层输出的 comm_total_mb（即下跳传输量）
                output_comm = self.layers[next_l - 1]['comm_total_mb'] if next_l < self.L else 0.0
                if output_comm <= min_comm_output:
                    min_comm_output = output_comm
                    best_next_l = next_l

            # 计算当前跳开销
            input_comm: float = self.layers[curr_l - 1]['comm_total_mb'] if curr_l > 0 else self.input_size_raw
            t_trans: float = self._comm_delay_ms(input_comm, self.B[k])

            if best_next_l > curr_l:
                t_comp = self._compute_delay(node_idx, curr_l, best_next_l)
                plan[self.nodes[node_idx]['id']] = [curr_l, best_next_l - 1]
                total_latency += t_comp + t_trans
                curr_l = best_next_l
            else:
                # 中继：仅传输，不计算
                total_latency += t_trans

        return total_latency, plan

    # =================== 3. Uniform Partition ===================
    def solve_uniform_partition(self) -> Tuple[float, Dict]:
        plan = {}
        curr_l = 0
        total_latency = 0.0
        avg_layers = max(1, self.L // self.K)

        for k in range(self.K):
            start: int = curr_l
            end: int = min(start + avg_layers, self.L) if k < self.K - 1 else self.L

            node_idx = k
            # 内存退避：从 end 往回缩
            while end > start and not self._check_memory(node_idx, start, end)[0]:
                end -= 1

            input_comm = self.layers[start - 1]['comm_total_mb'] if start > 0 else self.input_size_raw
            t_trans = self._comm_delay_ms(input_comm, self.B[k])

            if end > start:
                t_comp = self._compute_delay(node_idx, start, end)
                plan[self.nodes[node_idx]['id']] = [start, end - 1]
                total_latency += t_comp + t_trans
                curr_l = end
            else:
                total_latency += t_trans  # 中继

        return total_latency, plan

    # =================== 4. Bent-Pipe ===================
    def solve_bent_pipe(self) -> Tuple[float, Dict]:
        # 所有链路传输原始输入
        t_trans_total = sum(self._comm_delay_ms(self.input_size_raw, b) for b in self.B)
        gs_node = self.nodes[-1]
        gs_idx = len(self.nodes) - 1
        t_comp = self._compute_delay(gs_idx, 0, self.L)
        return t_trans_total + t_comp, {gs_node['id']: [0, self.L - 1]}

    # =================== 5. Random Split ===================
    def solve_random_split(self, n_trials: int = 50) -> Tuple[float, Dict]:
        best_lat = float('inf')
        best_plan: Dict = {}

        for _ in range(n_trials):
            cuts_raw: np.ndarray = np.random.choice(range(0, self.L + 1), size=self.K - 1, replace=True)
            cuts = np.unique(np.sort(cuts_raw))
            splits: List[int] = [0] + cuts.tolist() + [self.L]

            # 补足/截断至 K+1 个点
            if len(splits) > self.K + 1:
                splits = [0] + cuts.tolist()[:self.K - 1] + [self.L]
            elif len(splits) < self.K + 1:
                while len(splits) < self.K + 1:
                    gaps = []
                    for i in range(len(splits) - 1):
                        if splits[i + 1] - splits[i] > 1:
                            gaps.append((splits[i] + 1, splits[i + 1] - 1))
                    if gaps:
                        left, right = random.choice(gaps)
                        splits.append(random.randint(left, right))
                        splits.sort()
                    else:
                        splits.append(splits[-1])
                        splits.sort()

            total_lat = 0.0
            plan = {}
            valid = True

            for k in range(self.K):
                start, end = splits[k], splits[k + 1]
                if start >= end:
                    # 中继：传输上一跳输出
                    input_comm: float = self.layers[end - 1]['comm_total_mb'] if end > 0 else self.input_size_raw
                    total_lat += self._comm_delay_ms(input_comm, self.B[k])
                    continue

                node_idx = k
                mem_ok, _ = self._check_memory(node_idx, start, end)
                if not mem_ok:
                    valid = False
                    break

                t_comp: float = self._compute_delay(node_idx, start, end)
                input_comm: float = self.layers[start - 1]['comm_total_mb'] if start > 0 else self.input_size_raw
                t_trans: float = self._comm_delay_ms(input_comm, self.B[k])
                total_lat += t_comp + t_trans
                plan[self.nodes[node_idx]['id']] = [start, end - 1]

            if valid and total_lat < best_lat:
                best_lat = total_lat
                best_plan = plan.copy()

        return best_lat, best_plan

    # =================== 6. Genetic Algorithm ===================
    def solve_ga(self, pop_size: int = 30, generations: int = 100, mutation_rate: float = 0.2) -> Tuple[float, Dict]:
        def decode(individual: np.ndarray) -> List[int]:
            cuts: np.ndarray = np.sort(individual)
            cuts = np.clip(cuts, 1, self.L - 1)
            cuts = np.unique(cuts)
            while len(cuts) < self.K - 1:
                gaps = []
                for i in range(len(cuts) + 1):
                    left = cuts[i - 1] if i > 0 else 0
                    right = cuts[i] if i < len(cuts) else self.L
                    if right - left > 1:
                        gaps.append((left + 1, right - 1))
                if not gaps:
                    break
                left, right = random.choice(gaps)
                cuts = np.append(cuts, random.randint(left, right))
                cuts = np.sort(cuts)
            cuts = cuts[:self.K - 1]
            return [0] + cuts.tolist() + [self.L]

        def fitness(individual: np.ndarray) -> float:
            splits = decode(individual)
            total_lat = 0.0
            for k in range(self.K):
                start, end = splits[k], splits[k + 1]
                if start == end:
                    input_comm: float = self.layers[end - 1]['comm_total_mb'] if end > 0 else self.input_size_raw
                    total_lat += self._comm_delay_ms(input_comm, self.B[k])
                    continue
                node_idx = k
                mem_ok, _ = self._check_memory(node_idx, start, end)
                if not mem_ok:
                    return -float('inf')
                t_comp: float = self._compute_delay(node_idx, start, end)
                input_comm: float = self.layers[start - 1]['comm_total_mb'] if start > 0 else self.input_size_raw
                t_trans: float = self._comm_delay_ms(input_comm, self.B[k])
                total_lat += t_comp + t_trans
            return -total_lat  # 负值，最大化适应度 = 最小化时延

        population = [np.random.randint(1, self.L, size=self.K - 1) for _ in range(pop_size)]
        best_score = -float('inf')
        best_plan: Dict = {}

        for gen in range(generations):
            scores = [fitness(indiv) for indiv in population]
            best_idx: int = int(np.argmax(scores))
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_indiv: np.ndarray = population[best_idx]
                splits = decode(best_indiv)
                plan = {}
                for k in range(self.K):
                    s, e = splits[k], splits[k + 1]
                    if s < e:
                        plan[self.nodes[k]['id']] = [s, e - 1]
                best_plan = plan

            # 选择（取 top 50%）
            sorted_idx = np.argsort(scores)[::-1]
            selected = [population[i] for i in sorted_idx[:pop_size // 2]]

            # 交叉
            next_gen = selected.copy()
            while len(next_gen) < pop_size:
                p1, p2 = random.sample(selected, 2)
                cut = random.randint(1, len(p1) - 1)
                child: np.ndarray = np.concatenate([p1[:cut], p2[cut:]])
                next_gen.append(child)

            # 变异
            for i, indiv in enumerate(next_gen):
                if random.random() < mutation_rate:
                    mut_idx: int = random.randint(0, len(indiv) - 1)
                    next_gen[i][mut_idx] = random.randint(1, self.L - 1)

            population = next_gen

        return -best_score, best_plan
