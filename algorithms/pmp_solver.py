import numpy as np
import random

class PMPSolver:
    def __init__(self, model_profile, env_status):
        self.layers = model_profile['layers']
        self.L = len(self.layers)
        self.nodes = env_status['nodes']
        self.K = len(self.nodes)
        self.B = env_status['bandwidths']
        self.input_size_raw = model_profile.get('input_size_raw', 3.0)
        
        #  [算法核心改进] F_REF 是你在本地做 Profiling 时那台电脑的算力等级标定量
        # 比如你的开发机(显卡)对应 100 GFLOPs/ms，而卫星节点可能只有 5 GFLOPs/ms
        # 这个常数是作为异构硬件等效放缩的基石（具体数值需和你的 network_config 统一度量衡）
        self.F_REF = env_status.get("reference_compute_speed", 100.0)

    # ================= 1. LA-DP (负载感知动态规划) — 支持中继节点 =================
    def solve_la_dp(self):
        import numpy as np

        # ================= 前缀和优化 (Prefix Sums) =================
        # prefix_latency[i] 表示 前 i 层 base_latency_ms 的总和
        # prefix_mem[i] 表示 前 i 层 params_mb 的总和
        # 预计算后，查任意区间 [prev_l, l) 的和只要常数时间 O(1)
        prefix_latency = [0.0] * (self.L + 1)
        prefix_mem = [0.0] * (self.L + 1)
        for i in range(self.L):
            prefix_latency[i+1] = prefix_latency[i] + self.layers[i].get('base_latency_ms', 0)
            prefix_mem[i+1] = prefix_mem[i] + self.layers[i].get('params_mb', 0)
        # =========================================================

        dp = np.full((self.K + 1, self.L + 1), float('inf'))
        parent = np.full((self.K + 1, self.L + 1), -1)
        dp[0][0] = 0

        for k in range(1, self.K + 1):
            node_f = self.nodes[k-1]['compute_speed_gflops_per_ms']
            node_mem = self.nodes[k-1]['memory_mb']

            for l in range(0, self.L + 1):  # l 从 0 开始（允许中继）
                for prev_l in range(l + 1):  # prev_l ∈ [0, l]
                    if dp[k-1][prev_l] == float('inf'):
                        continue

                    # === 中继模式: prev_l == l ===
                    if prev_l == l:
                        t_comp = 0.0
                        input_idx = l - 1 if l > 0 else -1
                        input_size = self.layers[input_idx]['comm_size_mb'] if input_idx >= 0 else self.input_size_raw
                        t_trans = input_size / self.B[k-1]
                        cost = dp[k-1][prev_l] + t_comp + t_trans
                        if cost < dp[k][l]:
                            dp[k][l] = cost
                            parent[k][l] = prev_l
                        continue

                    # === 计算模式: prev_l < l ===
                    # 1. 内存约束检查 (用前缀和 O(1) 代替 sum)
                    required_mem = prefix_mem[l] - prefix_mem[prev_l]
                    if required_mem > node_mem:
                        continue

                    # 2. 计算时延 (用前缀和 O(1) 代替 sum)
                    sum_latency = prefix_latency[l] - prefix_latency[prev_l]
                    t_comp = sum_latency * (self.F_REF / node_f) if node_f > 0 else float('inf')

                    # 3. 传输时延
                    input_size = self.layers[prev_l-1]['comm_size_mb'] if prev_l > 0 else self.input_size_raw
                    t_trans = input_size / self.B[k-1]

                    cost = dp[k-1][prev_l] + t_comp + t_trans
                    if cost < dp[k][l]:
                        dp[k][l] = cost
                        parent[k][l] = prev_l

        total_latency = dp[self.K][self.L]
        if np.isinf(total_latency):
            raise RuntimeError("LA-DP: 无可行解！请检查内存/算力约束")

        # === 回溯生成 plan ===
        plan = {}
        curr_l = self.L
        for k in range(self.K, 0, -1):
            prev_l = parent[k][curr_l]
            if prev_l != curr_l:  # 非中继节点
                plan[self.nodes[k-1]['id']] = [int(prev_l), int(curr_l) - 1]
            curr_l = prev_l

        plan = dict(list(plan.items())[::-1])
        return total_latency, plan

        # ================= 2. Communication-Greedy (真正的贪婪：通信最小化) =================
    def solve_communication_greedy(self):
        """
        每一跳策略：在满足内存约束的前进范围内，寻找令【当前通信延迟】最小的最远切分点。
        """
        plan = {}
        curr_l = 0
        total_latency = 0
        
        for k in range(self.K):
            if curr_l >= self.L: # 已经分完了，剩下的变中继
                input_size = self.layers[self.L-1]['comm_size_mb']
                total_latency += input_size / self.B[k]
                continue

            node_f = self.nodes[k]['compute_speed_gflops_per_ms']
            node_mem = self.nodes[k]['memory_mb']
            
            best_next_l = curr_l # 默认中继
            min_comm_size = float('inf')
            
            # 尝试在当前节点塞入多少层 [curr_l, next_l)
            for next_l in range(curr_l + 1, self.L + 1):
                if (self.prefix_mem[next_l] - self.prefix_mem[curr_l]) > node_mem:
                    break # 内存爆了，停止探测
                
                # 贪婪目标：寻找输出张量（通信量）最小的地方
                # next_l 层执行完后的输出大小
                out_comm = self.layers[next_l-1]['comm_size_mb'] if next_l < self.L else 0
                
                if out_comm <= min_comm_size: # 贪婪选择通信更小的点
                    min_comm_size = out_comm
                    best_next_l = next_l

            # 计算这一跳的开销
            input_size = self.layers[curr_l-1]['comm_size_mb'] if curr_l > 0 else self.input_size_raw
            t_trans = input_size / self.B[k]
            
            if best_next_l > curr_l: # 节点执行了计算
                t_comp = (self.prefix_latency[best_next_l] - self.prefix_latency[curr_l]) * (self.F_REF / node_f)
                plan[self.nodes[k]['id']] = [curr_l, best_next_l - 1]
                total_latency += t_comp + t_trans
                curr_l = best_next_l
            else: # 节点沦为中继
                total_latency += t_trans
                
        return total_latency, plan

    # ================= 3. Uniform Partition (原 Greedy：平均切分) =================
    def solve_uniform_partition(self):
        plan = {}
        curr_l = 0
        avg_layers = max(1, self.L // self.K)
        total_latency = 0

        for k in range(self.K):
            start = curr_l
            # 尽量平均分
            end = min(start + avg_layers, self.L) if k < self.K-1 else self.L
            
            node_f = self.nodes[k]['compute_speed_gflops_per_ms']
            node_mem = self.nodes[k]['memory_mb']

            # 内存退避检查
            while end > start and (self.prefix_mem[end] - self.prefix_mem[start]) > node_mem:
                end -= 1

            input_size = self.layers[start-1]['comm_size_mb'] if start > 0 else self.input_size_raw
            t_trans = input_size / self.B[k]
            
            if end > start:
                t_comp = (self.prefix_latency[end] - self.prefix_latency[start]) * (self.F_REF / node_f)
                plan[self.nodes[k]['id']] = [start, end - 1]
                total_latency += t_comp + t_trans
                curr_l = end
            else: # 中继
                total_latency += t_trans
                
        return total_latency, plan

    # ================= 3. Bent-Pipe (弯管回传基线: 纯传图至地计算) =================
    def solve_bent_pipe(self):
        t_trans_total = sum(self.input_size_raw / b for b in self.B)
        gs_node = self.nodes[-1]
        f_gs = gs_node['compute_speed_gflops_per_ms']
        
        t_comp_gs = sum(l['base_latency_ms'] for l in self.layers) * (self.F_REF / f_gs) if f_gs > 0 else float('inf')
        
        return (t_trans_total + t_comp_gs), {gs_node['id']: [0, self.L - 1]}
    
    # ================= 4. Random Split (随机基线) =================
    def solve_random_split(self, n_trials=50):
        best_latency = float('inf')
        best_plan = {}
        
        for _ in range(n_trials):
            # ✅ 关键修改：允许重复切分点（模拟非严格切分），用 np.random.choice
            # 注意：切分点范围是 [0, L]（包含端点），但需保证严格递增 → 先生成再排序去重
            cuts_raw = np.random.choice(range(0, self.L + 1), size=self.K - 1, replace=True)
            cuts = np.unique(np.sort(cuts_raw))
            
            # 补头尾：[0] + cuts + [L]
            splits = [0] + cuts.tolist() + [self.L]
            
            # 如果切分点太多（> K），截断；太少（< K），补缺（避免空段）
            if len(splits) > self.K + 1:
                splits = [0] + cuts.tolist()[:self.K-1] + [self.L]
            elif len(splits) < self.K + 1:
                # 补充缺失点（在空隙中随机插）
                for _ in range(self.K + 1 - len(splits)):
                    gaps = []
                    for i in range(len(splits) - 1):
                        if splits[i+1] - splits[i] > 1:
                            gaps.append((splits[i] + 1, splits[i+1] - 1))
                    if gaps:
                        left, right = random.choice(gaps)
                        new_pt = random.randint(left, right)
                        splits.append(new_pt)
                        splits.sort()
                    else:
                        # 无空隙可插 → 重复末尾（中继）
                        splits.append(splits[-1])
                        splits.sort()
            
            total_latency = 0
            plan = {}
            valid = True
            
            for k in range(self.K):
                start = splits[k]
                end = splits[k + 1]
                if start >= end:
                    continue  # 中继
                
                node, f, mem = self.nodes[k], self.nodes[k]['compute_speed_gflops_per_ms'], self.nodes[k]['memory_mb']
                req_mem = sum(l.get('params_mb', 0) for l in self.layers[start:end])
                if req_mem > mem:
                    valid = False
                    break
                
                t_comp = sum(l['base_latency_ms'] for l in self.layers[start:end]) * (self.F_REF / f) if f > 0 else float('inf')
                input_size = self.layers[start-1]['comm_size_mb'] if start > 0 else self.input_size_raw
                t_trans = input_size / self.B[k]
                
                total_latency += t_comp + t_trans
                plan[node['id']] = [start, end - 1]
            
            if valid and total_latency < best_latency:
                best_latency = total_latency
                best_plan = plan.copy()
        
        return best_latency, best_plan
    
    # ================= 5. Genetic Algorithm (GA) =================
    def solve_ga(self, pop_size=30, generations=100, mutation_rate=0.2):
        import numpy as np
        
        # 辅助函数：个体转为切分方案
        def decode(individual):
            # individual: [e1, e2, ..., e_{K-1}], 严格递增
            cuts = np.sort(individual)
            cuts = np.clip(cuts, 1, self.L - 1)  # 保护边界
            # 去重 + 保证严格递增
            cuts = np.unique(cuts)
            while len(cuts) < self.K - 1:
                # 补充缺失点（在空隙中随机插入）
                gaps = []
                for i in range(len(cuts) + 1):
                    left = cuts[i-1] if i > 0 else 0
                    right = cuts[i] if i < len(cuts) else self.L
                    if right - left > 1:
                        gaps.append((left + 1, right - 1))
                if not gaps: break
                left, right = random.choice(gaps)
                cuts = np.append(cuts, random.randint(left, right))
                cuts = np.sort(cuts)
            cuts = cuts[:self.K - 1]  # 截断
            splits = [0] + cuts.tolist() + [self.L]
            return splits

        # 适应度函数：负时延（最大化适应度 = 最小化时延）
        def fitness(individual):
            splits = decode(individual)
            total_latency = 0
            for k in range(self.K):
                start, end = splits[k], splits[k+1]
                if start >= end:
                    continue  # 中继节点
                
                node = self.nodes[k]
                f, mem = node['compute_speed_gflops_per_ms'], node['memory_mb']
                
                # 内存检查
                req_mem = sum(l.get('params_mb', 0) for l in self.layers[start:end])
                if req_mem > mem:
                    return -float('inf')  # 无效解惩罚
                
                t_comp = sum(l['base_latency_ms'] for l in self.layers[start:end]) * (self.F_REF / f) if f > 0 else float('inf')
                input_size = self.layers[start-1]['comm_size_mb'] if start > 0 else self.input_size_raw
                t_trans = input_size / self.B[k]
                
                total_latency += t_comp + t_trans
            
            return -total_latency  # 负时延，越大越好

        # 初始化种群：随机生成合法切分点
        population = []
        for _ in range(pop_size):
            # 在 (1, L) 中均匀采样 K-1 个点
            indiv = np.random.randint(1, self.L, size=self.K - 1)
            population.append(indiv)
        
        best_score = -float('inf')
        best_plan = {}

        for gen in range(generations):
            # 评估
            scores = [fitness(indiv) for indiv in population]
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_indiv = population[best_idx]
                splits = decode(best_indiv)
                # 构建 plan
                plan = {}
                for k in range(self.K):
                    start, end = splits[k], splits[k+1]
                    if start < end:
                        plan[self.nodes[k]['id']] = [start, end - 1]
                best_plan = plan

            # 选择：轮盘赌 or top-k
            sorted_idx = np.argsort(scores)[::-1]
            selected = [population[i] for i in sorted_idx[:pop_size//2]]

            # 交叉（单点交叉）
            next_gen = selected.copy()
            while len(next_gen) < pop_size:
                p1, p2 = random.sample(selected, 2)
                cut = random.randint(1, len(p1)-1)
                child = np.concatenate([p1[:cut], p2[cut:]])
                next_gen.append(child)

            # 变异
            for i in range(len(next_gen)):
                if random.random() < mutation_rate:
                    mut_idx = random.randint(0, len(next_gen[i]) - 1)
                    next_gen[i][mut_idx] = random.randint(1, self.L - 1)

            population = next_gen

        return -best_score, best_plan
    
    def solve_exhaustive(self):
        """穷举法：遍历所有切分方案（支持透传中继，加入前缀和公平比对）"""
        import itertools
        import time

        start_time = time.perf_counter()
        best_latency = float('inf')
        best_plan = {}

        # 1. 同样加上前缀和，免得因为 Python 内置 sum() 运行太慢而带来不公平的对比
        prefix_latency = [0.0] * (self.L + 1)
        prefix_mem = [0.0] * (self.L + 1)
        for i in range(self.L):
            prefix_latency[i+1] = prefix_latency[i] + self.layers[i].get('base_latency_ms', 0)
            prefix_mem[i+1] = prefix_mem[i] + self.layers[i].get('params_mb', 0)

        # 2. 生成所有组合（重点修改：使用 combinations_with_replacement 支持透传）
        # 可以选取的切分点为 0 到 L（包含）。允许同一切分点被选取多次，即中间存在透传网络
        # 例如 K=3 时，可选出 [0, 0]，即 splits = [0, 0, 0, L]，代表前两个节点纯透传
        cut_positions = range(self.L + 1)
        all_combinations = itertools.combinations_with_replacement(cut_positions, self.K - 1)

        for cuts in all_combinations:
            splits = [0] + list(cuts) + [self.L]
            total_latency = 0
            plan = {}
            valid = True

            for k in range(self.K):
                start = splits[k]
                end = splits[k + 1]

                # 中继/透传节点逻辑校验
                if start == end:
                    t_comp = 0.0
                    input_idx = end - 1 if end > 0 else -1
                    input_size = self.layers[input_idx]['comm_size_mb'] if input_idx >= 0 else self.input_size_raw
                    t_trans = input_size / self.B[k]
                    total_latency += t_comp + t_trans
                    continue  

                node = self.nodes[k]
                f = node['compute_speed_gflops_per_ms']
                mem = node['memory_mb']

                # 内存检查 (O(1))
                req_mem = prefix_mem[end] - prefix_mem[start]
                if req_mem > mem:
                    valid = False
                    break

                # 计算时延 (O(1))
                sum_latency = prefix_latency[end] - prefix_latency[start]
                t_comp = sum_latency * (self.F_REF / f) if f > 0 else float('inf')
                
                input_size = self.layers[start-1]['comm_size_mb'] if start > 0 else self.input_size_raw
                t_trans = input_size / self.B[k]

                total_latency += t_comp + t_trans
                plan[node['id']] = [start, end - 1]

            if valid and total_latency < best_latency:
                best_latency = total_latency
                best_plan = plan.copy()

        search_time_ms = (time.perf_counter() - start_time) * 1000  # 转为毫秒
        return best_latency, best_plan, search_time_ms