class LAWAScheduler:
    """
    实验三：链路质量感知加权分配 (LAWA)
    在CDP模式下，根据各节点算力和带宽分配数据比例 alpha_i
    """
    def __init__(self, task, nodes, device_profiles, src_id='RS'):
        self.task = task
        self.nodes = nodes          # [node_id, ...]
        self.profiles = device_profiles
        self.src = src_id

    def _get_bandwidth(self, src, dst):
        """公网链路带宽 Mbps"""
        if 'SAT' in src and 'SAT' in dst:
            return 50.0
        return 80.0

    def calculate_weights(self):
        """
        目标: 使所有节点完成时间相等 T_1 = T_2 = ... = T_n
        T_i = alpha_i * D / BW_i + alpha_i * F / CP_i
            = alpha_i * K_i
        其中 K_i = D/BW_i + F/CP_i
        令所有 T_i 相等 => alpha_i = (1/K_i) / sum(1/K_j)
        """
        D = self.task.get('input_mb', 5.0)      # 总数据量 MB
        F = self.task.get('total_flops', 1e9)    # 总计算量 FLOPs

        k_vals = []
        for n_id in self.nodes:
            cp = self.profiles.get(n_id, {}).get('compute_gflops', 275)
            bw = self._get_bandwidth(self.src, n_id)

            trans_cost = (D * 8) / bw          # 秒 (MB -> Mb -> Mb/Mbps)
            calc_cost = F / (cp * 1e9)         # 秒
            k = trans_cost + calc_cost
            k_vals.append(k)

        # 防止极端值
        k_min = min(k_vals)
        k_vals = [min(k, k_min * 20) for k in k_vals]

        inv_k = [1.0 / k for k in k_vals]
        total_inv = sum(inv_k)
        weights = [ik / total_inv for ik in inv_k]

        # 保底: 每个节点至少 5%
        min_w = 0.05
        for i in range(len(weights)):
            if weights[i] < min_w:
                weights[i] = min_w
        # 归一化
        s = sum(weights)
        weights = [w / s for w in weights]

        return weights

    def get_allocation_plan(self):
        weights = self.calculate_weights()
        plan = []
        D = self.task.get('input_mb', 5.0)
        F = self.task.get('total_flops', 1e9)

        for i, n_id in enumerate(self.nodes):
            cp = self.profiles.get(n_id, {}).get('compute_gflops', 275)
            bw = self._get_bandwidth(self.src, n_id)

            est_latency = weights[i] * ((D * 8) / bw + F / (cp * 1e9))
            plan.append({
                'node_id': n_id,
                'data_ratio': round(weights[i], 4),
                'expected_latency_sec': round(est_latency, 6),
            })
        return plan