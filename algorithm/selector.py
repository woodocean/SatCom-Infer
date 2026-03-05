class InferenceSelector:
    """
    实验一：多因子加权决策模型
    决策: CDP(数据并行) vs PMP(流水线并行)
    """
    def __init__(self, node_ref, device_profiles):
        self.node = node_ref
        self.profiles = device_profiles

    def select_mode(self, task, available_nodes):
        model_mem = task.get('model_mem_mb', 0)

        # 1. 内存预筛选
        can_run_full = []
        for n_id in available_nodes:
            if n_id in self.profiles:
                if self.profiles[n_id].get('memory_mb', 0) > model_mem:
                    can_run_full.append(n_id)
            else:
                can_run_full.append(n_id)  # 无profile信息时默认可用

        if not can_run_full:
            return "PMP", available_nodes

        # 2. 预估CDP时延
        t_cdp = self._est_cdp(task, can_run_full)

        # 3. 预估PMP时延
        t_pmp = self._est_pmp(task, can_run_full)

        # 4. 任务类型加权
        task_type = task.get('type', 'compute_intensive')
        w_cdp, w_pmp = 1.0, 1.0
        if task_type == 'comm_sensitive':
            w_cdp *= 0.7
        elif task_type == 'memory_intensive':
            w_pmp *= 0.8

        score_cdp = t_cdp * w_cdp
        score_pmp = t_pmp * w_pmp

        print(f"  [Selector] CDP预估={t_cdp:.4f}s (score={score_cdp:.4f}), "
              f"PMP预估={t_pmp:.4f}s (score={score_pmp:.4f})")

        return ("CDP", can_run_full) if score_cdp < score_pmp else ("PMP", can_run_full)

    def _get_bw(self, src, dst):
        """公网带宽模型 (Mbps)"""
        if 'SAT' in src and 'SAT' in dst:
            return 50.0
        return 80.0

    def _est_cdp(self, task, nodes):
        """CDP预估: 分发 + 并行计算(取最慢) + 聚合"""
        input_mb = task.get('input_mb', 1.0)
        total_flops = task.get('total_flops', 1e9)
        output_mb = task.get('output_mb', 0.1)
        bw = 80.0

        dist_time = input_mb / bw  # 分发
        calc_times = []
        for n_id in nodes:
            cp = self.profiles.get(n_id, {}).get('compute_gflops', 275)
            calc_times.append(total_flops / (cp * 1e9))
        calc_time = max(calc_times) if calc_times else 0.1
        agg_time = output_mb * len(nodes) / bw

        return dist_time + calc_time + agg_time

    def _est_pmp(self, task, nodes):
        """PMP预估: 串行 (计算+传输) × 跳数"""
        total_flops = task.get('total_flops', 1e9)
        output_mb = task.get('output_mb', 0.1)
        n = len(nodes)
        bw = 80.0

        per_node_flops = total_flops / n
        cp = 275  # Orin GFLOPS
        time_per_hop = per_node_flops / (cp * 1e9) + output_mb / bw

        return time_per_hop * n