import numpy as np

class DPScheduler:
    """
    实验二：负载感知动态规划 (LA-DP)
    将 L 层分配给 K 个异构节点，最小化端到端时延
    """
    def __init__(self, profile_data, nodes, device_profiles, src_id='RS'):
        self.profile = profile_data  # [{flops, output_mb, mem_mb}, ...]
        self.nodes = nodes           # [node_id, ...]
        self.profiles = device_profiles
        self.src = src_id
        self.L = len(profile_data)
        self.K = len(nodes)

    def _get_compute_power(self, node_id):
        """节点算力 GFLOPS"""
        return self.profiles.get(node_id, {}).get('compute_gflops', 275)

    def _get_bandwidth(self, u, v):
        """链路带宽 Mbps"""
        if u == v:
            return 10000.0  # 本地
        if 'SAT' in u and 'SAT' in v:
            return 50.0
        return 80.0

    def run(self):
        L, K = self.L, self.K

        # dp[i][j] = 将前i层分配给前j个节点的最小时延
        dp = np.full((L + 1, K + 1), np.inf)
        cut = np.zeros((L + 1, K + 1), dtype=int)
        dp[0][0] = 0

        # 前缀和: 累计flops
        flops_sum = [0.0] * (L + 1)
        for i in range(L):
            flops_sum[i + 1] = flops_sum[i] + self.profile[i].get('flops', 1e6)

        for j in range(1, K + 1):
            node_id = self.nodes[j - 1]
            cp = self._get_compute_power(node_id) * 1e9  # FLOPS

            for i in range(1, L + 1):
                for k in range(0, i):
                    # 层 [k, i-1] 分配给节点 j
                    seg_flops = flops_sum[i] - flops_sum[k]
                    calc_time = seg_flops / cp  # 秒

                    # 传输时延: 第k层的输出到节点j
                    comm_time = 0
                    if k > 0 and j > 1:
                        prev_node = self.nodes[j - 2]
                        out_mb = self.profile[k - 1].get('output_mb', 0.1)
                        bw = self._get_bandwidth(prev_node, node_id)
                        comm_time = out_mb / bw  # 秒 (MB / Mbps * 8 => 更精确)
                        comm_time = (out_mb * 8) / bw  # Mb / Mbps = s

                    # 内存约束检查
                    node_mem = self.profiles.get(node_id, {}).get('memory_mb', 16384)
                    seg_mem = sum(self.profile[l].get('mem_mb', 1) for l in range(k, i))
                    if seg_mem > node_mem:
                        continue

                    cost = dp[k][j - 1] + calc_time + comm_time

                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        cut[i][j] = k

        # 回溯路径
        plan = []
        ci = L
        for j in range(K, 0, -1):
            ck = cut[ci][j]
            if ck < ci:
                node_id = self.nodes[j - 1]
                plan.append({
                    'layer': ck,
                    'layer_end': ci - 1,
                    'node': node_id,
                    'cost': dp[ci][j] - dp[ck][j - 1]
                })
            ci = ck

        plan.reverse()
        total_cost = dp[L][K]

        return plan, total_cost