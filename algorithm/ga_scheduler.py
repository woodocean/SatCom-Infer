import numpy as np
from sko.GA import GA

class GAScheduler:
    def __init__(self, net_config, src_id, dst_id, model_layers=25):
        self.nodes_cfg = net_config['nodes']
        self.src = src_id
        self.dst = dst_id
        self.layers = model_layers
        
        # 1. 模拟资源画像 (因为目前json里没有带宽算力信息)
        # 假设: SAT-01算力强但在内网，SAT-02带宽好但算力弱
        self.profile = {
            'RS':      {'cpu': 50,  'bw': 1000},
            'SAT-01':  {'cpu': 150, 'bw': 50},   # 算力强，带宽低
            'SAT-02':  {'cpu': 60,  'bw': 500},  # 算力弱，带宽高
            'SAT-AGG': {'cpu': 100, 'bw': 200},
            'GS':      {'cpu': 999, 'bw': 1000}
        }

        # 2. 拓扑发现：寻找所有从 RS 到 GS 的路径
        self.candidate_paths = self._find_all_paths(self.src, self.dst)
        
        # 3. 获取直接邻居 (用于并行模式)
        self.direct_neighbors = self.nodes_cfg[self.src]['neighbors']

    def _find_all_paths(self, current, target, path=None):
        """DFS 寻找所有路径"""
        if path is None: path = []
        path = path + [current]
        if current == target:
            return [path]
        if current not in self.nodes_cfg:
            return []
        paths = []
        for neighbor in self.nodes_cfg[current]['neighbors']:
            if neighbor not in path:
                new_paths = self._find_all_paths(neighbor, target, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def fitness_function(self, x):
        """
        基因: [模式(0/1), 路径/邻居选择, 切分比例]
        """
        mode = 0 if x[0] < 0.5 else 1 # 0:Pipeline, 1:Parallel
        
        if mode == 0: # Pipeline
            if not self.candidate_paths: return 9999
            
            # 1. 选路径
            path_idx = int(x[1] * (len(self.candidate_paths) - 0.001))
            path = self.candidate_paths[path_idx]
            
            # 2. 选切分点 (假设模型越深数据越小)
            split_layer = int(x[2] * (self.layers - 1)) + 1
            
            # 3. 模拟时延计算
            # 传输时延: 数据量 / 下一跳带宽
            next_hop = path[1]
            bw = self.profile.get(next_hop, {'bw':100})['bw']
            # 模拟数据量随层数衰减: 50MB * (0.9^layer)
            data_size = 50 * (0.9 ** split_layer)
            t_trans = data_size / bw * 1000
            
            # 计算时延
            local_cpu = self.profile.get(self.src, {'cpu':50})['cpu']
            t_comp = (split_layer * 10) / local_cpu * 1000
            
            return t_trans + t_comp + len(path)*50 # 加上跳数惩罚

        else: # Parallel
            if not self.direct_neighbors: return 9999
            
            # 1. 选邻居
            nbr_idx = int(x[1] * (len(self.direct_neighbors) - 0.001))
            target = self.direct_neighbors[nbr_idx]
            
            # 2. 卸载比例
            ratio = x[2] # 给邻居多少
            
            # 3. 模拟时延
            local_cpu = self.profile.get(self.src, {'cpu':50})['cpu']
            remote_cpu = self.profile.get(target, {'cpu':50})['cpu']
            bw = self.profile.get(target, {'bw':100})['bw']
            
            t_local = (1 - ratio) * 500 * (50/local_cpu)
            t_remote = ratio * 500 * (50/remote_cpu) + (ratio * 50 / bw * 1000)
            
            return max(t_local, t_remote)

    def run(self):
        # 运行 GA
        ga = GA(func=self.fitness_function, n_dim=3, size_pop=20, max_iter=10, 
                lb=[0,0,0], ub=[1,1,1], precision=1e-5)
        best_x, best_y = ga.run()
        return self._decode(best_x), best_y[0]

    def _decode(self, x):
        mode = "pipeline" if x[0] < 0.5 else "parallel"
        plan = {"mode": mode}
        
        if mode == "pipeline":
            path_idx = int(x[1] * (len(self.candidate_paths) - 0.001))
            plan["route"] = self.candidate_paths[path_idx]
            plan["split_point"] = int(x[2] * (self.layers - 1)) + 1
            # 去掉自己，只保留后续路由
            if plan["route"][0] == self.src:
                plan["route"] = plan["route"][1:]
        else:
            nbr_idx = int(x[1] * (len(self.direct_neighbors) - 0.001))
            plan["target"] = self.direct_neighbors[nbr_idx]
            plan["ratio"] = x[2]
            
        return plan