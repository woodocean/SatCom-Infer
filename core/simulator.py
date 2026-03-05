import pandas as pd

class SatelliteSimulator:
    def __init__(self, profile_csv='profile_database.csv'):
        self.lut = pd.read_csv(profile_csv)
        # 建立快速索引 (Model -> Size -> Layer -> Data)
        self.index = self.lut.set_index(['model', 'input_size', 'layer_idx']).to_dict('index')
        
    def estimate_latency(self, model_name, input_size, layer_idx, node_specs):
        """
        高标准仿真：基于查表 + 物理公式
        node_specs: {'flops_capacity': 1500 (GFLOPS), 'bandwidth_up': 100 (Mbps), ...}
        """
        # 1. 查找最接近的 Profile (Nearest Neighbor)
        # 真实工程中不可能穷举所有尺寸，需要插值或取最近
        avail_sizes = sorted(self.lut[self.lut['model'] == model_name]['input_size'].unique())
        target_size = min(avail_sizes, key=lambda x: abs(x - input_size))
        
        # 缩放因子 (假设复杂度是 O(N^2) 用于图像面积)
        scale_factor = (input_size / target_size) ** 2
        
        try:
            # 查表
            row = self.index[(model_name, target_size, layer_idx)]
            base_flops = row['flops_g']
            base_comm = row['comm_mb']
            
            # 2. 物理公式计算
            # 计算时延 = (基准FLOPs * 缩放) / (节点算力 * 效率因子)
            # 效率因子(alpha)体现了算力跑不满的情况
            t_comp = (base_flops * scale_factor) / (node_specs['flops_capacity'] * 0.8) 
            
            # 传输时延 = (基准MB * 缩放) / 带宽
            # 注意：这里只算发出去的时间
            t_trans = (base_comm * scale_factor * 8) / node_specs['bandwidth_up']
            
            return {
                't_comp': t_comp,       # 秒
                't_trans': t_trans,     # 秒
                'mem_req': row['mem_mb'] * scale_factor # MB
            }
            
        except KeyError:
            print(f"Error: 查表失败 {model_name} @ {layer_idx}")
            return {'t_comp': 0, 't_trans': 0, 'mem_req': 0}