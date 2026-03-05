import time
import uuid
import torch
import threading

from core.communicate import Communicator
from core.inference import InferenceEngine

class SatelliteNode:
    """
    卫星/地面站节点 - 支持真实跨设备推理
    """

    def __init__(self, node_id, ip, port, role,
                 device_profiles=None, simulate_bw=False,
                 model_name='alexnet', checkpoint=None):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.role = role
        self.device_profiles = device_profiles or {}
        self.task_buffer = {}  # 并行推理结果缓存

        # 通信模块
        self.comms = Communicator(node_id, ip, port, simulate_bw=simulate_bw)

        # 推理引擎
        self.engine = InferenceEngine(node_id, model_name)
        ckpt = checkpoint or self._find_checkpoint(model_name)
        self.engine.load_model(ckpt)

    def _find_checkpoint(self, model_name):
        """自动搜索 checkpoints/ 目录"""
        import os, glob
        patterns = [
            f'checkpoints/{model_name}*.pth',
            f'trained_models/{model_name}*.pth',
        ]
        for pat in patterns:
            files = glob.glob(pat)
            if files:
                return files[0]
        return None

    # ==================== 组网 ====================
    def join_network(self, neighbors):
        """neighbors: [(id, ip, port), ...]"""
        for n_id, n_ip, n_port in neighbors:
            self.comms.register_peer(n_id, n_ip, n_port)
            print(f"  已注册邻居: {n_id} @ {n_ip}:{n_port}")

    def start(self):
        """启动节点监听"""
        self.comms.start_listening(self._on_message)

    def stop(self):
        self.comms.stop()

    # ==================== 消息分发 ====================
    def _on_message(self, message):
        msg_type = message.get('type', 'unknown')
        src = message.get('src', '?')
        print(f"\n[{self.node_id}] 收到消息: type={msg_type}, from={src}")

        dispatch = {
            'PipelineForward': self._handle_pipeline_forward,
            'ParallelTask': self._handle_parallel_task,
            'ParallelResult': self._handle_parallel_result,
            'Para_GS': self._handle_para_gs,
            # 兼容旧消息类型
            'pip_forward': self._handle_pipeline_forward,
            'para_task': self._handle_parallel_task,
            'para_result': self._handle_parallel_result,
        }

        handler = dispatch.get(msg_type)
        if handler:
            handler(message)
        else:
            print(f"  [警告] 未知消息类型: {msg_type}")

    # ==================== 流水线推理 ====================
    def start_pip_task(self, route_list, input_data, split_point=10):
        """发起流水线推理 (RS节点调用)"""
        if not route_list:
            print("  路由为空, 本地全量推理")
            output, ms = self.engine.run_full(input_data)
            print(f"  本地推理完成: {ms:.2f}ms")
            return

        task_id = str(uuid.uuid4())[:8]
        next_hop = route_list[0]
        remain = route_list[1:]

        # 本地执行前半段
        print(f"  [PIP] 本地执行层 [0, {split_point-1}]")
        output, ms, details = self.engine.run_layers(input_data, 0, split_point - 1)
        print(f"  [PIP] 本地完成: {ms:.2f}ms, output={list(output.shape)}")

        # 发送给下一跳
        payload = {
            'task_id': task_id,
            'tensor': output,
            'start_layer': split_point,
            'end_layer': self.engine.num_layers - 1,
            'route_remain': remain,
            'split_history': [(self.node_id, 0, split_point - 1, ms)],
        }
        self.comms.send_message(next_hop, 'PipelineForward', payload)

    def _handle_pipeline_forward(self, message):
        """接收流水线推理的中间结果"""
        p = message.get('payload', message)
        tensor = p['tensor']
        start = p['start_layer']
        end = p['end_layer']
        route = p.get('route_remain', [])
        history = p.get('split_history', [])
        task_id = p.get('task_id', '?')

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to('cpu')

        # 计算本地执行的层范围
        if route:
            # 还有下一跳, 只执行一半
            mid = (start + end) // 2
            my_end = mid
        else:
            my_end = end

        print(f"  [PIP] 执行层 [{start}, {my_end}]")
        output, ms, details = self.engine.run_layers(tensor, start, my_end)
        print(f"  [PIP] 完成: {ms:.2f}ms, output={list(output.shape)}")

        history.append((self.node_id, start, my_end, ms))

        if route:
            next_hop = route[0]
            remain = route[1:]
            payload = {
                'task_id': task_id,
                'tensor': output,
                'start_layer': my_end + 1,
                'end_layer': end,
                'route_remain': remain,
                'split_history': history,
            }
            self.comms.send_message(next_hop, 'PipelineForward', payload)
        else:
            # 最终节点
            total_ms = sum(h[3] for h in history)
            print(f"\n{'='*50}")
            print(f"  [PIP] 流水线推理完成! Task={task_id}")
            print(f"  总计算时延: {total_ms:.2f}ms")
            for h in history:
                print(f"    {h[0]}: 层[{h[1]},{h[2]}] = {h[3]:.2f}ms")
            print(f"  输出shape: {list(output.shape)}")
            print(f"{'='*50}")

    # ==================== 并行推理 ====================
    def start_para_task(self, dist_map, input_data, aggregator_id):
        """发起数据并行推理 (RS节点调用)"""
        task_id = str(uuid.uuid4())[:8]
        total_parts = len(dist_map)

        for worker_id, (start_idx, end_idx) in dist_map.items():
            data_slice = input_data[start_idx:end_idx]
            payload = {
                'task_id': task_id,
                'tensor': data_slice,
                'total_parts': total_parts,
                'aggregator': aggregator_id,
                'part_range': (start_idx, end_idx),
            }
            print(f"  [PARA] 发送数据[{start_idx}:{end_idx}] -> {worker_id}")
            self.comms.send_message(worker_id, 'ParallelTask', payload)

    def _handle_parallel_task(self, message):
        """接收并执行数据并行推理"""
        p = message.get('payload', message)
        tensor = p['tensor']
        task_id = p['task_id']
        aggregator = p['aggregator']

        print(f"  [PARA] 执行推理, 输入shape={list(tensor.shape)}")
        output, ms = self.engine.run_full(tensor)
        print(f"  [PARA] 完成: {ms:.2f}ms")

        result_payload = {
            'task_id': task_id,
            'tensor': output,
            'total_parts': p['total_parts'],
            'exec_ms': ms,
            'worker': self.node_id,
            'part_range': p.get('part_range', (0, 0)),
        }
        self.comms.send_message(aggregator, 'ParallelResult', result_payload)

    def _handle_parallel_result(self, message):
        """聚合并行推理结果 (GS节点)"""
        p = message.get('payload', message)
        task_id = p['task_id']
        total = p['total_parts']

        if task_id not in self.task_buffer:
            self.task_buffer[task_id] = []
        self.task_buffer[task_id].append(p)

        current = len(self.task_buffer[task_id])
        print(f"  [AGG] 收到结果 {current}/{total} from {p['worker']}")

        if current >= total:
            results = self.task_buffer.pop(task_id)
            total_ms = max(r['exec_ms'] for r in results)
            print(f"\n{'='*50}")
            print(f"  [PARA] 并行推理聚合完成! Task={task_id}")
            print(f"  最大推理时延: {total_ms:.2f}ms")
            for r in results:
                print(f"    {r['worker']}: {r['exec_ms']:.2f}ms, range={r['part_range']}")
            print(f"{'='*50}")

    def _handle_para_gs(self, message):
        """兼容旧版 Para_GS"""
        p = message.get('payload', {})
        print(f"\n{'='*30}")
        print(f"  最终结果: {p.get('data', '?')}")
        print(f"{'='*30}")

    # ==================== 新增: 算法调度接口 ====================
    def run_baseline(self, images):
        """基准: 本地全量推理"""
        output, ms = self.engine.run_full(images)
        print(f"[Baseline] 推理完成: {ms:.2f}ms, output={list(output.shape)}")
        return output, ms

    def run_selector(self, images, net_config, device_profiles):
        """实验一: 多因子推理模式选择"""
        from algorithm.selector import InferenceSelector
        selector = InferenceSelector(self, device_profiles)

        task = {
            'model_mem_mb': 240,
            'input_mb': images.nelement() * images.element_size() / (1024 * 1024),
            'total_flops': 714e6,
            'output_mb': 0.04,
            'num_layers': self.engine.num_layers,
            'type': 'compute_intensive',
        }

        nodes = list(net_config['nodes'].keys())
        mode, selected = selector.select_mode(task, nodes)
        print(f"[Selector] 决策结果: {mode}, 使用节点: {selected}")

        if mode == 'PMP':
            route = [n for n in selected if n != self.node_id]
            self.start_pip_task(route, images, split_point=self.engine.num_layers // 2)
        else:
            # CDP
            workers = [n for n in selected if n != self.node_id and n != 'GS']
            if not workers:
                self.run_baseline(images)
                return
            batch = images.shape[0]
            per = max(1, batch // len(workers))
            dist_map = {}
            for i, w in enumerate(workers):
                s = i * per
                e = min((i + 1) * per, batch)
                dist_map[w] = (s, e)
            self.start_para_task(dist_map, images, 'GS')

    def run_dp_schedule(self, images, net_config, device_profiles):
        """实验二: DP流水线优化"""
        from algorithm.dp_scheduler import DPScheduler

        # 生成 profile (或读取缓存)
        profile_data = []
        sample = images[:1]
        _, _, layer_details = self.engine.run_layers(sample, 0, self.engine.num_layers - 1)
        for d in layer_details:
            profile_data.append({
                'flops': d['time_ms'] * 1e6,  # 用时间近似 (真实应用 thop)
                'output_mb': d['output_mb'],
                'mem_mb': d['output_mb'] * 2,
                'input_mb': d['output_mb'],
            })

        # 参与节点
        nodes = [nid for nid in net_config['nodes']
                 if net_config['nodes'][nid]['role'] in ['remote_sensing', 'leo_computing']]

        scheduler = DPScheduler(profile_data, nodes, device_profiles, src_id=self.node_id)
        plan, total_cost = scheduler.run()

        print(f"\n[DP] 最优划分方案 (预估时延: {total_cost:.4f}s):")
        for step in plan:
            print(f"  层{step['layer']} -> 节点{step['node']} (cost={step['cost']:.4f}s)")

        # 执行
        if plan:
            # 找出分配给自己的层
            my_layers = [s for s in plan if s['node'] == self.node_id]
            other_layers = [s for s in plan if s['node'] != self.node_id]

            if my_layers:
                start_l = my_layers[0]['layer']
                end_l = my_layers[-1]['layer']
                output, ms, _ = self.engine.run_layers(images, start_l, end_l)
                print(f"  [DP] 本地执行层[{start_l},{end_l}]: {ms:.2f}ms")

                if other_layers:
                    next_node = other_layers[0]['node']
                    next_start = other_layers[0]['layer']
                    next_end = other_layers[-1]['layer']
                    route = list(dict.fromkeys([s['node'] for s in other_layers]))
                    payload = {
                        'task_id': str(uuid.uuid4())[:8],
                        'tensor': output,
                        'start_layer': next_start,
                        'end_layer': next_end,
                        'route_remain': route[1:] if len(route) > 1 else [],
                        'split_history': [(self.node_id, start_l, end_l, ms)],
                    }
                    self.comms.send_message(next_node, 'PipelineForward', payload)

    def run_lawa_schedule(self, images, net_config, device_profiles):
        """实验三: LAWA链路感知加权并行"""
        from algorithm.lawa_scheduler import LAWAScheduler

        workers = [nid for nid in net_config['nodes']
                   if net_config['nodes'][nid]['role'] == 'leo_computing']

        task_info = {
            'input_mb': images.nelement() * images.element_size() / (1024 * 1024),
            'total_flops': 714e6,
        }

        scheduler = LAWAScheduler(task_info, workers, device_profiles, src_id=self.node_id)
        plan = scheduler.get_allocation_plan()

        print(f"\n[LAWA] 加权分配方案:")
        for p in plan:
            print(f"  {p['node_id']}: ratio={p['data_ratio']:.3f}, "
                  f"est_latency={p['expected_latency_sec']:.4f}s")

        # 执行: 按比例分配batch
        batch_size = images.shape[0]
        if batch_size < len(workers):
            images = images.repeat(len(workers), 1, 1, 1)
            batch_size = images.shape[0]

        dist_map = {}
        offset = 0
        for i, p in enumerate(plan):
            count = max(1, int(batch_size * p['data_ratio']))
            end = min(offset + count, batch_size)
            dist_map[p['node_id']] = (offset, end)
            offset = end

        aggregator = 'GS'
        self.start_para_task(dist_map, images, aggregator)

    def run_auto_schedule(self, images, net_config):
        """GA遗传算法调度 (兼容旧版)"""
        from algorithm.ga_scheduler import GAScheduler
        try:
            ga = GAScheduler(net_config, self.engine.num_layers)
            best = ga.run()
            print(f"[GA] 最优方案: {best}")
        except Exception as e:
            print(f"[GA] 调度失败: {e}")
            self.run_baseline(images)