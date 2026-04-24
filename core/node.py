import threading
import time
import uuid
import torch
import sys
import os
import json
import csv
from datetime import datetime

# 确保能找到同级或外层的模块
sys.path.append(os.path.dirname(__file__))
from communicate import Communicator
from inference import InferenceEngine

class ComputeNode:
    """
    计算节点 - 支持流水线(PMP)与数据并行(CDP)推理
    融合了原版高稳定性的路由接力与缓冲区设计
    """

    def __init__(self, node_id, ip, port, role="compute_node", model_name='swin_base', simulate_bw=False):
        self.node_id = node_id
        self.role = role

        # 【新增】：GPU/CPU 计算互斥锁，保护引擎不被并发冲垮
        self.compute_lock = threading.Lock()

        # 针对 CDP 模式下的并行推理结果缓存
        self.task_buffer = {}  
        self.buffer_lock = threading.Lock()

        # 初始化通信与引擎
        self.comms = Communicator(node_id, ip, port, simulate_bw=simulate_bw)
        self.engine = InferenceEngine(node_id, model_name=model_name)

         # [新增] 引入 ACK 任务流控事件锁，初始状态为 True，允许发第一个包
        self.task_ack_event = threading.Event()
        self.task_ack_event.set()
        
    def load_model(self, checkpoint_path=None):
        self.engine.load_model(checkpoint_path)

    def start(self):
        """启动节点监听网络事件"""
        self.comms.start_listening(self.handle_message)

    def stop(self):
        self.comms.stop()

    def join_network(self, neighbors):
        """配置网络拓扑. neighbors: [(id, ip, port), ...]"""
        for n_id, n_ip, n_port in neighbors:
            self.comms.register_peer(n_id, n_ip, n_port)
            print(f"[{self.node_id}] 已注册邻居: {n_id} @ {n_ip}:{n_port}")

    def _get_compute_scale_ratio(self, net_cfg):
        """返回计算时延缩放系数：基准算力 / 节点当前算力。"""
        node_info = net_cfg.get("nodes", {}).get(self.node_id, {})
        device_str = str(node_info.get("device", "PC")).lower()
        baseline_tflops = 5.0 if "jetson" in device_str else 11.6

        hardware = node_info.get("hardware", {})
        node_tflops = float(hardware.get("compute_speed_tflops", hardware.get("compute_speed_gflops_per_ms", baseline_tflops)))
        if node_tflops <= 0.0:
            return 1.0
        return baseline_tflops / node_tflops

    def _debug_cuda_mem(self, tag):
        """打印当前 CUDA 显存水位，辅助定位模型切换和大包转发时的峰值。"""
        if not torch.cuda.is_available():
            return
        try:
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            max_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(
                f"[{self.node_id}] [MEM] {tag}: "
                f"alloc={alloc_mb:.1f}MB, reserved={reserved_mb:.1f}MB, peak={max_alloc_mb:.1f}MB"
            )
        except Exception as e:
            print(f"[{self.node_id}] [MEM] {tag}: 无法读取显存统计 -> {e}")

    def _debug_tensor(self, name, tensor):
        """打印张量基础信息与估算大小，辅助定位 OOM 来源。"""
        if not isinstance(tensor, torch.Tensor):
            print(f"[{self.node_id}] [TENSOR] {name}: 非Tensor(type={type(tensor)})")
            return

        try:
            size_mb = (tensor.element_size() * tensor.nelement()) / (1024 * 1024)
            print(
                f"[{self.node_id}] [TENSOR] {name}: shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, device={tensor.device}, "
                f"contiguous={tensor.is_contiguous()}, size={size_mb:.2f}MB"
            )
            if size_mb >= 128:
                print(f"[{self.node_id}] [TENSOR] ⚠️ {name} 为超大张量，序列化时可能触发内存峰值")
        except Exception as e:
            print(f"[{self.node_id}] [TENSOR] {name}: 打印失败 -> {e}")

    def _extract_bw_metrics(self, net_cfg):
        """提取星间与星地带宽均值，用于统一长表记录。"""
        raw_links = net_cfg.get("links", {})
        isl_bws = []
        gsl_bws = []

        for link_name, info in raw_links.items():
            bw = float(info.get("bandwidth_mbps", 0.0))
            if "GS" in link_name:
                gsl_bws.append(bw)
            else:
                isl_bws.append(bw)

        isl_avg = sum(isl_bws) / len(isl_bws) if isl_bws else 0.0
        gsl_avg = sum(gsl_bws) / len(gsl_bws) if gsl_bws else 0.0
        return isl_avg, gsl_avg

    def _append_standardized_physical_row(self, exp_meta, task_id, alg, acc_lat, net_cfg):
        """将实物端到端结果追加到统一长表，支持与理论结果直接 join。"""
        output_csv = exp_meta.get("standardized_csv_file", "results_long.csv")
        file_exists = os.path.isfile(output_csv)
        timestamp = datetime.now().isoformat(timespec="seconds")
        isl_avg_bw, gsl_avg_bw = self._extract_bw_metrics(net_cfg)

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "run_id",
                    "exp_type",
                    "mode",
                    "task_id",
                    "algorithm",
                    "model_name",
                    "batch_size",
                    "input_h",
                    "input_w",
                    "isl_avg_bw_mbps",
                    "gsl_avg_bw_mbps",
                    "latency_ms",
                    "norm_latency_vs_gs",
                    "timestamp",
                ])

            writer.writerow([
                exp_meta.get("run_id", "default"),
                exp_meta.get("exp_type", "algo_effectiveness"),
                "physical",
                task_id,
                alg,
                exp_meta.get("model_name", ""),
                exp_meta.get("batch_size", ""),
                exp_meta.get("input_h", ""),
                exp_meta.get("input_w", ""),
                f"{isl_avg_bw:.4f}",
                f"{gsl_avg_bw:.4f}",
                acc_lat,
                "",
                timestamp,
            ])

    # ==========================================================
    # 网络消息分发总线
    # ==========================================================
    def handle_message(self, message):
        msg_type = message.get('type', 'unknown')
        src = message.get('src', '?')

        # =================【新增：拆解出 TASK_ACK 携带的真正名字】=================
        if msg_type == 'TASK_ACK':
            payload = message.get('payload', {})
            task_id = payload.get('task_id', 'unknown')
            alg_name = payload.get('algorithm', '未知算法') 
            print(f"\n[\033[92m流水线流控\033[0m] >>> {self.node_id} 收到终端 GS 传回的 【{task_id} ({alg_name})】 应答 (ACK)！ <<<", flush=True)
            self.task_ack_event.set()
            return
        # ====================================================================

        print(f"\n[{self.node_id}] 收到消息: type={msg_type}, from={src}")

        dispatch = {
            'NEW_TASK': self._handle_new_task,                 # 外部算法/主控下发新任务
            'PipelineForward': self._handle_pipeline_forward,  # PMP中间件接力
            'ParallelTask': self._handle_parallel_task,        # CDP子任务计算
            'ParallelResult': self._handle_parallel_result     # CDP汇聚中心聚合结果
        }

        handler = dispatch.get(msg_type)
        if handler:
            handler(message)
        else:
            print(f"  [警告] 未知消息类型: {msg_type}")

    # ==========================================================
    # 任务调度器起始下发入口 (替代原基准触发方法)
    # ==========================================================
    def _handle_new_task(self, message):
        """RS 收到新任务指令包后的操作分配"""
        payload = message.get('payload', {})
        task_mode = payload.get('mode', 'PMP')
        input_data = payload.get('tensor')

        if task_mode == 'PMP':
            route_list = payload.get('route', [])
            layer_plan = payload.get('layer_plan', {})
            exp_meta = payload.get('exp_meta', {})
            # 解析出新增的控制信息
            task_id = payload.get('task_id', str(uuid.uuid4())[:8])
            alg = payload.get('algorithm', 'Unknown')
            acc_lat = payload.get('accumulated_latency', 0.0)
            print(f"起始acc_lat为{acc_lat},类型为{type(acc_lat)}")
            req_model = payload.get('model_name')  # 提取模型名
            batch = payload.get('batch')
            print(f"batch = {batch}")
            self.start_pip_task(route_list, input_data, layer_plan, task_id, alg, acc_lat, req_model, batch, exp_meta)
            
        elif task_mode == 'CDP':
            dist_map = payload.get('dist_map', {})
            aggregator = payload.get('aggregator', 'GS')
            self.start_para_task(dist_map, input_data, aggregator)

    # ==========================================================
    # 流水线推理 (PMP) 核心逻辑
    # ==========================================================
    def start_pip_task(self, route_list, input_data, layer_plan, task_id, alg, acc_lat, req_model=None, batch=1, exp_meta=None):
        PC_BASE_GFLOPS = 100.0  
        net_cfg = None
        # =================【加入读冲突重试机制】=================
        for _ in range(10):  # 尝试10次
            try:
                with open("config/network_config.json", 'r') as f:
                    net_cfg = json.load(f)
                break  # 读取成功，跳出循环
            except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                # 遇到由于并发导致的 读写冲突 / 文件占用 / 空文件 时，等待并重试
                time.sleep(0.02)
        
        if net_cfg is None:
            # 万一非常极端的情况10次都失败了，用默认值或者放弃本次操作
            print(f"[{self.node_id}] 读取配置遇到严重阻塞，跳过解析")
            return
        # ========================================================

        # 【关键保护】：加锁！确保这个任务处理完前，别人不能换模型
        with self.compute_lock:
            # 1. 动态切换模型逻辑
            if req_model and req_model != self.engine.model_name:
                print(f"[{self.node_id}] 🔄 [加锁] 触发动态模型切换: {self.engine.model_name} -> {req_model}")
                self._debug_cuda_mem("switch-before-empty-cache")
                self.engine.model_name = req_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._debug_cuda_mem("switch-after-empty-cache")
                self.engine.load_model()
                self._debug_cuda_mem("switch-after-load-model")

            if not route_list:
                print(f"[{self.node_id}] [PIP] 路由为空, 本地全量推理!")
                output, ms = self.engine.run_full(input_data)
                return

            next_hop = route_list[0]
            remain = route_list[1:]

            # 获取源节点自身该跑的层数
            my_start, my_end = layer_plan.get(self.node_id, (-1, -1))
            my_gflops = net_cfg["nodes"][self.node_id]["hardware"].get("compute_speed_gflops_per_ms", 10.0)
            
            if my_start >= 0 and my_end >= 0:
                print(f"[{self.node_id}] [PIP] 任务 {task_id}({alg}) 首发，执行层 [{my_start} -> {my_end}]")
                output, ms = self.engine.exec_layers(input_data, my_start, my_end)
                
                # --- 部署到物理设备，使用真实的计算时延 (直接使用 ms) ---
                scale_ratio = self._get_compute_scale_ratio(net_cfg)
                sim_comp_ms = ms * scale_ratio
                acc_lat += sim_comp_ms
                history = [(self.node_id, my_start, my_end, sim_comp_ms)]
                print(f"  -> 真实计算耗时: {ms:.2f}ms (缩放后: {sim_comp_ms:.2f}ms, 系数={scale_ratio:.3f})")
            else:
                print(f"[{self.node_id}] [PIP] 首发节点作为纯数据源，直接派发数据...")
                output, ms = input_data, 0.0
                history = []
                my_end = -1

            # --- 模拟传输时延换算 ---
            sim_tx_ms = 0.0
            if next_hop:
                self._debug_tensor("start_pip_task.output-before-send", output)
                self._debug_cuda_mem("start_pip_task.before-send")
                tensor_mb = (output.element_size() * output.nelement()) / (1024 * 1024)
                print(f"  -> 产出: {tensor_mb:.4f}MB | 正通过物理层喷射给 {next_hop}，将由下一跳实测传输耗时...")

            # 读取下一跳需要的层数参数送出去
            next_start, next_end = layer_plan.get(next_hop, (-1, -1))

            
            # 打包送走
            payload = {
                'task_id': task_id,
                'algorithm': alg,                  # 透传算法名
                'model_name': req_model,           # 【透传模型名】，让下一跳知道是什么模型！
                'accumulated_latency': acc_lat,    # 透传累计时延 (首发节点这里就是 0.0)
                'tensor': output,
                'start_layer': next_start,
                'end_layer': next_end,
                'route_remain': remain,
                'layer_plan': layer_plan,
                'split_history': history,
                'batch':batch,
                'exp_meta': exp_meta or {},
            }
            print(f"accumulated latency: {acc_lat}")
            self.comms.send_message(next_hop, 'PipelineForward', payload)

    def _handle_pipeline_forward(self, message):
        """接收流水线推理并接力"""
        PC_BASE_GFLOPS = 100.0
        import json
        net_cfg = None
        # =================【加入读冲突重试机制】=================
        for _ in range(10):  # 尝试10次
            try:
                with open("config/network_config.json", 'r') as f:
                    net_cfg = json.load(f)
                break  # 读取成功，跳出循环
            except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                # 遇到由于并发导致的 读写冲突 / 文件占用 / 空文件 时，等待并重试
                time.sleep(0.02)
        
        if net_cfg is None:
            # 万一非常极端的情况10次都失败了，用默认值或者放弃本次操作
            print(f"[{self.node_id}] 读取配置遇到严重阻塞，跳过解析")
            return
        # ========================================================

        p = message.get('payload', message)
        tensor = p.get('tensor')
        start = p.get('start_layer', 0)
        end = p.get('end_layer', 0)
        route = p.get('route_remain', [])
        layer_plan = p.get('layer_plan', {})
        history = p.get('split_history', [])
        exp_meta = p.get('exp_meta', {})
        
        task_id = p.get('task_id', '?')
        alg = p.get('algorithm', 'Unknown')
        acc_lat = p.get('accumulated_latency', 0.0)
        batch = p.get('batch')
        print(f"batch = {batch}")

        # =================【核心修正】：累积上游物理线路的实测传输经历时间！=================
        measured_comm_ms = message.get('measured_comm_latency_ms', 0.0)
        print(f"measured_comm_ms={measured_comm_ms}")
        if measured_comm_ms > 0.0:
            acc_lat += measured_comm_ms
            print(f"  [+] UDP底层截获物理传输耗时，并已累加等效时延: {measured_comm_ms:.2f} ms")

        # =================【新增：动态切换模型逻辑】=================
        req_model = p.get('model_name')
        # 【关键保护】：加锁执行！
        with self.compute_lock:
            # 2. 动态切换模型逻辑
            if req_model and req_model != self.engine.model_name:
                print(f"[{self.node_id}]  [加锁] 触发动态模型切换: {self.engine.model_name} -> {req_model}")
                self._debug_cuda_mem("forward-switch-before-empty-cache")
                self.engine.model_name = req_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._debug_cuda_mem("forward-switch-after-empty-cache")
                self.engine.load_model()
                self._debug_cuda_mem("forward-switch-after-load-model")
        # =========================================================
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(self.engine.device)
                self._debug_tensor("pipeline_forward.input-after-to-device", tensor)
                self._debug_cuda_mem("pipeline_forward.after-input-to-device")

            # ============== 【新增核心逻辑：纯透传保护】 ==============
            # 如果收到的起始层 > 结束层，或者层数为负数，说明本节点不承担计算任务
            if start > end or start < 0 or end < 0:
                print(f"\n[{self.node_id}] [PIP] 当前节点未分配计算任务，启动 [纯透传模式]...')")
                output, ms = tensor, 0.0   # 直接把原图/原特征作为输出，耗时 0
            else:
                print(f"\n[{self.node_id}] [PIP] 收到任务 {task_id}({alg})，执行层 [{start} -> {end}]")
                output, ms = self.engine.exec_layers(tensor, start, end)

            my_gflops = net_cfg["nodes"][self.node_id]["hardware"].get("compute_speed_gflops_per_ms", 10.0)

            # =========================================================
            
            # --- 部署到物理设备，使用真实的计算时延 (直接使用 ms) ---
            scale_ratio = self._get_compute_scale_ratio(net_cfg)
            sim_comp_ms = ms * scale_ratio
            acc_lat += float(sim_comp_ms)
            history.append((self.node_id, start, end, sim_comp_ms))
            print(f"  -> 真实计算耗时: {ms:.2f}ms (缩放后: {sim_comp_ms:.2f}ms, 系数={scale_ratio:.3f})")

            if route: # 如果还有下一跳
                next_hop = route[0]
                remain = route[1:]
                
                # --- 移除虚假的理论通信时延累加 ---
                self._debug_tensor("pipeline_forward.output-before-send", output)
                self._debug_cuda_mem("pipeline_forward.before-send")
                tensor_mb = (output.element_size() * output.nelement()) / (1024 * 1024)
                print(f"  -> 产出: {tensor_mb:.4f}MB | 正通过物理层喷射给 {next_hop}，将由下一跳实测传输耗时...")
                
                # 定位下一跳要执行的起止层
                illegitimate_range = [-1, -2]
                plan_for_next = layer_plan.get(next_hop, illegitimate_range) if layer_plan else illegitimate_range
                next_start, next_end = plan_for_next[0], plan_for_next[1]
                
                payload = {
                    'task_id': task_id,
                    'algorithm': alg,
                    'model_name': req_model, 
                    'accumulated_latency': acc_lat,
                    'tensor': output,
                    'start_layer': next_start,
                    'end_layer': next_end,
                    'route_remain': remain,
                    'layer_plan': layer_plan,
                    'split_history': history,
                    'batch':batch,
                    'exp_meta': exp_meta,
                }
                self.comms.send_message(next_hop, 'PipelineForward', payload)
            else:
                # avg_lat = acc_lat/batch
                # 最终节点 (GS)
                print(f"\n{'='*50}")
                print(f"   [PIP] 流水线端到端到达终点! Task={task_id} ({alg})")
                print(f"   总计模拟时延 (计算+传输): {acc_lat:.2f} ms")
                print(f"   批次: {batch} 张图")
                print(f"   总计平均模拟时延 (计算+传输): {acc_lat:.2f} ms")
                print(f"{'='*50}\n")
                
                # =================【新增：向首发节点 RS 发送 ACK 】=================
                print(f"[{self.node_id}] 正在向 RS 节点回传 TASK_ACK 确认报文...")
                # 注意：你源码 communicator.py 发送形式是 send_message(target_id, msg_type, payload)
                self.comms.send_message("RS", "TASK_ACK", {
                    'task_id': task_id,
                    'algorithm': alg
                })

                # 将结果追加写入 CSV，直接拿去画图表！
                with open("experiment_results.csv", "a", encoding="utf-8") as f:
                    f.write(f"{task_id},{alg},{acc_lat:.2f}\n")

                # 同步写入标准化长表 (physical)，以便与 theory 直接按任务键对齐。
                self._append_standardized_physical_row(
                    exp_meta=exp_meta,
                    task_id=task_id,
                    alg=alg,
                    acc_lat=acc_lat,
                    net_cfg=net_cfg,
                )

    # ==========================================================
    # 数据并行推理 (CDP) 核心逻辑
    # ==========================================================
    def start_para_task(self, dist_map, input_data, aggregator_id):
        """发射数据并行推理 (由RS端切图派发)"""
        task_id = str(uuid.uuid4())[:8]
        total_parts = len(dist_map)

        for worker_id, (start_idx, end_idx) in dist_map.items():
            # 这里按照 batch_size 维度切片 (例如 [4,3,224,224] 划给两个节点处理)
            data_slice = input_data[start_idx:end_idx]
            payload = {
                'task_id': task_id,
                'tensor': data_slice,
                'total_parts': total_parts,
                'aggregator': aggregator_id,
                'part_range': (start_idx, end_idx),
            }
            print(f"[{self.node_id}] [CDP] 发送数据并行切片 [{start_idx}:{end_idx}] -> {worker_id}")
            self.comms.send_message(worker_id, 'ParallelTask', payload)

    def _handle_parallel_task(self, message):
        """计算节点：接收切片并执行全量推理"""
        p = message.get('payload', message)
        tensor = p['tensor']
        task_id = p['task_id']
        aggregator = p['aggregator']

        in_shape = list(tensor.shape) if hasattr(tensor, 'shape') else str(type(tensor))
        print(f"[{self.node_id}] [CDP] 执行数据切片推理, 输入形状={in_shape}")
        
        output, ms = self.engine.run_full(tensor)
        print(f"[{self.node_id}] [CDP] 切片完成: {ms:.2f}ms")

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
        """聚合节点 (通常是GS)：聚合并行结果"""
        p = message.get('payload', message)
        task_id = p['task_id']
        total = p['total_parts']

        with self.buffer_lock:
            if task_id not in self.task_buffer:
                self.task_buffer[task_id] = []
            self.task_buffer[task_id].append(p)

            current = len(self.task_buffer[task_id])
            print(f"[{self.node_id}] [CDP] 汇总结果 {current}/{total} 来源于 {p['worker']}")

            if current >= total:
                results = self.task_buffer.pop(task_id) # 取出并删除缓存
                
                # 按照 part_range 排序以便用 torch.cat 合并
                results.sort(key=lambda x: x['part_range'][0])
                total_ms = max(r['exec_ms'] for r in results) # 理想并行情况总耗时近似最慢节点耗时
                
                # 如果返回的张量有效，进行真实的数据物理拼接
                try:
                    final_tensor = torch.cat([r['tensor'] for r in results], dim=0)
                    final_shape = list(final_tensor.shape)
                except Exception as e:
                    final_shape = f"Data concat failed: {e}"

                print(f"\n{'='*50}")
                print(f"  [CDP]  数据并行推理合并完成! Task={task_id}")
                print(f"  全网最大单点计算计算时延: {total_ms:.2f}ms")
                for r in results:
                    print(f"    来源 {r['worker']}: 耗时 {r['exec_ms']:.2f}ms, 处理了数据段 {r['part_range']}")
                print(f"  回传拼接特征形状: {final_shape}")
                print(f"{'='*50}\n")