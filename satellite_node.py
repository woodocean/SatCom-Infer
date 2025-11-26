import socket
import torch
import pickle
import time
import threading
from enum import Enum
from utils.inference_utils import get_dnn_model, model_partition
from predictor import predictor_utils


class SatelliteType(Enum):
    REMOTE_SENSING = "remote_sensing"  # 遥感卫星
    LEO_COMPUTING = "leo_computing"  # 计算卫星


class SatelliteNode:
    def __init__(self, node_id: str, satellite_type: SatelliteType,
                 ip: str, port: int, compute_capacity: float, device: str = "cpu"):
        self.node_id = node_id
        self.satellite_type = satellite_type
        self.ip = ip
        self.port = port
        self.compute_capacity = compute_capacity
        self.device = device

        # 网络信息
        self.neighbor_nodes = {}  # 存储发现的邻居

        # 添加模型和预测器
        self.current_model = None
        self.assigned_layers = None
        self.predictor_dict = {}  # 时延预测器字典

        # 任务状态
        self.current_task = None
        self.is_busy = False

        self.model_type = None
        self.input_data = None
        self.max_latency = None

        # 添加地面站信息
        self.ground_stations = {}

        print(f"创建卫星节点: {self.node_id} ({self.satellite_type.value})")

    def add_ground_station(self, station_id: str, station_info: dict):
        """添加地面站信息"""
        self.ground_stations[station_id] = station_info
        print(f"{self.node_id} 添加地面站: {station_id}")

    def get_info(self):
        """获取卫星完整信息"""
        return {
            'node_id': self.node_id,
            'type': self.satellite_type.value,
            'ip': self.ip,
            'port': self.port,
            'compute_capacity': self.compute_capacity,
            'device': self.device
        }

    def start_discovery_service(self):
        """启动发现服务"""

        def discovery_handler():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.ip, self.port))
                server_socket.listen(5)

                print(f"{self.node_id} 发现服务启动在 {self.ip}:{self.port}")

                while True:
                    conn, addr = server_socket.accept()
                    threading.Thread(
                        target=self._handle_neighbor_connection,
                        args=(conn, addr)
                    ).start()

            except Exception as e:
                print(f"发现服务错误: {e}")

        thread = threading.Thread(target=discovery_handler, daemon=True)
        thread.start()
        return thread

    def _handle_neighbor_connection(self, conn, addr):
        """处理所有连接请求（邻居发现 + 任务提交）"""
        try:
            # 接收完整数据
            data = b""
            conn.settimeout(5.0)  # 设置接收超时

            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                # 尝试解析数据，如果成功则停止接收
                try:
                    message = pickle.loads(data)
                    break  # 成功解析，停止接收
                except:
                    continue  # 数据不完整，继续接收

            if not data:
                return

            try:
                message = pickle.loads(data)

                # 判断消息类型：如果是任务消息
                if isinstance(message, dict) and 'task_id' in message:
                    print(f"{self.node_id} 接收到任务: {message['task_id']}")

                    if self.satellite_type == SatelliteType.REMOTE_SENSING and not self.is_busy:
                        result = self.assign_task(message)
                    else:
                        result = {"status": "rejected", "message": "节点繁忙或非协调节点"}

                    # 发送响应
                    response_data = pickle.dumps(result)
                    conn.sendall(response_data)

                # 判断消息类型：如果是邻居发现消息
                elif isinstance(message, dict) and 'node_id' in message and 'type' in message:
                    print(f"{self.node_id} 发现邻居: {message['node_id']}")

                    self.neighbor_nodes[message['node_id']] = {
                        'node_id': message['node_id'],
                        'type': message['type'],
                        'ip': message['ip'],
                        'port': message['port'],
                        'compute_capacity': message['compute_capacity'],
                        'device': message['device']
                    }

                    ack = self.get_info()
                    ack['status'] = 'ack'
                    conn.sendall(pickle.dumps(ack))

                else:
                    conn.sendall(pickle.dumps({"status": "error", "message": "未知消息格式"}))

            except Exception as e:
                print(f"解析消息错误: {e}")
                conn.sendall(pickle.dumps({"status": "error", "message": f"解析错误: {str(e)}"}))

        except socket.timeout:
            print("接收数据超时")
        except Exception as e:
            print(f"处理连接消息错误: {e}")
        finally:
            conn.close()

    def discover_neighbor(self, neighbor_ip: str, neighbor_port: int):
        """主动发现一个邻居"""
        try:
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.connect((neighbor_ip, neighbor_port))

            # 发送本节点完整信息
            hello_msg = self.get_info()
            conn.send(pickle.dumps(hello_msg))

            # 等待回复（现在包含对方完整信息）
            response_data = conn.recv(1024)
            response = pickle.loads(response_data)

            if response['status'] == 'ack':
                print(f"{self.node_id} 成功连接到 {response['node_id']}")

                # 修复：保存完整的邻居信息（从回复中获取）
                self.neighbor_nodes[response['node_id']] = {
                    'node_id': response['node_id'],
                    'type': response['type'],
                    'ip': response['ip'],
                    'port': response['port'],
                    'compute_capacity': response['compute_capacity'],
                    'device': response['device']
                }

                return True
            else:
                print(f"{self.node_id} 连接被拒绝")
                return False

        except Exception as e:
            print(f"{self.node_id} 连接失败: {e}")
            return False
        finally:
            conn.close()

    def get_network_status(self):
        """获取网络状态信息"""
        return {
            'node_id': self.node_id,
            'neighbors_count': len(self.neighbor_nodes),
            'neighbors': list(self.neighbor_nodes.keys()),
            'compute_capacity': self.compute_capacity,
            'device': self.device
        }

    def print_network_info(self):
        """打印网络拓扑信息"""
        print(f"\n{self.node_id} 的网络拓扑:")
        print(f"本节点: {self.satellite_type.value}, 算力: {self.compute_capacity}, 设备: {self.device}")
        print(f"邻居节点 ({len(self.neighbor_nodes)} 个):")

        for neighbor_id, info in self.neighbor_nodes.items():
            print(f"    {neighbor_id}: {info['type']}, 算力: {info['compute_capacity']}, 设备: {info['device']}")

    def assign_task(self, task_spec):
        """接收并分配推理任务"""
        if self.is_busy:
            return {"status": "busy", "message": "节点正忙"}

        print(f"{self.node_id} 接收到任务: {task_spec['task_id']}")
        self.is_busy = True
        self.current_task = task_spec

        # 存储任务信息
        self.model_type = task_spec['model_type']
        self.input_data = task_spec['input_data']
        self.max_latency = task_spec['max_latency']

        # 加载模型
        self.current_model = get_dnn_model(self.model_type)

        # 如果是遥感卫星，启动协调流程
        if self.satellite_type == SatelliteType.REMOTE_SENSING:
            result = self.coordinate_inference()
        else:
            # 其他卫星等待分配
            result = {"status": "waiting_for_assignment"}

        self.is_busy = False
        return result

    def coordinate_inference(self):
        """协调推理过程（遥感卫星专用）"""
        print(f"{self.node_id} 开始协调推理...")

        # 1. 收集网络拓扑信息
        network_info = self.collect_network_info()

        # 2. 使用智能分割算法（而不是简单平均分配）
        partition_plan = self.calculate_optimal_partition(network_info)

        # 3. 分发任务到各卫星
        execution_result = self.distribute_and_execute(partition_plan)

        return execution_result

    def collect_network_info(self):
        """收集网络拓扑和节点信息（包含地面站）"""
        print(f"{self.node_id} 收集网络信息...")

        network_info = {
            'nodes': {},
            'links': {},
            'ground_stations': self.ground_stations.copy(),
            'timestamp': time.time()
        }

        # 添加本节点信息
        network_info['nodes'][self.node_id] = {
            'node_id': self.node_id,
            'type': self.satellite_type.value,
            'compute_capacity': self.compute_capacity,
            'device': self.device,
            'ip': self.ip,
            'port': self.port
        }

        # 添加邻居节点信息
        for neighbor_id, neighbor_info in self.neighbor_nodes.items():
            network_info['nodes'][neighbor_id] = {
                'node_id': neighbor_id,
                'type': neighbor_info['type'],
                'compute_capacity': neighbor_info['compute_capacity'],
                'device': neighbor_info['device'],
                'ip': neighbor_info['ip'],
                'port': neighbor_info['port']
            }

            # 添加链路信息（简化估计）
            link_key = f"{self.node_id}->{neighbor_id}"
            network_info['links'][link_key] = {
                'bandwidth': self.estimate_link_bandwidth(neighbor_info),
                'latency': self.estimate_link_latency(neighbor_info)
            }

        # 添加地面站到节点的链路
        for station_id, station_info in self.ground_stations.items():
            link_key = f"{self.node_id}->{station_id}"
            network_info['links'][link_key] = {
                'bandwidth': station_info.get('bandwidth', 200.0),  # 地面站带宽更高
                'latency': station_info.get('latency', 50.0)  # 卫星到地面延迟
            }

        print(
            f"收集到 {len(network_info['nodes'])} 个节点, {len(network_info['links'])} 条链路, {len(network_info['ground_stations'])} 个地面站")
        return network_info

    def estimate_link_bandwidth(self, neighbor_info):
        """估计链路带宽（基于卫星类型和距离）"""
        # 简化估计：遥感卫星到计算卫星的带宽
        if self.satellite_type == SatelliteType.REMOTE_SENSING:
            if neighbor_info['type'] == 'leo_computing':
                return 100.0  # Mbps
            else:
                return 50.0  # Mbps
        else:
            return 80.0  # Mbps

    def estimate_link_latency(self, neighbor_info):
        """估计链路延迟（基于卫星类型）"""
        # 简化估计
        if self.satellite_type == SatelliteType.REMOTE_SENSING:
            return 10.0  # ms
        else:
            return 5.0  # ms

    def calculate_optimal_partition(self, network_info):
        """简单的分割方案计算（临时替代）"""
        print(f"{self.node_id} 计算简单分割方案...")

        # 获取可用节点
        available_nodes = list(network_info['nodes'].keys())
        print(f"可用节点: {available_nodes}")

        # 简单的分割策略：按节点算力分配层数
        total_layers = len(self.current_model)
        print(f"模型总层数: {total_layers}")

        # 计算总算力
        total_compute = sum(network_info['nodes'][node]['compute_capacity'] for node in available_nodes)

        partition_plan = []
        current_layer = 0

        for i, node_id in enumerate(available_nodes):
            if current_layer >= total_layers:
                break

            # 基于节点算力比例分配层数
            node_compute = network_info['nodes'][node_id]['compute_capacity']
            compute_ratio = node_compute / total_compute
            layers_to_assign = max(1, int(total_layers * compute_ratio))

            # 确保不会超过总层数
            end_layer = min(current_layer + layers_to_assign, total_layers)

            partition_plan.append({
                'node_id': node_id,
                'layer_range': (current_layer, end_layer),
                'compute_load': end_layer - current_layer
            })

            current_layer = end_layer

        # 如果还有剩余层数，分配给最后一个节点
        if current_layer < total_layers and partition_plan:
            last_partition = partition_plan[-1]
            last_partition['layer_range'] = (last_partition['layer_range'][0], total_layers)
            last_partition['compute_load'] = total_layers - last_partition['layer_range'][0]

        print(f"分割方案: {partition_plan}")
        return partition_plan

    def distribute_and_execute(self, partition_plan):
        """分发任务并执行协同推理（包含地面站）"""
        print(f"{self.node_id} 分发任务到 {len(partition_plan)} 个节点...")

        execution_results = {
            'success': False,
            'total_latency': 0.0,
            'partition_plan': partition_plan,
            'node_results': {},
            'final_output': None
        }

        current_data = self.input_data
        total_latency = 0.0

        try:
            # 执行各卫星节点的分配任务
            for i, partition in enumerate(partition_plan):
                node_id = partition['node_id']
                layer_range = partition['layer_range']

                print(f"  分区 {i + 1}: {node_id} 执行层 {layer_range}")

                if node_id == self.node_id:
                    # 本节点执行
                    output_data, exec_time = self.execute_assigned_layers(current_data, layer_range)
                    execution_results['node_results'][node_id] = {
                        'execution_time': exec_time,
                        'status': 'success'
                    }
                    total_latency += exec_time
                else:
                    # 远程节点执行（模拟）
                    print(f"    远程执行 {node_id}")
                    exec_time = self.estimate_remote_execution(current_data, layer_range, node_id)
                    execution_results['node_results'][node_id] = {
                        'execution_time': exec_time,
                        'status': 'simulated'
                    }
                    total_latency += exec_time

                # 更新中间数据
                current_data = self.estimate_output_size(current_data, layer_range)

                # 传输时延
                if i < len(partition_plan) - 1:
                    next_node = partition_plan[i + 1]['node_id']
                    transmit_time = self.estimate_transmission_time(current_data, node_id, next_node)
                    total_latency += transmit_time

            # 关键补充：将最终结果传输到地面站
            if current_data is not None and self.ground_stations:
                # 选择第一个地面站作为最终接收点
                ground_station_id = list(self.ground_stations.keys())[0]
                print(f"  传输最终结果到地面站: {ground_station_id}")

                # 估算传输时延
                ground_transmit_time = self.estimate_ground_transmission_time(current_data)
                total_latency += ground_transmit_time

                execution_results['ground_station'] = ground_station_id
                execution_results['ground_transmit_time'] = ground_transmit_time
                execution_results['final_output'] = current_data  # 模拟最终输出

            execution_results['total_latency'] = total_latency
            execution_results['success'] = total_latency <= self.max_latency

            print(f"协同推理完成，总时延: {total_latency:.2f}ms")

        except Exception as e:
            print(f"协同推理失败: {e}")
            execution_results['success'] = False
            execution_results['error'] = str(e)

        return execution_results

    def extract_layers_range(self, start_layer, end_layer):
        """提取指定层范围的模型（适配自定义模型结构）"""
        layers = []
        for i in range(start_layer, end_layer):
            layer = self.current_model[i]  # 使用模型自定义的 __getitem__ 方法
            layers.append(layer)

        # 将层序列包装成 Sequential
        return torch.nn.Sequential(*layers)

    def estimate_remote_execution(self, data, layer_range, remote_node_id):
        """估计远程节点执行时间"""
        # 简化估计，基于节点算力和层数
        node_info = self.neighbor_nodes.get(remote_node_id, {})
        compute_capacity = node_info.get('compute_capacity', 4.0)
        num_layers = layer_range[1] - layer_range[0]

        base_time = num_layers * 15.0  # 每层基础时间
        adjusted_time = base_time * (8.0 / compute_capacity)  # 基于算力调整

        return adjusted_time

    def estimate_transmission_time(self, data, from_node, to_node):
        """估计数据传输时间"""
        # 简化估计
        if from_node == self.node_id or to_node == self.node_id:
            return 5.0  # ms
        else:
            return 10.0  # ms

    def estimate_ground_transmission_time(self, data):
        """估算到地面站的传输时延"""
        # 卫星到地面站的传输通常有更高的延迟
        if torch.is_tensor(data):
            data_size = data.nelement() * 4 / (1024 * 1024)  # MB
        else:
            data_size = 1.0

        # 地面站带宽通常较高，但延迟也较高
        bandwidth = 200.0  # Mbps
        transmission_time = (data_size * 8) / bandwidth  # 秒
        base_latency = 50.0  # 基础卫星到地面延迟

        return transmission_time * 1000 + base_latency  # 转换为毫秒

    def generate_partition_strategies(self, total_layers, available_nodes):
        """生成可能的分割策略"""
        strategies = []
        num_nodes = len(available_nodes)

        # 策略1：均匀分配
        layers_per_node = max(1, total_layers // num_nodes)
        uniform_plan = []
        current_layer = 0

        for i, node_id in enumerate(available_nodes):
            if current_layer >= total_layers:
                break
            end_layer = min(current_layer + layers_per_node, total_layers)
            uniform_plan.append({
                'node_id': node_id,
                'layer_range': (current_layer, end_layer),
                'compute_load': end_layer - current_layer
            })
            current_layer = end_layer
        strategies.append(uniform_plan)

        # 策略2：基于算力加权分配
        weighted_plan = []
        current_layer = 0

        # 计算总算力
        total_compute = sum(1.0 for _ in available_nodes)  # 简化

        for i, node_id in enumerate(available_nodes):
            if current_layer >= total_layers:
                break
            # 基于算力分配层数
            compute_ratio = 1.0 / num_nodes  # 简化
            layers_to_assign = max(1, int(total_layers * compute_ratio))
            end_layer = min(current_layer + layers_to_assign, total_layers)
            weighted_plan.append({
                'node_id': node_id,
                'layer_range': (current_layer, end_layer),
                'compute_load': end_layer - current_layer
            })
            current_layer = end_layer
        strategies.append(weighted_plan)

        return strategies

    def predict_strategy_latency(self, partition_plan, network_info):
        """预测分割策略的总时延"""
        total_latency = 0.0
        current_data = self.input_data

        for i, partition in enumerate(partition_plan):
            node_id = partition['node_id']
            layer_range = partition['layer_range']

            # 使用你现有的时延预测模型
            if node_id == self.node_id:
                # 本节点执行
                comp_latency = self.predict_computation_latency(current_data, layer_range)
            else:
                # 其他节点执行（简化预测）
                node_info = network_info['nodes'][node_id]
                comp_latency = self.estimate_remote_computation(layer_range, node_info)

            total_latency += comp_latency

            # 预测传输时延
            if i < len(partition_plan) - 1:
                next_node = partition_plan[i + 1]['node_id']
                transmit_latency = self.predict_transmission_latency(current_data, node_id, next_node, network_info)
                total_latency += transmit_latency

            # 更新数据大小估计（简化）
            current_data = self.estimate_output_size(current_data, layer_range)

        return total_latency

    def predict_computation_latency(self, input_data, layer_range):
        """使用你现有的时延预测模型预测计算时延"""
        start_layer, end_layer = layer_range
        target_model = self.extract_layers_range(start_layer, end_layer)

        # 使用你现有的 predictor_utils
        predicted_latency = predictor_utils.predict_model_latency(
            input_data, target_model, self.device, self.predictor_dict
        )

        return predicted_latency

    def estimate_remote_computation(self, layer_range, node_info):
        """估算远程节点的计算时延"""
        start_layer, end_layer = layer_range
        num_layers = end_layer - start_layer

        # 基于节点算力和设备类型估算
        base_time_per_layer = 10.0  # ms
        device_factor = 1.0 if node_info['device'] == "cpu" else 0.3
        compute_capacity_factor = 8.0 / node_info['compute_capacity']  # 相对于基准算力8.0

        estimated_time = num_layers * base_time_per_layer * device_factor * compute_capacity_factor
        return estimated_time

    def predict_transmission_latency(self, data, from_node, to_node, network_info):
        """预测传输时延"""
        # 估算数据大小
        if torch.is_tensor(data):
            data_size = data.nelement() * 4 / (1024 * 1024)  # MB
        else:
            data_size = 1.0  # 默认1MB

        # 获取链路带宽
        link_key = f"{from_node}->{to_node}"
        bandwidth = network_info['links'].get(link_key, {}).get('bandwidth', 50.0)  # Mbps

        # 计算传输时间
        transmission_time = (data_size * 8) / bandwidth  # 秒
        return transmission_time * 1000  # 转换为毫秒

    def estimate_output_size(self, input_data, layer_range):
        """估算层输出数据大小（简化）"""
        # 返回模拟的输出数据（保持形状）
        return torch.rand_like(input_data)

    def execute_assigned_layers(self, input_data, layer_range):
        """执行分配到的模型层（真实执行）"""
        print(f"    {self.node_id} 执行层 {layer_range}")

        # 提取指定层范围的模型
        start_layer, end_layer = layer_range
        target_model = self.extract_layers_range(start_layer, end_layer)

        # 预热
        self.warm_up_model(target_model, input_data)

        # 记录真实执行时间
        start_time = time.time()
        with torch.no_grad():
            output_data = target_model(input_data)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000  # 转换为毫秒

        print(f"       执行完成, 耗时: {execution_time:.2f}ms")
        return output_data, execution_time

    def warm_up_model(self, model, input_data):
        """模型预热"""
        model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            input_data = input_data.cuda()

        # 预热运行
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)

    def start_task_service(self):
        """启动任务接收服务"""

        def task_handler():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.ip, self.port + 1000))  # 任务服务使用不同端口
                server_socket.listen(5)

                print(f"{self.node_id} 任务服务启动在 {self.ip}:{self.port + 1000}")

                while True:
                    conn, addr = server_socket.accept()
                    threading.Thread(
                        target=self._handle_task_connection,
                        args=(conn, addr)
                    ).start()

            except Exception as e:
                print(f"任务服务错误: {e}")

        thread = threading.Thread(target=task_handler, daemon=True)
        thread.start()
        return thread

    def _handle_task_connection(self, conn, addr):
        """处理任务连接请求"""
        try:
            data = conn.recv(4096)
            if data:
                message = pickle.loads(data)
                if message['type'] == 'task':
                    print(f"{self.node_id} 接收到任务: {message['data']['task_id']}")

                    # 处理任务
                    if self.satellite_type == SatelliteType.REMOTE_SENSING:
                        result = self.assign_task(message['data'])
                    else:
                        result = {"status": "not_coordinator", "message": "非协调节点"}

                    # 返回结果
                    conn.send(pickle.dumps(result))
                else:
                    conn.send(pickle.dumps({"status": "error", "message": "未知消息类型"}))

        except Exception as e:
            print(f"处理任务消息错误: {e}")
            try:
                conn.send(pickle.dumps({"status": "error", "message": str(e)}))
            except:
                pass
        finally:
            conn.close()