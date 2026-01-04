import time
from .communicate import CommManager
import threading
import uuid
from .inference import InferenceManager
from utils.data_utils import decode_prediction
import torch
from utils.data_utils import decode_prediction, save_result_image


class SatelliteNode:
    def __init__(self, node_id, ip, port, role="leo_computing"):
        self.node_id = node_id
        self.role = role  # 'remote_sensing', 'leo_computing', 'ground_station'

        # 初始化通信和推理运算模块
        self.comms = CommManager(ip, port, node_id)
        self.infer = InferenceManager()
        # 注册基础消息处理
        self.comms.register_handler("PING", self._handle_ping)
        # 注册任务消息处理
        self.comms.register_handler("PIPLINE", self._handle_pip_infer)
        self.comms.register_handler("Para_infer", self._handle_para_infer)
        self.comms.register_handler("Para_agg", self._handle_para_agg)
        self.comms.register_handler("Para_GS", self._handle_para_GS)

        self.task_buffer = {}

    def start(self):
        self.comms.start_server()
        print(f"节点 {self.node_id} ({self.role}) 已就绪.")

    def join_network(self, neighbors_list):
        """
        neighbors_list: [('SAT-02', '127.0.0.1', 7002), ...]
        """
        for nid, nip, nport in neighbors_list:
            self.comms.add_neighbor(nid, nip, nport)

    # ---------------------- 业务逻辑处理 -----------------------------------------------
    # --------基础PING测试功能-----------

    def ping_neighbor(self, neighbor_id):
        print(f"[{self.node_id}] Pinging {neighbor_id}...")
        self.comms.send_message(neighbor_id, "PING", payload="Hello World")

    def _handle_ping(self, message):
        src = message['source']
        print(f"[{self.node_id}] 收到来自 {src} 的 PING: {message['payload']}")

    # ---------并行推理功能---------------
    # 这个函数可以作为启动并行推理的原型函数
    def start_para_task(self, distribute_map, agg_dest="SAT_AGG", destination="GS"):
        """
                发起并行任务
                :param distribute_map: 字典 { "SAT-01": "上半图数据", "SAT-02": "下半图数据" }
                :param agg_dest: 最终汇聚结果的节点，默认是地面站 GS，但其实可以指定是哪颗聚合卫星
                """

        if self.role != 'remote_sensing':
            print(f" [{self.node_id}] 我不是遥感卫星，不能发起任务")
            return

        threads = []
        print(f"[{self.node_id}] 启动多线程并行分发...")

        # 生成唯一任务ID (UUID截取前8位，类似 'a1b2c3d4')
        task_id = str(uuid.uuid4())[:8]
        total_parts = len(distribute_map)

        print(f"[{self.node_id}] 启动并行任务 Task[{task_id}] -> 汇聚于 {agg_dest},最终回传到{destination}")

        def send_target(target, data):
            payload = {
                "task_id": task_id,
                "data": data,
                "agg_dest": agg_dest,
                "dest":destination,
                "total_parts": total_parts,
                "timestamp": time.time()
            }

            self.comms.send_message(target, "Para_infer", payload)
            print(f"开始分发-->{target}的数据")

        for target, target_data in distribute_map.items():
            t = threading.Thread(target=send_target, args=(target, target_data))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print(f"[{self.node_id}] 所有并行任务已发出")

    # 这个函数可以作为并行推理中继节点的原型函数
    def _handle_para_infer(self, message):
        payload = message['payload']
        data = payload['data']
        agg_dest = payload['agg_dest']
        task_id = payload['task_id']
        destination = payload['dest']
        # 这里可以写处理数据的逻辑
        output_tensor, cost = self.infer.exec_full(data)
        print(f"[{self.node_id}] 推理完成，耗时 {cost:.2f}ms，输出形状 {output_tensor.shape}")
        # 构造发送给聚合卫星的数据包
        result_pack = {
            "task_id": task_id,
            "data": output_tensor,
            "source": self.node_id,
            "dest":destination,
            "total_parts": payload['total_parts']
        }

        # 发送给聚合卫星
        print(f"[{self.node_id}] 推理完成，结果回传 -> {agg_dest}")
        self.comms.send_message(agg_dest, "Para_agg", result_pack)

    # 这个函数可以作为并行推理聚合节点的原型函数
    def _handle_para_agg(self, message):
        payload = message['payload']
        task_id = payload['task_id']
        result_part = payload['data']
        total_parts = payload['total_parts']
        source = payload['source']
        destination = payload['dest']

        print(f"[{self.node_id}] 收到 Task[{task_id}] 结果 (来自 {source})")

        # 懒加载：如果是新任务，先在字典里开辟空间
        if task_id not in self.task_buffer:
            self.task_buffer[task_id] = []

        self.task_buffer[task_id].append(result_part)

        current_count = len(self.task_buffer[task_id])
        if current_count == total_parts:
            print(f"[{self.node_id}] Task[{task_id}] 全部收齐! ({current_count}/{total_parts})")
            # 1. 拼接结果: [Batch_A, 10] + [Batch_B, 10] -> [Total_Batch, 10]
            # 注意：task_buffer 里存的是 Tensor
            final_tensor = torch.cat(self.task_buffer[task_id], dim=0)

            # 2. 解码结果
            pred_names = decode_prediction(final_tensor)

            print("=" * 30)
            print(f"最终识别结果: {pred_names}")
            print("=" * 30)

            # img_name = f"result_{task_id}.png"
            # save_result_image(final_tensor, pred_names, filename=img_name)

            payload={
                "task_id": task_id,
                "data": pred_names,
                "timestamp": time.time()
            }
            self.comms.send_message(destination, "Para_GS", payload)
            del self.task_buffer[task_id]
        else:
            print(f"    等待其他分片... ({current_count}/{total_parts})")

    # 这个函数可以作为并行推理地面节点的原型函数
    def _handle_para_GS(self, message):
        payload = message['payload']
        pred_names = payload['data']

        print("=" * 30)
        print(f"最终识别结果: {pred_names}")
        print("=" * 30)

    # --------流水线推理功能--------------
    # 这个函数可以作为启动流水线推理的原型函数
    def start_pip_task(self, route_list,input_data):
        if self.role != 'remote_sensing':
            print(f" [{self.node_id}] 我不是遥感卫星，不能发起任务")
            return
        if not route_list:
            print(" 路由列表为空")
            return

        # 生成任务ID
        task_id = str(uuid.uuid4())[:8]

        split_point = 10

        # 获取下一跳
        next_hop = route_list[0]
        # 剩余路由
        remain_list = route_list[1:]
        # 数据包及其路由
        pay_load = {
            "task_id": task_id,
            "data": input_data,
            "route": remain_list,
            "start_layer": 0,
            "end_layer": split_point
        }
        # 发送数据
        self.comms.send_message(next_hop, "PIPLINE", payload=pay_load)

    # 这个函数可以作为流水线推理中继节点的原型函数
    def _handle_pip_infer(self, message):
        """
                流水线节点逻辑：接收中间结果 -> 执行指定层 -> 转发给下一跳
                """
        payload = message["payload"]
        task_id = payload["task_id"]
        data = payload["data"]
        route_list = payload["route"]

        # 获取这一跳该跑的层范围
        start = payload.get("start_layer", 0)
        end = payload.get("end_layer", 999)  # 999代表跑到最后

        # 1. 真正的推理 (Inference)
        # 注意：这里的 data 可能是原始图片(第一跳)，也可能是中间特征图(Feature Map)
        print(f"[{self.node_id}] 流水线计算: Layer {start} -> {end} ...")

        # 调用 InferenceManager 执行切片
        output_tensor, cost = self.infer.exec_layers(data, start, end)
        print(f"   耗时: {cost:.2f}ms | 输出形状: {output_tensor.shape}")

        if not route_list:
            print(f" 路由列表为空,本跳{self.node_id}为终点")
            preds = decode_prediction(output_tensor)
            print("=" * 30)
            print(f"最终预测: {preds}")
            print("=" * 30)

            return
        else:
            # 获取下一跳
            next_hop = route_list[0]
            # 剩余路由
            remain_list = route_list[1:]
            next_start = end
            next_end = 999
            print(f" [{self.node_id}] 传递中间结果 -> {next_hop} (Layer {next_start}-End)")
            # 数据包及其路由
            pay_load = {
                "data": output_tensor,
                "route": remain_list,
                "start_layer": next_start,
                "end_layer": next_end,
                "task_id": task_id
            }
            # 发送数据
            self.comms.send_message(next_hop, "PIPLINE", payload=pay_load)
