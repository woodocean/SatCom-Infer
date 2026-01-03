import time
from .communicate import CommManager
import threading
import uuid
from .inference import InferenceManager
class SatelliteNode:
    def __init__(self, node_id, ip, port, role="leo_computing"):
        self.node_id = node_id
        self.role = role  # 'remote_sensing', 'leo_computing', 'ground_station'

        # 初始化通信和推理运算模块
        self.comms = CommManager(ip, port, node_id)
        self.infer = InferenceManager()
        # 注册基础消息处理
        self.comms.register_handler("PING", self._handle_ping)
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
    def start_para_task(self, distribute_map, agg_dest="GS"):
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

        print(f"[{self.node_id}] 启动并行任务 Task[{task_id}] -> 汇聚于 {agg_dest}")

        def send_target(target, data):
            payload = {
                "task_id": task_id,
                "data": data,
                "agg_dest": agg_dest,
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

        # 这里可以写处理数据的逻辑
        data, cost = self.infer.exec_full(data)

        # 构造发送给聚合卫星的数据包
        result_pack = {
            "task_id": task_id,
            "data": data,
            "source": self.node_id,
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

        print(f"[{self.node_id}] 收到 Task[{task_id}] 结果 (来自 {source})")

        # 懒加载：如果是新任务，先在字典里开辟空间
        if task_id not in self.task_buffer:
            self.task_buffer[task_id] = []

        self.task_buffer[task_id].append(result_part)

        current_count = len(self.task_buffer[task_id])
        if current_count == total_parts:
            print(f"[{self.node_id}] Task[{task_id}] 全部收齐! ({current_count}/{total_parts})")
            final_output = " + ".join(self.task_buffer[task_id])
            print(f"最终落盘数据: [{final_output}]")

            self.comms.send_message("GS","Para_GS", final_output)
            # 清理内存，防止内存泄漏
            del self.task_buffer[task_id]
        else:
            print(f"   ⏳ 等待其他分片... ({current_count}/{total_parts})")

    def _handle_para_GS(self, message):
        data = message['payload']
        print(data)

    # --------流水线推理功能--------------
    # 这个函数可以作为启动流水线推理的原型函数
    def start_pip_task(self, route_list):
        if self.role != 'remote_sensing':
            print(f" [{self.node_id}] 我不是遥感卫星，不能发起任务")
            return

        if not route_list:
            print(" 路由列表为空")
            return

        # 获取下一跳
        next_hop = route_list[0]
        # 剩余路由
        remain_list = route_list[1:]
        # 数据包及其路由
        pay_load = {
            "data":"这是遥感图像数据",
            "route":remain_list
        }
        # 发送数据
        self.comms.send_message(next_hop, "PIPLINE", payload=pay_load)

    # 这个函数可以作为流水线推理中继节点的原型函数
    def _handle_pip_infer(self, message):
        route_list = message["payload"]["route"]
        if not route_list:
            print(f" 路由列表为空,本跳{self.node_id}为终点")
            print(f"测试传输的数据为{message['payload']['data']}")
            return
        else:
            # 获取下一跳
            next_hop = route_list[0]
            # 剩余路由
            remain_list = route_list[1:]
            # 数据包及其路由
            pay_load = {
                "data": message["payload"]["data"],
                "route": remain_list
            }
            # 发送数据
            self.comms.send_message(next_hop, "PIPLINE", payload=pay_load)
