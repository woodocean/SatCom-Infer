import time
import torch
from satellite_node import SatelliteNode, SatelliteType


def test_integrated_system():
    """测试集成后的完整系统"""
    print("测试集成时延预测的卫星协同推理系统...")

    # 创建卫星网络
    remote_sat = SatelliteNode("SAT-001", SatelliteType.REMOTE_SENSING, "127.0.0.1", 10001, 8.0, "cuda")
    leo_sat1 = SatelliteNode("SAT-002", SatelliteType.LEO_COMPUTING, "127.0.0.1", 10002, 6.0, "cuda")
    leo_sat2 = SatelliteNode("SAT-003", SatelliteType.LEO_COMPUTING, "127.0.0.1", 10003, 4.0, "cpu")

    # 添加地面站
    remote_sat.add_ground_station("GROUND-001", {
        'bandwidth': 200.0,
        'latency': 50.0,
        'location': 'Beijing'
    })

    # 启动发现服务
    remote_sat.start_discovery_service()
    leo_sat1.start_discovery_service()
    leo_sat2.start_discovery_service()

    # 等待服务启动
    time.sleep(1)

    # 建立网络连接
    print("\n建立卫星网络...")
    remote_sat.discover_neighbor("127.0.0.1", 10002)
    time.sleep(0.5)
    remote_sat.discover_neighbor("127.0.0.1", 10003)
    time.sleep(1)

    # 创建测试任务
    print("\n创建推理任务...")
    task = {
        'task_id': 'urban_detection_001',
        'model_type': 'alex_net',  # 使用你现有的模型
        'input_data': torch.rand(1, 3, 224, 224),
        'max_latency': 5000,  # 5秒
        'priority': 'high'
    }

    # 执行任务
    print("\n开始智能协同推理...")
    start_time = time.time()
    result = remote_sat.assign_task(task)
    end_time = time.time()

    # 输出结果
    print(f"\n智能协同推理结果:")
    print(f"任务ID: {task['task_id']}")
    print(f"预测总时延: {result.get('total_latency', 0):.2f}ms")
    print(f"实际执行时间: {(end_time - start_time) * 1000:.2f}ms")
    print(f"是否满足时限: {'是' if result.get('success', False) else '否'}")
    print(f"最大允许延迟: {task['max_latency']}ms")

    if 'ground_station' in result:
        print(f"最终接收地面站: {result['ground_station']}")
        print(f"地面站传输时延: {result.get('ground_transmit_time', 0):.2f}ms")

    if 'partition_plan' in result:
        print(f"\n智能分割方案:")
        for i, partition in enumerate(result['partition_plan']):
            print(f"  {i + 1}. {partition['node_id']}: 层{partition['layer_range']}")


if __name__ == "__main__":
    test_integrated_system()