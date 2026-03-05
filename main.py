import argparse
import time
import os
import sys
import json
import torch

sys.path.append(os.getcwd())

from core.node import SatelliteNode
from utils.data_utils import get_cifar10_batch

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='卫星协同推理 - PC/Jetson联合仿真')

    # 基础参数
    parser.add_argument('--id', required=True, help='节点ID (RS, SAT-01, SAT-02, GS)')
    parser.add_argument('--net_cfg', default='config/network_config.json')
    parser.add_argument('--task_cfg', default='config/task_config.json')
    parser.add_argument('--dev_cfg', default='config/device_profile.json')

    # 运行模式
    parser.add_argument('--run_task', required=False, help='手动模式: 运行指定任务名')
    parser.add_argument('--algo', choices=['dp', 'lawa', 'ga', 'selector', 'baseline'],
                        default=None, help='自动模式: 选择调度算法')
    parser.add_argument('--auto', action='store_true', help='自动模式开关(兼容旧版)')
    parser.add_argument('--model', default='alexnet', help='模型名称')
    parser.add_argument('--profile', action='store_true', help='运行设备性能测试')
    parser.add_argument('--simulate_bw', action='store_true', help='模拟带宽限制')

    args = parser.parse_args()

    # ========== 1. 加载配置 ==========
    try:
        net_config = load_config(args.net_cfg)
        my_config = net_config['nodes'][args.id]
    except KeyError:
        print(f"[错误] 节点 [{args.id}] 未在 {args.net_cfg} 中定义!")
        print(f"  可用节点: {list(net_config['nodes'].keys())}")
        return
    except FileNotFoundError:
        print(f"[错误] 找不到配置文件 {args.net_cfg}")
        return

    # 加载设备profile (可选)
    device_profiles = {}
    if os.path.exists(args.dev_cfg):
        device_profiles = load_config(args.dev_cfg)

    print("=" * 60)
    print(f"  节点 [{args.id}] 启动中...")
    print(f"  角色: {my_config['role']}")
    print(f"  地址: {my_config['ip']}:{my_config['port']}")
    print(f"  CUDA: {'可用 - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用(CPU模式)'}")
    print("=" * 60)

    # ========== 2. 设备性能测试模式 ==========
    if args.profile:
        from predictor.profiler import DeviceProfiler
        profiler = DeviceProfiler(args.id)
        hw = profiler.detect_hardware()
        print(f"\n[硬件检测]\n{json.dumps(hw, indent=2, ensure_ascii=False)}")

        bench = profiler.benchmark_model(args.model)
        print(f"\n[推理性能]\n{json.dumps(bench, indent=2)}")

        layers = profiler.profile_layers(args.model)
        out_path = f'profile_{args.id}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(layers, f, indent=2, ensure_ascii=False)
        print(f"\n[层级Profile] 已保存到 {out_path}, 共 {len(layers)} 层")
        return

    # ========== 3. 创建节点并组网 ==========
    node = SatelliteNode(
        node_id=args.id,
        ip=my_config['ip'],
        port=my_config['port'],
        role=my_config['role'],
        device_profiles=device_profiles,
        simulate_bw=args.simulate_bw
    )

    # 组网: 注册邻居
    if my_config.get('neighbors'):
        neighbors_parsed = []
        for n_id in my_config['neighbors']:
            if n_id not in net_config['nodes']:
                print(f"  [警告] 邻居 [{n_id}] 未在配置中定义, 跳过")
                continue
            n_info = net_config['nodes'][n_id]
            neighbors_parsed.append((n_id, n_info['ip'], n_info['port']))
        node.join_network(neighbors_parsed)

    # ========== 4. 启动监听 ==========
    node.start()

    # ========== 5. 任务触发 (仅 RS 节点) ==========
    is_trigger = my_config['role'] == 'remote_sensing'
    has_task = args.auto or args.algo or args.run_task

    if is_trigger and has_task:
        print(f"\n[{args.id}] 等待 5秒 让其他节点就绪...")
        time.sleep(5)

        # 准备输入数据
        print(f"[{args.id}] 正在加载输入数据...")
        images = _load_input_data(args.model)

        # -------- 分支A: 算法模式 --------
        if args.algo:
            _run_algorithm(args, node, images, net_config, device_profiles)

        # -------- 分支B: 自动GA调度 (兼容旧版) --------
        elif args.auto:
            print(f"[{args.id}] 自动GA调度模式")
            node.run_auto_schedule(images, net_config)

        # -------- 分支C: 手动任务 --------
        elif args.run_task:
            _run_manual_task(args, node, images, net_config)

    # ========== 6. 保持运行 ==========
    print(f"\n[{args.id}] 节点运行中... (Ctrl+C退出)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{args.id}] 正在关闭...")
        node.stop()

def _load_input_data(model_name):
    """加载/生成输入数据"""
    data_path = './data/cifar-10-batches-py'
    try:
        images, labels = get_cifar10_batch(data_path, batch_size=16)
        print(f"  真实数据加载成功: {images.shape}")
        return images
    except Exception as e:
        print(f"  数据加载失败({e}), 使用随机数据...")
        if model_name in ['alexnet', 'lenet']:
            return torch.randn(1, 3, 32, 32)
        else:
            return torch.randn(1, 3, 224, 224)

def _run_algorithm(args, node, images, net_config, device_profiles):
    """算法调度模式"""
    algo = args.algo

    if algo == 'baseline':
        print(f"[Baseline] 本地完整推理...")
        node.run_baseline(images)

    elif algo == 'selector':
        print(f"[Selector] 多因子推理模式选择...")
        node.run_selector(images, net_config, device_profiles)

    elif algo == 'dp':
        print(f"[DP] 流水线优化调度...")
        node.run_dp_schedule(images, net_config, device_profiles)

    elif algo == 'lawa':
        print(f"[LAWA] 链路感知加权并行...")
        node.run_lawa_schedule(images, net_config, device_profiles)

    elif algo == 'ga':
        print(f"[GA] 遗传算法调度...")
        node.run_auto_schedule(images, net_config)

def _run_manual_task(args, node, images, net_config):
    """手动任务模式"""
    task_config = load_config(args.task_cfg)
    if args.run_task not in task_config:
        print(f"[错误] 任务 '{args.run_task}' 未在 task_config.json 中定义")
        return

    task = task_config[args.run_task]
    if task['type'] == 'pipeline':
        route = task['route']
        split = task.get('split_point', 10)
        if route and route[0] == args.id:
            route = route[1:]
        node.start_pip_task(route, images, split_point=split)

    elif task['type'] == 'parallel':
        dist_map = {}
        workers = task.get('workers', [])
        batch = images.shape[0]
        per = max(1, batch // len(workers))
        for i, w_id in enumerate(workers):
            start = i * per
            end = min((i + 1) * per, batch)
            dist_map[w_id] = (start, end)
        aggregator = task.get('aggregator', 'GS')
        node.start_para_task(dist_map, images, aggregator)

if __name__ == '__main__':
    main()