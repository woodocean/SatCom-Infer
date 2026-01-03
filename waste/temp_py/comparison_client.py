import socket
import pickle
import torch
import time
from utils.dataset_utils import load_test_dataset, evaluate_model_accuracy, get_sample_batch


def connect_satellite_network():
    """在测试前连接所有卫星节点建立网络拓扑"""
    print("🛰️ 建立卫星网络拓扑...")

    # 节点配置
    nodes = [
        {'id': 'SAT-001', 'ip': '127.0.0.1', 'port': 10001},
        {'id': 'SAT-002', 'ip': '127.0.0.1', 'port': 10002},
        {'id': 'SAT-003', 'ip': '127.0.0.1', 'port': 10003},
        {'id': 'GROUND-001', 'ip': '127.0.0.1', 'port': 20001}
    ]

    # 让SAT-001连接其他所有节点
    coordinator = nodes[0]
    connected_nodes = []

    for node in nodes[1:]:
        try:
            # 发送连接请求
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.settimeout(5.0)
            conn.connect((node['ip'], node['port']))

            # 发送SAT-001的信息
            hello_msg = {
                'node_id': coordinator['id'],
                'type': 'remote_sensing',
                'ip': coordinator['ip'],
                'port': coordinator['port'],
                'compute_capacity': 8.0,
                'device': 'cuda'
            }
            conn.send(pickle.dumps(hello_msg))

            # 等待回复
            response_data = conn.recv(1024)
            response = pickle.loads(response_data)

            if response['status'] == 'ack':
                print(f"✅ {coordinator['id']} 成功连接到 {node['id']}")
                connected_nodes.append(node['id'])
            else:
                print(f"❌ {coordinator['id']} 连接 {node['id']} 被拒绝")

            conn.close()
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 连接 {node['id']} 失败: {e}")

    print(f"📡 网络拓扑建立完成，连接了 {len(connected_nodes)} 个节点: {connected_nodes}")
    return len(connected_nodes) > 0


def evaluate_satellite_accuracy(satellite_ip, satellite_port, testloader, test_type, num_samples=50):
    """评估卫星推理精度 - 修复设备不匹配问题"""
    print(f"   正在评估{test_type}精度...")

    correct = 0
    total = 0

    for i, (images, labels) in enumerate(testloader):
        if i >= num_samples:
            break

        task = {
            'task_id': f'accuracy_test_{i}',
            'model_type': 'alex_net',
            'input_data': images,
            'max_latency': 30000,
            'priority': 'low',
            'test_type': test_type,
            'return_output': True
        }

        try:
            result = submit_task_with_output(task, satellite_ip, satellite_port)

            if result and result.get('success') and 'final_output' in result and result['final_output'] is not None:
                outputs = result['final_output']

                # 🎯 修复设备不匹配：确保 outputs 和 labels 在相同设备上
                if outputs.device != labels.device:
                    # 将 outputs 移动到 labels 所在的设备
                    outputs = outputs.to(labels.device)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 10 == 0:
                    current_acc = 100 * correct / total if total > 0 else 0
                    print(f"      进度: {i + 1}/{num_samples}, 当前精度: {current_acc:.1f}%")
            else:
                print(f"      样本{i}: 推理失败或没有输出")

        except Exception as e:
            print(f"      样本{i}推理异常: {e}")
            continue

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"      {test_type}精度评估完成: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


def submit_task_with_output(task, satellite_ip, satellite_port):
    """提交任务到卫星节点 - 修复版本：处理设备问题"""
    try:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(60.0)
        conn.connect((satellite_ip, satellite_port))

        # 🎯 确保输入数据在 CPU 上，避免序列化问题
        if 'input_data' in task and task['input_data'] is not None:
            if task['input_data'].device.type != 'cpu':
                task['input_data'] = task['input_data'].cpu()

        task['return_output'] = True
        task_data = pickle.dumps(task)
        conn.sendall(task_data)

        response_data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            response_data += chunk
            try:
                response = pickle.loads(response_data)
                break
            except:
                continue

        conn.close()

        # 🎯 确保返回的 tensor 在正确的设备上
        if response and 'final_output' in response and response['final_output'] is not None:
            # 如果有 CUDA，将输出移动到 CUDA
            if torch.cuda.is_available():
                response['final_output'] = response['final_output'].cuda()

        return response

    except Exception as e:
        print(f"❌ 任务提交失败: {e}")
        return None


def debug_device_issues(satellite_ip, satellite_port, testloader):
    """调试设备问题"""
    print("🔍 调试设备问题...")

    # 获取一个测试样本
    sample_image, sample_label = next(iter(testloader))

    print(f"  输入图像设备: {sample_image.device}")
    print(f"  标签设备: {sample_label.device}")

    task = {
        'task_id': 'device_debug',
        'model_type': 'alex_net',
        'input_data': sample_image,
        'max_latency': 30000,
        'priority': 'low',
        'test_type': 'single_satellite',
        'return_output': True
    }

    result = submit_task_with_output(task, satellite_ip, satellite_port)

    if result and result.get('success') and 'final_output' in result:
        print(f"  输出设备: {result['final_output'].device}")
        print(f"  输出形状: {result['final_output'].shape}")

        # 测试设备兼容性
        try:
            # 确保设备一致
            if result['final_output'].device != sample_label.device:
                adjusted_output = result['final_output'].to(sample_label.device)
                _, predicted = torch.max(adjusted_output.data, 1)
                print(f"  ✅ 设备调整成功，预测: {predicted.item()}, 真实: {sample_label.item()}")
            else:
                _, predicted = torch.max(result['final_output'].data, 1)
                print(f"  ✅ 设备一致，预测: {predicted.item()}, 真实: {sample_label.item()}")
        except Exception as e:
            print(f"  ❌ 设备调整失败: {e}")

    return result


def test_single_satellite_with_accuracy(satellite_ip, satellite_port, testloader):
    """测试单星推理并评估精度 - 修复设备问题"""
    print("=== 测试单星推理 ===")

    # 先评估精度
    accuracy = evaluate_satellite_accuracy(satellite_ip, satellite_port, testloader, 'single_satellite')
    print(f"🎯 单星推理精度: {accuracy:.2f}%")

    # 测试单个样本的时延
    sample_batch, true_label = get_sample_batch(testloader, batch_size=1)

    # 🎯 确保输入数据在 CPU 上
    if sample_batch.device.type != 'cpu':
        sample_batch = sample_batch.cpu()

    task = {
        'task_id': 'single_satellite_test',
        'model_type': 'alex_net',
        'input_data': sample_batch,
        'max_latency': 30000,
        'priority': 'high',
        'test_type': 'single_satellite',
        'return_output': True
    }

    start_time = time.perf_counter()
    result = submit_task_with_output(task, satellite_ip, satellite_port)
    end_time = time.perf_counter()

    total_time = (end_time - start_time) * 1000

    if result and result.get('success'):
        print(f"✅ 单星推理成功 - 总时延: {total_time:.2f}ms")
        return {
            'type': 'single_satellite',
            'success': True,
            'total_latency': total_time,
            'accuracy': accuracy,
            'node_latencies': result.get('node_results', {}),
            'partition_plan': result.get('partition_plan', [])
        }
    else:
        print(f"❌ 单星推理失败: {result}")
        return {'type': 'single_satellite', 'success': False, 'accuracy': accuracy}


def test_multi_satellite_with_accuracy(satellite_ip, satellite_port, testloader):
    """测试多星协同推理并评估精度 - 修复设备问题"""
    print("=== 测试多星协同推理 ===")

    # 先评估精度
    accuracy = evaluate_satellite_accuracy(satellite_ip, satellite_port, testloader, 'multi_satellite')
    print(f"🎯 多星协同推理精度: {accuracy:.2f}%")

    # 测试单个样本的时延
    sample_batch, true_label = get_sample_batch(testloader, batch_size=1)

    # 🎯 确保输入数据在 CPU 上
    if sample_batch.device.type != 'cpu':
        sample_batch = sample_batch.cpu()

    task = {
        'task_id': 'multi_satellite_test',
        'model_type': 'alex_net',
        'input_data': sample_batch,
        'max_latency': 30000,
        'priority': 'high',
        'test_type': 'multi_satellite',
        'return_output': True
    }

    start_time = time.perf_counter()
    result = submit_task_with_output(task, satellite_ip, satellite_port)
    end_time = time.perf_counter()

    total_time = (end_time - start_time) * 1000

    if result and result.get('success'):
        print(f"✅ 多星协同推理成功 - 总时延: {total_time:.2f}ms")
        return {
            'type': 'multi_satellite',
            'success': True,
            'total_latency': total_time,
            'accuracy': accuracy,
            'node_latencies': result.get('node_results', {}),
            'partition_plan': result.get('partition_plan', []),
            'ground_station_time': result.get('ground_transmit_time', 0)
        }
    else:
        print(f"❌ 多星协同推理失败: {result}")
        return {'type': 'multi_satellite', 'success': False, 'accuracy': accuracy}
def test_local_server_with_accuracy(testloader, model_type='alex_net'):
    """测试本地服务器推理并评估精度"""
    print("=== 测试本地服务器推理 ===")

    from utils.inference_utils import get_dnn_model

    # 加载训练好的模型
    model = get_dnn_model(model_type)

    # 评估精度 - 使用更多样本
    accuracy = evaluate_model_accuracy_extended(model, testloader, 'cuda', num_batches=50)
    print(f"🎯 本地服务器精度: {accuracy:.2f}%")

    # 测试推理速度
    sample_batch, true_label = get_sample_batch(testloader, batch_size=1)
    if sample_batch is not None:
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        sample_batch = sample_batch.to(device)

        # 验证单次推理正确性
        with torch.no_grad():
            output = model(sample_batch)
            pred = torch.argmax(output, 1)
            print(f"🔍 单样本测试 - 预测: {pred.item()}, 真实: {true_label.item()}")

        # 预热
        for _ in range(10):
            _ = model(sample_batch)

        # 计时
        num_runs = 50
        execution_times = []

        for run in range(num_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_batch)
            end_time = time.perf_counter()
            execution_times.append((end_time - start_time) * 1000)

        execution_times.sort()
        median_time = execution_times[len(execution_times) // 2]

        print(f"⚡ 本地服务器推理时延: {median_time:.2f}ms")

        return {
            'type': 'local_server',
            'success': True,
            'total_latency': median_time,
            'accuracy': accuracy,
            'node_latencies': {'local_server': median_time}
        }

    return {'type': 'local_server', 'success': False}


def evaluate_model_accuracy_extended(model, testloader, device='cuda', num_batches=50):
    """扩展的精度评估"""
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i >= num_batches:
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


def run_comparison():
    """运行完整的对比测试 - 修复设备问题"""
    print("🚀 开始卫星协同推理对比测试（完整精度评估）")
    print("=" * 60)

    # 第一步：建立卫星网络拓扑
    network_ready = connect_satellite_network()
    if not network_ready:
        print("❌ 网络拓扑建立失败，无法进行对比测试")
        return

    # 第二步：加载测试数据
    print("\n📊 加载测试数据...")
    testloader = load_test_dataset('cifar10', batch_size=1)

    # 🎯 第三步：调试设备问题
    print("\n🔍 调试设备兼容性...")
    debug_result = debug_device_issues('127.0.0.1', 10001, testloader)

    if not debug_result or not debug_result.get('success'):
        print("❌ 设备调试失败，无法进行精度评估")
        return

    print("📊 使用50个测试样本评估每个方案的精度...")

    results = []

    # 测试1: 本地服务器基准
    print("\n1️⃣ 本地服务器基准测试...")
    local_result = test_local_server_with_accuracy(testloader)
    results.append(local_result)

    # 测试2: 单星推理
    print("\n2️⃣ 单星推理测试...")
    single_result = test_single_satellite_with_accuracy('127.0.0.1', 10001, testloader)
    results.append(single_result)

    # 测试3: 多星协同推理
    print("\n3️⃣ 多星协同推理测试...")
    multi_result = test_multi_satellite_with_accuracy('127.0.0.1', 10001, testloader)
    results.append(multi_result)

    # 输出完整的对比结果
    print("\n" + "=" * 60)
    print("📈 完整对比测试结果:")
    print("=" * 60)

    for result in results:
        if result.get('success'):
            print(
                f"{result['type']:20} | 时延: {result['total_latency']:8.2f}ms | 精度: {result.get('accuracy', 0):6.2f}%")

            if 'node_latencies' in result:
                for node, latency in result['node_latencies'].items():
                    if isinstance(latency, dict):
                        exec_time = latency.get('execution_time', 0)
                        print(f"{' ':20} | {node:15}: {exec_time:7.2f}ms")
                    else:
                        print(f"{' ':20} | {node:15}: {latency:7.2f}ms")

            if 'partition_plan' in result and result['partition_plan']:
                print(f"{' ':20} | 分割方案: {result['partition_plan']}")
        else:
            print(f"{result['type']:20} | ❌ 测试失败 | 精度: {result.get('accuracy', 0):6.2f}%")

        print("-" * 60)

    # 计算性能指标
    if all(r.get('success') for r in results):
        single_time = results[1]['total_latency']
        multi_time = results[2]['total_latency']
        local_time = results[0]['total_latency']

        single_accuracy = results[1].get('accuracy', 0)
        multi_accuracy = results[2].get('accuracy', 0)
        local_accuracy = results[0].get('accuracy', 0)

        print(f"🚀 性能总结:")
        print(f"   精度对比: 本地{local_accuracy:.1f}% vs 单星{single_accuracy:.1f}% vs 多星{multi_accuracy:.1f}%")

        if multi_time > 0:
            speedup_vs_single = single_time / multi_time
            print(f"   时延对比: 单星{single_time:.1f}ms vs 多星{multi_time:.1f}ms")
            print(f"   加速比(多星vs单星): {speedup_vs_single:.2f}x")

        # 精度损失分析
        accuracy_loss_single = local_accuracy - single_accuracy
        accuracy_loss_multi = local_accuracy - multi_accuracy
        print(f"   精度损失: 单星{accuracy_loss_single:.1f}% 多星{accuracy_loss_multi:.1f}%")


if __name__ == "__main__":
    run_comparison()