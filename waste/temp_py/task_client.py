import socket
import pickle
import torch


def submit_task(task, satellite_ip, satellite_port):
    """提交任务到卫星节点"""
    try:
        # 创建socket连接
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(30)  # 设置更长的超时时间

        print(f"尝试连接到 {satellite_ip}:{satellite_port}")
        conn.connect((satellite_ip, satellite_port))
        print("连接成功")

        # 序列化任务数据
        task_data = pickle.dumps(task)

        # 发送任务数据
        print("发送任务数据...")
        conn.sendall(task_data)  # 使用 sendall 确保所有数据发送

        # 接收响应
        print("等待响应...")
        response_data = b""
        conn.settimeout(30.0)  # 设置接收超时

        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            response_data += chunk
            # 尝试解析，如果成功则停止接收
            try:
                response = pickle.loads(response_data)
                break
            except:
                continue

        if response_data:
            response = pickle.loads(response_data)
            print(f"任务提交成功: {response}")
            return response
        else:
            print("收到空响应")
            return None

    except socket.timeout:
        print("连接或接收超时")
        return None
    except ConnectionRefusedError:
        print(f"连接被拒绝，请确保卫星节点正在运行")
        return None
    except Exception as e:
        print(f"任务提交失败: {e}")
        return None
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    # 创建测试任务
    task = {
        'task_id': 'urban_detection_001',
        'model_type': 'alex_net',
        'input_data': torch.rand(1, 3, 1024, 1024),  # 🎯 大幅增大到高分辨率
        'max_latency': 30000,  # 🎯 相应增加时限到30秒
        'priority': 'high'
    }

    print("开始提交任务...")
    # 向遥感卫星提交任务（使用发现服务端口）
    result = submit_task(task, '127.0.0.1', 10001)

    if result:
        print("\n=== 任务执行结果 ===")
        print(f"状态: {result.get('status', 'unknown')}")
        print(f"总时延: {result.get('total_latency', 0):.2f}ms")
        print(f"是否成功: {result.get('success', False)}")
    else:
        print("任务执行失败")