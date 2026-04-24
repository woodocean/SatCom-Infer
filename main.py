import argparse
import time
import json
import logging
from datetime import datetime
from core.node import ComputeNode
from core.router_qos import RouterQoSClient
import sys
import os
import torch
import random  # 用于生成多样化任务
import paramiko

sys.path.append(os.getcwd())

# --- 全局 SSH 连接池，避免频繁握手导致的性能损耗 ---
class SSHSessionPool:
    def __init__(self):
        self.clients = {}  # {host: ssh_client}
        self.sftps = {}    # {host: sftp_client}

    def get_ssh(self, host, user="nvidia", pw="nvidia"):
        if host in self.clients:
            transport = self.clients[host].get_transport()
            if transport and transport.is_active():
                return self.clients[host]
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=host, port=22, username=user, password=pw, timeout=4)
            self.clients[host] = ssh
            return ssh
        except Exception as e:
            print(f"[POOL] 无法建立到 {host} 的 SSH 连接: {e}")
            return None

    def get_sftp(self, host, user="nvidia", pw="nvidia"):
        if host in self.sftps:
            try:
                self.sftps[host].stat('.') 
                return self.sftps[host]
            except:
                del self.sftps[host]
        
        ssh = self.get_ssh(host, user, pw)
        if ssh:
            try:
                sftp = ssh.open_sftp()
                self.sftps[host] = sftp
                return sftp
            except Exception as e:
                print(f"[POOL] 创建 SFTP 失败 {host}: {e}")
        return None

    def close_all(self):
        for sftp in list(self.sftps.values()):
            try:
                sftp.close()
            except Exception:
                pass
        self.sftps.clear()

        for ssh in list(self.clients.values()):
            try:
                ssh.close()
            except Exception:
                pass
        self.clients.clear()

GLOBAL_POOL = SSHSessionPool()

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def _sync_config_to_jetsons(config, config_path):
    """使用长连接池同步 network_config.json 到所有 Jetson。"""
    jetson_ips = sorted({
        info.get('ip')
        for _, info in config.get('nodes', {}).items()
        if 'jetson' in str(info.get('device', '')).lower() and info.get('ip')
    })

    if not jetson_ips:
        return

    cfg_text = json.dumps(config, indent=2)
    remote_candidates = [
        "/home/nvidia/satinfer/SatCom-Infer/config/network_config.json",
        "/home/nvidia/satinfer/SatCom-Infer/config/network_config.json",
    ]

    for host in jetson_ips:
        sftp = GLOBAL_POOL.get_sftp(host)
        if not sftp:
            print(f"[SYNC] 无法获取 Jetson {host} 的长连接，跳过同步")
            continue
            
        try:
            # 自动探测路径（仅在第一次或路径丢失时做，此处简化为直接写入探测到的路径）
            # 实际生产中可以缓存 target_path 以进一步提速
            ssh = GLOBAL_POOL.get_ssh(host)
            if not ssh:
                print(f"[SYNC] 无法获取 Jetson {host} 的 SSH 通道，跳过同步")
                continue
            target_path = remote_candidates[0]
            for candidate in remote_candidates:
                _, stdout, _ = ssh.exec_command(f"test -f {candidate} && echo exists || echo missing")
                if stdout.read().decode("utf-8").strip() == "exists":
                    target_path = candidate
                    break

            with sftp.file(target_path, 'w') as f:
                f.write(cfg_text)
            print(f"[SYNC] [POOL] 已通过长连接同步配置到 {host}")
        except Exception as e:
            print(f"[SYNC] 同步到 {host} 失败: {e}")

def update_network_topology(config_path, qos_client=None):
    try:
        config = load_config(config_path)

        # 同步刷新节点算力（单位：TFLOPS）
        for node_id, node_info in config.get('nodes', {}).items():
            hw = node_info.setdefault('hardware', {})
            if node_id == 'GS':
                node_tflops = 300.0
            elif node_id.startswith('SAT'):
                node_tflops = round(random.uniform(0.5, 10.0), 3)
            else:
                node_tflops = 0.0

            # 新字段 + 兼容旧字段（旧字段名仅用于兼容历史代码）
            hw['compute_speed_tflops'] = node_tflops
            hw['compute_speed_gflops_per_ms'] = node_tflops
        
        # --- 路由器长连接优化 ---
        should_close_qos = False
        if qos_client is None:
            # 只有在 main 外部单独调用或没传长连接时，才建立临时连接
            router_ip = "192.168.10.1"
            router_user = "root"
            router_password = "wslhy110"
            qos_client = RouterQoSClient(router_ip, router_user, router_password, ssh_timeout=3)
            should_close_qos = True
        
        for link_name, info in config['links'].items():
            # 1. 产生新的带宽与时延
            if "GS" in link_name: 
                new_bw = random.randint(50, 200)
                new_delay = random.uniform(1.0, 2.0)
            else:
                new_bw = random.randint(1000, 20000)
                new_delay = random.uniform(2.0, 5.0)
                
            info['bandwidth_mbps'] = new_bw
            info['propagation_delay_ms'] = round(new_delay, 2)
            
            # 2. 解析设备 IP
            dst_node_id = link_name.split("_to_")[-1]
            if dst_node_id in config['nodes']:
                dst_ip = config['nodes'][dst_node_id]['ip']
                
                # # 只有非本机的远端路由才去下发硬件操作
                # if not dst_ip.startswith("127.0.0.1") and not dst_ip.startswith("localhost"):
                #     # 使用传入的或新建的长连接执行命令
                #     qos_client.set_root_delay(dst_ip, delay_ms=int(new_delay))

        if should_close_qos:
            qos_client.close()
            
        tmp_path = config_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # =================【加入写冲突重试机制】=================
        for _ in range(10):  # 最多重试10次
            try:
                os.replace(tmp_path, config_path)
                break  # 替换成功，跳出循环
            except PermissionError:
                # 在Windows上，如果恰好别的节点在读，会报 PermissionError，稍微等几毫秒
                time.sleep(0.02)
        # ========================================================

        # 将最新拓扑配置实时下发到 Jetson，保证其本地读取到与 PC 一致的参数。
        _sync_config_to_jetsons(config, config_path)
        
    except Exception as e:
        print(f"更新带宽配置失败: {e}")


def _build_scheduler(net_config_path):
    """按当前网络配置创建调度器实例。"""
    from core.scheduler import Scheduler
    return Scheduler(
        net_config_path=net_config_path,
        pc_profiles_path="config/dnn_profiles_database_pc.json",
        jetson_profiles_path="config/dnn_profiles_database_jetson.json"
    )


def _pick_task_profile(task_index, model_pool, batch_pool, res_pool):
    """为当前任务挑选模型与输入规格，保持与历史逻辑一致。"""
    model_idx = (task_index // 10) % len(model_pool)
    chosen_model = model_pool[model_idx]
    chosen_bs = random.choice(batch_pool)
    chosen_res = random.choice(res_pool[chosen_model])
    return chosen_model, chosen_bs, chosen_res


def _run_single_rs_task_iteration(
    node,
    net_config_path,
    qos_client,
    task_index,
    model_pool,
    batch_pool,
    res_pool,
    run_id,
    exp_type,
    exp_mode,
):
    """执行 RS 单个任务迭代：更新环境、生成计划、构造输入。"""
    task_id = f"Task_{task_index:03d}"

    # 步骤 A: 环境动态演进
    update_network_topology(net_config_path, qos_client=qos_client)
    time.sleep(0.5)

    # 步骤 B: 调度器感知更新
    scheduler = _build_scheduler(net_config_path)

    # 任务采样与输入构造
    chosen_model, chosen_bs, chosen_res = _pick_task_profile(
        task_index, model_pool, batch_pool, res_pool
    )

    print(
        f"\n[任务生成] ==> {task_id} | 模型: {chosen_model} | "
        f"Batch: {chosen_bs} | 尺寸: {chosen_res[0]}x{chosen_res[1]}"
    )

    # 调度器内部测算：hybrid/theory 写理论结果，physical 仅求解计划不落理论文件。
    plans = scheduler.generate_task_and_schedule(
        task_id=task_id,
        model_name=chosen_model,
        batch_size=chosen_bs,
        target_h=chosen_res[0],
        target_w=chosen_res[1],
        run_id=run_id,
        exp_type=exp_type,
        mode="theory",
        standardized_csv_file="results_long.csv",
        persist_theory=(exp_mode in ("hybrid", "theory")),
    )

    # 保持与历史代码一致：预先构造真实任务输入
    fake_img = torch.randn(chosen_bs, 3, chosen_res[0], chosen_res[1])

    # theory-only 不触发实物下发。
    if exp_mode == "theory":
        return {
            "task_id": task_id,
            "model_name": chosen_model,
            "batch_size": chosen_bs,
            "resolution": chosen_res,
            "plans": plans,
            "input_tensor": fake_img,
        }

    # --- 步骤 C: 顺序分发给管道，一次执行一个算法进行公平仿真 ---
    for alg, plan in plans.items():
        if plan is None:
            print(f"  [RS] 🛑 算法 [{alg}] 当前由于硬件约束无解，直接跳过管道仿真。")
            continue
    
        if "simulation_paths" in scheduler.net_config and "pipeline" in scheduler.net_config["simulation_paths"]:
            ordered_route = scheduler.net_config["simulation_paths"]["pipeline"][1:]
        else:
            ordered_route = [n["id"] for n in scheduler.net_config["nodes"] if "RS" not in n["id"]]
    
        if hasattr(node, "task_ack_event"):
            if not node.task_ack_event.is_set():
                print(f"  [RS] ⏳ 管道占用中: 正在等待前序算法应答...")
    
            node.task_ack_event.wait()
            node.task_ack_event.clear()
    
        time.sleep(1.0)
    
        print(f"  [RS] 🚀 正在向网络下发任务: [{task_id}] | 驱动策略: {alg}")
    
        rs_payload = {
            'mode': 'PMP',
            'task_id': task_id,
            'algorithm': alg,
            'model_name': chosen_model,
            'accumulated_latency': 0.0,
            'tensor': fake_img,
            'batch': chosen_bs,
            'route': ordered_route,
            'layer_plan': plan,
            'exp_meta': {
                'run_id': run_id,
                'exp_type': exp_type,
                'mode': 'physical',
                'model_name': chosen_model,
                'batch_size': chosen_bs,
                'input_h': chosen_res[0],
                'input_w': chosen_res[1],
                'standardized_csv_file': 'results_long.csv',
            }
        }
    
        node.handle_message({
            'type': 'NEW_TASK',
            'src': 'system_trigger',
            'payload': rs_payload
        })

    # 预留返回值，便于后续实验编排层复用
    return {
        "task_id": task_id,
        "model_name": chosen_model,
        "batch_size": chosen_bs,
        "resolution": chosen_res,
        "plans": plans,
        "input_tensor": fake_img,
    }


def run_rs_pmp_dynamic_bandwidth_experiment(
    node,
    net_config_path,
    num_tasks=50,
    exp_mode="hybrid",
    run_id=None,
    exp_type="algo_effectiveness",
):
    """RS 侧实验入口：保留原有行为，后续可被独立实验脚本直接调用。"""
    router_ip = "192.168.10.1"
    router_user = "root"
    router_password = "wslhy110"
    global_qos_client = RouterQoSClient(router_ip, router_user, router_password, ssh_timeout=3)

    print("\n" + "="*50)
    print("--- 发起 PMP 动态带宽 六大算法 对比实验 (100个任务) ---")
    print("="*50)

    # 异构任务池：通过控制分辨率和 batch 模拟真实的复杂网络负荷
    model_pool = ["vit_huge", "vgg19", "yolov5", "swin_base", "resnet101"]
    batch_pool = [16, 32, 64]
    res_pool = {
        "yolov5": [(640, 640)],
        "resnet101": [(224, 224)],
        "vgg19": [(224, 224)],
        "swin_base": [(224, 224)],
        "vit_huge": [(224, 224)],
    }

    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        for i in range(0, num_tasks):
            _run_single_rs_task_iteration(
                node=node,
                net_config_path=net_config_path,
                qos_client=global_qos_client,
                task_index=i,
                model_pool=model_pool,
                batch_pool=batch_pool,
                res_pool=res_pool,
                run_id=run_id,
                exp_type=exp_type,
                exp_mode=exp_mode,
            )
    finally:
        global_qos_client.close()

    print("\n🎉 [RS] 第一阶段：100组对比实验所有任务已经全部轰炸分发完毕。")
    print("🎉 [RS] 你可以前往目录查看自动生成的 `experiment_results.csv`。")

def main():
    parser = argparse.ArgumentParser(description="Distributed Satellite Node")
    parser.add_argument('--id', type=str, required=True, help="节点ID，如 Sat_1, RS, GS")
    parser.add_argument('--exp-mode', type=str, default='hybrid', choices=['hybrid', 'theory', 'physical'], help="RS 实验模式")
    parser.add_argument('--run-id', type=str, default=None, help="实验批次ID，不传则自动生成")
    parser.add_argument('--exp-type', type=str, default='algo_effectiveness', help="实验类型标签")
    args = parser.parse_args()

    # 1. 直接固定加载网络拓扑配置
    net_config_path = 'config/network_config.json'
    try:
        net_config = load_config(net_config_path)
    except FileNotFoundError:
        print(f"Error: 找不到网络配置文件 {net_config_path}")
        return
    
    if args.id not in net_config['nodes']:
        print(f"Error: 节点 {args.id} 不在网络配置中！")
        return

    my_net_info = net_config['nodes'][args.id]

    print(f"--- 正在启动节点: {args.id} ---")

    # ==========================================================
    # 2. 一键实例化 ComputeNode 
    # ==========================================================
    node = ComputeNode(
        node_id=args.id, 
        ip=my_net_info['ip'],               
        port=my_net_info['port'],           
        role=my_net_info.get('role', 'Sat')
    )

    # 3. 加载计算引擎 (RS 节点本身不需要真跑模型)
    if "RS" not in args.id:
        print(f"[{args.id}] 检测为工作节点/地面站，初始化推理引擎...")
        node.load_model()                                        
    else:
        print(f"[{args.id}] 检测为遥感卫星RS，仅作为任务源。")

    # 4. 查表组网
    print(f"[{args.id}] 正在构建邻居路由表...")
    neighbors_parsed = []
    if "neighbors" in my_net_info:
        for neighbor_id in my_net_info["neighbors"]:
            if neighbor_id in net_config["nodes"]:
                n_info = net_config["nodes"][neighbor_id]
                neighbors_parsed.append((neighbor_id, n_info["ip"], n_info["port"]))
            else:
                print(f"  [警告] 邻居 {neighbor_id} 未在配置中定义！")
    
    node.join_network(neighbors_parsed)  # 批量注册邻居

    # 5. 启动网络监听
    node.start()

    # ==========================================================
    # 6. 任务触发逻辑 (仅遥感卫星 RS 执行)
    # ==========================================================
    if args.id == "RS":
        run_rs_pmp_dynamic_bandwidth_experiment(
            node=node,
            net_config_path=net_config_path,
            num_tasks=50,
            exp_mode=args.exp_mode,
            run_id=args.run_id,
            exp_type=args.exp_type,
        )

    # ========== 7. 保持进程循环以维持监听 ==========
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{args.id}] 正在安全终止节点...")
        node.stop()
    finally:
        GLOBAL_POOL.close_all()
        
if __name__ == "__main__":
    # 为了避免输出过于眼花缭乱，这里将日志限制在 INFO 级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    main()