import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import paramiko
import torch

from core.node import ComputeNode
from core.router_qos import RouterQoSClient

sys.path.append(os.getcwd())


class SSHSessionPool:
    """全局 SSH 连接池，避免频繁握手导致性能损耗。"""

    def __init__(self):
        self.clients = {}
        self.sftps = {}

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
            except Exception:
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
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _sync_config_to_jetsons(config):
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
    """动态更新带宽/算力并同步配置。"""
    try:
        config = load_config(config_path)

        for node_id, node_info in config.get('nodes', {}).items():
            hw = node_info.setdefault('hardware', {})
            if node_id == 'GS':
                node_tflops = 300.0
            elif node_id.startswith('SAT'):
                node_tflops = round(random.uniform(0.5, 10.0), 3)
            else:
                node_tflops = 0.0

            hw['compute_speed_tflops'] = node_tflops
            hw['compute_speed_gflops_per_ms'] = node_tflops

        should_close_qos = False
        if qos_client is None:
            qos_client = RouterQoSClient("192.168.10.1", "root", "wslhy110", ssh_timeout=3)
            should_close_qos = True

        for link_name, info in config['links'].items():
            if "GS" in link_name:
                new_bw = random.randint(50, 200)
                new_delay = random.uniform(1.0, 2.0)
            else:
                new_bw = random.randint(1000, 20000)
                new_delay = random.uniform(2.0, 5.0)

            info['bandwidth_mbps'] = new_bw
            info['propagation_delay_ms'] = round(new_delay, 2)

        if should_close_qos:
            qos_client.close()

        tmp_path = config_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        for _ in range(10):
            try:
                os.replace(tmp_path, config_path)
                break
            except PermissionError:
                time.sleep(0.02)

        _sync_config_to_jetsons(config)

    except Exception as e:
        print(f"更新带宽配置失败: {e}")


def _build_scheduler(net_config_path):
    from core.scheduler import Scheduler

    return Scheduler(
        net_config_path=net_config_path,
        pc_profiles_path="config/dnn_profiles_database_pc.json",
        jetson_profiles_path="config/dnn_profiles_database_jetson.json"
    )


def _pick_task_profile(task_index, model_pool, batch_pool, res_pool):
    model_idx = (task_index // 10) % len(model_pool)
    chosen_model = model_pool[model_idx]
    chosen_bs = random.choice(batch_pool)
    chosen_res = random.choice(res_pool[chosen_model])
    return chosen_model, chosen_bs, chosen_res


def _dispatch_one_task_to_rs(rs_node, scheduler, task_id, chosen_model, chosen_bs, chosen_res, plans, run_id, exp_type):
    """将单任务的多算法计划依次送入 RS。"""
    fake_img = torch.randn(chosen_bs, 3, chosen_res[0], chosen_res[1])

    for alg, plan in plans.items():
        if plan is None:
            print(f"  [RS] 🛑 算法 [{alg}] 当前由于硬件约束无解，直接跳过管道仿真。")
            continue

        if "simulation_paths" in scheduler.net_config and "pipeline" in scheduler.net_config["simulation_paths"]:
            ordered_route = scheduler.net_config["simulation_paths"]["pipeline"][1:]
        else:
            ordered_route = [n["id"] for n in scheduler.net_config["nodes"] if "RS" not in n["id"]]

        if hasattr(rs_node, "task_ack_event"):
            if not rs_node.task_ack_event.is_set():
                print("  [RS] ⏳ 管道占用中: 正在等待前序算法应答...")
            rs_node.task_ack_event.wait()
            rs_node.task_ack_event.clear()

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

        rs_node.handle_message({
            'type': 'NEW_TASK',
            'src': 'experiment_runner',
            'payload': rs_payload
        })


def run_experiment(rs_node, net_config_path, num_tasks, exp_mode, run_id, exp_type):
    router_client = RouterQoSClient("192.168.10.1", "root", "wslhy110", ssh_timeout=3)

    model_pool = ["vit_huge", "vgg19", "yolov5", "swin_base", "resnet101"]
    batch_pool = [16, 32, 64]
    res_pool = {
        "yolov5": [(640, 640)],
        "resnet101": [(224, 224)],
        "vgg19": [(224, 224)],
        "swin_base": [(224, 224)],
        "vit_huge": [(224, 224)],
    }

    print("\n" + "=" * 50)
    print(f"--- 实验开始: mode={exp_mode}, run_id={run_id}, exp_type={exp_type} ---")
    print("=" * 50)

    try:
        for i in range(num_tasks):
            task_id = f"Task_{i:03d}"
            update_network_topology(net_config_path, qos_client=router_client)
            time.sleep(0.5)

            scheduler = _build_scheduler(net_config_path)
            chosen_model, chosen_bs, chosen_res = _pick_task_profile(i, model_pool, batch_pool, res_pool)
            print(
                f"\n[任务生成] ==> {task_id} | 模型: {chosen_model} | "
                f"Batch: {chosen_bs} | 尺寸: {chosen_res[0]}x{chosen_res[1]}"
            )

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

            if exp_mode in ("hybrid", "physical"):
                _dispatch_one_task_to_rs(
                    rs_node=rs_node,
                    scheduler=scheduler,
                    task_id=task_id,
                    chosen_model=chosen_model,
                    chosen_bs=chosen_bs,
                    chosen_res=chosen_res,
                    plans=plans,
                    run_id=run_id,
                    exp_type=exp_type,
                )
    finally:
        router_client.close()



def start_rs_node(net_config_path, rs_id):
    config = load_config(net_config_path)
    if rs_id not in config.get('nodes', {}):
        raise ValueError(f"节点 {rs_id} 不在网络配置中")

    rs_info = config['nodes'][rs_id]
    rs_node = ComputeNode(
        node_id=rs_id,
        ip=rs_info['ip'],
        port=rs_info['port'],
        role=rs_info.get('role', 'RS')
    )

    neighbors_parsed = []
    for neighbor_id in rs_info.get("neighbors", []):
        if neighbor_id in config["nodes"]:
            n_info = config["nodes"][neighbor_id]
            neighbors_parsed.append((neighbor_id, n_info["ip"], n_info["port"]))

    rs_node.join_network(neighbors_parsed)
    rs_node.start()
    print(f"[{rs_id}] 已启动并监听，准备接收编排任务")
    return rs_node



def main():
    parser = argparse.ArgumentParser(description="Experiment Runner (RS orchestration entry)")
    parser.add_argument('--config', type=str, default='config/network_config.json', help='网络配置路径')
    parser.add_argument('--rs-id', type=str, default='RS', help='RS 节点ID')
    parser.add_argument('--num-tasks', type=int, default=50, help='任务数量')
    parser.add_argument('--exp-mode', type=str, default='hybrid', choices=['hybrid', 'theory', 'physical'], help='实验模式')
    parser.add_argument('--run-id', type=str, default=None, help='实验批次ID')
    parser.add_argument('--exp-type', type=str, default='algo_effectiveness', help='实验类型标签')
    args = parser.parse_args()

    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rs_node = None
    try:
        rs_node = start_rs_node(args.config, args.rs_id)
        run_experiment(
            rs_node=rs_node,
            net_config_path=args.config,
            num_tasks=args.num_tasks,
            exp_mode=args.exp_mode,
            run_id=run_id,
            exp_type=args.exp_type,
        )
    finally:
        if rs_node is not None:
            rs_node.stop()
        GLOBAL_POOL.close_all()


if __name__ == '__main__':
    main()
