import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime

import paramiko
import numpy as np
import torch

from core.node import ComputeNode
from core.experiment_archive import (
    append_experiment_index,
    build_artifact_stem,
    create_run_archive,
    export_run_rows,
    now_stamp,
    update_run_metadata,
)

sys.path.append(os.getcwd())


class SSHSessionPool:
    """Lightweight SSH/SFTP pool to avoid reconnecting on every sync."""

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
            print(f"[POOL] Failed to create SSH connection to {host}: {e}")
            return None

    def get_sftp(self, host, user="nvidia", pw="nvidia"):
        if host in self.sftps:
            try:
                self.sftps[host].stat(".")
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
                print(f"[POOL] Failed to open SFTP for {host}: {e}")
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

DEFAULT_MODEL_POOL = ["vit_huge", "vgg19", "yolov5", "swin_base", "resnet101"]
DEFAULT_BATCH_POOL = [16, 32, 64]
DEFAULT_RES_POOL = {
    "yolov5": [(640, 640)],
    "resnet101": [(224, 224)],
    "vgg19": [(224, 224)],
    "swin_base": [(224, 224)],
    "vit_huge": [(224, 224)],
}
DEFAULT_ISL_SWEEP_VALUES = [round(float(v), 6) for v in np.linspace(500, 20000, 20)]
DEFAULT_GSL_SWEEP_VALUES = [round(float(v), 6) for v in np.linspace(20, 200, 20)]
DEFAULT_NODE_COUNT_SWEEP_VALUES = [1, 2, 3, 4, 5]
DEFAULT_PRESET_FILE = "config/experiment_presets.json"
DEFAULT_PRESETS = {
    "algo": {
        "exp_type": "algo_effectiveness",
        "exp_mode": "theory",
        "num_tasks": 50,
    },
    "isl": {
        "exp_type": "isl_bandwidth_sensitivity",
        "exp_mode": "theory",
        "sweep_start": 500,
        "sweep_stop": 20000,
        "sweep_points": 20,
        "repeat_per_point": 10,
    },
    "gsl": {
        "exp_type": "gsl_bandwidth_sensitivity",
        "exp_mode": "theory",
        "sweep_start": 20,
        "sweep_stop": 200,
        "sweep_points": 20,
        "repeat_per_point": 10,
    },
    "nodes": {
        "exp_type": "node_count_sensitivity",
        "exp_mode": "theory",
        "sweep_values": "1,2,3,4,5",
        "repeat_per_point": 10,
    },
    "energy": {
        "exp_type": "energy_comparison",
        "exp_mode": "theory",
        "num_tasks": 30,
        "fixed_model": "yolov5",
        "fixed_batch_size": 32,
        "fixed_input_h": 640,
        "fixed_input_w": 640,
    },
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clone_config(base_config):
    return copy.deepcopy(base_config)


def _stable_int_seed(*parts):
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _seed_everything(seed):
    seed = int(seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_config_atomic(config_path, config):
    tmp_path = config_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for _ in range(10):
        try:
            os.replace(tmp_path, config_path)
            return
        except PermissionError:
            time.sleep(0.02)

    os.replace(tmp_path, config_path)


def _default_bandwidth_sweep_values(exp_type):
    if exp_type == "isl_bandwidth_sensitivity":
        return DEFAULT_ISL_SWEEP_VALUES
    if exp_type == "gsl_bandwidth_sensitivity":
        return DEFAULT_GSL_SWEEP_VALUES
    raise ValueError(f"Unsupported bandwidth sensitivity exp_type: {exp_type}")


def _default_sweep_values(exp_type):
    if exp_type == "node_count_sensitivity":
        return DEFAULT_NODE_COUNT_SWEEP_VALUES
    return _default_bandwidth_sweep_values(exp_type)


def _parse_bandwidth_sweep_values(raw_values):
    if not raw_values:
        return None

    values = []
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))

    if not values:
        raise ValueError("Empty sweep values after parsing --sweep-values")
    return values


def _build_sweep_values(raw_values=None, sweep_start=None, sweep_stop=None, sweep_points=None):
    explicit_values = _parse_bandwidth_sweep_values(raw_values)
    if explicit_values is not None:
        return explicit_values

    range_args = [sweep_start, sweep_stop, sweep_points]
    if all(value is None for value in range_args):
        return None
    if any(value is None for value in range_args):
        raise ValueError("--sweep-start, --sweep-stop and --sweep-points must be provided together")

    points = int(sweep_points)
    if points <= 0:
        raise ValueError("--sweep-points must be greater than 0")
    if points == 1:
        return [float(sweep_start)]

    values = np.linspace(float(sweep_start), float(sweep_stop), points)
    return [round(float(value), 6) for value in values]


def _build_run_metadata(args, run_id, sweep_values, started_at, started_at_compact):
    return {
        "run_id": run_id,
        "started_at": started_at,
        "started_at_compact": started_at_compact,
        "status": "running",
        "command": " ".join(sys.argv),
        "config": args.config,
        "preset": args.preset,
        "preset_file": args.preset_file,
        "seed": args.seed,
        "exp_type": args.exp_type,
        "exp_mode": args.exp_mode,
        "num_tasks": args.num_tasks,
        "sweep_values": sweep_values,
        "sweep_start": args.sweep_start,
        "sweep_stop": args.sweep_stop,
        "sweep_points": args.sweep_points,
        "fixed_model": args.fixed_model,
        "fixed_batch_size": args.fixed_batch_size,
        "fixed_input_h": args.fixed_input_h,
        "fixed_input_w": args.fixed_input_w,
        "repeat_per_point": args.repeat_per_point,
    }


def _load_presets(preset_file):
    presets = _clone_config(DEFAULT_PRESETS)
    if not preset_file or not os.path.exists(preset_file):
        return presets

    with open(preset_file, "r", encoding="utf-8") as f:
        user_presets = json.load(f)

    if not isinstance(user_presets, dict):
        raise ValueError(f"Preset file must be a JSON object: {preset_file}")

    for name, values in user_presets.items():
        if not isinstance(values, dict):
            raise ValueError(f"Preset '{name}' must be a JSON object")
        presets[name] = values
    return presets


def _collect_explicit_cli_args(raw_args):
    explicit = set()
    option_to_dest = {
        "--config": "config",
        "--rs-id": "rs_id",
        "--num-tasks": "num_tasks",
        "--exp-mode": "exp_mode",
        "--run-id": "run_id",
        "--exp-type": "exp_type",
        "--sweep-values": "sweep_values",
        "--sweep-start": "sweep_start",
        "--sweep-stop": "sweep_stop",
        "--sweep-points": "sweep_points",
        "--fixed-model": "fixed_model",
        "--fixed-batch-size": "fixed_batch_size",
        "--fixed-input-h": "fixed_input_h",
        "--fixed-input-w": "fixed_input_w",
        "--repeat-per-point": "repeat_per_point",
        "--preset-file": "preset_file",
        "--seed": "seed",
    }

    for raw_arg in raw_args:
        option = raw_arg.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def _apply_preset(args, explicit_cli_args=None):
    if not args.preset:
        return args
    explicit_cli_args = explicit_cli_args or set()

    presets = _load_presets(args.preset_file)
    if args.preset not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown preset '{args.preset}'. Available presets: {available}")

    range_keys = {"sweep_start", "sweep_stop", "sweep_points"}
    explicit_has_range = bool(range_keys & explicit_cli_args)
    explicit_has_values = "sweep_values" in explicit_cli_args

    for key, value in presets[args.preset].items():
        if key == "sweep_values" and explicit_has_range:
            continue
        if key in range_keys and explicit_has_values:
            continue
        if key not in explicit_cli_args:
            setattr(args, key, value)

    has_range_sweep = (
        args.sweep_start is not None
        or args.sweep_stop is not None
        or args.sweep_points is not None
    )
    if (
        args.exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity", "node_count_sensitivity")
        and args.sweep_values is None
        and not has_range_sweep
    ):
        args.sweep_values = ",".join(str(v) for v in _default_sweep_values(args.exp_type))

    return args


def _sync_config_to_jetsons(config):
    """Synchronize the latest network_config.json to all Jetson nodes."""
    jetson_ips = sorted(
        {
            info.get("ip")
            for _, info in config.get("nodes", {}).items()
            if "jetson" in str(info.get("device", "")).lower() and info.get("ip")
        }
    )

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
            print(f"[SYNC] Skip Jetson {host}: cannot open SFTP")
            continue

        try:
            ssh = GLOBAL_POOL.get_ssh(host)
            if not ssh:
                print(f"[SYNC] Skip Jetson {host}: cannot open SSH")
                continue

            target_path = remote_candidates[0]
            for candidate in remote_candidates:
                _, stdout, _ = ssh.exec_command(f"test -f {candidate} && echo exists || echo missing")
                if stdout.read().decode("utf-8").strip() == "exists":
                    target_path = candidate
                    break

            with sftp.file(target_path, "w") as f:
                f.write(cfg_text)
            print(f"[SYNC] Config synced to Jetson {host}")
        except Exception as e:
            print(f"[SYNC] Failed to sync config to {host}: {e}")


def _apply_bandwidth_scale(base_config, target, desired_avg_bw):
    """Scale only the chosen link class to the desired average bandwidth.

    This preserves the original heterogeneity pattern across hops while making
    the sweep value represent the class-level average bandwidth.
    """
    config = _clone_config(base_config)
    links = config.get("links", {})

    selected = []
    for link_name, info in links.items():
        link_type = "gsl" if "GS" in link_name else "isl"
        if link_type == target:
            selected.append((link_name, float(info.get("bandwidth_mbps", 0.0))))

    if not selected:
        raise ValueError(f"No {target.upper()} links found in config")

    base_avg = sum(bw for _, bw in selected) / len(selected)
    if base_avg <= 0:
        for link_name, _ in selected:
            links[link_name]["bandwidth_mbps"] = float(desired_avg_bw)
        return config

    scale = float(desired_avg_bw) / base_avg
    for link_name, old_bw in selected:
        links[link_name]["bandwidth_mbps"] = round(max(1.0, old_bw * scale), 4)
    return config


def _apply_random_resource_profile(base_config):
    config = _clone_config(base_config)
    for node_id, node_info in config.get("nodes", {}).items():
        hw = node_info.setdefault("hardware", {})
        if node_id == "GS":
            node_tflops = 300.0
        elif node_id.startswith("SAT"):
            node_tflops = round(random.uniform(0.5, 10.0), 3)
        else:
            node_tflops = 0.0

        hw["compute_speed_tflops"] = node_tflops
        hw["compute_speed_gflops_per_ms"] = node_tflops

    for link_name, info in config.get("links", {}).items():
        if "GS" in link_name:
            new_bw = random.randint(50, 200)
            new_delay = random.uniform(1.0, 2.0)
        else:
            new_bw = random.randint(1000, 20000)
            new_delay = random.uniform(2.0, 5.0)

        info["bandwidth_mbps"] = new_bw
        info["propagation_delay_ms"] = round(new_delay, 2)
    return config


def _link_class(link_name):
    return "gsl" if "GS" in link_name else "isl"


def _link_class_defaults(links):
    stats = {
        "isl": {"bandwidths": [], "delays": []},
        "gsl": {"bandwidths": [], "delays": []},
    }
    for link_name, info in links.items():
        cls = _link_class(link_name)
        stats[cls]["bandwidths"].append(float(info.get("bandwidth_mbps", 100.0)))
        stats[cls]["delays"].append(float(info.get("propagation_delay_ms", 1.0)))

    defaults = {}
    for cls, values in stats.items():
        bws = values["bandwidths"]
        delays = values["delays"]
        defaults[cls] = {
            "bandwidth_mbps": sum(bws) / len(bws) if bws else (150.0 if cls == "gsl" else 10000.0),
            "propagation_delay_ms": sum(delays) / len(delays) if delays else (1.5 if cls == "gsl" else 3.0),
        }
    return defaults


def _find_existing_link(links, src, dst):
    forward_key = f"{src}_to_{dst}"
    backward_key = f"{dst}_to_{src}"
    if forward_key in links:
        return forward_key, links[forward_key]
    if backward_key in links:
        return backward_key, links[backward_key]
    return None, None


def _apply_pipeline_node_count(base_config, node_count):
    """Build a linear RS -> SAT-* -> GS pipeline for node-count sensitivity.

    Missing adjacent links are synthesized from the average bandwidth/delay of
    the same link class. This makes the theory experiment runnable today while
    keeping the generated topology explicit for future physical orchestration.
    """
    config = _clone_config(base_config)
    nodes = config.get("nodes", {})
    links = config.setdefault("links", {})

    sat_ids = sorted(node_id for node_id in nodes if node_id.startswith("SAT"))
    node_count = int(node_count)
    if node_count < 1:
        raise ValueError("node_count must be at least 1")
    if node_count > len(sat_ids):
        raise ValueError(f"node_count={node_count} exceeds available satellites: {len(sat_ids)}")

    selected_sats = sat_ids[:node_count]
    pipeline = ["RS"] + selected_sats + ["GS"]
    defaults = _link_class_defaults(links)

    for src, dst in zip(pipeline[:-1], pipeline[1:]):
        _, existing = _find_existing_link(links, src, dst)
        if existing is not None:
            continue

        link_name = f"{src}_to_{dst}"
        cls = _link_class(link_name)
        links[link_name] = {
            "bandwidth_mbps": round(float(defaults[cls]["bandwidth_mbps"]), 4),
            "propagation_delay_ms": round(float(defaults[cls]["propagation_delay_ms"]), 4),
            "synthetic_for_node_count_sweep": True,
        }

    for node_id, info in nodes.items():
        if node_id not in pipeline:
            info["neighbors"] = []
            continue

        idx = pipeline.index(node_id)
        neighbors = []
        if idx > 0:
            neighbors.append(pipeline[idx - 1])
        if idx < len(pipeline) - 1:
            neighbors.append(pipeline[idx + 1])
        info["neighbors"] = neighbors

    simulation_paths = config.setdefault("simulation_paths", {})
    simulation_paths["pipeline"] = pipeline
    simulation_paths["node_count_sweep"] = {
        "pipeline_node_count": node_count,
        "pipeline_hop_count": len(pipeline) - 1,
        "selected_satellites": selected_sats,
    }
    return config


def update_network_topology(
    config_path,
    qos_client=None,
    topology_mode="random",
    base_config=None,
    sweep_target=None,
    sweep_value=None,
    sync_remote=False,
):
    """Update the topology in-place and sync it to remote nodes.

    topology_mode:
        - random: preserve old behavior, randomize compute and link conditions
        - bandwidth_sweep: keep everything from base_config except the selected bandwidth class
        - node_count_sweep: keep resources fixed while changing simulation_paths.pipeline
    """
    try:
        config = _clone_config(base_config) if base_config is not None else load_config(config_path)

        if topology_mode == "random":
            config = _apply_random_resource_profile(config)

        elif topology_mode == "bandwidth_sweep":
            if sweep_target not in ("isl", "gsl"):
                raise ValueError(f"Invalid sweep_target: {sweep_target}")
            if sweep_value is None:
                raise ValueError("sweep_value is required for bandwidth_sweep mode")

            config = _apply_bandwidth_scale(config, sweep_target, sweep_value)

        elif topology_mode == "node_count_sweep":
            if sweep_value is None:
                raise ValueError("sweep_value is required for node_count_sweep mode")
            config = _apply_pipeline_node_count(config, int(round(float(sweep_value))))

        else:
            raise ValueError(f"Unsupported topology_mode: {topology_mode}")

        _write_config_atomic(config_path, config)
        if sync_remote:
            _sync_config_to_jetsons(config)
        return config
    except Exception as e:
        print(f"[TOPOLOGY] Failed to update network topology: {e}")
        return None


def _build_scheduler(net_config_path):
    from core.scheduler import Scheduler

    return Scheduler(
        net_config_path=net_config_path,
        pc_profiles_path="config/dnn_profiles_database_pc.json",
        jetson_profiles_path="config/dnn_profiles_database_jetson.json",
    )


def _pick_task_profile(task_index, model_pool, batch_pool, res_pool):
    model_idx = (task_index // 10) % len(model_pool)
    chosen_model = model_pool[model_idx]
    chosen_bs = random.choice(batch_pool)
    chosen_res = random.choice(res_pool[chosen_model])
    return chosen_model, chosen_bs, chosen_res


def _dispatch_one_task_to_rs(
    rs_node,
    scheduler,
    task_id,
    chosen_model,
    chosen_bs,
    chosen_res,
    plans,
    run_id,
    exp_type,
):
    """Send every algorithm plan of one task to RS in sequence."""
    fake_img = torch.randn(chosen_bs, 3, chosen_res[0], chosen_res[1])

    for alg, plan in plans.items():
        if plan is None:
            print(f"  [RS] Skip algorithm [{alg}] because no valid plan was produced")
            continue

        if "simulation_paths" in scheduler.net_config and "pipeline" in scheduler.net_config["simulation_paths"]:
            ordered_route = scheduler.net_config["simulation_paths"]["pipeline"][1:]
        else:
            ordered_route = [n["id"] for n in scheduler.net_config["nodes"] if "RS" not in n["id"]]

        if hasattr(rs_node, "task_ack_event"):
            if not rs_node.task_ack_event.is_set():
                print("  [RS] Channel busy, waiting for previous algorithm ack...")
            rs_node.task_ack_event.wait()
            rs_node.task_ack_event.clear()

        time.sleep(1.0)
        print(f"  [RS] Dispatching task [{task_id}] with algorithm: {alg}")

        rs_payload = {
            "mode": "PMP",
            "task_id": task_id,
            "algorithm": alg,
            "model_name": chosen_model,
            "accumulated_latency": 0.0,
            "tensor": fake_img,
            "batch": chosen_bs,
            "route": ordered_route,
            "layer_plan": plan,
            "exp_meta": {
                "run_id": run_id,
                "exp_type": exp_type,
                "mode": "physical",
                "model_name": chosen_model,
                "batch_size": chosen_bs,
                "input_h": chosen_res[0],
                "input_w": chosen_res[1],
                "standardized_csv_file": "results_long.csv",
            },
        }

        rs_node.handle_message(
            {
                "type": "NEW_TASK",
                "src": "experiment_runner",
                "payload": rs_payload,
            }
        )


def _run_algorithm_effectiveness_experiment(rs_node, net_config_path, num_tasks, exp_mode, run_id, exp_type):
    model_pool = DEFAULT_MODEL_POOL
    batch_pool = DEFAULT_BATCH_POOL
    res_pool = DEFAULT_RES_POOL

    print("\n" + "=" * 50)
    print(f"--- Experiment start: mode={exp_mode}, run_id={run_id}, exp_type={exp_type} ---")
    print("=" * 50)

    for i in range(num_tasks):
        task_id = f"Task_{i:03d}"
        seed = _stable_int_seed(run_id, exp_type, task_id)
        _seed_everything(seed)
        update_network_topology(net_config_path, topology_mode="random", sync_remote=(exp_mode in ("hybrid", "physical")))
        time.sleep(0.5)

        scheduler = _build_scheduler(net_config_path)
        chosen_model, chosen_bs, chosen_res = _pick_task_profile(i, model_pool, batch_pool, res_pool)
        print(
            f"\n[Task] {task_id} | model={chosen_model} | "
            f"batch={chosen_bs} | input={chosen_res[0]}x{chosen_res[1]}"
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

        if exp_mode in ("hybrid", "physical") and rs_node is not None:
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


def _run_bandwidth_sensitivity_experiment(
    net_config_path,
    run_id,
    exp_type,
    sweep_values,
    fixed_model,
    fixed_batch_size,
    fixed_input_h,
    fixed_input_w,
    repeat_per_point,
):
    sweep_target = "isl" if exp_type == "isl_bandwidth_sensitivity" else "gsl"
    base_config = load_config(net_config_path)
    repeat_per_point = max(1, int(repeat_per_point))

    print("\n" + "=" * 50)
    print(
        f"--- Bandwidth sweep start: run_id={run_id}, exp_type={exp_type}, "
        f"target={sweep_target}, repeat_per_point={repeat_per_point} ---"
    )
    print("=" * 50)

    deterministic_algorithms = ["LA-DP", "Greedy", "Uniform", "GS-Only"]
    stochastic_algorithms = ["Random", "GA"]

    for idx, bandwidth_mbps in enumerate(sweep_values):
        point_id = f"{exp_type}_{idx:03d}"
        print(f"\n[Sweep] {point_id} | target={sweep_target} | bandwidth={bandwidth_mbps} Mbps")
        update_network_topology(
            net_config_path,
            topology_mode="bandwidth_sweep",
            base_config=base_config,
            sweep_target=sweep_target,
            sweep_value=bandwidth_mbps,
            sync_remote=False,
        )

        det_task_id = f"{point_id}_det"
        _seed_everything(_stable_int_seed(run_id, exp_type, point_id, bandwidth_mbps, "deterministic"))
        scheduler = _build_scheduler(net_config_path)
        deterministic_plans = scheduler.generate_task_and_schedule(
            task_id=det_task_id,
            model_name=fixed_model,
            batch_size=fixed_batch_size,
            target_h=fixed_input_h,
            target_w=fixed_input_w,
            run_id=run_id,
            exp_type=exp_type,
            mode="theory",
            standardized_csv_file="results_long.csv",
            persist_theory=True,
            algorithm_names=deterministic_algorithms,
            persist_algorithms=deterministic_algorithms,
            return_full_plans=True,
        )
        gs_only_latency = deterministic_plans.get("GS-Only", {}).get("latency", None)
        gs_only_energy = deterministic_plans.get("GS-Only", {}).get("satellite_energy_j", None)
        print(f"[Sweep] Done {det_task_id}, algorithms={list(deterministic_plans.keys())}")

        for repeat_idx in range(repeat_per_point):
            task_id = f"{point_id}_stoch_rep{repeat_idx:02d}"
            seed = _stable_int_seed(run_id, exp_type, point_id, bandwidth_mbps, repeat_idx)
            _seed_everything(seed)

            scheduler = _build_scheduler(net_config_path)
            plans = scheduler.generate_task_and_schedule(
                task_id=task_id,
                model_name=fixed_model,
                batch_size=fixed_batch_size,
                target_h=fixed_input_h,
                target_w=fixed_input_w,
                run_id=run_id,
                exp_type=exp_type,
                mode="theory",
                standardized_csv_file="results_long.csv",
                persist_theory=True,
                algorithm_names=stochastic_algorithms,
                persist_algorithms=stochastic_algorithms,
                normalization_baseline_latency=gs_only_latency,
                normalization_baseline_energy=gs_only_energy,
                return_full_plans=True,
            )

            print(
                f"[Sweep] Done {task_id} ({repeat_idx + 1}/{repeat_per_point}), "
                f"algorithms={list(plans.keys())}"
            )


def _run_node_count_sensitivity_experiment(
    net_config_path,
    run_id,
    exp_type,
    sweep_values,
    fixed_model,
    fixed_batch_size,
    fixed_input_h,
    fixed_input_w,
    repeat_per_point,
):
    base_config = load_config(net_config_path)
    repeat_per_point = max(1, int(repeat_per_point))
    node_counts = [int(round(float(value))) for value in sweep_values]

    print("\n" + "=" * 50)
    print(
        f"--- Node-count sweep start: run_id={run_id}, exp_type={exp_type}, "
        f"node_counts={node_counts}, repeat_per_point={repeat_per_point} ---"
    )
    print("=" * 50)

    algorithms = ["LA-DP", "Greedy", "Uniform", "GS-Only", "Random", "GA"]

    for idx, node_count in enumerate(node_counts):
        point_id = f"{exp_type}_{idx:03d}"
        print(f"\n[Sweep] {point_id} | pipeline_node_count={node_count}")

        for repeat_idx in range(repeat_per_point):
            task_id = f"{point_id}_rep{repeat_idx:02d}"
            seed = _stable_int_seed(run_id, exp_type, point_id, node_count, repeat_idx)
            _seed_everything(seed)

            scenario_base_config = _apply_random_resource_profile(base_config)
            config = update_network_topology(
                net_config_path,
                topology_mode="node_count_sweep",
                base_config=scenario_base_config,
                sweep_value=node_count,
                sync_remote=False,
            )
            if config is None:
                raise RuntimeError(f"Failed to build node-count topology for node_count={node_count}")

            pipeline = config.get("simulation_paths", {}).get("pipeline", [])
            metadata_extra = {
                "pipeline_node_count": max(0, len(pipeline) - 2),
                "pipeline_hop_count": max(0, len(pipeline) - 1),
                "pipeline_path": "->".join(pipeline),
                "sweep_param": "pipeline_node_count",
                "sweep_value": node_count,
            }

            scheduler = _build_scheduler(net_config_path)
            plans = scheduler.generate_task_and_schedule(
                task_id=task_id,
                model_name=fixed_model,
                batch_size=fixed_batch_size,
                target_h=fixed_input_h,
                target_w=fixed_input_w,
                run_id=run_id,
                exp_type=exp_type,
                mode="theory",
                standardized_csv_file="results_long.csv",
                persist_theory=True,
                algorithm_names=algorithms,
                persist_algorithms=algorithms,
                return_full_plans=True,
                metadata_extra=metadata_extra,
            )

            print(
                f"[Sweep] Done {task_id} ({repeat_idx + 1}/{repeat_per_point}), "
                f"path={' -> '.join(pipeline)}, algorithms={list(plans.keys())}"
            )


def _run_energy_comparison_experiment(
    net_config_path,
    num_tasks,
    run_id,
    exp_type,
    fixed_model,
    fixed_batch_size,
    fixed_input_h,
    fixed_input_w,
):
    print("\n" + "=" * 50)
    print(
        f"--- Energy comparison start: run_id={run_id}, exp_type={exp_type}, "
        f"model={fixed_model}, batch={fixed_batch_size}, tasks={num_tasks} ---"
    )
    print("=" * 50)

    algorithms = ["LA-DP", "Greedy", "Uniform", "GS-Only", "Random", "GA"]

    for i in range(num_tasks):
        task_id = f"energy_{i:03d}"
        seed = _stable_int_seed(run_id, exp_type, task_id)
        _seed_everything(seed)
        update_network_topology(net_config_path, topology_mode="random", sync_remote=False)
        time.sleep(0.1)

        scheduler = _build_scheduler(net_config_path)
        scheduler.generate_task_and_schedule(
            task_id=task_id,
            model_name=fixed_model,
            batch_size=fixed_batch_size,
            target_h=fixed_input_h,
            target_w=fixed_input_w,
            run_id=run_id,
            exp_type=exp_type,
            mode="theory",
            standardized_csv_file="results_long.csv",
            persist_theory=True,
            algorithm_names=algorithms,
            persist_algorithms=algorithms,
            return_full_plans=True,
        )
        print(f"[Energy] Done {task_id} ({i + 1}/{num_tasks})")


def run_experiment(
    rs_node,
    net_config_path,
    num_tasks,
    exp_mode,
    run_id,
    exp_type,
    sweep_values=None,
    fixed_model="yolov5",
    fixed_batch_size=32,
    fixed_input_h=640,
    fixed_input_w=640,
    repeat_per_point=10,
):
    if exp_type == "algo_effectiveness":
        return _run_algorithm_effectiveness_experiment(rs_node, net_config_path, num_tasks, exp_mode, run_id, exp_type)

    if exp_type == "energy_comparison":
        if exp_mode != "theory":
            print(f"[WARN] exp_type={exp_type} is theory-only for now; forcing theory mode and ignoring exp_mode={exp_mode}")
        return _run_energy_comparison_experiment(
            net_config_path=net_config_path,
            num_tasks=num_tasks,
            run_id=run_id,
            exp_type=exp_type,
            fixed_model=fixed_model,
            fixed_batch_size=fixed_batch_size,
            fixed_input_h=fixed_input_h,
            fixed_input_w=fixed_input_w,
        )

    if exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"):
        if exp_mode != "theory":
            print(f"[WARN] exp_type={exp_type} is theory-only; forcing theory mode and ignoring exp_mode={exp_mode}")

        values = sweep_values or _default_bandwidth_sweep_values(exp_type)
        return _run_bandwidth_sensitivity_experiment(
            net_config_path=net_config_path,
            run_id=run_id,
            exp_type=exp_type,
            sweep_values=values,
            fixed_model=fixed_model,
            fixed_batch_size=fixed_batch_size,
            fixed_input_h=fixed_input_h,
            fixed_input_w=fixed_input_w,
            repeat_per_point=repeat_per_point,
        )

    if exp_type == "node_count_sensitivity":
        if exp_mode != "theory":
            print(f"[WARN] exp_type={exp_type} is theory-only for now; forcing theory mode and ignoring exp_mode={exp_mode}")

        values = sweep_values or DEFAULT_NODE_COUNT_SWEEP_VALUES
        return _run_node_count_sensitivity_experiment(
            net_config_path=net_config_path,
            run_id=run_id,
            exp_type=exp_type,
            sweep_values=values,
            fixed_model=fixed_model,
            fixed_batch_size=fixed_batch_size,
            fixed_input_h=fixed_input_h,
            fixed_input_w=fixed_input_w,
            repeat_per_point=repeat_per_point,
        )

    raise ValueError(f"Unsupported exp_type: {exp_type}")


def start_rs_node(net_config_path, rs_id):
    config = load_config(net_config_path)
    if rs_id not in config.get("nodes", {}):
        raise ValueError(f"Node {rs_id} not found in network config")

    rs_info = config["nodes"][rs_id]
    rs_node = ComputeNode(
        node_id=rs_id,
        ip=rs_info["ip"],
        port=rs_info["port"],
        role=rs_info.get("role", "RS"),
    )

    neighbors_parsed = []
    for neighbor_id in rs_info.get("neighbors", []):
        if neighbor_id in config["nodes"]:
            n_info = config["nodes"][neighbor_id]
            neighbors_parsed.append((neighbor_id, n_info["ip"], n_info["port"]))

    rs_node.join_network(neighbors_parsed)
    rs_node.start()
    print(f"[{rs_id}] Node started and waiting for tasks")
    return rs_node


def main():
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--config", type=str, default="config/network_config.json", help="Path to network config")
    parser.add_argument("--rs-id", type=str, default="RS", help="RS node ID")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks for algo effectiveness experiments")
    parser.add_argument(
        "--exp-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "theory", "physical"],
        help="Experiment mode",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Experiment batch ID")
    parser.add_argument(
        "--exp-type",
        type=str,
        default="algo_effectiveness",
        choices=[
            "algo_effectiveness",
            "energy_comparison",
            "isl_bandwidth_sensitivity",
            "gsl_bandwidth_sensitivity",
            "node_count_sensitivity",
        ],
        help="Experiment type",
    )
    parser.add_argument(
        "--sweep-values",
        type=str,
        default=None,
        help="Comma-separated bandwidth sweep values, e.g. 500,1000,2000",
    )
    parser.add_argument("--sweep-start", type=float, default=None, help="Sweep range start bandwidth in Mbps")
    parser.add_argument("--sweep-stop", type=float, default=None, help="Sweep range stop bandwidth in Mbps")
    parser.add_argument("--sweep-points", type=int, default=None, help="Number of evenly spaced sweep points")
    parser.add_argument("--fixed-model", type=str, default="yolov5", help="Fixed model for bandwidth sensitivity")
    parser.add_argument("--fixed-batch-size", type=int, default=32, help="Fixed batch size for bandwidth sensitivity")
    parser.add_argument("--fixed-input-h", type=int, default=640, help="Fixed input height for bandwidth sensitivity")
    parser.add_argument("--fixed-input-w", type=int, default=640, help="Fixed input width for bandwidth sensitivity")
    parser.add_argument(
        "--repeat-per-point",
        type=int,
        default=10,
        help="Repeated measurements per bandwidth point for sensitivity experiments",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Shortcut preset for common experiments",
    )
    parser.add_argument("--preset-file", type=str, default=DEFAULT_PRESET_FILE, help="JSON preset file path")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic runs")
    explicit_cli_args = _collect_explicit_cli_args(sys.argv[1:])
    args = parser.parse_args()

    args = _apply_preset(args, explicit_cli_args)

    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rs_node = None
    archive_dir = None
    metadata = None
    sweep_values = _build_sweep_values(
        raw_values=args.sweep_values,
        sweep_start=args.sweep_start,
        sweep_stop=args.sweep_stop,
        sweep_points=args.sweep_points,
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    started_at_compact = now_stamp()

    try:
        metadata = _build_run_metadata(args, run_id, sweep_values, started_at, started_at_compact)
        archive_dir = create_run_archive(
            metadata,
            snapshot_paths={
                "network_config": args.config,
                "experiment_presets": args.preset_file,
            },
        )
        print(f"[ARCHIVE] Run archive: {archive_dir}")

        _seed_everything(_stable_int_seed(args.seed, run_id, args.exp_type))
        needs_rs_node = args.exp_type == "algo_effectiveness" and args.exp_mode in ("hybrid", "physical")
        if needs_rs_node:
            rs_node = start_rs_node(args.config, args.rs_id)

        run_experiment(
            rs_node=rs_node,
            net_config_path=args.config,
            num_tasks=args.num_tasks,
            exp_mode=args.exp_mode,
            run_id=run_id,
            exp_type=args.exp_type,
            sweep_values=sweep_values,
            fixed_model=args.fixed_model,
            fixed_batch_size=args.fixed_batch_size,
            fixed_input_h=args.fixed_input_h,
            fixed_input_w=args.fixed_input_w,
            repeat_per_point=args.repeat_per_point,
        )
        if archive_dir is not None:
            stem = build_artifact_stem(metadata)
            data_path = archive_dir / "data" / f"results_long_{stem}.csv"
            rows = export_run_rows("results_long.csv", run_id, data_path)
            metadata = update_run_metadata(
                archive_dir,
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "exported_results_csv": str(data_path),
                    "exported_rows": rows,
                },
            )
            append_experiment_index(metadata, archive_dir)
            print(f"[ARCHIVE] Exported {rows} rows to {data_path}")
    except Exception:
        if archive_dir is not None:
            metadata = update_run_metadata(
                archive_dir,
                {
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            append_experiment_index(metadata, archive_dir)
        raise
    finally:
        if rs_node is not None:
            rs_node.stop()
        GLOBAL_POOL.close_all()


if __name__ == "__main__":
    main()
