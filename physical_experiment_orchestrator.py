"""One-click orchestration for PC + Jetson semi-physical experiments.

This launcher starts logical satellite nodes inside Jetson Docker containers,
starts local PC nodes when needed, synchronizes the active network_config.json,
and then runs the existing physical/hybrid experiment runner.

Current stable path:
    PMP physical/hybrid verification through experiments_runner.py.

Why RS is not launched as a separate process in run-pmp:
    experiments_runner.py creates an in-process RS node and dispatches tasks by
    calling handle_message directly. Launching another RS process would compete
    for the same UDP port and would not receive runner tasks.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List

import paramiko


DEFAULT_JETSONS = {
    "Jetson_1": {
        "host": "192.168.10.181",
        "user": "nvidia",
        "password": "nvidia",
        "repo": "/home/nvidia/satinfer/SatCom-Infer",
    },
    "Jetson_2": {
        "host": "192.168.10.178",
        "user": "nvidia",
        "password": "nvidia",
        "repo": "/home/nvidia/satinfer/SatCom-Infer",
    },
}


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-").lower()


class SshSession:
    def __init__(self, host: str, user: str, password: str):
        self.host = host
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=host, username=user, password=password, port=22, timeout=8)

    def run(self, command: str, timeout_s: int = 60) -> str:
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout_s)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            out = out + ("\n" if out else "") + err
        return out

    def put_text(self, remote_path: str, text: str) -> None:
        sftp = self.client.open_sftp()
        try:
            with sftp.file(remote_path, "w") as f:
                f.write(text)
        finally:
            sftp.close()

    def close(self) -> None:
        self.client.close()


class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_id = args.run_id or f"semi_physical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logs_dir = Path(args.logs_dir) / self.run_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.ssh: Dict[str, SshSession] = {}
        self.local_processes: List[subprocess.Popen] = []
        self.started_containers: List[tuple[str, str]] = []

    def close(self) -> None:
        for process in self.local_processes:
            if process.poll() is None:
                process.terminate()
        for process in self.local_processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        self.local_processes.clear()

        if self.args.keep_remote:
            return
        for device_name, container in self.started_containers:
            try:
                device = self._device_by_name(device_name)
                ssh = self._ssh_for_device(device_name, device)
                ssh.run(f"docker rm -f {container} >/dev/null 2>&1 || true", timeout_s=20)
            except Exception as exc:
                print(f"[ORCH] failed to stop {container} on {device_name}: {exc}")
        for session in self.ssh.values():
            session.close()
        self.ssh.clear()

    def _device_by_name(self, device_name: str) -> dict:
        devices = _load_devices(self.args.jetson_config)
        if device_name not in devices:
            raise KeyError(f"Device {device_name} not found in Jetson config")
        return devices[device_name]

    def _ssh_for_device(self, device_name: str, device: dict) -> SshSession:
        if device_name not in self.ssh:
            self.ssh[device_name] = SshSession(
                host=device["host"],
                user=device.get("user", "nvidia"),
                password=device.get("password", "nvidia"),
            )
        return self.ssh[device_name]

    def prepare_config(self) -> dict:
        source = Path(self.args.source_config)
        runtime = Path(self.args.runtime_config)
        config = _load_json(source)
        runtime.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != runtime.resolve():
            shutil.copyfile(source, runtime)
        print(f"[ORCH] runtime config: {runtime}")
        return config

    def sync_config_to_jetsons(self, config: dict) -> None:
        cfg_text = json.dumps(config, ensure_ascii=False, indent=2)
        devices = _load_devices(self.args.jetson_config)
        touched = sorted(
            {
                str(info.get("device", ""))
                for info in config.get("nodes", {}).values()
                if "jetson" in str(info.get("device", "")).lower()
            }
        )
        for device_name in touched:
            if device_name not in devices:
                print(f"[ORCH] skip sync for unknown device {device_name}")
                continue
            device = devices[device_name]
            remote_config = f"{device['repo'].rstrip('/')}/config/network_config.json"
            if self.args.dry_run:
                print(f"[DRY] sync config -> {device_name}:{remote_config}")
                continue
            ssh = self._ssh_for_device(device_name, device)
            ssh.run(f"mkdir -p {device['repo'].rstrip('/')}/config", timeout_s=20)
            ssh.put_text(remote_config, cfg_text)
            print(f"[ORCH] synced config -> {device_name}:{remote_config}")

    def launch_remote_satellites(self, config: dict, sat_nodes: Iterable[str]) -> None:
        devices = _load_devices(self.args.jetson_config)
        for node_id in sat_nodes:
            node = config.get("nodes", {}).get(node_id)
            if not node:
                print(f"[ORCH] skip unknown SAT node {node_id}")
                continue
            device_name = str(node.get("device", ""))
            if device_name not in devices:
                print(f"[ORCH] skip {node_id}: unknown device {device_name}")
                continue
            device = devices[device_name]
            container = f"satinfer-{_safe_name(self.run_id)}-{_safe_name(node_id)}"
            log_path = f"{device['repo'].rstrip('/')}/logs/{self.run_id}_{node_id}.log"
            command = (
                f"cd {device['repo']} && mkdir -p logs; "
                f"docker rm -f {container} >/dev/null 2>&1 || true; "
                f"cid=$(docker run -d --rm --name {container} "
                f"--runtime nvidia --network host "
                f"-v {device['repo']}:/workspace -w /workspace "
                f"{self.args.docker_image} "
                f"bash -lc 'python main.py --id {node_id}'); "
                f"echo $cid; "
                f"nohup docker logs -f $cid > {log_path} 2>&1 < /dev/null &"
            )
            if self.args.dry_run:
                print(f"[DRY] {device_name} launch {node_id}: {command}")
                continue
            ssh = self._ssh_for_device(device_name, device)
            output = ssh.run(command, timeout_s=60)
            self.started_containers.append((device_name, container))
            print(f"[ORCH] launched {node_id} on {device_name}: {container}")
            if output.strip():
                print(output.strip())

    def launch_local_nodes(self, config: dict, pc_nodes: Iterable[str]) -> None:
        for node_id in pc_nodes:
            if node_id not in config.get("nodes", {}):
                print(f"[ORCH] skip unknown PC node {node_id}")
                continue
            log_file = self.logs_dir / f"{node_id}.log"
            if self.args.dry_run:
                print(f"[DRY] local launch {node_id}: {sys.executable} main.py --id {node_id}")
                continue
            f = log_file.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "main.py", "--id", node_id],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
            )
            self.local_processes.append(process)
            print(f"[ORCH] launched local {node_id}, log={log_file}")

    def run_existing_pmp_runner(self) -> None:
        command = [
            sys.executable,
            "experiments_runner.py",
            "--config",
            self.args.runtime_config,
            "--exp-mode",
            self.args.exp_mode,
            "--exp-type",
            "algo_effectiveness",
            "--num-tasks",
            str(self.args.num_tasks),
            "--run-id",
            self.run_id,
            "--fixed-model",
            self.args.model_name,
            "--fixed-batch-size",
            str(self.args.batch_size),
            "--fixed-input-h",
            str(self.args.input_h),
            "--fixed-input-w",
            str(self.args.input_w),
        ]
        if self.args.dry_run:
            print("[DRY] runner:", " ".join(command))
            return
        print("[ORCH] runner:", " ".join(command))
        subprocess.run(command, check=True, cwd=Path.cwd())


def _load_devices(path: str | Path) -> dict:
    if path and Path(path).exists():
        payload = _load_json(path)
        return payload.get("jetsons", payload.get("devices", payload))
    return DEFAULT_JETSONS


def write_default_jetson_config(path: Path) -> None:
    _write_json(path, {"jetsons": DEFAULT_JETSONS})
    print(f"[ORCH] wrote {path}")


def _sat_nodes_from_config(config: dict) -> List[str]:
    return [
        node_id
        for node_id, info in config.get("nodes", {}).items()
        if "jetson" in str(info.get("device", "")).lower()
    ]


def command_launch(args: argparse.Namespace) -> None:
    orch = Orchestrator(args)
    try:
        config = orch.prepare_config()
        sat_nodes = _parse_csv(args.sat_nodes) or _sat_nodes_from_config(config)
        pc_nodes = _parse_csv(args.pc_nodes)
        orch.sync_config_to_jetsons(config)
        orch.launch_remote_satellites(config, sat_nodes)
        orch.launch_local_nodes(config, pc_nodes)
        if not args.dry_run and not args.no_wait:
            print("[ORCH] nodes are running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    finally:
        if not args.no_cleanup:
            orch.close()


def command_run_pmp(args: argparse.Namespace) -> None:
    orch = Orchestrator(args)
    try:
        config = orch.prepare_config()
        sat_nodes = _parse_csv(args.sat_nodes) or _sat_nodes_from_config(config)
        # GS is external; RS is in-process inside experiments_runner.py.
        pc_nodes = _parse_csv(args.pc_nodes) or ["GS"]
        orch.sync_config_to_jetsons(config)
        orch.launch_remote_satellites(config, sat_nodes)
        orch.launch_local_nodes(config, pc_nodes)
        if not args.dry_run:
            time.sleep(args.startup_wait_s)
        orch.run_existing_pmp_runner()
    finally:
        if not args.no_cleanup:
            orch.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch PC + Jetson semi-physical experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("write-template", help="Write default Jetson SSH config.")
    template.add_argument("--output", default="config/physical_jetsons.example.json")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source-config", default="config/network_config.json")
        p.add_argument("--runtime-config", default="config/network_config.json")
        p.add_argument("--jetson-config", default="config/physical_jetsons.local.json")
        p.add_argument("--docker-image", default="satinfer:v4.0")
        p.add_argument("--run-id", default="")
        p.add_argument("--logs-dir", default="result/semi_physical/node_logs")
        p.add_argument("--sat-nodes", default="", help="Comma-separated SAT node ids. Default: all Jetson nodes in config.")
        p.add_argument("--pc-nodes", default="", help="Comma-separated local PC node ids.")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--no-cleanup", action="store_true")
        p.add_argument("--keep-remote", action="store_true")

    launch = sub.add_parser("launch", help="Only launch node processes, useful for manual debugging.")
    add_common(launch)
    launch.add_argument("--no-wait", action="store_true")
    launch.set_defaults(func=command_launch)

    run_pmp = sub.add_parser("run-pmp", help="Launch nodes and run existing PMP physical/hybrid experiment.")
    add_common(run_pmp)
    run_pmp.add_argument("--exp-mode", default="physical", choices=["physical", "hybrid"])
    run_pmp.add_argument("--num-tasks", type=int, default=3)
    run_pmp.add_argument("--model-name", default="yolov5")
    run_pmp.add_argument("--batch-size", type=int, default=32)
    run_pmp.add_argument("--input-h", type=int, default=640)
    run_pmp.add_argument("--input-w", type=int, default=640)
    run_pmp.add_argument("--startup-wait-s", type=float, default=8.0)
    run_pmp.set_defaults(func=command_run_pmp)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "write-template":
        write_default_jetson_config(Path(args.output))
        return
    args.func(args)


if __name__ == "__main__":
    main()
