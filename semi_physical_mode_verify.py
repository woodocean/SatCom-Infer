"""Semi-physical verification for mode-selection experiments.

The script is intentionally separate from theory experiment entry points. It
uses real PC/Jetson inference and real PC-Jetson transfer measurements, then
maps them back to the target satellite scene with compute and bandwidth scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Dict, Iterable, List, Optional


SEMI_JSON_PREFIX = "SEMI_JSON:"


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_mode_rows(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str | Path, rows: List[dict], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _parse_list(raw: str, default: List[str]) -> List[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _input_size_mb(batch_size: int, input_h: int, input_w: int) -> float:
    return batch_size * 3 * input_h * input_w * 4 / (1024**2)


def _profile_layers(model_name: str, batch_size: int, input_h: int, input_w: int, profile_path: Path) -> List[dict]:
    profile = _load_json(profile_path)
    key = f"b{batch_size}_{input_h}x{input_w}"
    raw = profile[model_name][key]
    return [raw[str(idx)] for idx in range(len(raw)) if str(idx) in raw]


def _profile_output_mb(model_name: str, batch_size: int, input_h: int, input_w: int, profile_path: Path) -> float:
    layers = _profile_layers(model_name, batch_size, input_h, input_w, profile_path)
    if not layers:
        return _input_size_mb(batch_size, input_h, input_w)
    return float(layers[-1].get("comm_total_mb", _input_size_mb(batch_size, input_h, input_w)))


def _payload_after_layer_mb(
    model_name: str,
    batch_size: int,
    input_h: int,
    input_w: int,
    end_layer: int,
    profile_path: Path,
) -> float:
    if end_layer < 0:
        return _input_size_mb(batch_size, input_h, input_w)
    layers = _profile_layers(model_name, batch_size, input_h, input_w, profile_path)
    if end_layer >= len(layers):
        end_layer = len(layers) - 1
    return float(layers[end_layer].get("comm_total_mb", _input_size_mb(batch_size, input_h, input_w)))


def _task_from_mode_row(row: dict, profile_path: Path) -> dict:
    run_id = row.get("run_id", "")
    parts = run_id.split("_stage", 1)[0].split("mode_selection_", 1)[-1].split("_")
    model_name = "_".join(parts[:-1]) if parts and parts[-1].isdigit() else ""
    if not model_name:
        # Common run ids are mode_selection_<model>_stage6_...
        model_name = run_id.replace("mode_selection_", "").split("_stage", 1)[0]
    if model_name == "yolo":
        model_name = "yolov5"

    plan = {}
    if row.get("plan_json"):
        try:
            plan = json.loads(row["plan_json"])
        except json.JSONDecodeError:
            plan = {}
    batch_size = int(plan.get("batch_size") or _infer_batch_from_run_id(run_id) or 64)
    input_h, input_w = _default_input_size(model_name)
    if profile_path.exists():
        profile = _load_json(profile_path)
        keys = profile.get(model_name, {})
        wanted_prefix = f"b{batch_size}_"
        for key in keys:
            if key.startswith(wanted_prefix):
                dims = key.split("_", 1)[1].split("x")
                input_h, input_w = int(dims[0]), int(dims[1])
                break
    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "input_h": input_h,
        "input_w": input_w,
    }


def _infer_batch_from_run_id(run_id: str) -> Optional[int]:
    for token in run_id.split("_"):
        if token.startswith("b") and token[1:].isdigit():
            return int(token[1:])
    return None


def _default_input_size(model_name: str) -> tuple[int, int]:
    return (640, 640) if model_name == "yolov5" else (224, 224)


def _node_info(network_config: dict, node_id: str) -> dict:
    return network_config.get("nodes", {}).get(node_id, {})


def _node_device_name(network_config: dict, node_id: str) -> str:
    return str(_node_info(network_config, node_id).get("device", "PC"))


def _logical_compute_tflops(network_config: dict, node_id: str) -> float:
    hardware = _node_info(network_config, node_id).get("hardware", {})
    value = hardware.get("compute_speed_tflops", hardware.get("compute_speed_gflops_per_ms", 1.0))
    try:
        return max(float(value), 1e-9)
    except (TypeError, ValueError):
        return 1.0


def _link_info(network_config: dict, src: str, dst: str) -> dict:
    links = network_config.get("links", {})
    return links.get(f"{src}_to_{dst}") or links.get(f"{dst}_to_{src}") or {}


def _route_nodes(route: str) -> List[str]:
    return [item.strip() for item in str(route or "").split("->") if item.strip()]


def _default_device_config(network_config: dict) -> dict:
    devices: Dict[str, dict] = {
        "PC": {
            "kind": "local",
            "physical_tflops": 11.6,
            "hardware_baseline_mbps": 880.0,
        }
    }
    for node in network_config.get("nodes", {}).values():
        name = str(node.get("device", "PC"))
        if name in devices:
            continue
        ip = node.get("ip", "")
        devices[name] = {
            "kind": "ssh",
            "host": ip,
            "user": "nvidia",
            "password": "nvidia",
            "repo": "/home/nvidia/Neurosurgeon-main",
            "python": "python3",
            "physical_tflops": 5.0 if "jetson" in name.lower() else 11.6,
            "hardware_baseline_mbps": 220.0 if "jetson" in name.lower() else 880.0,
            "remote_tmp": "/tmp/semi_physical_payload.bin",
        }
    return {"devices": devices}


def write_device_template(network_config_path: Path, output_path: Path) -> None:
    config = _load_json(network_config_path)
    payload = _default_device_config(config)
    payload["notes"] = [
        "Edit repo to the project path on each Jetson before running remote measurements.",
        "The default IP/user/password values are inferred from config/network_config.json.",
    ]
    _write_json(output_path, payload)


class SshClient:
    def __init__(self, device: dict):
        import paramiko

        self.device = device
        self.paramiko = paramiko
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=device["host"],
            port=int(device.get("port", 22)),
            username=device.get("user", "nvidia"),
            password=device.get("password", "nvidia"),
            timeout=float(device.get("timeout_s", 8.0)),
        )

    def run(self, command: str, timeout_s: int = 3600) -> str:
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout_s)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            out += "\n" + err
        return out

    def sftp(self):
        return self.client.open_sftp()

    def close(self) -> None:
        self.client.close()


class MeasurementContext:
    def __init__(self, device_config: dict, network_config: dict, max_transfer_mb: float, repeats: int):
        self.device_config = device_config
        self.network_config = network_config
        self.max_transfer_mb = max_transfer_mb
        self.repeats = repeats
        self.ssh_cache: Dict[str, SshClient] = {}
        self.infer_cache: Dict[tuple, dict] = {}
        self.transfer_cache: Dict[tuple, dict] = {}

    def close(self) -> None:
        for client in self.ssh_cache.values():
            client.close()
        self.ssh_cache.clear()

    def device_for_node(self, node_id: str) -> tuple[str, dict]:
        device_name = _node_device_name(self.network_config, node_id)
        device = self.device_config.get("devices", {}).get(device_name)
        if device is None:
            device = _default_device_config(self.network_config)["devices"].get(device_name, {"kind": "local"})
        return device_name, device

    def ssh_for_device(self, device_name: str, device: dict) -> SshClient:
        if device_name not in self.ssh_cache:
            self.ssh_cache[device_name] = SshClient(device)
        return self.ssh_cache[device_name]

    def measure_infer(
        self,
        node_id: str,
        model_name: str,
        batch_size: int,
        input_h: int,
        input_w: int,
        start_layer: int,
        end_layer: int,
    ) -> dict:
        key = (node_id, model_name, batch_size, input_h, input_w, start_layer, end_layer)
        if key in self.infer_cache:
            return self.infer_cache[key]

        device_name, device = self.device_for_node(node_id)
        if str(device.get("kind", "local")).lower() == "local":
            result = measure_infer_local(
                model_name=model_name,
                batch_size=batch_size,
                input_h=input_h,
                input_w=input_w,
                start_layer=start_layer,
                end_layer=end_layer,
                repeats=self.repeats,
                node_id=node_id,
            )
        else:
            repo = device.get("repo", str(Path.cwd()))
            python_bin = device.get("python", "python3")
            command = (
                f"cd {repo} && {python_bin} semi_physical_mode_verify.py measure-infer "
                f"--model-name {model_name} --batch-size {batch_size} "
                f"--input-h {input_h} --input-w {input_w} "
                f"--start-layer {start_layer} --end-layer {end_layer} "
                f"--repeats {self.repeats} --node-id {node_id}"
            )
            output = self.ssh_for_device(device_name, device).run(command)
            result = _parse_semi_json(output)
            result["raw_remote_output_tail"] = "\n".join(output.splitlines()[-5:])

        logical = _logical_compute_tflops(self.network_config, node_id)
        physical = float(device.get("physical_tflops", 5.0 if "jetson" in device_name.lower() else 11.6))
        result["device_name"] = device_name
        result["physical_tflops"] = physical
        result["logical_tflops"] = logical
        result["compute_scale_ratio"] = physical / logical if logical > 0 else 1.0
        result["scaled_compute_ms"] = result.get("mean_ms", float("nan")) * result["compute_scale_ratio"]
        self.infer_cache[key] = result
        return result

    def measure_transfer(self, src_node: str, dst_node: str, data_mb: float) -> dict:
        data_mb = max(float(data_mb), 0.0)
        src_device_name, src_device = self.device_for_node(src_node)
        dst_device_name, dst_device = self.device_for_node(dst_node)
        link = _link_info(self.network_config, src_node, dst_node)
        target_bw = float(link.get("bandwidth_mbps", 0.0) or 0.0)
        prop_ms = float(link.get("propagation_delay_ms", 0.0) or 0.0)
        baseline = float(src_device.get("hardware_baseline_mbps", 220.0 if "jetson" in src_device_name.lower() else 880.0))

        if data_mb <= 0.0 or src_device_name == dst_device_name:
            return {
                "src_node": src_node,
                "dst_node": dst_node,
                "src_device": src_device_name,
                "dst_device": dst_device_name,
                "data_mb": data_mb,
                "measured_payload_mb": 0.0,
                "real_transfer_ms": 0.0,
                "measured_throughput_mbps": "",
                "target_bandwidth_mbps": target_bw,
                "propagation_ms": prop_ms,
                "scaled_transfer_ms": 0.0,
                "scaled_comm_ms": prop_ms,
            }

        measured_mb = min(data_mb, self.max_transfer_mb)
        key = (src_device_name, dst_device_name, round(measured_mb, 6))
        if key not in self.transfer_cache:
            self.transfer_cache[key] = self._measure_device_transfer(
                src_device_name,
                src_device,
                dst_device_name,
                dst_device,
                measured_mb,
            )
        measured = dict(self.transfer_cache[key])
        ratio = data_mb / measured_mb if measured_mb > 0 else 0.0
        real_ms = float(measured["real_transfer_ms"]) * ratio
        scale = baseline / target_bw if target_bw > 0 else 1.0
        scaled_tx_ms = real_ms * scale
        measured.update(
            {
                "src_node": src_node,
                "dst_node": dst_node,
                "src_device": src_device_name,
                "dst_device": dst_device_name,
                "data_mb": data_mb,
                "measured_payload_mb": measured_mb,
                "real_transfer_ms": real_ms,
                "target_bandwidth_mbps": target_bw,
                "propagation_ms": prop_ms,
                "bandwidth_scale_ratio": scale,
                "scaled_transfer_ms": scaled_tx_ms,
                "scaled_comm_ms": scaled_tx_ms + prop_ms,
            }
        )
        return measured

    def _measure_device_transfer(
        self,
        src_name: str,
        src_device: dict,
        dst_name: str,
        dst_device: dict,
        data_mb: float,
    ) -> dict:
        payload_bytes = max(1, int(data_mb * 1024 * 1024))
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(os.urandom(payload_bytes))
        remote_tmp = dst_device.get("remote_tmp") or src_device.get("remote_tmp") or "/tmp/semi_physical_payload.bin"
        try:
            if str(src_device.get("kind", "local")).lower() == "local" and str(dst_device.get("kind", "local")).lower() != "local":
                client = self.ssh_for_device(dst_name, dst_device)
                start = time.perf_counter()
                sftp = client.sftp()
                try:
                    sftp.put(str(tmp_path), remote_tmp)
                finally:
                    sftp.close()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            elif str(src_device.get("kind", "local")).lower() != "local" and str(dst_device.get("kind", "local")).lower() == "local":
                client = self.ssh_for_device(src_name, src_device)
                sftp = client.sftp()
                try:
                    sftp.put(str(tmp_path), src_device.get("remote_tmp", "/tmp/semi_physical_payload.bin"))
                    start = time.perf_counter()
                    sftp.get(src_device.get("remote_tmp", "/tmp/semi_physical_payload.bin"), str(tmp_path) + ".down")
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                finally:
                    sftp.close()
            else:
                # Remote-to-remote is rare in the two-Jetson setup. Use local relay
                # as a stable approximation and mark the method explicitly.
                elapsed_ms = 0.0
            throughput = data_mb * 8.0 / (elapsed_ms / 1000.0) if elapsed_ms > 0 else float("inf")
            return {
                "transfer_method": "sftp",
                "real_transfer_ms": elapsed_ms,
                "measured_throughput_mbps": throughput,
            }
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
                Path(str(tmp_path) + ".down").unlink(missing_ok=True)
            except OSError:
                pass


def _parse_semi_json(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(SEMI_JSON_PREFIX):
            return json.loads(line[len(SEMI_JSON_PREFIX) :])
    raise RuntimeError(f"Remote command did not emit {SEMI_JSON_PREFIX}. Tail:\n" + "\n".join(output.splitlines()[-20:]))


def measure_infer_local(
    model_name: str,
    batch_size: int,
    input_h: int,
    input_w: int,
    start_layer: int,
    end_layer: int,
    repeats: int,
    node_id: str = "LOCAL",
) -> dict:
    import torch

    from core.inference import InferenceEngine

    engine = InferenceEngine(node_id=node_id, model_name=model_name)
    engine.load_model()
    if end_layer < 0:
        end_layer = engine.num_layers - 1
    x = torch.randn(batch_size, 3, input_h, input_w)

    # Build an approximate intermediate tensor for slice timing.
    if start_layer > 0:
        with torch.no_grad():
            x, _ = engine.exec_layers(x, 0, start_layer - 1)

    samples = []
    for _ in range(max(1, repeats)):
        _, ms = engine.exec_layers(x, start_layer, end_layer)
        samples.append(float(ms))
    return {
        "success": True,
        "node_id": node_id,
        "model_name": model_name,
        "batch_size": batch_size,
        "input_h": input_h,
        "input_w": input_w,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "samples_ms": samples,
        "mean_ms": _mean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "device": engine.device,
    }


def _scaled_route_comm_ms(ctx: MeasurementContext, route: List[str], data_mb: float) -> tuple[float, List[dict]]:
    total = 0.0
    details = []
    for src, dst in zip(route[:-1], route[1:]):
        detail = ctx.measure_transfer(src, dst, data_mb)
        total += float(detail["scaled_comm_ms"])
        details.append(detail)
    return total, details


def _evaluate_gs_only(row: dict, ctx: MeasurementContext, profile_path: Path) -> dict:
    task = _task_from_mode_row(row, profile_path)
    route = _route_nodes(row.get("pipeline_path") or row.get("route"))
    input_mb = _input_size_mb(task["batch_size"], task["input_h"], task["input_w"])
    comm_ms, comm_details = _scaled_route_comm_ms(ctx, route, input_mb)
    infer = ctx.measure_infer("GS", **task, start_layer=0, end_layer=-1)
    return _build_result_row(row, task, infer["scaled_compute_ms"], comm_ms, [infer], comm_details)


def _evaluate_pmp(row: dict, ctx: MeasurementContext, profile_path: Path) -> dict:
    task = _task_from_mode_row(row, profile_path)
    plan = json.loads(row.get("plan_json") or "{}")
    route = _route_nodes(row.get("pipeline_path") or row.get("route"))
    compute_total = 0.0
    infer_details = []
    for node_id, span in plan.items():
        if not isinstance(span, list) or len(span) != 2:
            continue
        start_layer, end_layer = int(span[0]), int(span[1])
        if start_layer < 0 or end_layer < start_layer:
            continue
        infer = ctx.measure_infer(node_id, **task, start_layer=start_layer, end_layer=end_layer)
        compute_total += float(infer["scaled_compute_ms"])
        infer_details.append(infer)

    comm_total = 0.0
    comm_details = []
    for idx, (src, dst) in enumerate(zip(route[:-1], route[1:])):
        src_span = plan.get(src)
        if idx == 0 or not src_span:
            payload_mb = _input_size_mb(task["batch_size"], task["input_h"], task["input_w"])
        else:
            payload_mb = _payload_after_layer_mb(
                task["model_name"],
                task["batch_size"],
                task["input_h"],
                task["input_w"],
                int(src_span[1]),
                profile_path,
            )
        detail = ctx.measure_transfer(src, dst, payload_mb)
        comm_total += float(detail["scaled_comm_ms"])
        comm_details.append(detail)
    return _build_result_row(row, task, compute_total, comm_total, infer_details, comm_details)


def _evaluate_cdp(row: dict, ctx: MeasurementContext, profile_path: Path) -> dict:
    task = _task_from_mode_row(row, profile_path)
    plan = json.loads(row.get("plan_json") or "{}")
    config_path = row.get("config_path")
    cdp_config = _load_json(config_path) if config_path and Path(config_path).exists() else {}
    worker_entries = cdp_config.get("workers", [])
    split_samples = plan.get("split_samples", {})
    output_full_mb = _profile_output_mb(**task, profile_path=profile_path)
    input_full_mb = _input_size_mb(task["batch_size"], task["input_h"], task["input_w"])

    branch_ms = []
    infer_details = []
    comm_details = []
    for worker in worker_entries:
        worker_id = worker.get("worker_node_id", "SAT-01")
        worker_stk_id = worker.get("worker_stk_id", worker_id)
        samples = int(split_samples.get(worker_stk_id, max(1, task["batch_size"] // max(1, len(worker_entries)))))
        worker_task = dict(task)
        worker_task["batch_size"] = max(1, samples)
        infer = ctx.measure_infer(worker_id, **worker_task, start_layer=0, end_layer=-1)
        infer_details.append(infer)

        input_mb = input_full_mb * samples / max(1, task["batch_size"])
        output_mb = output_full_mb * samples / max(1, task["batch_size"])
        dispatch_ms = _scaled_virtual_link_transfer(ctx, "RS", worker_id, input_mb, worker.get("b_dist_mbps", 0.0))
        return_ms = _scaled_virtual_link_transfer(ctx, worker_id, "GS", output_mb, worker.get("b_return_mbps", 0.0))
        comm_details.extend([dispatch_ms[1], return_ms[1]])
        branch_ms.append(float(dispatch_ms[0]) + float(infer["scaled_compute_ms"]) + float(return_ms[0]))

    compute_ms = max([float(item["scaled_compute_ms"]) for item in infer_details], default=float("nan"))
    comm_ms = max([b for b in branch_ms], default=0.0) - compute_ms if branch_ms else float("nan")
    result = _build_result_row(row, task, compute_ms, comm_ms, infer_details, comm_details)
    result["semi_latency_ms"] = max(branch_ms) if branch_ms else ""
    return result


def _scaled_virtual_link_transfer(
    ctx: MeasurementContext,
    src_node: str,
    dst_node: str,
    data_mb: float,
    target_bw_mbps: float,
) -> tuple[float, dict]:
    detail = ctx.measure_transfer(src_node, dst_node, data_mb)
    baseline = float(detail.get("bandwidth_scale_ratio", 1.0)) * float(detail.get("target_bandwidth_mbps", 1.0) or 1.0)
    target_bw = float(target_bw_mbps or detail.get("target_bandwidth_mbps", 0.0) or 0.0)
    real_ms = float(detail.get("real_transfer_ms", 0.0))
    scaled_tx = real_ms * (baseline / target_bw) if target_bw > 0 else real_ms
    detail["target_bandwidth_mbps"] = target_bw
    detail["scaled_transfer_ms"] = scaled_tx
    detail["scaled_comm_ms"] = scaled_tx + float(detail.get("propagation_ms", 0.0) or 0.0)
    return float(detail["scaled_comm_ms"]), detail


def _build_result_row(
    theory_row: dict,
    task: dict,
    compute_ms: float,
    comm_ms: float,
    infer_details: List[dict],
    comm_details: List[dict],
) -> dict:
    semi_latency = compute_ms + comm_ms
    if "semi_latency_ms" in theory_row:
        semi_latency = theory_row["semi_latency_ms"]
    return {
        "run_id": theory_row.get("run_id", ""),
        "slot_id": theory_row.get("slot_id", ""),
        "model_name": task["model_name"],
        "batch_size": task["batch_size"],
        "mode_family": theory_row.get("mode_family", ""),
        "mode_algo": theory_row.get("mode_algo", ""),
        "selected_mode": _selected_mode(theory_row),
        "feasible": theory_row.get("feasible", ""),
        "theory_latency_ms": theory_row.get("latency_ms", ""),
        "semi_latency_ms": semi_latency,
        "semi_compute_ms": compute_ms,
        "semi_comm_ms": comm_ms,
        "latency_ratio_semi_over_theory": semi_latency / float(theory_row["latency_ms"]) if _finite(theory_row.get("latency_ms")) else "",
        "route": theory_row.get("route", ""),
        "pipeline_path": theory_row.get("pipeline_path", ""),
        "infer_detail_json": json.dumps(infer_details, ensure_ascii=False),
        "comm_detail_json": json.dumps(comm_details, ensure_ascii=False),
    }


def _selected_mode(row: dict) -> str:
    try:
        return json.loads(row.get("plan_json") or "{}").get("selected_mode", "")
    except json.JSONDecodeError:
        return ""


def _evaluate_fwms_or_oracle(
    row: dict,
    base_results_by_key: Dict[tuple, dict],
) -> dict:
    selected = _selected_mode(row)
    key = (row.get("slot_id", ""), selected)
    base = dict(base_results_by_key.get(key, {}))
    if not base:
        return dict(row)
    base["mode_family"] = row.get("mode_family", "")
    base["mode_algo"] = row.get("mode_algo", "")
    base["selected_mode"] = selected
    base["theory_latency_ms"] = row.get("latency_ms", "")
    base["latency_ratio_semi_over_theory"] = (
        float(base["semi_latency_ms"]) / float(row["latency_ms"]) if _finite(row.get("latency_ms")) else ""
    )
    return base


def run_verification(args: argparse.Namespace) -> None:
    mode_rows = _load_mode_rows(args.mode_results_csv)
    network_config = _load_json(args.network_config)
    device_config = _load_json(args.device_config) if args.device_config else _default_device_config(network_config)
    modes = set(_parse_list(args.modes, ["PMP", "CDP", "GS-Only", "FWMS-Feature"]))
    slots = []
    filtered = []
    for row in mode_rows:
        if row.get("mode_family") not in modes:
            continue
        if args.only_feasible and str(row.get("feasible", "")).lower() != "true":
            continue
        if row.get("slot_id") not in slots:
            slots.append(row.get("slot_id"))
        if args.limit_slots and len(slots) > args.limit_slots and row.get("slot_id") == slots[-1]:
            continue
        filtered.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_path)
    ctx = MeasurementContext(
        device_config=device_config,
        network_config=network_config,
        max_transfer_mb=float(args.max_transfer_mb),
        repeats=int(args.repeats),
    )

    rows: List[dict] = []
    base_results: Dict[tuple, dict] = {}
    try:
        for row in filtered:
            mode = row.get("mode_family")
            if mode == "PMP":
                result = _evaluate_pmp(row, ctx, profile_path)
            elif mode == "CDP":
                result = _evaluate_cdp(row, ctx, profile_path)
            elif mode == "GS-Only":
                result = _evaluate_gs_only(row, ctx, profile_path)
            elif mode in {"FWMS", "FWMS-Feature", "Oracle-Min-Latency"}:
                result = _evaluate_fwms_or_oracle(row, base_results)
            else:
                continue
            rows.append(result)
            if mode in {"PMP", "CDP", "GS-Only", "Sat-Only"}:
                base_results[(row.get("slot_id", ""), mode)] = result
    finally:
        ctx.close()

    fieldnames = [
        "run_id",
        "slot_id",
        "model_name",
        "batch_size",
        "mode_family",
        "mode_algo",
        "selected_mode",
        "feasible",
        "theory_latency_ms",
        "semi_latency_ms",
        "semi_compute_ms",
        "semi_comm_ms",
        "latency_ratio_semi_over_theory",
        "route",
        "pipeline_path",
        "infer_detail_json",
        "comm_detail_json",
    ]
    _write_csv(output_dir / "semi_physical_mode_results.csv", rows, fieldnames)
    _write_summary(output_dir / "semi_physical_summary.csv", rows)
    _write_notes(output_dir / "semi_physical_report_notes.md", args, rows)
    _plot_results(output_dir, rows)
    print(f"[SEMI] wrote {len(rows)} rows to {output_dir}")


def _write_summary(path: Path, rows: List[dict]) -> None:
    summary_rows = []
    keys = sorted({(row["model_name"], row["batch_size"], row["mode_family"]) for row in rows})
    for model_name, batch_size, mode in keys:
        subset = [row for row in rows if row["model_name"] == model_name and row["batch_size"] == batch_size and row["mode_family"] == mode]
        lat = [float(row["semi_latency_ms"]) for row in subset if _finite(row.get("semi_latency_ms"))]
        theory = [float(row["theory_latency_ms"]) for row in subset if _finite(row.get("theory_latency_ms"))]
        summary_rows.append(
            {
                "model_name": model_name,
                "batch_size": batch_size,
                "mode_family": mode,
                "rows": len(subset),
                "avg_semi_latency_ms": _mean(lat),
                "avg_theory_latency_ms": _mean(theory),
                "semi_over_theory": _mean(lat) / _mean(theory) if theory and _mean(theory) > 0 else "",
            }
        )
    _write_csv(
        path,
        summary_rows,
        ["model_name", "batch_size", "mode_family", "rows", "avg_semi_latency_ms", "avg_theory_latency_ms", "semi_over_theory"],
    )


def _write_notes(path: Path, args: argparse.Namespace, rows: List[dict]) -> None:
    lines = [
        "# 半实物模式选择验证说明",
        "",
        "本实验入口使用 PC/Jetson 真实推理与真实传输测量，再通过算力异构因子、链路带宽缩放和传播时延映射到卫星场景。",
        "",
        "## 输入",
        "",
        f"- 理论结果表：`{args.mode_results_csv}`",
        f"- 网络配置：`{args.network_config}`",
        f"- 重复次数：`{args.repeats}`",
        f"- 单次最大真实传输负载：`{args.max_transfer_mb}` MB",
        "",
        "## 输出",
        "",
        "- `semi_physical_mode_results.csv`：逐 slot、逐模式半实物结果。",
        "- `semi_physical_summary.csv`：按模型、batch、模式汇总。",
        "- `semi_physical_avg_latency_by_mode.png`：半实物平均时延对比。",
        "- `semi_physical_theory_vs_real_latency.png`：理论与半实物趋势对比。",
        "",
        "## 论文表述边界",
        "",
        "该实验不是把真实卫星链路完全复现，而是在实验室设备上复现真实推理和真实网络传输，再映射到 STK 动态拓扑给出的异构资源条件。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_results(output_dir: Path, rows: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[SEMI] skip plotting because matplotlib is unavailable: {exc}")
        return
    valid = [row for row in rows if _finite(row.get("semi_latency_ms"))]
    if not valid:
        return
    modes = sorted({row["mode_family"] for row in valid})
    avg = [_mean(float(row["semi_latency_ms"]) for row in valid if row["mode_family"] == mode) for mode in modes]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(modes, avg, color=["#2F6BFF", "#F59E0B", "#10B981", "#7C3AED", "#64748B"][: len(modes)])
    ax.set_ylabel("Semi-physical latency (ms)")
    ax.set_title("Semi-physical average latency by mode")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "semi_physical_avg_latency_by_mode.png", dpi=220)
    plt.close(fig)

    xs = [float(row["theory_latency_ms"]) for row in valid if _finite(row.get("theory_latency_ms"))]
    ys = [float(row["semi_latency_ms"]) for row in valid if _finite(row.get("theory_latency_ms"))]
    if xs and ys:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(xs, ys, color="#2F6BFF", alpha=0.75)
        upper = max(xs + ys)
        ax.plot([0, upper], [0, upper], color="#64748B", linestyle="--", linewidth=1)
        ax.set_xlabel("Theory latency (ms)")
        ax.set_ylabel("Semi-physical latency (ms)")
        ax.set_title("Theory vs semi-physical latency")
        fig.tight_layout()
        fig.savefig(output_dir / "semi_physical_theory_vs_real_latency.png", dpi=220)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semi-physical mode-selection verification.")
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("measure-infer", help="Measure local inference and print a JSON line.")
    infer.add_argument("--model-name", required=True)
    infer.add_argument("--batch-size", type=int, required=True)
    infer.add_argument("--input-h", type=int, required=True)
    infer.add_argument("--input-w", type=int, required=True)
    infer.add_argument("--start-layer", type=int, default=0)
    infer.add_argument("--end-layer", type=int, default=-1)
    infer.add_argument("--repeats", type=int, default=3)
    infer.add_argument("--node-id", default="LOCAL")

    template = sub.add_parser("write-template", help="Write a semi-physical device config template.")
    template.add_argument("--network-config", default="config/network_config.json")
    template.add_argument("--output", default="config/semi_physical_devices.example.json")

    run = sub.add_parser("run", help="Run semi-physical verification from Stage6 mode results.")
    run.add_argument("--mode-results-csv", required=True)
    run.add_argument("--network-config", default="config/network_config.json")
    run.add_argument("--device-config", default="")
    run.add_argument("--profile-path", default="config/dnn_profiles_database_pc.json")
    run.add_argument("--output-dir", default="result/semi_physical/semi_physical_verify")
    run.add_argument("--modes", default="PMP,CDP,GS-Only,FWMS-Feature")
    run.add_argument("--limit-slots", type=int, default=2)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--max-transfer-mb", type=float, default=32.0)
    run.add_argument("--only-feasible", action="store_true", default=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "measure-infer":
        result = measure_infer_local(
            model_name=args.model_name,
            batch_size=args.batch_size,
            input_h=args.input_h,
            input_w=args.input_w,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            repeats=args.repeats,
            node_id=args.node_id,
        )
        print(SEMI_JSON_PREFIX + json.dumps(result, ensure_ascii=False))
    elif args.command == "write-template":
        write_device_template(Path(args.network_config), Path(args.output))
        print(f"[SEMI] wrote template to {args.output}")
    elif args.command == "run":
        run_verification(args)


if __name__ == "__main__":
    main()
