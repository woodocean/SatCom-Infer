"""Mode evaluators used by the mode-selection experiment pipeline."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple

from algorithms.cdp_solver import CDPSolver
from algorithms.pmp_solver import PMPSolver
from core.mode_scene_builder import SlotScene
from core.scheduler import Scheduler
from core.stk_scenario_builder import (
    CandidatePath,
    build_candidate_paths,
    build_network_config_for_path,
    load_stk_reports,
)


@dataclass(frozen=True)
class ModeEvaluation:
    source_run_id: str
    slot_id: str
    mode_family: str
    mode_algo: str
    candidate_id: str
    route_policy: str
    feasible: bool
    reason: str
    latency_ms: float | str
    satellite_energy_j: float | str
    energy_compute_j: float | str
    energy_comm_j: float | str
    satellite_compute_time_ms: float | str
    satellite_tx_time_ms: float | str
    active_sat_count: int | str
    hop_count: int | str
    route: str
    pipeline_path: str
    plan_json: str
    config_path: str
    candidate_path: str

    def to_row(self) -> Dict:
        return asdict(self)


def _route_to_str(route) -> str:
    return "->".join(route or [])


def _split_route_str(value: str) -> List[str]:
    return [item for item in str(value or "").split("->") if item]


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count_active_satellites(plan: Dict) -> int:
    return sum(1 for node_id in plan if str(node_id).upper().startswith("SAT"))


def _scene_slot_index(scene: SlotScene) -> int:
    try:
        return int(scene.slot_id.split("_", 2)[1])
    except (IndexError, ValueError):
        digest = hashlib.sha256(scene.slot_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)


def _slot_rng(scene: SlotScene) -> random.Random:
    base_seed = int(scene.metadata.get("seed", 42))
    return random.Random(base_seed + _scene_slot_index(scene) * 1009)


def _path_key(path: Sequence[str]) -> str:
    return "->".join(path)


def _metadata_ranges(scene: SlotScene) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    isl = scene.metadata.get("isl_range_mbps", [1000.0, 20000.0])
    gsl = scene.metadata.get("gsl_range_mbps", [50.0, 300.0])
    return (float(isl[0]), float(isl[1])), (float(gsl[0]), float(gsl[1]))


class StkPathResolver:
    """Rebuild STK candidate paths with link metrics for mode-specific routes."""

    def __init__(self) -> None:
        self._report_cache: Dict[Tuple[str, ...], Tuple[List, List]] = {}
        self._candidate_cache: Dict[Tuple[str, str], List[CandidatePath]] = {}

    def _report_paths(self, scene: SlotScene) -> Tuple[str, ...]:
        reports = scene.metadata.get("stk_reports", {})
        keys = (
            "chain2_access_data",
            "chain2_aer",
            "chain4_access_data",
            "chain4_aer",
            "chain5_access_data",
            "chain5_aer",
        )
        return tuple(str(reports.get(key, "")) for key in keys)

    def _load_reports(self, scene: SlotScene):
        report_paths = self._report_paths(scene)
        if report_paths not in self._report_cache:
            self._report_cache[report_paths] = load_stk_reports(
                chain2_access_data=report_paths[0] or None,
                chain2_aer=report_paths[1] or None,
                chain4_access_data=report_paths[2] or None,
                chain4_aer=report_paths[3] or None,
                chain5_access_data=report_paths[4] or None,
                chain5_aer=report_paths[5] or None,
            )
        return self._report_cache[report_paths]

    def candidate_paths(self, scene: SlotScene) -> List[CandidatePath]:
        cache_key = (scene.source_run_id, scene.slot_id)
        if cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]

        windows, samples = self._load_reports(scene)
        max_paths = max(len(scene.candidate_paths), 50)
        source_node = scene.selected_stk_path[0] if scene.selected_stk_path else "RS"
        ground_node = scene.selected_stk_path[-1] if scene.selected_stk_path else "Shenzhen"
        paths = build_candidate_paths(
            windows=windows,
            samples=samples,
            time_start=scene.slot_start,
            time_stop=scene.slot_stop,
            source_node=source_node,
            ground_node=ground_node,
            max_hops=int(scene.metadata.get("max_hops", 6)),
            max_paths=max_paths,
            max_neighbors_per_node=int(scene.metadata.get("max_neighbors_per_node", 24)),
            beam_width_per_node=int(scene.metadata.get("beam_width_per_node", 8)),
            sort_policy="shortest_delay",
        )
        self._candidate_cache[cache_key] = paths
        return paths


def _existing_link_by_stk_pair(scene: SlotScene) -> Dict[Tuple[str, str], dict]:
    links = {}
    for info in scene.network_config.get("links", {}).values():
        pair = (info.get("stk_from"), info.get("stk_to"))
        if all(pair):
            links[pair] = info
    return links


def _stable_link_rng(scene: SlotScene, pair: Tuple[str, str], link_type: str) -> random.Random:
    base_seed = int(scene.metadata.get("seed", 42))
    payload = f"{base_seed}|{scene.slot_id}|{pair[0]}|{pair[1]}|{link_type}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _sample_missing_bandwidths(config: dict, scene: SlotScene) -> None:
    existing = _existing_link_by_stk_pair(scene)
    isl_range, gsl_range = _metadata_ranges(scene)

    for link in config.get("links", {}).values():
        pair = (link.get("stk_from"), link.get("stk_to"))
        if pair in existing:
            source = existing[pair]
            for key in (
                "bandwidth_mbps",
                "effective_bandwidth_class",
                "bandwidth_sampling_range_mbps",
                "bandwidth_model",
            ):
                if key in source:
                    link[key] = source[key]
            continue

        link_type = str(link.get("stk_link_type", ""))
        rng = _stable_link_rng(scene, pair, link_type)
        if link_type == "SAT-GS":
            bw = rng.uniform(*gsl_range)
            link["effective_bandwidth_class"] = "GSL"
            link["bandwidth_sampling_range_mbps"] = list(gsl_range)
        else:
            bw = rng.uniform(*isl_range)
            link["effective_bandwidth_class"] = "ISL"
            link["bandwidth_sampling_range_mbps"] = list(isl_range)
        link["bandwidth_mbps"] = round(float(bw), 4)
        link["bandwidth_model"] = "effective_random_uniform"


def _write_network_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _base_config_path(scene: SlotScene) -> Path:
    default_path = Path("config/network_config.json")
    return default_path if default_path.exists() else scene.config_path


def _network_config_for_candidate(scene: SlotScene, candidate: CandidatePath) -> dict:
    if _path_key(candidate.path) == _path_key(scene.selected_stk_path):
        return _apply_resource_overrides(copy.deepcopy(scene.network_config), scene)

    isl_range, gsl_range = _metadata_ranges(scene)
    config = build_network_config_for_path(
        candidate,
        base_config_path=_base_config_path(scene),
        isl_bandwidth_mbps=isl_range[0],
        gsl_bandwidth_mbps=gsl_range[0],
    )
    _sample_missing_bandwidths(config, scene)
    return _apply_resource_overrides(config, scene)


def _route_policy_with_min_hops(base_policy: str, min_hops: int) -> str:
    if int(min_hops) <= 0:
        return base_policy
    return f"{base_policy}_ge{int(min_hops)}hops"


_PROFILE_CACHE: Optional[Dict[str, Dict]] = None
_PROFILE_DEVICE_OVERRIDE: Optional[str] = None
_SAT_MEMORY_RANGE_MB: Optional[Tuple[float, float]] = None


def set_profile_device_override(device: str | None) -> None:
    """Force all mode evaluators to use one profiled device family."""
    global _PROFILE_DEVICE_OVERRIDE
    if device in (None, "", "mixed"):
        _PROFILE_DEVICE_OVERRIDE = None
        return
    normalized = str(device).lower()
    if normalized not in {"pc", "jetson"}:
        raise ValueError(f"Unsupported profile device override: {device}")
    _PROFILE_DEVICE_OVERRIDE = normalized


def set_sat_memory_range_mb(value: str | Sequence[float] | None) -> None:
    """Override every LEO satellite memory with a stable value in the range."""
    global _SAT_MEMORY_RANGE_MB
    if value in (None, ""):
        _SAT_MEMORY_RANGE_MB = None
        return
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    else:
        parts = [str(item) for item in value]
    if len(parts) != 2:
        raise ValueError("sat memory range must have two values: min,max")
    lo, hi = float(parts[0]), float(parts[1])
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid sat memory range: {value}")
    _SAT_MEMORY_RANGE_MB = (lo, hi)


def _sat_memory_for_stk_id(scene: SlotScene, stk_id: str) -> float:
    if _SAT_MEMORY_RANGE_MB is None:
        return 0.0
    lo, hi = _SAT_MEMORY_RANGE_MB
    if math.isclose(lo, hi):
        return lo
    base_seed = int(scene.metadata.get("seed", 42))
    payload = f"{base_seed}|{stk_id}|sat_memory_mb"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return float(round(rng.uniform(lo, hi)))


def _apply_resource_overrides(config: dict, scene: SlotScene) -> dict:
    if _SAT_MEMORY_RANGE_MB is None:
        return config

    for node_id, node in config.get("nodes", {}).items():
        if not str(node_id).startswith("SAT-") and str(node.get("role", "")) != "leo_computing":
            continue
        hardware = node.setdefault("hardware", {})
        stk_id = str(node.get("stk_id", node_id))
        memory_mb = _sat_memory_for_stk_id(scene, stk_id)
        hardware["memory_mb"] = memory_mb
        node["memory_mb"] = memory_mb

    sim_paths = config.setdefault("simulation_paths", {})
    sim_paths["resource_override"] = {
        "sat_memory_range_mb": list(_SAT_MEMORY_RANGE_MB),
        "memory_assignment": "stable_uniform_by_stk_id",
    }
    return config


def _load_profiles() -> Dict[str, Dict]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        with Path("config/dnn_profiles_database_pc.json").open("r", encoding="utf-8") as f:
            pc_profiles = json.load(f)
        with Path("config/dnn_profiles_database_jetson.json").open("r", encoding="utf-8") as f:
            jetson_profiles = json.load(f)
        _PROFILE_CACHE = {"pc": pc_profiles, "jetson": jetson_profiles}
    return _PROFILE_CACHE


def _build_model_profile(scene: SlotScene) -> Dict:
    profiles = _load_profiles()
    config_key = f"b{scene.task.batch_size}_{scene.task.input_h}x{scene.task.input_w}"

    if _PROFILE_DEVICE_OVERRIDE:
        raw_profile = profiles[_PROFILE_DEVICE_OVERRIDE][scene.task.model_name].get(config_key)
        if raw_profile is None:
            raise KeyError(
                f"Missing profile for {scene.task.model_name}:{config_key} "
                f"on device={_PROFILE_DEVICE_OVERRIDE}"
            )
        layers = [raw_profile[str(idx)] for idx in range(len(raw_profile)) if str(idx) in raw_profile]
        input_mb = (scene.task.batch_size * 3 * scene.task.input_h * scene.task.input_w * 4) / (1024 ** 2)
        return {"layers": {"pc": layers, "jetson": layers}, "input_size_raw": input_mb}

    layers_dict = {"pc": [], "jetson": []}

    for device in ("pc", "jetson"):
        raw_profile = profiles[device][scene.task.model_name].get(config_key)
        if raw_profile is None and device == "jetson":
            raw_profile = profiles["pc"][scene.task.model_name].get(config_key)
        if raw_profile is None:
            raise KeyError(
                f"Missing profile for {scene.task.model_name}:{config_key} on device={device}"
            )
        for idx in range(len(raw_profile)):
            if str(idx) in raw_profile:
                layers_dict[device].append(raw_profile[str(idx)])

    input_mb = (scene.task.batch_size * 3 * scene.task.input_h * scene.task.input_w * 4) / (1024 ** 2)
    return {"layers": layers_dict, "input_size_raw": input_mb}


def _build_env_status(config: dict) -> Dict:
    raw_nodes = config.get("nodes", {})
    raw_links = config.get("links", {})
    pipeline = config.get("simulation_paths", {}).get("pipeline", [])
    compute_node_ids = pipeline[1:] if pipeline else [nid for nid in raw_nodes if "RS" not in nid]

    nodes = []
    for node_id in compute_node_ids:
        node_info = raw_nodes[node_id].copy()
        node_info["id"] = node_id
        if _PROFILE_DEVICE_OVERRIDE:
            node_info["device"] = "PC" if _PROFILE_DEVICE_OVERRIDE == "pc" else "Jetson"
        nodes.append(node_info)

    bandwidths = []
    propagation_delays = []
    current_source = pipeline[0] if pipeline else "RS"
    for node in nodes:
        target_node = node["id"]
        forward_key = f"{current_source}_to_{target_node}"
        backward_key = f"{target_node}_to_{current_source}"
        link = raw_links.get(forward_key) or raw_links.get(backward_key) or {}
        bandwidths.append(float(link.get("bandwidth_mbps", 100.0)))
        propagation_delays.append(float(link.get("propagation_delay_ms", 0.0)))
        current_source = target_node

    return {
        "nodes": nodes,
        "bandwidths": bandwidths,
        "propagation_delays_ms": propagation_delays,
        "reference_compute_speed": config.get("reference_compute_speed", 100.0),
    }


def evaluate_pmp_slot(
    scene: SlotScene,
    run_id: str,
    algorithm: str = "LA-DP",
    shared_route_eval: ModeEvaluation | None = None,
    route_policy: str = "selected_path",
) -> ModeEvaluation:
    """Evaluate PMP on the slot's already selected route."""

    config_path = scene.config_path
    route = scene.selected_stk_path
    pipeline_path = scene.pipeline_path
    candidate_id = scene.candidate_id
    candidate_path = str(scene.candidate_path) if scene.candidate_path else ""
    if shared_route_eval is not None:
        if (
            not shared_route_eval.feasible
            or not str(shared_route_eval.config_path).strip()
            or not str(shared_route_eval.route).strip()
        ):
            return ModeEvaluation(
                source_run_id=scene.source_run_id,
                slot_id=scene.slot_id,
                mode_family="PMP",
                mode_algo=algorithm,
                candidate_id="",
                route_policy=route_policy,
                feasible=False,
                reason="no_shared_gs_only_route_candidate",
                latency_ms="",
                satellite_energy_j="",
                energy_compute_j="",
                energy_comm_j="",
                satellite_compute_time_ms="",
                satellite_tx_time_ms="",
                active_sat_count="",
                hop_count="",
                route="",
                pipeline_path="",
                plan_json="{}",
                config_path="",
                candidate_path=candidate_path,
            )
        config_path = Path(str(shared_route_eval.config_path))
        route = _split_route_str(shared_route_eval.route)
        pipeline_path = _split_route_str(shared_route_eval.pipeline_path)
        candidate_id = f"shared_{shared_route_eval.candidate_id}" if shared_route_eval.candidate_id else "shared_gs_only_route"
    scheduler = Scheduler(net_config_path=str(config_path))
    plans = scheduler.generate_task_and_schedule(
        task_id=f"{scene.slot_id}_pmp",
        model_name=scene.task.model_name,
        batch_size=scene.task.batch_size,
        target_h=scene.task.input_h,
        target_w=scene.task.input_w,
        run_id=run_id,
        exp_type="mode_selection",
        mode="theory",
        persist_theory=False,
        algorithm_names=[algorithm],
        return_full_plans=True,
        profile_device=_PROFILE_DEVICE_OVERRIDE,
        metadata_extra={
            "sweep_param": "time_slot",
            "sweep_value": scene.slot_id,
        },
    )

    data = plans.get(algorithm, {})
    plan = data.get("plan") or {}
    latency = data.get("latency", float("inf"))
    feasible = bool(plan) and _finite(latency)
    reason = "" if feasible else "no_feasible_pmp_plan"

    return ModeEvaluation(
        source_run_id=scene.source_run_id,
        slot_id=scene.slot_id,
        mode_family="PMP",
        mode_algo=algorithm,
        candidate_id=candidate_id,
        route_policy=route_policy,
        feasible=feasible,
        reason=reason,
        latency_ms=float(latency) if _finite(latency) else "",
        satellite_energy_j=data.get("satellite_energy_j", "") if feasible else "",
        energy_compute_j=data.get("energy_compute_j", "") if feasible else "",
        energy_comm_j=data.get("energy_comm_j", "") if feasible else "",
        satellite_compute_time_ms=data.get("satellite_compute_time_ms", "") if feasible else "",
        satellite_tx_time_ms=data.get("satellite_tx_time_ms", "") if feasible else "",
        active_sat_count=_count_active_satellites(plan) if feasible else "",
        hop_count=len(pipeline_path) - 1 if pipeline_path else "",
        route=_route_to_str(route),
        pipeline_path=_route_to_str(pipeline_path),
        plan_json=json.dumps(plan, ensure_ascii=False, sort_keys=True),
        config_path=str(config_path),
        candidate_path=candidate_path,
    )


def _best_gs_only_candidate(
    scene: SlotScene,
    resolver: StkPathResolver,
    min_hops: int = 0,
):
    candidates = resolver.candidate_paths(scene)
    model_profile = _build_model_profile(scene)
    best = None

    for candidate in candidates:
        if int(candidate.hop_count) < int(min_hops):
            continue
        config = _network_config_for_candidate(scene, candidate)
        solver = PMPSolver(model_profile, _build_env_status(config))
        try:
            latency, plan = solver.solve_bent_pipe()
        except Exception:
            continue
        if not plan or not _finite(latency):
            continue
        energy = solver.estimate_satellite_energy(plan)
        pipeline = config.get("simulation_paths", {}).get("pipeline", [])
        candidate_row = {
            "candidate": candidate,
            "config": config,
            "pipeline": pipeline,
            "latency_ms": float(latency),
            "plan": plan,
            "energy": energy,
        }
        key = (
            candidate_row["latency_ms"],
            energy.get("satellite_energy_j", float("inf")),
            candidate.hop_count,
            _path_key(candidate.path),
        )
        if best is None or key < best[0]:
            best = (key, candidate_row)
    return best


def evaluate_gs_only_slot(
    scene: SlotScene,
    run_id: str,
    config_output_dir: str | Path,
    resolver: Optional[StkPathResolver] = None,
    min_hops: int = 0,
) -> ModeEvaluation:
    """Evaluate GS-Only on its own lowest predicted-latency route."""

    resolver = resolver or StkPathResolver()
    best = _best_gs_only_candidate(scene, resolver, min_hops=min_hops)
    route_policy = _route_policy_with_min_hops("min_predicted_gs_only_latency", min_hops)

    if best is None:
        reason = "no_gs_only_candidate"
        if int(min_hops) > 0:
            reason = f"no_gs_only_candidate_ge{int(min_hops)}hops"
        return ModeEvaluation(
            source_run_id=scene.source_run_id,
            slot_id=scene.slot_id,
            mode_family="GS-Only",
            mode_algo="Min-Latency-Route",
            candidate_id="",
            route_policy=route_policy,
            feasible=False,
            reason=reason,
            latency_ms="",
            satellite_energy_j="",
            energy_compute_j="",
            energy_comm_j="",
            satellite_compute_time_ms="",
            satellite_tx_time_ms="",
            active_sat_count="",
            hop_count="",
            route="",
            pipeline_path="",
            plan_json="{}",
            config_path="",
            candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
        )

    selected = best[1]
    candidate = selected["candidate"]
    config = selected["config"]
    config["simulation_paths"]["mode_selection"] = {
        "slot_id": scene.slot_id,
        "mode_family": "GS-Only",
        "route_policy": route_policy,
        "source_stk_run_id": scene.source_run_id,
    }
    config_path = Path(config_output_dir) / f"{scene.slot_id}_gs_only_network_config.json"
    _write_network_config(config_path, config)

    plan = selected["plan"]
    latency = selected["latency_ms"]
    energy = selected["energy"]
    pipeline = selected["pipeline"]

    return ModeEvaluation(
        source_run_id=scene.source_run_id,
        slot_id=scene.slot_id,
        mode_family="GS-Only",
        mode_algo="Min-Latency-Route",
        candidate_id=f"gs_only_rank_{int(candidate.rank):03d}",
        route_policy=route_policy,
        feasible=True,
        reason="",
        latency_ms=float(latency),
        satellite_energy_j=energy.get("satellite_energy_j", ""),
        energy_compute_j=energy.get("energy_compute_j", ""),
        energy_comm_j=energy.get("energy_comm_j", ""),
        satellite_compute_time_ms=energy.get("satellite_compute_time_ms", ""),
        satellite_tx_time_ms=energy.get("satellite_tx_time_ms", ""),
        active_sat_count=0,
        hop_count=len(pipeline) - 1 if pipeline else "",
        route=_route_to_str(candidate.path),
        pipeline_path=_route_to_str(pipeline),
        plan_json=json.dumps(plan, ensure_ascii=False, sort_keys=True),
        config_path=str(config_path),
        candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
    )


def _sat_only_metrics(solver: PMPSolver, compute_idx: int) -> Tuple[float, Dict, Dict]:
    feasible, reason = solver._check_segment_constraints(compute_idx, 0, solver.L)
    if not feasible:
        return float("inf"), {}, reason

    final_output_mb = solver.layers[solver.L - 1].get("comm_total_mb", solver.input_size_raw)
    compute_node_id = solver.nodes[compute_idx]["id"]
    compute_ms = solver._compute_delay(compute_idx, 0, solver.L)
    latency_ms = compute_ms
    satellite_tx_ms = 0.0
    comm_energy_j = 0.0

    for hop_idx in range(solver.K):
        source_id = "RS" if hop_idx == 0 else solver.nodes[hop_idx - 1]["id"]
        comm_mb = solver.input_size_raw if hop_idx <= compute_idx else final_output_mb
        latency_ms += solver._comm_delay_ms(comm_mb, solver.B[hop_idx], solver.P[hop_idx])
        if solver._is_satellite_node(source_id):
            tx_ms = solver._tx_delay_ms(comm_mb, solver.B[hop_idx])
            satellite_tx_ms += tx_ms
            comm_energy_j += 10.0 * (tx_ms / 1000.0)

    compute_energy_j = 15.0 * (compute_ms / 1000.0)
    plan = {compute_node_id: [0, solver.L - 1]}
    metrics = {
        "energy_compute_j": compute_energy_j,
        "energy_comm_j": comm_energy_j,
        "satellite_energy_j": compute_energy_j + comm_energy_j,
        "satellite_compute_time_ms": compute_ms,
        "satellite_tx_time_ms": satellite_tx_ms,
    }
    return latency_ms, plan, metrics


def _candidate_link_infos(config: dict) -> List[dict]:
    pipeline = config.get("simulation_paths", {}).get("pipeline", [])
    links = config.get("links", {})
    link_infos = []
    for src, dst in zip(pipeline[:-1], pipeline[1:]):
        link = links.get(f"{src}_to_{dst}") or links.get(f"{dst}_to_{src}") or {}
        link_infos.append(
            {
                "src": src,
                "dst": dst,
                "bandwidth_mbps": float(link.get("bandwidth_mbps", 100.0)),
                "propagation_delay_ms": float(link.get("propagation_delay_ms", 0.0)),
                "stk_from": link.get("stk_from"),
                "stk_to": link.get("stk_to"),
                "stk_link_type": link.get("stk_link_type"),
            }
        )
    return link_infos


def _equivalent_bandwidth_mbps(link_infos: Sequence[dict]) -> float:
    inverse_sum = 0.0
    for link in link_infos:
        bw = float(link.get("bandwidth_mbps", 0.0))
        if bw <= 0.0:
            return 0.0
        inverse_sum += 1.0 / bw
    return 1.0 / inverse_sum if inverse_sum > 0.0 else 0.0


def _path_delay_ms(data_mb: float, link_infos: Sequence[dict]) -> float:
    total = 0.0
    for link in link_infos:
        bw = float(link.get("bandwidth_mbps", 0.0))
        if bw <= 0.0:
            return float("inf")
        total += (data_mb * 8.0 / bw) * 1000.0
        total += float(link.get("propagation_delay_ms", 0.0))
    return total


def _satellite_tx_energy(data_mb: float, link_infos: Sequence[dict], tx_power_w: float = 10.0) -> Tuple[float, float]:
    energy_j = 0.0
    tx_time_ms = 0.0
    for link in link_infos:
        src = str(link.get("src", ""))
        if not src.upper().startswith("SAT"):
            continue
        bw = float(link.get("bandwidth_mbps", 0.0))
        if bw <= 0.0:
            return float("inf"), float("inf")
        tx_ms = (data_mb * 8.0 / bw) * 1000.0
        tx_time_ms += tx_ms
        energy_j += tx_power_w * (tx_ms / 1000.0)
    return energy_j, tx_time_ms


def _build_cdp_worker_options(scene: SlotScene, candidates: Sequence[CandidatePath]) -> Dict[str, dict]:
    model_profile = _build_model_profile(scene)
    options: Dict[str, dict] = {}

    for candidate in candidates:
        satellite_count = max(0, len(candidate.path) - 2)
        if satellite_count <= 0:
            continue

        config = _network_config_for_candidate(scene, candidate)
        env_status = _build_env_status(config)
        solver = PMPSolver(model_profile, env_status)
        pipeline = config.get("simulation_paths", {}).get("pipeline", [])
        link_infos = _candidate_link_infos(config)

        for sat_idx in range(satellite_count):
            feasible, _ = solver._check_segment_constraints(sat_idx, 0, solver.L)
            if not feasible:
                continue

            worker_stk_id = candidate.path[sat_idx + 1]
            worker_node_id = pipeline[sat_idx + 1]
            dispatch_links = link_infos[: sat_idx + 1]
            return_links = link_infos[sat_idx + 1 :]
            if not dispatch_links or not return_links:
                continue

            compute_full_ms = solver._compute_delay(sat_idx, 0, solver.L)
            if not _finite(compute_full_ms):
                continue

            input_mb = model_profile["input_size_raw"]
            output_mb = solver.layers[solver.L - 1].get("comm_total_mb", input_mb)
            single_worker_latency = (
                _path_delay_ms(input_mb, dispatch_links)
                + compute_full_ms
                + _path_delay_ms(output_mb, return_links)
            )
            option = {
                "worker_stk_id": worker_stk_id,
                "worker_node_id": worker_node_id,
                "candidate": candidate,
                "config": config,
                "pipeline": pipeline,
                "dispatch_links": dispatch_links,
                "return_links": return_links,
                "dispatch_route": candidate.path[: sat_idx + 2],
                "return_route": candidate.path[sat_idx + 1 :],
                "compute_full_model_ms": compute_full_ms,
                "b_dist_mbps": _equivalent_bandwidth_mbps(dispatch_links),
                "b_return_mbps": _equivalent_bandwidth_mbps(return_links),
                "dist_prop_ms": sum(link["propagation_delay_ms"] for link in dispatch_links),
                "return_prop_ms": sum(link["propagation_delay_ms"] for link in return_links),
                "single_worker_latency_ms": single_worker_latency,
            }
            current = options.get(worker_stk_id)
            key = (single_worker_latency, candidate.hop_count, _path_key(candidate.path), worker_node_id)
            current_key = None
            if current is not None:
                current_key = (
                    current["single_worker_latency_ms"],
                    current["candidate"].hop_count,
                    _path_key(current["candidate"].path),
                    current["worker_node_id"],
                )
            if current is None or key < current_key:
                options[worker_stk_id] = option

    return options


def _cdp_gamma(option: dict, input_mb: float, output_mb: float) -> float:
    if input_mb <= 0.0:
        return 0.0
    b_dist = float(option["b_dist_mbps"])
    b_return = float(option["b_return_mbps"])
    if b_dist <= 0.0 or b_return <= 0.0:
        return 0.0
    unit_cost = (
        8.0 * 1000.0 / b_dist
        + float(option["compute_full_model_ms"]) / input_mb
        + (output_mb / input_mb) * 8.0 * 1000.0 / b_return
    )
    return 1.0 / unit_cost if unit_cost > 0.0 else 0.0


def _evaluate_cdp_exact(options: Sequence[dict], allocation_plan: dict, input_mb: float, output_mb: float, batch_size: int) -> Dict:
    samples_by_worker = allocation_plan["samples"]
    delays = {}
    compute_energy_j = 0.0
    comm_energy_j = 0.0
    satellite_compute_time_ms = 0.0
    satellite_tx_time_ms = 0.0

    for option in options:
        worker_id = option["worker_stk_id"]
        samples = int(samples_by_worker.get(worker_id, 0))
        input_slice_mb = input_mb * (samples / batch_size)
        output_slice_mb = output_mb * (samples / batch_size)
        dispatch_ms = _path_delay_ms(input_slice_mb, option["dispatch_links"])
        compute_ms = float(option["compute_full_model_ms"]) * (samples / batch_size)
        return_ms = _path_delay_ms(output_slice_mb, option["return_links"])
        worker_delay = dispatch_ms + compute_ms + return_ms
        delays[worker_id] = worker_delay

        satellite_compute_time_ms += compute_ms
        compute_energy_j += 15.0 * (compute_ms / 1000.0)

        dispatch_energy, dispatch_tx_ms = _satellite_tx_energy(input_slice_mb, option["dispatch_links"])
        return_energy, return_tx_ms = _satellite_tx_energy(output_slice_mb, option["return_links"])
        comm_energy_j += dispatch_energy + return_energy
        satellite_tx_time_ms += dispatch_tx_ms + return_tx_ms

    latency_ms = max(delays.values()) if delays else float("inf")
    return {
        "latency_ms": latency_ms,
        "worker_delays_ms": delays,
        "energy_compute_j": compute_energy_j,
        "energy_comm_j": comm_energy_j,
        "satellite_energy_j": compute_energy_j + comm_energy_j,
        "satellite_compute_time_ms": satellite_compute_time_ms,
        "satellite_tx_time_ms": satellite_tx_time_ms,
    }


def evaluate_sat_only_slot(
    scene: SlotScene,
    run_id: str,
    config_output_dir: str | Path,
    resolver: Optional[StkPathResolver] = None,
    shared_route_eval: ModeEvaluation | None = None,
    route_policy: str = "best_single_satellite_over_candidate_paths",
) -> ModeEvaluation:
    """Evaluate Sat-Only by choosing the fastest single compute satellite."""

    model_profile = _build_model_profile(scene)
    best = None
    if shared_route_eval is not None:
        if (
            not shared_route_eval.feasible
            or not str(shared_route_eval.config_path).strip()
            or not str(shared_route_eval.route).strip()
        ):
            return ModeEvaluation(
                source_run_id=scene.source_run_id,
                slot_id=scene.slot_id,
                mode_family="Sat-Only",
                mode_algo="Min-Latency-Single-Sat",
                candidate_id="",
                route_policy=route_policy,
                feasible=False,
                reason="no_shared_route_for_sat_only",
                latency_ms="",
                satellite_energy_j="",
                energy_compute_j="",
                energy_comm_j="",
                satellite_compute_time_ms="",
                satellite_tx_time_ms="",
                active_sat_count="",
                hop_count="",
                route="",
                pipeline_path="",
                plan_json="{}",
                config_path="",
                candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
            )
        config_path = Path(str(shared_route_eval.config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        env_status = _build_env_status(config)
        solver = PMPSolver(model_profile, env_status)
        pipeline = config.get("simulation_paths", {}).get("pipeline", [])
        satellite_count = max(0, len(pipeline) - 2)
        for compute_idx in range(satellite_count):
            latency_ms, plan, metrics = _sat_only_metrics(solver, compute_idx)
            if not plan or not _finite(latency_ms):
                continue
            sat_node_id = pipeline[compute_idx + 1]
            candidate_row = {
                "candidate": None,
                "config": config,
                "config_path": config_path,
                "pipeline": pipeline,
                "route": _split_route_str(shared_route_eval.route),
                "compute_node": sat_node_id,
                "latency_ms": float(latency_ms),
                "plan": plan,
                "metrics": metrics,
                "hop_count": len(pipeline) - 1 if pipeline else 0,
                "candidate_id": f"shared_{shared_route_eval.candidate_id}" if shared_route_eval.candidate_id else "shared_route",
            }
            key = (
                candidate_row["latency_ms"],
                metrics["satellite_energy_j"],
                candidate_row["hop_count"],
                _path_key(candidate_row["route"]),
                sat_node_id,
            )
            if best is None or key < best[0]:
                best = (key, candidate_row)
    else:
        resolver = resolver or StkPathResolver()
        candidates = resolver.candidate_paths(scene)
        for candidate in candidates:
            satellite_count = max(0, len(candidate.path) - 2)
            if satellite_count <= 0:
                continue

            config = _network_config_for_candidate(scene, candidate)
            env_status = _build_env_status(config)
            solver = PMPSolver(model_profile, env_status)
            pipeline = config.get("simulation_paths", {}).get("pipeline", [])

            for compute_idx in range(satellite_count):
                latency_ms, plan, metrics = _sat_only_metrics(solver, compute_idx)
                if not plan or not _finite(latency_ms):
                    continue
                sat_node_id = pipeline[compute_idx + 1]
                candidate_row = {
                    "candidate": candidate,
                    "config": config,
                    "config_path": None,
                    "pipeline": pipeline,
                    "route": candidate.path,
                    "compute_node": sat_node_id,
                    "latency_ms": float(latency_ms),
                    "plan": plan,
                    "metrics": metrics,
                    "hop_count": candidate.hop_count,
                    "candidate_id": f"sat_only_rank_{int(candidate.rank):03d}",
                }
                key = (
                    candidate_row["latency_ms"],
                    metrics["satellite_energy_j"],
                    candidate.hop_count,
                    _path_key(candidate.path),
                    sat_node_id,
                )
                if best is None or key < best[0]:
                    best = (key, candidate_row)

    if best is None:
        return ModeEvaluation(
            source_run_id=scene.source_run_id,
            slot_id=scene.slot_id,
            mode_family="Sat-Only",
            mode_algo="Min-Latency-Single-Sat",
            candidate_id="",
            route_policy=route_policy,
            feasible=False,
            reason="no_feasible_sat_only_candidate",
            latency_ms="",
            satellite_energy_j="",
            energy_compute_j="",
            energy_comm_j="",
            satellite_compute_time_ms="",
            satellite_tx_time_ms="",
            active_sat_count="",
            hop_count="",
            route="",
            pipeline_path="",
            plan_json="{}",
            config_path="",
            candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
        )

    selected = best[1]
    config = selected["config"]
    config_path = selected.get("config_path")
    if config_path is None:
        config["simulation_paths"]["mode_selection"] = {
            "slot_id": scene.slot_id,
            "mode_family": "Sat-Only",
            "route_policy": route_policy,
            "compute_node": selected["compute_node"],
            "source_stk_run_id": scene.source_run_id,
        }
        config_path = Path(config_output_dir) / f"{scene.slot_id}_sat_only_network_config.json"
        _write_network_config(config_path, config)

    metrics = selected["metrics"]
    return ModeEvaluation(
        source_run_id=scene.source_run_id,
        slot_id=scene.slot_id,
        mode_family="Sat-Only",
        mode_algo="Min-Latency-Single-Sat",
        candidate_id=f"{selected['candidate_id']}_{selected['compute_node']}",
        route_policy=route_policy,
        feasible=True,
        reason="",
        latency_ms=selected["latency_ms"],
        satellite_energy_j=metrics["satellite_energy_j"],
        energy_compute_j=metrics["energy_compute_j"],
        energy_comm_j=metrics["energy_comm_j"],
        satellite_compute_time_ms=metrics["satellite_compute_time_ms"],
        satellite_tx_time_ms=metrics["satellite_tx_time_ms"],
        active_sat_count=1,
        hop_count=selected["hop_count"],
        route=_route_to_str(selected["route"]),
        pipeline_path=_route_to_str(selected["pipeline"]),
        plan_json=json.dumps(selected["plan"], ensure_ascii=False, sort_keys=True),
        config_path=str(config_path),
        candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
    )


def evaluate_cdp_slot(
    scene: SlotScene,
    run_id: str,
    config_output_dir: str | Path,
    resolver: Optional[StkPathResolver] = None,
    max_workers: int = 4,
    worker_pool_limit: int = 8,
) -> ModeEvaluation:
    """Evaluate no-aggregator CDP with discrete LAWA allocation."""

    resolver = resolver or StkPathResolver()
    candidates = resolver.candidate_paths(scene)
    model_profile = _build_model_profile(scene)
    input_mb = float(model_profile["input_size_raw"])

    # Use a one-hop solver only to access the profile's final output size.
    probe_config = _network_config_for_candidate(scene, candidates[0]) if candidates else scene.network_config
    probe_solver = PMPSolver(model_profile, _build_env_status(probe_config))
    output_mb = float(probe_solver.layers[probe_solver.L - 1].get("comm_total_mb", input_mb))
    batch_size = int(scene.task.batch_size)

    options_by_worker = _build_cdp_worker_options(scene, candidates)
    if len(options_by_worker) < 2:
        return ModeEvaluation(
            source_run_id=scene.source_run_id,
            slot_id=scene.slot_id,
            mode_family="CDP",
            mode_algo="LAWA-Discrete",
            candidate_id="",
            route_policy="best_lawa_worker_set_no_aggregator",
            feasible=False,
            reason="fewer_than_two_feasible_workers",
            latency_ms="",
            satellite_energy_j="",
            energy_compute_j="",
            energy_comm_j="",
            satellite_compute_time_ms="",
            satellite_tx_time_ms="",
            active_sat_count="",
            hop_count="",
            route="",
            pipeline_path="",
            plan_json="{}",
            config_path="",
            candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
        )

    ranked_options = sorted(
        options_by_worker.values(),
        key=lambda option: (
            -_cdp_gamma(option, input_mb, output_mb),
            option["single_worker_latency_ms"],
            option["worker_stk_id"],
        ),
    )[: max(2, worker_pool_limit)]

    max_workers = max(2, min(int(max_workers), len(ranked_options), batch_size))
    best = None
    for worker_count in range(2, max_workers + 1):
        for combo in itertools.combinations(ranked_options, worker_count):
            cdp_env = {
                "nodes": [
                    {
                        "id": option["worker_stk_id"],
                        "compute_full_model_ms": option["compute_full_model_ms"],
                        "b_dist_mbps": option["b_dist_mbps"],
                        "b_return_mbps": option["b_return_mbps"],
                        "dist_prop_ms": option["dist_prop_ms"],
                        "return_prop_ms": option["return_prop_ms"],
                    }
                    for option in combo
                ]
            }
            cdp_profile = {
                "input_size_mb": input_mb,
                "output_size_mb": output_mb,
                "batch_size": batch_size,
            }
            try:
                _, allocation = CDPSolver(cdp_profile, cdp_env).solve_lawa_discrete(batch_size=batch_size)
            except ValueError:
                continue
            exact = _evaluate_cdp_exact(combo, allocation, input_mb, output_mb, batch_size)
            if not _finite(exact["latency_ms"]):
                continue
            key = (
                exact["latency_ms"],
                exact["satellite_energy_j"],
                worker_count,
                "|".join(option["worker_stk_id"] for option in combo),
            )
            if best is None or key < best[0]:
                best = (key, combo, allocation, exact)

    if best is None:
        return ModeEvaluation(
            source_run_id=scene.source_run_id,
            slot_id=scene.slot_id,
            mode_family="CDP",
            mode_algo="LAWA-Discrete",
            candidate_id="",
            route_policy="best_lawa_worker_set_no_aggregator",
            feasible=False,
            reason="no_feasible_lawa_worker_set",
            latency_ms="",
            satellite_energy_j="",
            energy_compute_j="",
            energy_comm_j="",
            satellite_compute_time_ms="",
            satellite_tx_time_ms="",
            active_sat_count="",
            hop_count="",
            route="",
            pipeline_path="",
            plan_json="{}",
            config_path="",
            candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
        )

    _, selected_options, allocation, exact = best
    worker_ids = [option["worker_stk_id"] for option in selected_options]
    plan = {
        "workers": worker_ids,
        "batch_size": batch_size,
        "split_samples": allocation["samples"],
        "split_ratio": allocation["split_ratio"],
        "alloc_mb": allocation["alloc_mb"],
        "continuous_alloc_mb": allocation["continuous_alloc_mb"],
        "worker_delays_ms": exact["worker_delays_ms"],
        "dispatch_routes": {
            option["worker_stk_id"]: option["dispatch_route"]
            for option in selected_options
        },
        "return_routes": {
            option["worker_stk_id"]: option["return_route"]
            for option in selected_options
        },
    }

    config_payload = {
        "source_run_id": scene.source_run_id,
        "slot_id": scene.slot_id,
        "mode_family": "CDP",
        "mode_algo": "LAWA-Discrete",
        "route_policy": "best_lawa_worker_set_no_aggregator",
        "no_aggregator": True,
        "workers": [
            {
                "worker_stk_id": option["worker_stk_id"],
                "worker_node_id": option["worker_node_id"],
                "candidate_rank": option["candidate"].rank,
                "candidate_path": option["candidate"].path,
                "dispatch_route": option["dispatch_route"],
                "return_route": option["return_route"],
                "b_dist_mbps": option["b_dist_mbps"],
                "b_return_mbps": option["b_return_mbps"],
                "compute_full_model_ms": option["compute_full_model_ms"],
            }
            for option in selected_options
        ],
    }
    config_path = Path(config_output_dir) / f"{scene.slot_id}_cdp_network_config.json"
    _write_network_config(config_path, config_payload)

    return ModeEvaluation(
        source_run_id=scene.source_run_id,
        slot_id=scene.slot_id,
        mode_family="CDP",
        mode_algo="LAWA-Discrete",
        candidate_id="cdp_workers_" + "_".join(worker_ids),
        route_policy="best_lawa_worker_set_no_aggregator",
        feasible=True,
        reason="",
        latency_ms=exact["latency_ms"],
        satellite_energy_j=exact["satellite_energy_j"],
        energy_compute_j=exact["energy_compute_j"],
        energy_comm_j=exact["energy_comm_j"],
        satellite_compute_time_ms=exact["satellite_compute_time_ms"],
        satellite_tx_time_ms=exact["satellite_tx_time_ms"],
        active_sat_count=len(worker_ids),
        hop_count=max(option["candidate"].hop_count for option in selected_options),
        route=";".join(worker_ids),
        pipeline_path=";".join(
            _route_to_str(option["candidate"].path)
            for option in selected_options
        ),
        plan_json=json.dumps(plan, ensure_ascii=False, sort_keys=True),
        config_path=str(config_path),
        candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
    )
