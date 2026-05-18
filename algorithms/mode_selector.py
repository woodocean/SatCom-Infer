"""Mode-selection algorithms for collaborative inference experiments.

This module keeps selector logic separate from experiment orchestration:

* FWMS-Feature is the thesis-facing resource-driven mode selector.
* Oracle-Min-Latency is an offline upper-bound baseline, not an online policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from algorithms.pmp_solver import PMPSolver
from core.mode_evaluators import ModeEvaluation
from core.mode_evaluators import (
    _build_cdp_worker_options,
    _build_env_status,
    _build_model_profile,
    _network_config_for_candidate,
    StkPathResolver,
)
from core.mode_scene_builder import SlotScene


_PROFILE_CACHE: dict | None = None


def finite_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _mode_candidate_summary(evaluations: List[ModeEvaluation]) -> List[dict]:
    candidates = []
    for evaluation in evaluations:
        candidates.append(
            {
                "mode_family": evaluation.mode_family,
                "mode_algo": evaluation.mode_algo,
                "feasible": bool(evaluation.feasible),
                "reason": evaluation.reason,
                "latency_ms": evaluation.latency_ms,
                "satellite_energy_j": evaluation.satellite_energy_j,
                "candidate_id": evaluation.candidate_id,
                "route_policy": evaluation.route_policy,
            }
        )
    return candidates


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0.0:
        return default
    return float(numerator) / float(denominator)


def _harmonic_mean(values: Sequence[float]) -> float:
    usable = [float(value) for value in values if float(value) > 0.0]
    if not usable:
        return 0.0
    return len(usable) / sum(1.0 / value for value in usable)


def _load_json_file(path_value: str | Path | None) -> dict:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _plan_dict(evaluation: ModeEvaluation | None) -> dict:
    if evaluation is None or not evaluation.plan_json:
        return {}
    try:
        return json.loads(evaluation.plan_json)
    except json.JSONDecodeError:
        return {}


def _node_memory_mb(node: dict) -> float:
    hardware = node.get("hardware", {}) if isinstance(node.get("hardware", {}), dict) else {}
    return float(node.get("memory_mb", hardware.get("memory_mb", 0.0)))


def _node_compute_tflops(node: dict) -> float:
    hardware = node.get("hardware", {}) if isinstance(node.get("hardware", {}), dict) else {}
    return float(
        hardware.get(
            "compute_speed_tflops",
            hardware.get("compute_speed_gflops_per_ms", 0.0),
        )
    )


def _load_pc_profile(scene: SlotScene) -> List[dict]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        with Path("config/dnn_profiles_database_pc.json").open("r", encoding="utf-8") as f:
            _PROFILE_CACHE = json.load(f)
    config_key = f"b{scene.task.batch_size}_{scene.task.input_h}x{scene.task.input_w}"
    raw_profile = _PROFILE_CACHE[scene.task.model_name][config_key]
    return [raw_profile[str(idx)] for idx in range(len(raw_profile)) if str(idx) in raw_profile]


def _input_size_mb(scene: SlotScene) -> float:
    return (scene.task.batch_size * 3 * scene.task.input_h * scene.task.input_w * 4) / (1024**2)


def _data_expansion_ratio(scene: SlotScene) -> float:
    input_mb = _input_size_mb(scene)
    if input_mb <= 0.0:
        return 0.0
    layers = _load_pc_profile(scene)
    comm_values = [float(layer.get("comm_total_mb", 0.0)) for layer in layers]
    comm_values = [value for value in comm_values if value > 0.0]
    if not comm_values:
        return 0.0
    return sum(value / input_mb for value in comm_values) / len(comm_values)


def _average_pmp_route_bandwidth(scene: SlotScene) -> float:
    pipeline = scene.network_config.get("simulation_paths", {}).get("pipeline", [])
    links = scene.network_config.get("links", {})
    bandwidths = []
    for src, dst in zip(pipeline[:-1], pipeline[1:]):
        link = links.get(f"{src}_to_{dst}") or links.get(f"{dst}_to_{src}") or {}
        bw = float(link.get("bandwidth_mbps", 0.0))
        if bw > 0.0:
            bandwidths.append(bw)
    return sum(bandwidths) / len(bandwidths) if bandwidths else 0.0


def _compute_heterogeneity_from_cdp(cdp_evaluation: ModeEvaluation | None) -> float:
    if not cdp_evaluation or not cdp_evaluation.feasible or not cdp_evaluation.config_path:
        return 0.0
    config_path = Path(cdp_evaluation.config_path)
    if not config_path.exists():
        return 0.0
    try:
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0.0

    speeds = []
    for worker in payload.get("workers", []):
        compute_ms = float(worker.get("compute_full_model_ms", 0.0))
        if compute_ms > 0.0:
            speeds.append(1.0 / compute_ms)
    if len(speeds) < 2:
        return 0.0
    mean_speed = sum(speeds) / len(speeds)
    if mean_speed <= 0.0:
        return 0.0
    variance = sum((speed - mean_speed) ** 2 for speed in speeds) / len(speeds)
    return (variance**0.5) / mean_speed


def normalized_feature_values(scene: SlotScene, cdp_evaluation: ModeEvaluation | None) -> dict:
    eta = _data_expansion_ratio(scene)
    rho = _compute_heterogeneity_from_cdp(cdp_evaluation)
    b_bar = _average_pmp_route_bandwidth(scene)
    return {
        "eta_data_expansion_ratio": eta,
        "rho_compute_heterogeneity": rho,
        "b_bar_route_avg_bandwidth_mbps": b_bar,
        "eta_norm": eta / (1.0 + eta) if eta >= 0.0 else 0.0,
        "rho_norm": min(max(rho, 0.0), 1.0),
        "b_bar_norm": b_bar / (b_bar + 1000.0) if b_bar > 0.0 else 0.0,
    }


def _model_structure_features(scene: SlotScene) -> dict:
    model_profile = _build_model_profile(scene)
    layers = model_profile["layers"]["pc"]
    input_mb = float(model_profile["input_size_raw"])
    if not layers:
        return {
            "layer_count": 0,
            "input_mb": input_mb,
            "output_mb": input_mb,
            "return_ratio": 1.0,
            "return_lightness": 0.0,
            "min_intermediate_ratio": 1.0,
            "global_compression_gain": 0.0,
        }

    output_mb = float(layers[-1].get("comm_total_mb", input_mb))
    min_intermediate_mb = min(float(layer.get("comm_total_mb", input_mb)) for layer in layers)
    return_ratio = _safe_ratio(output_mb, input_mb, 1.0)
    min_intermediate_ratio = _safe_ratio(min_intermediate_mb, input_mb, 1.0)
    return {
        "layer_count": len(layers),
        "input_mb": input_mb,
        "output_mb": output_mb,
        "return_ratio": return_ratio,
        "return_lightness": _clamp01(1.0 - min(1.0, return_ratio)),
        "min_intermediate_ratio": min_intermediate_ratio,
        "global_compression_gain": _clamp01(1.0 - min(1.0, min_intermediate_ratio)),
    }


def _pipeline_link_stats(config: dict) -> dict:
    pipeline = config.get("simulation_paths", {}).get("pipeline", [])
    links = config.get("links", {})
    isl_bandwidths: list[float] = []
    gsl_bandwidths: list[float] = []
    all_bandwidths: list[float] = []

    for src, dst in zip(pipeline[:-1], pipeline[1:]):
        link = links.get(f"{src}_to_{dst}") or links.get(f"{dst}_to_{src}") or {}
        bandwidth = float(link.get("bandwidth_mbps", 0.0))
        if bandwidth <= 0.0:
            continue
        all_bandwidths.append(bandwidth)
        link_class = str(link.get("effective_bandwidth_class", "")).upper()
        link_type = str(link.get("stk_link_type", "")).upper()
        if link_class == "GSL" or link_type == "SAT-GS":
            gsl_bandwidths.append(bandwidth)
        else:
            isl_bandwidths.append(bandwidth)

    return {
        "route_bottleneck_bw_mbps": min(all_bandwidths) if all_bandwidths else 0.0,
        "route_isl_bw_mbps": min(isl_bandwidths) if isl_bandwidths else 0.0,
        "route_gsl_bw_mbps": min(gsl_bandwidths) if gsl_bandwidths else 0.0,
    }


def _memory_margin_norm(memory_limit_mb: float, memory_required_mb: float) -> float:
    if memory_limit_mb <= 0.0 or memory_required_mb <= 0.0:
        return 0.0
    ratio = memory_limit_mb / memory_required_mb
    return _clamp01((ratio - 1.0) / 1.5)


def _transfer_time_ms(data_mb: float, bandwidth_mbps: float) -> float:
    if data_mb <= 0.0 or bandwidth_mbps <= 0.0:
        return 0.0
    return float(data_mb) * 8.0 * 1000.0 / float(bandwidth_mbps)


def _pmp_route_features(scene: SlotScene, shared_route_eval: ModeEvaluation | None) -> dict:
    config = _load_json_file(shared_route_eval.config_path if shared_route_eval else "")
    if not config:
        config = scene.network_config

    model_profile = _build_model_profile(scene)
    env_status = _build_env_status(config)
    solver = PMPSolver(model_profile, env_status)
    link_stats = _pipeline_link_stats(config)
    nodes = env_status.get("nodes", [])
    route_sat_indices = [
        idx
        for idx, node in enumerate(nodes)
        if str(node.get("role", "")).lower() == "leo_computing"
    ]
    input_mb = float(model_profile["input_size_raw"])
    layer_count = int(solver.L)

    full_model_required_mb = 0.0
    if route_sat_indices:
        _, full_details = solver._check_segment_constraints(route_sat_indices[0], 0, layer_count)
        full_model_required_mb = float(full_details.get("memory_required_mb", 0.0))

    gs_indices = [
        idx
        for idx, node in enumerate(nodes)
        if str(node.get("role", "")).lower() == "ground_station"
    ]
    gs_full_compute_ms = 0.0
    if gs_indices:
        gs_full_compute_ms = float(solver._compute_delay(gs_indices[0], 0, layer_count))
    gs_input_tx_ms = _transfer_time_ms(input_mb, link_stats["route_gsl_bw_mbps"])

    best_prefix = {
        "score": float("-inf"),
        "compression_gain": 0.0,
        "depth_norm": 0.0,
        "output_ratio": 1.0,
        "memory_margin_norm": 0.0,
        "compute_tflops": 0.0,
        "node_id": "",
        "end_layer": 0,
    }
    best_single = {
        "compute_tflops": 0.0,
        "memory_margin_norm": 0.0,
        "node_id": "",
        "full_compute_ms": 0.0,
    }
    full_fit_route_count = 0

    for node_idx in route_sat_indices:
        node = nodes[node_idx]
        node_mem_mb = _node_memory_mb(node)
        node_tflops = _node_compute_tflops(node)
        node_full_compute_ms = float(solver._compute_delay(node_idx, 0, layer_count))

        full_fit, full_details = solver._check_segment_constraints(node_idx, 0, layer_count)
        if full_fit:
            full_fit_route_count += 1
            margin_norm = _memory_margin_norm(
                node_mem_mb,
                float(full_details.get("memory_required_mb", 0.0)),
            )
            single_key = (node_tflops, margin_norm, -node_full_compute_ms, str(node.get("id", "")))
            current_key = (
                best_single["compute_tflops"],
                best_single["memory_margin_norm"],
                -best_single["full_compute_ms"],
                best_single["node_id"],
            )
            if single_key > current_key:
                best_single = {
                    "compute_tflops": node_tflops,
                    "memory_margin_norm": margin_norm,
                    "node_id": str(node.get("id", "")),
                    "full_compute_ms": node_full_compute_ms,
                }

        for end_layer in range(1, layer_count):
            feasible, details = solver._check_segment_constraints(node_idx, 0, end_layer)
            if not feasible:
                continue
            output_mb = float(solver.layers[end_layer - 1].get("comm_total_mb", input_mb))
            output_ratio = _safe_ratio(output_mb, input_mb, 1.0)
            compression_gain = _clamp01(1.0 - min(1.0, output_ratio))
            depth_norm = _safe_ratio(end_layer, layer_count, 0.0)
            margin_norm = _memory_margin_norm(
                node_mem_mb,
                float(details.get("memory_required_mb", 0.0)),
            )
            score = 0.65 * compression_gain + 0.25 * depth_norm + 0.10 * margin_norm
            key = (score, compression_gain, depth_norm, node_tflops)
            current = (
                best_prefix["score"],
                best_prefix["compression_gain"],
                best_prefix["depth_norm"],
                best_prefix["compute_tflops"],
            )
            if key > current:
                best_prefix = {
                    "score": score,
                    "compression_gain": compression_gain,
                    "depth_norm": depth_norm,
                    "output_ratio": output_ratio,
                    "memory_margin_norm": margin_norm,
                    "compute_tflops": node_tflops,
                    "node_id": str(node.get("id", "")),
                    "end_layer": end_layer,
                }

    strongest_route_compute = max((_node_compute_tflops(nodes[idx]) for idx in route_sat_indices), default=0.0)
    best_sat_full_compute_ms = best_single["full_compute_ms"]
    gs_direct_total_ms = gs_input_tx_ms + gs_full_compute_ms
    gs_direct_advantage_norm = 0.0
    if best_sat_full_compute_ms > 0.0 and gs_direct_total_ms > 0.0:
        gs_direct_advantage_norm = _clamp01(
            best_sat_full_compute_ms / (best_sat_full_compute_ms + gs_direct_total_ms)
        )

    return {
        "route_sat_count": len(route_sat_indices),
        "route_bottleneck_bw_mbps": link_stats["route_bottleneck_bw_mbps"],
        "route_isl_bw_mbps": link_stats["route_isl_bw_mbps"],
        "route_gsl_bw_mbps": link_stats["route_gsl_bw_mbps"],
        "strongest_route_compute_tflops": strongest_route_compute,
        "full_fit_route_count": full_fit_route_count,
        "full_model_required_mb": full_model_required_mb,
        "best_prefix_compression_gain": best_prefix["compression_gain"],
        "best_prefix_depth_norm": best_prefix["depth_norm"],
        "best_prefix_output_ratio": best_prefix["output_ratio"],
        "best_prefix_memory_margin_norm": best_prefix["memory_margin_norm"],
        "best_prefix_compute_tflops": best_prefix["compute_tflops"],
        "best_prefix_node_id": best_prefix["node_id"],
        "best_prefix_end_layer": best_prefix["end_layer"],
        "best_single_compute_tflops": best_single["compute_tflops"],
        "best_single_memory_margin_norm": best_single["memory_margin_norm"],
        "best_single_node_id": best_single["node_id"],
        "best_single_full_compute_ms": best_sat_full_compute_ms,
        "gs_full_compute_ms": gs_full_compute_ms,
        "gs_input_tx_ms": gs_input_tx_ms,
        "gs_direct_total_ms": gs_direct_total_ms,
        "gs_direct_advantage_norm": gs_direct_advantage_norm,
    }


def _cdp_pool_features(
    scene: SlotScene,
    resolver: StkPathResolver | None,
    max_workers: int,
) -> dict:
    resolver = resolver or StkPathResolver()
    candidates = resolver.candidate_paths(scene)
    options = _build_cdp_worker_options(scene, candidates)
    if not options:
        return {
            "cdp_worker_option_count": 0,
            "cdp_pool_worker_count": 0,
            "cdp_parallel_gain": 0.0,
            "cdp_dispatch_hmean_bw_mbps": 0.0,
            "cdp_return_hmean_bw_mbps": 0.0,
            "cdp_dispatch_min_bw_mbps": 0.0,
            "cdp_return_min_bw_mbps": 0.0,
            "cdp_best_worker_compute_ms": 0.0,
            "cdp_pool_workers": [],
        }

    ranked = sorted(
        options.values(),
        key=lambda item: (
            item["single_worker_latency_ms"],
            item["compute_full_model_ms"],
            item["worker_stk_id"],
        ),
    )
    pool = ranked[: max(2, min(max_workers, len(ranked), int(scene.task.batch_size)))]
    compute_ms = [float(item["compute_full_model_ms"]) for item in pool]
    best_compute_ms = min(compute_ms) if compute_ms else 0.0
    parallel_gain = 0.0
    if best_compute_ms > 0.0:
        parallel_gain = sum(best_compute_ms / value for value in compute_ms if value > 0.0)

    dispatch_bandwidths = [float(item["b_dist_mbps"]) for item in pool]
    return_bandwidths = [float(item["b_return_mbps"]) for item in pool]

    return {
        "cdp_worker_option_count": len(options),
        "cdp_pool_worker_count": len(pool),
        "cdp_parallel_gain": parallel_gain,
        "cdp_dispatch_hmean_bw_mbps": _harmonic_mean(dispatch_bandwidths),
        "cdp_return_hmean_bw_mbps": _harmonic_mean(return_bandwidths),
        "cdp_dispatch_min_bw_mbps": min(dispatch_bandwidths) if dispatch_bandwidths else 0.0,
        "cdp_return_min_bw_mbps": min(return_bandwidths) if return_bandwidths else 0.0,
        "cdp_best_worker_compute_ms": best_compute_ms,
        "cdp_pool_workers": [str(item["worker_stk_id"]) for item in pool],
    }


def _resource_driven_feature_values(
    scene: SlotScene,
    mode_evaluations: List[ModeEvaluation],
    resolver: StkPathResolver | None = None,
    max_workers: int = 4,
) -> dict:
    gs_only = _mode_by_family(mode_evaluations, "GS-Only")
    pmp = _mode_by_family(mode_evaluations, "PMP")
    sat_only = _mode_by_family(mode_evaluations, "Sat-Only")
    cdp = _mode_by_family(mode_evaluations, "CDP")
    model_features = _model_structure_features(scene)
    pmp_features = _pmp_route_features(scene, pmp or gs_only)
    cdp_features = _cdp_pool_features(scene, resolver=resolver, max_workers=max_workers)

    features = {
        **model_features,
        **pmp_features,
        **cdp_features,
        "pmp_feasible": bool(pmp and pmp.feasible),
        "sat_only_feasible": bool(sat_only and sat_only.feasible),
        "cdp_feasible": bool(cdp and cdp.feasible),
        "gs_only_feasible": bool(gs_only and gs_only.feasible),
        "compute_norm_single": _safe_ratio(
            pmp_features["best_single_compute_tflops"],
            pmp_features["best_single_compute_tflops"] + 6.0,
            0.0,
        ),
        "compute_norm_prefix": _safe_ratio(
            pmp_features["best_prefix_compute_tflops"],
            pmp_features["best_prefix_compute_tflops"] + 6.0,
            0.0,
        ),
        "route_isl_norm": _safe_ratio(
            pmp_features["route_isl_bw_mbps"],
            pmp_features["route_isl_bw_mbps"] + 5000.0,
            0.0,
        ),
        "route_gsl_norm": _safe_ratio(
            pmp_features["route_gsl_bw_mbps"],
            pmp_features["route_gsl_bw_mbps"] + 150.0,
            0.0,
        ),
        "cdp_dispatch_norm": _safe_ratio(
            cdp_features["cdp_dispatch_hmean_bw_mbps"],
            cdp_features["cdp_dispatch_hmean_bw_mbps"] + 5000.0,
            0.0,
        ),
        "cdp_return_norm": _safe_ratio(
            cdp_features["cdp_return_hmean_bw_mbps"],
            cdp_features["cdp_return_hmean_bw_mbps"] + 150.0,
            0.0,
        ),
        "cdp_parallel_gain_norm": _clamp01((cdp_features["cdp_parallel_gain"] - 1.0) / 3.0),
        "cdp_worker_count_norm": _clamp01((cdp_features["cdp_pool_worker_count"] - 1.0) / 3.0),
        "single_route_count_norm": _clamp01(pmp_features["full_fit_route_count"] / 2.0),
        "gs_ground_bias_norm": _clamp01((pmp_features["gs_direct_advantage_norm"] - 0.70) / 0.20),
    }
    return features


def _resource_mode_scores(features: dict) -> dict:
    cdp_score = (
        1.35 * features["cdp_parallel_gain_norm"]
        + 1.00 * features["cdp_worker_count_norm"]
        + 0.80 * features["cdp_return_norm"]
        + 0.35 * features["cdp_dispatch_norm"]
        + 0.55 * features["return_lightness"]
        + 0.25 * features["compute_norm_single"]
        - 0.95 * features["gs_ground_bias_norm"]
    )

    pmp_score = (
        1.35 * features["best_prefix_compression_gain"]
        + 0.85 * features["best_prefix_depth_norm"]
        + 0.70 * features["route_isl_norm"]
        + 0.45 * features["compute_norm_prefix"]
        + 0.35 * features["best_prefix_memory_margin_norm"]
        + 0.20 * features["return_lightness"]
        - 0.45 * features["single_route_count_norm"]
        + 0.45 * features["gs_ground_bias_norm"]
    )

    sat_only_score = (
        1.10 * features["compute_norm_single"]
        + 0.95 * features["best_single_memory_margin_norm"]
        + 0.70 * features["route_gsl_norm"]
        + 0.55 * (1.0 - features["best_prefix_compression_gain"])
        + 0.35 * features["return_lightness"]
        + 0.20 * features["single_route_count_norm"]
        - 0.80 * features["gs_ground_bias_norm"]
    )

    sat_resource_strength = max(
        features["compute_norm_single"],
        features["best_prefix_memory_margin_norm"],
        features["cdp_worker_count_norm"],
    )
    gs_only_score = (
        0.35
        + 0.85 * (1.0 - sat_resource_strength)
        + 0.55 * (1.0 - features["best_prefix_compression_gain"])
        + 0.35 * (1.0 - features["cdp_parallel_gain_norm"])
        + 0.20 * (1.0 - features["return_lightness"])
        + 1.20 * features["gs_ground_bias_norm"]
    )

    return {
        "CDP": cdp_score,
        "PMP": pmp_score,
        "Sat-Only": sat_only_score,
        "GS-Only": gs_only_score,
    }


def _mode_by_family(mode_evaluations: List[ModeEvaluation], mode_family: str) -> ModeEvaluation | None:
    for evaluation in mode_evaluations:
        if evaluation.mode_family == mode_family:
            return evaluation
    return None


def _build_selector_row(
    scene: SlotScene,
    selector_family: str,
    selector_algo: str,
    route_policy: str,
    selected: ModeEvaluation | None,
    plan_payload: dict,
    candidate_id: str,
    reason: str = "",
) -> ModeEvaluation:
    if selected is None:
        return ModeEvaluation(
            source_run_id=scene.source_run_id,
            slot_id=scene.slot_id,
            mode_family=selector_family,
            mode_algo=selector_algo,
            candidate_id=candidate_id,
            route_policy=route_policy,
            feasible=False,
            reason=reason or "no_feasible_mode",
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
            plan_json=json.dumps(plan_payload, ensure_ascii=False, sort_keys=True),
            config_path="",
            candidate_path=str(scene.candidate_path) if scene.candidate_path else "",
        )

    return ModeEvaluation(
        source_run_id=scene.source_run_id,
        slot_id=scene.slot_id,
        mode_family=selector_family,
        mode_algo=selector_algo,
        candidate_id=f"{candidate_id}_{selected.mode_family}_{selected.candidate_id}",
        route_policy=route_policy,
        feasible=True,
        reason=reason,
        latency_ms=selected.latency_ms,
        satellite_energy_j=selected.satellite_energy_j,
        energy_compute_j=selected.energy_compute_j,
        energy_comm_j=selected.energy_comm_j,
        satellite_compute_time_ms=selected.satellite_compute_time_ms,
        satellite_tx_time_ms=selected.satellite_tx_time_ms,
        active_sat_count=selected.active_sat_count,
        hop_count=selected.hop_count,
        route=selected.route,
        pipeline_path=selected.pipeline_path,
        plan_json=json.dumps(plan_payload, ensure_ascii=False, sort_keys=True),
        config_path=selected.config_path,
        candidate_path=selected.candidate_path,
    )


def select_oracle_min_latency(scene: SlotScene, mode_evaluations: List[ModeEvaluation]) -> ModeEvaluation:
    """Select one feasible mode using prediction-based minimum latency."""

    candidate_summary = _mode_candidate_summary(mode_evaluations)
    feasible = [
        item
        for item in mode_evaluations
        if item.feasible and finite_number(item.latency_ms)
    ]

    if not feasible:
        return _build_selector_row(
            scene=scene,
            selector_family="Oracle-Min-Latency",
            selector_algo="Prediction-Min-Latency",
            route_policy="min_predicted_latency_over_all_feasible_modes",
            selected=None,
            candidate_id="oracle_no_feasible_mode",
            reason="no_feasible_mode",
            plan_payload={
                "selection_rule": "filter infeasible modes, then choose minimum predicted latency",
                "selected_mode": "",
                "candidate_modes": candidate_summary,
            },
        )

    selected = min(
        feasible,
        key=lambda item: (
            float(item.latency_ms),
            float(item.satellite_energy_j) if finite_number(item.satellite_energy_j) else float("inf"),
            item.mode_family,
        ),
    )
    plan_payload = {
        "selection_rule": "filter infeasible modes, then choose minimum predicted latency",
        "selected_mode": selected.mode_family,
        "selected_algo": selected.mode_algo,
        "selected_candidate_id": selected.candidate_id,
        "selected_route_policy": selected.route_policy,
        "candidate_modes": candidate_summary,
    }
    return _build_selector_row(
        scene=scene,
        selector_family="Oracle-Min-Latency",
        selector_algo="Prediction-Min-Latency",
        route_policy="min_predicted_latency_over_all_feasible_modes",
        selected=selected,
        candidate_id="oracle_selected",
        plan_payload=plan_payload,
    )


def select_fwms_feature(
    scene: SlotScene,
    mode_evaluations: List[ModeEvaluation],
    resolver: StkPathResolver | None = None,
    max_workers: int = 4,
) -> ModeEvaluation:
    """Select the execution mode using resource and model features only."""

    candidate_summary = _mode_candidate_summary(mode_evaluations)
    mode_map = {
        evaluation.mode_family: evaluation
        for evaluation in mode_evaluations
    }
    feature_values = _resource_driven_feature_values(
        scene,
        mode_evaluations,
        resolver=resolver,
        max_workers=max_workers,
    )
    mode_scores = _resource_mode_scores(feature_values)
    feasible_modes = {
        mode_family: evaluation
        for mode_family, evaluation in mode_map.items()
        if evaluation.feasible
    }

    selected: Optional[ModeEvaluation] = None
    reason = ""
    if feasible_modes:
        ranked = sorted(
            feasible_modes.items(),
            key=lambda item: (
                mode_scores.get(item[0], float("-inf")),
                1 if item[0] == "CDP" else 0,
                item[0],
            ),
            reverse=True,
        )
        predicted_mode = ranked[0][0]
        selected = ranked[0][1]
        reason = f"selected_by_resource_score_{predicted_mode.lower().replace('-', '_')}"

        if predicted_mode == "CDP" and feature_values["cdp_pool_worker_count"] < 2:
            selected = feasible_modes.get("PMP") or feasible_modes.get("Sat-Only") or feasible_modes.get("GS-Only")
            reason = "cdp_score_won_but_worker_pool_too_small_fallback"
        elif predicted_mode == "PMP" and feature_values["best_prefix_end_layer"] <= 0:
            selected = feasible_modes.get("Sat-Only") or feasible_modes.get("GS-Only") or feasible_modes.get("CDP")
            reason = "pmp_score_won_but_no_useful_prefix_fallback"

    if selected is None:
        oracle = select_oracle_min_latency(scene, mode_evaluations)
        selected = _mode_by_family(mode_evaluations, json.loads(oracle.plan_json).get("selected_mode", ""))
        reason = "fwms_feature_no_feasible_score_fallback_to_oracle"

    plan_payload = {
        "selection_rule": (
            "first filter infeasible modes, then score feasible modes using only "
            "resource and model-structure features: memory fit, prefix compression, "
            "parallel worker richness, and route bandwidth quality"
        ),
        "selected_mode": selected.mode_family if selected else "",
        "selected_algo": selected.mode_algo if selected else "",
        "selected_candidate_id": selected.candidate_id if selected else "",
        "selected_route_policy": selected.route_policy if selected else "",
        "decision_reason": reason,
        "feature_values": feature_values,
        "mode_scores": mode_scores,
        "candidate_modes": candidate_summary,
    }
    return _build_selector_row(
        scene=scene,
        selector_family="FWMS-Feature",
        selector_algo="Resource-Driven",
        route_policy="resource_guided_multi_mode_boundary",
        selected=selected,
        candidate_id="fwms_feature_selected" if selected else "fwms_feature_no_feasible_mode",
        reason="" if selected else reason,
        plan_payload=plan_payload,
    )
