"""Mode-selection algorithms for collaborative inference experiments.

This module keeps selector logic separate from experiment orchestration:

* FWMS-Feature is the thesis-facing feature-weighted boundary selector.
* Oracle-Min-Latency is an offline upper-bound baseline, not an online policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from core.mode_evaluators import ModeEvaluation
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
    weight_rho: float = 1.0,
    weight_eta: float = 1.0,
    weight_bandwidth: float = 1.0,
) -> ModeEvaluation:
    """Select PMP or CDP using the feature-weighted FWMS boundary rule."""

    candidate_summary = _mode_candidate_summary(mode_evaluations)
    pmp = _mode_by_family(mode_evaluations, "PMP")
    cdp = _mode_by_family(mode_evaluations, "CDP")
    feature_values = normalized_feature_values(scene, cdp)
    score = (
        weight_rho * feature_values["rho_norm"]
        - weight_eta * feature_values["eta_norm"]
        + weight_bandwidth * feature_values["b_bar_norm"]
    )
    weights = {
        "weight_rho": weight_rho,
        "weight_eta": weight_eta,
        "weight_bandwidth": weight_bandwidth,
    }

    selected = None
    reason = ""
    if not cdp or not cdp.feasible:
        selected = pmp if pmp and pmp.feasible else None
        reason = "cdp_infeasible_fallback_to_pmp"
    elif not pmp or not pmp.feasible:
        selected = cdp
        reason = "pmp_infeasible_fallback_to_cdp"
    elif score >= 0.0:
        selected = pmp
        reason = "feature_score_prefers_pmp"
    else:
        selected = cdp
        reason = "feature_score_prefers_cdp"

    if selected is None:
        oracle = select_oracle_min_latency(scene, mode_evaluations)
        selected = _mode_by_family(mode_evaluations, json.loads(oracle.plan_json).get("selected_mode", ""))
        reason = "fwms_feature_no_pmp_or_cdp_fallback_to_oracle"

    plan_payload = {
        "selection_rule": "memory-feasibility gate, then U = w_rho*rho_norm - w_eta*eta_norm + w_bandwidth*b_bar_norm",
        "selected_mode": selected.mode_family if selected else "",
        "selected_algo": selected.mode_algo if selected else "",
        "selected_candidate_id": selected.candidate_id if selected else "",
        "selected_route_policy": selected.route_policy if selected else "",
        "decision_reason": reason,
        "feature_values": feature_values,
        "feature_weights": weights,
        "feature_score_u": score,
        "candidate_modes": candidate_summary,
    }
    return _build_selector_row(
        scene=scene,
        selector_family="FWMS-Feature",
        selector_algo="Feature-Weighted",
        route_policy="feature_weighted_pmp_cdp_boundary",
        selected=selected,
        candidate_id="fwms_feature_selected" if selected else "fwms_feature_no_feasible_mode",
        reason="" if selected else reason,
        plan_payload=plan_payload,
    )
