"""Run the mode-selection experiment framework.

The current stage implements:
    1. STK slot_scene loading/materialization.
    2. PMP evaluation on the already selected route.
    3. GS-Only evaluation on its own minimum-latency route.
    4. Sat-Only evaluation on the best single satellite.
    5. CDP evaluation with no aggregator and discrete LAWA allocation.
    6. Feature-weighted FWMS selection between PMP and CDP.
    7. Oracle-Min-Latency selection as a prediction-based upper-bound baseline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, List

from core.mode_evaluators import (
    ModeEvaluation,
    StkPathResolver,
    evaluate_cdp_slot,
    evaluate_gs_only_slot,
    evaluate_pmp_slot,
    evaluate_sat_only_slot,
)
from core.mode_scene_builder import SlotScene, load_stk_slot_scenes, write_slot_scene


MODE_RESULT_FIELDS = [
    "run_id",
    "source_run_id",
    "slot_id",
    "mode_family",
    "mode_algo",
    "candidate_id",
    "route_policy",
    "feasible",
    "reason",
    "latency_ms",
    "satellite_energy_j",
    "energy_compute_j",
    "energy_comm_j",
    "satellite_compute_time_ms",
    "satellite_tx_time_ms",
    "active_sat_count",
    "hop_count",
    "route",
    "pipeline_path",
    "plan_json",
    "config_path",
    "candidate_path",
    "timestamp",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_mode_results(path: Path, run_id: str, evaluations: Iterable[ModeEvaluation]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODE_RESULT_FIELDS)
        writer.writeheader()
        for evaluation in evaluations:
            row = evaluation.to_row()
            row["run_id"] = run_id
            row["timestamp"] = timestamp
            writer.writerow({field: row.get(field, "") for field in MODE_RESULT_FIELDS})
            count += 1
    return count


def _average(values: List[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _write_summary_by_mode(path: Path, evaluations: List[ModeEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = sorted({item.mode_family for item in evaluations})
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "mode_family",
            "rows",
            "feasible_rows",
            "feasible_rate",
            "avg_latency_ms",
            "avg_satellite_energy_j",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in modes:
            rows = [item for item in evaluations if item.mode_family == mode]
            feasible = [item for item in rows if item.feasible]
            latencies = [float(item.latency_ms) for item in feasible if _finite_number(item.latency_ms)]
            energies = [
                float(item.satellite_energy_j)
                for item in feasible
                if _finite_number(item.satellite_energy_j)
            ]
            writer.writerow(
                {
                    "mode_family": mode,
                    "rows": len(rows),
                    "feasible_rows": len(feasible),
                    "feasible_rate": len(feasible) / len(rows) if rows else "",
                    "avg_latency_ms": _average(latencies),
                    "avg_satellite_energy_j": _average(energies),
                }
            )


def _write_fwms_selection_distribution(path: Path, evaluations: List[ModeEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str], int] = {}
    for item in evaluations:
        if item.mode_family not in {"FWMS", "FWMS-Feature", "Oracle-Min-Latency"}:
            continue
        selected_mode = "infeasible"
        if item.plan_json:
            try:
                selected_mode = json.loads(item.plan_json).get("selected_mode") or "infeasible"
            except json.JSONDecodeError:
                selected_mode = "parse_error"
        key = (item.mode_family, selected_mode)
        counts[key] = counts.get(key, 0) + 1

    totals: dict[str, int] = {}
    for selector_family, selected_mode in counts:
        totals[selector_family] = totals.get(selector_family, 0) + counts[(selector_family, selected_mode)]
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["selector_family", "selected_mode", "count", "ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (selector_family, selected_mode), count in sorted(counts.items()):
            writer.writerow(
                {
                    "selector_family": selector_family,
                    "selected_mode": selected_mode,
                    "count": count,
                    "ratio": count / totals[selector_family] if totals.get(selector_family) else "",
                }
            )


def _finite_number(value) -> bool:
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


_PROFILE_CACHE: dict | None = None


def _load_pc_profile(scene: SlotScene) -> List[dict]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        with Path("config/dnn_profiles_database_pc.json").open("r", encoding="utf-8") as f:
            _PROFILE_CACHE = json.load(f)
    config_key = f"b{scene.task.batch_size}_{scene.task.input_h}x{scene.task.input_w}"
    raw_profile = _PROFILE_CACHE[scene.task.model_name][config_key]
    return [raw_profile[str(idx)] for idx in range(len(raw_profile)) if str(idx) in raw_profile]


def _input_size_mb(scene: SlotScene) -> float:
    return (scene.task.batch_size * 3 * scene.task.input_h * scene.task.input_w * 4) / (1024 ** 2)


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
    return (variance ** 0.5) / mean_speed


def _normalized_feature_values(scene: SlotScene, cdp_evaluation: ModeEvaluation | None) -> dict:
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


def _select_oracle_min_latency(scene: SlotScene, mode_evaluations: List[ModeEvaluation]) -> ModeEvaluation:
    """Select one feasible mode using prediction-based minimum latency."""

    candidate_summary = _mode_candidate_summary(mode_evaluations)
    feasible = [
        item
        for item in mode_evaluations
        if item.feasible and _finite_number(item.latency_ms)
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
            float(item.satellite_energy_j) if _finite_number(item.satellite_energy_j) else float("inf"),
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


def _select_fwms_feature(
    scene: SlotScene,
    mode_evaluations: List[ModeEvaluation],
    weight_rho: float = 1.0,
    weight_eta: float = 1.0,
    weight_bandwidth: float = 1.0,
) -> ModeEvaluation:
    """Select PMP or CDP using the thesis feature-weighted FWMS rule."""

    candidate_summary = _mode_candidate_summary(mode_evaluations)
    pmp = _mode_by_family(mode_evaluations, "PMP")
    cdp = _mode_by_family(mode_evaluations, "CDP")
    feature_values = _normalized_feature_values(scene, cdp)
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
        oracle = _select_oracle_min_latency(scene, mode_evaluations)
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


def _run_mode_stage(
    scenes: List[SlotScene],
    run_id: str,
    config_dir: Path,
    cdp_max_workers: int,
) -> List[ModeEvaluation]:
    evaluations: List[ModeEvaluation] = []
    resolver = StkPathResolver()
    for scene in scenes:
        scene_evaluations: List[ModeEvaluation] = []
        print(f"[MODE] PMP/LA-DP | {scene.slot_id} | route={'->'.join(scene.selected_stk_path)}")
        scene_evaluations.append(evaluate_pmp_slot(scene, run_id=run_id, algorithm="LA-DP"))
        print(f"[MODE] GS-Only | {scene.slot_id} | route_policy=min_predicted_gs_only_latency")
        scene_evaluations.append(
            evaluate_gs_only_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
            )
        )
        print(f"[MODE] Sat-Only | {scene.slot_id} | route_policy=best_single_satellite_over_candidate_paths")
        scene_evaluations.append(
            evaluate_sat_only_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
            )
        )
        print(f"[MODE] CDP/LAWA | {scene.slot_id} | route_policy=best_lawa_worker_set_no_aggregator")
        scene_evaluations.append(
            evaluate_cdp_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
                max_workers=cdp_max_workers,
            )
        )
        fwms_feature = _select_fwms_feature(scene, scene_evaluations)
        print(
            "[MODE] FWMS-Feature | "
            f"{scene.slot_id} | selected={json.loads(fwms_feature.plan_json).get('selected_mode', '')}"
        )
        scene_evaluations.append(fwms_feature)

        oracle = _select_oracle_min_latency(scene, scene_evaluations[:-1])
        print(
            "[MODE] Oracle-Min-Latency | "
            f"{scene.slot_id} | selected={json.loads(oracle.plan_json).get('selected_mode', '')}"
        )
        scene_evaluations.append(oracle)
        evaluations.extend(scene_evaluations)
    return evaluations


def _override_task_spec(
    scenes: List[SlotScene],
    batch_size: int | None,
    model_name: str | None,
    input_h: int | None,
    input_w: int | None,
) -> List[SlotScene]:
    if batch_size is None and model_name is None and input_h is None and input_w is None:
        return scenes
    updated_scenes = []
    for scene in scenes:
        updated_task = replace(
            scene.task,
            model_name=str(model_name) if model_name is not None else scene.task.model_name,
            batch_size=int(batch_size) if batch_size is not None else scene.task.batch_size,
            input_h=int(input_h) if input_h is not None else scene.task.input_h,
            input_w=int(input_w) if input_w is not None else scene.task.input_w,
        )
        updated_scenes.append(replace(scene, task=updated_task))
    return updated_scenes


def _task_override_suffix(
    model_name: str | None,
    batch_size: int | None,
    input_h: int | None,
    input_w: int | None,
) -> str:
    parts = []
    if model_name:
        parts.append(str(model_name))
    if batch_size:
        parts.append(f"b{batch_size}")
    if input_h and input_w:
        parts.append(f"{input_h}x{input_w}")
    return "_".join(parts) if parts else "source_task"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mode-selection experiment framework through feature FWMS.")
    parser.add_argument(
        "--stk-run-dir",
        type=str,
        default="result/stk_dynamic/stk_dynamic_yolo_001",
        help="Existing STK dynamic run directory to use as the scene source.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Mode-selection run id.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--slot-id",
        action="append",
        default=None,
        help="Specific slot id to evaluate. Can be provided multiple times.",
    )
    parser.add_argument("--limit-slots", type=int, default=None, help="Limit completed slots for smoke runs.")
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Override the source STK run batch size for all mode evaluations.",
    )
    parser.add_argument(
        "--model-name-override",
        type=str,
        default=None,
        help="Override the source STK run model name while reusing the same STK topology.",
    )
    parser.add_argument(
        "--input-h-override",
        type=int,
        default=None,
        help="Override the source STK run input height.",
    )
    parser.add_argument(
        "--input-w-override",
        type=int,
        default=None,
        help="Override the source STK run input width.",
    )
    parser.add_argument("--cdp-max-workers", type=int, default=4, help="Maximum CDP worker satellites to try.")
    args = parser.parse_args()

    stk_run_dir = Path(args.stk_run_dir)
    source_run_id = stk_run_dir.name
    suffix = _task_override_suffix(
        args.model_name_override,
        args.batch_size_override,
        args.input_h_override,
        args.input_w_override,
    )
    run_id = args.run_id or f"mode_selection_{source_run_id}_stage6_feature_fwms_oracle_{suffix}"
    output_dir = Path(args.output_dir or Path("result") / "mode_selection" / run_id)
    scenes_dir = output_dir / "scenes"
    config_dir = output_dir / "configs"
    data_dir = output_dir / "data"
    results_csv = data_dir / "slot_mode_results.csv"
    summary_csv = data_dir / "summary_by_mode.csv"
    fwms_distribution_csv = data_dir / "fwms_selection_distribution.csv"
    metadata_path = output_dir / "metadata.json"

    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[MODE] Loading STK scenes from {stk_run_dir}")
    scenes = load_stk_slot_scenes(
        run_dir=stk_run_dir,
        slot_ids=args.slot_id,
        limit=args.limit_slots,
    )
    scenes = _override_task_spec(
        scenes,
        batch_size=args.batch_size_override,
        model_name=args.model_name_override,
        input_h=args.input_h_override,
        input_w=args.input_w_override,
    )
    if not scenes:
        raise SystemExit("[MODE] No completed slot scenes found.")

    scenes_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        write_slot_scene(scene, scenes_dir / f"{scene.slot_id}_scene.json")

    evaluations = _run_mode_stage(
        scenes,
        run_id=run_id,
        config_dir=config_dir,
        cdp_max_workers=args.cdp_max_workers,
    )
    row_count = _write_mode_results(results_csv, run_id=run_id, evaluations=evaluations)
    _write_summary_by_mode(summary_csv, evaluations)
    _write_fwms_selection_distribution(fwms_distribution_csv, evaluations)
    feasible_count = sum(1 for item in evaluations if item.feasible)

    metadata = {
        "run_id": run_id,
        "exp_type": "mode_selection",
        "stage": "stage6_slot_scene_pmp_gs_sat_cdp_fwms_feature_oracle",
        "source_stk_run_dir": str(stk_run_dir),
        "source_run_id": scenes[0].source_run_id,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "implemented_modes": ["PMP", "GS-Only", "Sat-Only", "CDP", "FWMS-Feature", "Oracle-Min-Latency"],
        "implemented_algorithms": {
            "PMP": "LA-DP",
            "GS-Only": "Min-Latency-Route",
            "Sat-Only": "Min-Latency-Single-Sat",
            "CDP": "LAWA-Discrete",
            "FWMS-Feature": "Feature-Weighted",
            "Oracle-Min-Latency": "Prediction-Min-Latency",
        },
        "pending_modes": [],
        "route_policy": {
            "PMP": "selected_path",
            "GS-Only": "min_predicted_gs_only_latency",
            "Sat-Only": "best_single_satellite_over_candidate_paths",
            "CDP": "best_lawa_worker_set_no_aggregator",
            "FWMS-Feature": "feature_weighted_pmp_cdp_boundary",
            "Oracle-Min-Latency": "min_predicted_latency_over_all_feasible_modes",
        },
        "batch_size_override": args.batch_size_override,
        "model_name_override": args.model_name_override,
        "input_h_override": args.input_h_override,
        "input_w_override": args.input_w_override,
        "effective_task": {
            "model_name": scenes[0].task.model_name,
            "batch_size": scenes[0].task.batch_size,
            "input_h": scenes[0].task.input_h,
            "input_w": scenes[0].task.input_w,
        },
        "cdp_max_workers": args.cdp_max_workers,
        "slot_scene_count": len(scenes),
        "mode_result_rows": row_count,
        "feasible_rows": feasible_count,
        "scenes_dir": str(scenes_dir),
        "configs_dir": str(config_dir),
        "slot_mode_results_csv": str(results_csv),
        "summary_by_mode_csv": str(summary_csv),
        "fwms_selection_distribution_csv": str(fwms_distribution_csv),
    }
    _write_json(metadata_path, metadata)

    print(
        "[MODE] Completed mode-selection stage: "
        f"scenes={len(scenes)} mode_rows={row_count} feasible={feasible_count} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
