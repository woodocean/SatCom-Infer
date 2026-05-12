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

from algorithms.mode_selector import finite_number, select_fwms_feature, select_oracle_min_latency
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
            latencies = [float(item.latency_ms) for item in feasible if finite_number(item.latency_ms)]
            energies = [
                float(item.satellite_energy_j)
                for item in feasible
                if finite_number(item.satellite_energy_j)
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


def _run_mode_stage(
    scenes: List[SlotScene],
    run_id: str,
    config_dir: Path,
    cdp_max_workers: int,
    shared_pmp_min_hops: int,
) -> List[ModeEvaluation]:
    evaluations: List[ModeEvaluation] = []
    resolver = StkPathResolver()
    for scene in scenes:
        scene_evaluations: List[ModeEvaluation] = []
        gs_route_policy = f"min_predicted_gs_only_latency_ge{int(shared_pmp_min_hops)}hops"
        print(f"[MODE] GS-Only | {scene.slot_id} | route_policy={gs_route_policy}")
        gs_only = evaluate_gs_only_slot(
            scene,
            run_id=run_id,
            config_output_dir=config_dir,
            resolver=resolver,
            min_hops=shared_pmp_min_hops,
        )
        scene_evaluations.append(gs_only)
        pmp_route_policy = f"shared_gs_only_route_ge{int(shared_pmp_min_hops)}hops"
        print(f"[MODE] PMP/LA-DP | {scene.slot_id} | route_policy={pmp_route_policy}")
        scene_evaluations.append(
            evaluate_pmp_slot(
                scene,
                run_id=run_id,
                algorithm="LA-DP",
                shared_route_eval=gs_only,
                route_policy=pmp_route_policy,
            )
        )
        sat_only_route_policy = f"shared_gs_only_route_single_sat_ge{int(shared_pmp_min_hops)}hops"
        print(f"[MODE] Sat-Only | {scene.slot_id} | route_policy={sat_only_route_policy}")
        scene_evaluations.append(
            evaluate_sat_only_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
                shared_route_eval=gs_only,
                route_policy=sat_only_route_policy,
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
        fwms_feature = select_fwms_feature(scene, scene_evaluations)
        print(
            "[MODE] FWMS-Feature | "
            f"{scene.slot_id} | selected={json.loads(fwms_feature.plan_json).get('selected_mode', '')}"
        )
        scene_evaluations.append(fwms_feature)

        oracle = select_oracle_min_latency(scene, scene_evaluations[:-1])
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
    parser.add_argument(
        "--shared-pmp-min-hops",
        type=int,
        default=3,
        help="Require GS-Only/PMP shared comparison routes to have at least this many hops.",
    )
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
        shared_pmp_min_hops=args.shared_pmp_min_hops,
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
            "PMP": f"shared_gs_only_route_ge{int(args.shared_pmp_min_hops)}hops",
            "GS-Only": f"min_predicted_gs_only_latency_ge{int(args.shared_pmp_min_hops)}hops",
            "Sat-Only": f"shared_gs_only_route_single_sat_ge{int(args.shared_pmp_min_hops)}hops",
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
        "shared_pmp_min_hops": args.shared_pmp_min_hops,
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
