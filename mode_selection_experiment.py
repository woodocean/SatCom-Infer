"""Run the first-stage mode-selection experiment framework.

The current stage implements:
    1. STK slot_scene loading/materialization.
    2. PMP evaluation on the already selected route.
    3. GS-Only evaluation on its own minimum-latency route.
    4. Sat-Only evaluation on the best single satellite.
    5. CDP evaluation with no aggregator and discrete LAWA allocation.

FWMS is added on top of the same result table in a later stage.
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


def _run_mode_stage(
    scenes: List[SlotScene],
    run_id: str,
    config_dir: Path,
    cdp_max_workers: int,
) -> List[ModeEvaluation]:
    evaluations: List[ModeEvaluation] = []
    resolver = StkPathResolver()
    for scene in scenes:
        print(f"[MODE] PMP/LA-DP | {scene.slot_id} | route={'->'.join(scene.selected_stk_path)}")
        evaluations.append(evaluate_pmp_slot(scene, run_id=run_id, algorithm="LA-DP"))
        print(f"[MODE] GS-Only | {scene.slot_id} | route_policy=min_predicted_gs_only_latency")
        evaluations.append(
            evaluate_gs_only_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
            )
        )
        print(f"[MODE] Sat-Only | {scene.slot_id} | route_policy=best_single_satellite_over_candidate_paths")
        evaluations.append(
            evaluate_sat_only_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
            )
        )
        print(f"[MODE] CDP/LAWA | {scene.slot_id} | route_policy=best_lawa_worker_set_no_aggregator")
        evaluations.append(
            evaluate_cdp_slot(
                scene,
                run_id=run_id,
                config_output_dir=config_dir,
                resolver=resolver,
                max_workers=cdp_max_workers,
            )
        )
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
    parser = argparse.ArgumentParser(description="Run mode-selection experiment framework through CDP.")
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
    run_id = args.run_id or f"mode_selection_{source_run_id}_stage4_pmp_gs_sat_cdp_{suffix}"
    output_dir = Path(args.output_dir or Path("result") / "mode_selection" / run_id)
    scenes_dir = output_dir / "scenes"
    config_dir = output_dir / "configs"
    data_dir = output_dir / "data"
    results_csv = data_dir / "slot_mode_results.csv"
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
    feasible_count = sum(1 for item in evaluations if item.feasible)

    metadata = {
        "run_id": run_id,
        "exp_type": "mode_selection",
        "stage": "stage4_slot_scene_pmp_gs_sat_cdp",
        "source_stk_run_dir": str(stk_run_dir),
        "source_run_id": scenes[0].source_run_id,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "implemented_modes": ["PMP", "GS-Only", "Sat-Only", "CDP"],
        "implemented_algorithms": {
            "PMP": "LA-DP",
            "GS-Only": "Min-Latency-Route",
            "Sat-Only": "Min-Latency-Single-Sat",
            "CDP": "LAWA-Discrete",
        },
        "pending_modes": ["FWMS"],
        "route_policy": {
            "PMP": "selected_path",
            "GS-Only": "min_predicted_gs_only_latency",
            "Sat-Only": "best_single_satellite_over_candidate_paths",
            "CDP": "best_lawa_worker_set_no_aggregator",
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
    }
    _write_json(metadata_path, metadata)

    print(
        "[MODE] Completed mode-selection stage: "
        f"scenes={len(scenes)} mode_rows={row_count} feasible={feasible_count} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
