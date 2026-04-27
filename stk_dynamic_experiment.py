"""Run dynamic PMP theory experiments from STK visibility reports.

This script keeps STK usage at the topology layer:
    - Access Data decides whether a link is visible.
    - AER range samples determine propagation delay.
    - Effective bandwidth is sampled from configured engineering ranges.

It does not start physical nodes or modify ``config/network_config.json``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy exists in the project env, fallback keeps script usable.
    np = None

from core.scheduler import Scheduler
from core.stk_parser import format_stk_time, parse_stk_time
from core.stk_scenario_builder import (
    CandidatePath,
    build_candidate_paths,
    build_network_config_for_path,
    load_stk_reports,
)


ALGORITHMS = ["LA-DP", "Greedy", "Uniform", "GS-Only", "Random", "GA"]

SLOT_FIELDS = [
    "run_id",
    "slot_id",
    "slot_start",
    "slot_stop",
    "status",
    "selected_path",
    "pipeline_path",
    "hop_count",
    "satellite_count",
    "common_start",
    "common_stop",
    "common_duration_s",
    "total_range_km",
    "total_propagation_delay_ms",
    "isl_avg_bw_mbps",
    "gsl_avg_bw_mbps",
    "config_path",
    "note",
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32 - 1))


def _slot_ranges(start: datetime, stop: datetime, slot_minutes: float) -> List[Tuple[datetime, datetime]]:
    if slot_minutes <= 0:
        raise ValueError("slot_minutes must be positive")
    slots = []
    delta = timedelta(minutes=slot_minutes)
    current = start
    while current < stop:
        nxt = min(current + delta, stop)
        if nxt > current:
            slots.append((current, nxt))
        current = nxt
    return slots


def _safe_time_token(value: datetime) -> str:
    return value.strftime("%H%M%S")


def _sample_effective_bandwidths(
    config: dict,
    rng: random.Random,
    isl_range_mbps: Tuple[float, float],
    gsl_range_mbps: Tuple[float, float],
) -> Tuple[float, float]:
    """Assign per-link effective bandwidths in-place.

    RS-SAT and SAT-SAT are treated as inter-satellite/space links.  SAT-GS is
    treated as the satellite-ground bottleneck.
    """

    isl_values: List[float] = []
    gsl_values: List[float] = []

    for link in config.get("links", {}).values():
        link_type = str(link.get("stk_link_type", ""))
        if link_type == "SAT-GS":
            bw = rng.uniform(*gsl_range_mbps)
            gsl_values.append(bw)
            link["effective_bandwidth_class"] = "GSL"
            link["bandwidth_sampling_range_mbps"] = list(gsl_range_mbps)
        else:
            bw = rng.uniform(*isl_range_mbps)
            isl_values.append(bw)
            link["effective_bandwidth_class"] = "ISL"
            link["bandwidth_sampling_range_mbps"] = list(isl_range_mbps)
        link["bandwidth_mbps"] = round(float(bw), 4)
        link["bandwidth_model"] = "effective_random_uniform"

    isl_avg = sum(isl_values) / len(isl_values) if isl_values else 0.0
    gsl_avg = sum(gsl_values) / len(gsl_values) if gsl_values else 0.0
    return isl_avg, gsl_avg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_slot_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SLOT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SLOT_FIELDS})


def _path_to_str(path: Optional[Sequence[str]]) -> str:
    return "->".join(path or [])


def _select_path(paths: Sequence[CandidatePath]) -> Optional[CandidatePath]:
    if not paths:
        return None
    return max(
        paths,
        key=lambda item: (
            item.common_duration_s,
            -item.hop_count,
            -item.total_propagation_delay_ms,
            "->".join(item.path),
        ),
    )


def _run_scheduler_for_slot(
    config_path: Path,
    output_csv: Path,
    run_id: str,
    slot_id: str,
    slot_start: datetime,
    slot_stop: datetime,
    model_name: str,
    batch_size: int,
    input_h: int,
    input_w: int,
    repeat_per_slot: int,
    seed: int,
    path: CandidatePath,
) -> None:
    scheduler = Scheduler(net_config_path=str(config_path))
    raw_links = scheduler.net_config.get("links", {})
    isl_avg_bw, gsl_avg_bw = scheduler._extract_bw_metrics(raw_links)
    pipeline = scheduler.net_config.get("simulation_paths", {}).get("pipeline", [])

    for repeat_idx in range(repeat_per_slot):
        _seed_everything(seed + repeat_idx)
        task_id = f"{slot_id}_rep{repeat_idx:02d}"
        plans = scheduler.generate_task_and_schedule(
            task_id=task_id,
            model_name=model_name,
            batch_size=batch_size,
            target_h=input_h,
            target_w=input_w,
            run_id=run_id,
            exp_type="stk_dynamic_pmp",
            mode="theory",
            standardized_csv_file=str(output_csv),
            persist_theory=False,
            algorithm_names=ALGORITHMS,
            return_full_plans=True,
        )
        scheduler._append_standardized_theory_rows(
            task_id=task_id,
            model_name=model_name,
            batch_size=batch_size,
            target_h=input_h,
            target_w=input_w,
            plans=plans,
            isl_avg_bw=isl_avg_bw,
            gsl_avg_bw=gsl_avg_bw,
            run_id=run_id,
            exp_type="stk_dynamic_pmp",
            mode="theory",
            output_csv=str(output_csv),
            metadata_extra={
                "pipeline_node_count": path.satellite_count,
                "pipeline_hop_count": path.hop_count,
                "pipeline_path": _path_to_str(pipeline),
                "sweep_param": "time_slot",
                "sweep_value": slot_id,
            },
        )


def _parse_range(value: str, label: str) -> Tuple[float, float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"{label} must be formatted as min,max")
    low, high = float(parts[0]), float(parts[1])
    if low <= 0 or high <= 0 or high < low:
        raise ValueError(f"{label} must satisfy 0 < min <= max")
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STK-driven dynamic PMP theory experiment.")
    parser.add_argument(
        "--stk-dir",
        type=str,
        default="data/stk",
        help="Directory containing the six STK TXT reports",
    )
    parser.add_argument("--chain2-access-data", default=None, help="STK Chain2 Access Data TXT: SAT -> GS")
    parser.add_argument("--chain2-aer", default=None, help="STK Chain2 Access AER TXT: SAT -> GS")
    parser.add_argument("--chain4-access-data", default=None, help="STK Chain4 Access Data TXT: SAT -> SAT")
    parser.add_argument("--chain4-aer", default=None, help="STK Chain4 Access AER TXT: SAT -> SAT")
    parser.add_argument("--chain5-access-data", default=None, help="STK Chain5 Access Data TXT: RS -> SAT")
    parser.add_argument("--chain5-aer", default=None, help="STK Chain5 Access AER TXT: RS -> SAT")
    parser.add_argument("--time-start", required=True, help='Scenario start, e.g. "14 Apr 2026 04:00:00.000"')
    parser.add_argument("--time-stop", required=True, help='Scenario stop, e.g. "14 Apr 2026 08:00:00.000"')
    parser.add_argument("--slot-minutes", type=float, default=5.0, help="Time-slot length in minutes")
    parser.add_argument("--max-hops", type=int, default=6, help="Maximum route hops from RS to GS")
    parser.add_argument("--max-paths-per-slot", type=int, default=50, help="Candidate paths retained per slot")
    parser.add_argument("--max-neighbors-per-node", type=int, default=24, help="Path-search expansion cap")
    parser.add_argument("--beam-width-per-node", type=int, default=8, help="Beam width retained for each node at each hop")
    parser.add_argument("--source-node", type=str, default="RS")
    parser.add_argument("--ground-node", type=str, default="Shenzhen")
    parser.add_argument("--base-config", type=str, default="config/network_config.json")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Standardized result CSV. Defaults to <output-dir>/results_long_stk_dynamic.csv",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--isl-range-mbps", type=str, default="1000,20000", help="Uniform ISL range, min,max Mbps")
    parser.add_argument("--gsl-range-mbps", type=str, default="50,300", help="Uniform GSL range, min,max Mbps")
    parser.add_argument("--model-name", type=str, default="yolov5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-h", type=int, default=640)
    parser.add_argument("--input-w", type=int, default=640)
    parser.add_argument("--repeat-per-slot", type=int, default=1)
    args = parser.parse_args()

    stk_dir = Path(args.stk_dir)
    args.chain2_access_data = args.chain2_access_data or str(stk_dir / "Chain2_Access_Data.txt")
    args.chain2_aer = args.chain2_aer or str(stk_dir / "Chain2_Access_AER.txt")
    args.chain4_access_data = args.chain4_access_data or str(stk_dir / "Chain4_Access_Data.txt")
    args.chain4_aer = args.chain4_aer or str(stk_dir / "Chain4_Access_AER.txt")
    args.chain5_access_data = args.chain5_access_data or str(stk_dir / "Chain5_Access_Data.txt")
    args.chain5_aer = args.chain5_aer or str(stk_dir / "Chain5_Access_AER.txt")

    run_id = args.run_id or f"stk_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir or Path("result") / "stk_dynamic" / run_id)
    config_dir = output_dir / "configs"
    results_csv = Path(args.output_csv) if args.output_csv else output_dir / "results_long_stk_dynamic.csv"
    slots_csv = output_dir / "stk_dynamic_slots.csv"
    candidates_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    isl_range = _parse_range(args.isl_range_mbps, "--isl-range-mbps")
    gsl_range = _parse_range(args.gsl_range_mbps, "--gsl-range-mbps")
    time_start = parse_stk_time(args.time_start)
    time_stop = parse_stk_time(args.time_stop)
    slots = _slot_ranges(time_start, time_stop, args.slot_minutes)

    metadata = {
        "run_id": run_id,
        "exp_type": "stk_dynamic_pmp",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "time_start": args.time_start,
        "time_stop": args.time_stop,
        "slot_minutes": args.slot_minutes,
        "max_hops": args.max_hops,
        "path_selection": "max_common_duration_s",
        "bandwidth_model": "effective_random_uniform",
        "isl_range_mbps": list(isl_range),
        "gsl_range_mbps": list(gsl_range),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "input_h": args.input_h,
        "input_w": args.input_w,
        "repeat_per_slot": args.repeat_per_slot,
        "seed": args.seed,
        "stk_reports": {
            "chain2_access_data": args.chain2_access_data,
            "chain2_aer": args.chain2_aer,
            "chain4_access_data": args.chain4_access_data,
            "chain4_aer": args.chain4_aer,
            "chain5_access_data": args.chain5_access_data,
            "chain5_aer": args.chain5_aer,
        },
    }
    _write_json(output_dir / "metadata.json", metadata)

    print("[STK-DYN] Loading STK reports...")
    windows, samples = load_stk_reports(
        chain2_access_data=args.chain2_access_data,
        chain2_aer=args.chain2_aer,
        chain4_access_data=args.chain4_access_data,
        chain4_aer=args.chain4_aer,
        chain5_access_data=args.chain5_access_data,
        chain5_aer=args.chain5_aer,
    )
    print(f"[STK-DYN] Access windows: {len(windows)}")
    print(f"[STK-DYN] AER samples: {len(samples)}")
    print(f"[STK-DYN] Slots: {len(slots)} | output={output_dir}")

    completed_slots = 0
    skipped_slots = 0

    for idx, (slot_start, slot_stop) in enumerate(slots):
        slot_id = f"slot_{idx:03d}_{_safe_time_token(slot_start)}_{_safe_time_token(slot_stop)}"
        print(f"\n[STK-DYN] {slot_id}: {format_stk_time(slot_start)} -> {format_stk_time(slot_stop)}")

        paths = build_candidate_paths(
            windows=windows,
            samples=samples,
            time_start=slot_start,
            time_stop=slot_stop,
            source_node=args.source_node,
            ground_node=args.ground_node,
            max_hops=args.max_hops,
            max_paths=args.max_paths_per_slot,
            max_neighbors_per_node=args.max_neighbors_per_node,
            beam_width_per_node=args.beam_width_per_node,
            sort_policy="longest_visibility",
        )
        candidate_path = candidates_dir / f"{slot_id}_candidate_paths.json"
        _write_json(
            candidate_path,
            {
                "slot_id": slot_id,
                "slot_start": format_stk_time(slot_start),
                "slot_stop": format_stk_time(slot_stop),
                "paths": [
                    {
                        "rank": path.rank,
                        "path": path.path,
                        "hop_count": path.hop_count,
                        "satellite_count": path.satellite_count,
                        "common_start": format_stk_time(path.common_start),
                        "common_stop": format_stk_time(path.common_stop),
                        "common_duration_s": path.common_duration_s,
                        "total_range_km": path.total_range_km,
                        "total_propagation_delay_ms": path.total_propagation_delay_ms,
                    }
                    for path in paths
                ],
            },
        )

        selected = _select_path(paths)
        if selected is None:
            skipped_slots += 1
            _append_slot_row(
                slots_csv,
                {
                    "run_id": run_id,
                    "slot_id": slot_id,
                    "slot_start": format_stk_time(slot_start),
                    "slot_stop": format_stk_time(slot_stop),
                    "status": "no_path",
                    "note": "No RS-to-GS path found in this slot.",
                },
            )
            print("[STK-DYN] No path found; skipped.")
            continue

        slot_seed = args.seed + idx * 1009
        rng = random.Random(slot_seed)
        config = build_network_config_for_path(
            selected,
            base_config_path=args.base_config,
            isl_bandwidth_mbps=isl_range[0],
            gsl_bandwidth_mbps=gsl_range[0],
            rs_sat_bandwidth_mbps=None,
        )
        isl_avg, gsl_avg = _sample_effective_bandwidths(config, rng, isl_range, gsl_range)
        config["simulation_paths"]["stk_dynamic_slot"] = {
            "slot_id": slot_id,
            "slot_start_utcg": format_stk_time(slot_start),
            "slot_stop_utcg": format_stk_time(slot_stop),
            "bandwidth_seed": slot_seed,
            "path_selection": "max_common_duration_s",
            "isl_range_mbps": list(isl_range),
            "gsl_range_mbps": list(gsl_range),
        }
        config_path = config_dir / f"{slot_id}_network_config.json"
        _write_json(config_path, config)

        pipeline = config.get("simulation_paths", {}).get("pipeline", [])
        _append_slot_row(
            slots_csv,
            {
                "run_id": run_id,
                "slot_id": slot_id,
                "slot_start": format_stk_time(slot_start),
                "slot_stop": format_stk_time(slot_stop),
                "status": "completed",
                "selected_path": _path_to_str(selected.path),
                "pipeline_path": _path_to_str(pipeline),
                "hop_count": selected.hop_count,
                "satellite_count": selected.satellite_count,
                "common_start": format_stk_time(selected.common_start),
                "common_stop": format_stk_time(selected.common_stop),
                "common_duration_s": f"{selected.common_duration_s:.3f}",
                "total_range_km": f"{selected.total_range_km:.6f}",
                "total_propagation_delay_ms": f"{selected.total_propagation_delay_ms:.6f}",
                "isl_avg_bw_mbps": f"{isl_avg:.4f}",
                "gsl_avg_bw_mbps": f"{gsl_avg:.4f}",
                "config_path": str(config_path),
                "note": "",
            },
        )

        print(
            "[STK-DYN] Selected "
            f"path={_path_to_str(selected.path)} "
            f"duration={selected.common_duration_s:.1f}s "
            f"delay={selected.total_propagation_delay_ms:.3f}ms "
            f"ISLavg={isl_avg:.1f}Mbps GSLavg={gsl_avg:.1f}Mbps"
        )
        _run_scheduler_for_slot(
            config_path=config_path,
            output_csv=results_csv,
            run_id=run_id,
            slot_id=slot_id,
            slot_start=slot_start,
            slot_stop=slot_stop,
            model_name=args.model_name,
            batch_size=args.batch_size,
            input_h=args.input_h,
            input_w=args.input_w,
            repeat_per_slot=args.repeat_per_slot,
            seed=slot_seed,
            path=selected,
        )
        completed_slots += 1

    metadata.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "completed_slots": completed_slots,
            "skipped_slots": skipped_slots,
            "slots_csv": str(slots_csv),
            "results_csv": str(results_csv),
            "configs_dir": str(config_dir),
        }
    )
    _write_json(output_dir / "metadata.json", metadata)
    print("\n[STK-DYN] Done.")
    print(f"[STK-DYN] Completed slots: {completed_slots}, skipped slots: {skipped_slots}")
    print(f"[STK-DYN] Slot table: {slots_csv}")
    print(f"[STK-DYN] Result table: {results_csv}")


if __name__ == "__main__":
    main()
