"""Command line entry for building network_config files from STK reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.stk_parser import format_stk_time
from core.stk_scenario_builder import (
    build_candidate_paths,
    load_stk_reports,
    write_stk_network_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PMP network_config from STK Chain reports.")
    parser.add_argument(
        "--stk-dir",
        type=str,
        default="data/stk",
        help="Directory containing the six STK TXT reports",
    )
    parser.add_argument("--chain2-access-data", type=str, default=None, help="STK Chain2 Access Data TXT: SAT -> GS")
    parser.add_argument("--chain2-aer", type=str, default=None, help="STK Chain2 Access AER TXT: SAT -> GS")
    parser.add_argument("--chain4-access-data", type=str, default=None, help="STK Chain4 Access Data TXT: SAT -> SAT")
    parser.add_argument("--chain4-aer", type=str, default=None, help="STK Chain4 Access AER TXT: SAT -> SAT")
    parser.add_argument("--chain5-access-data", type=str, default=None, help="STK Chain5 Access Data TXT: RS -> SAT")
    parser.add_argument("--chain5-aer", type=str, default=None, help="STK Chain5 Access AER TXT: RS -> SAT")
    parser.add_argument("--time-start", type=str, default=None, help='Filter start time, e.g. "14 Apr 2026 04:00:00.000"')
    parser.add_argument("--time-stop", type=str, default=None, help='Filter stop time, e.g. "14 Apr 2026 08:00:00.000"')
    parser.add_argument("--source-node", type=str, default="RS", help="Source object name in STK reports")
    parser.add_argument("--ground-node", type=str, default="Shenzhen", help="Ground object name in STK reports")
    parser.add_argument("--max-hops", type=int, default=4, help="Maximum routing hops, including RS-SAT and SAT-GS")
    parser.add_argument("--exact-hops", type=int, default=None, help="Require exact hop count instead of <= max-hops")
    parser.add_argument("--max-paths", type=int, default=20, help="Maximum candidate paths to export")
    parser.add_argument(
        "--sort-policy",
        type=str,
        default="shortest_delay",
        choices=["shortest_delay", "longest_visibility"],
        help="Candidate path ranking policy",
    )
    parser.add_argument(
        "--max-neighbors-per-node",
        type=int,
        default=16,
        help="Per-node expansion cap for path search; set <=0 to disable",
    )
    parser.add_argument("--beam-width-per-node", type=int, default=8, help="Beam width retained for each node at each hop")
    parser.add_argument("--base-config", type=str, default="config/network_config.json", help="Base config for node templates")
    parser.add_argument("--output-dir", type=str, default="config/generated/stk_network", help="Output directory")
    parser.add_argument("--num-configs", type=int, default=1, help="How many top candidate paths to materialize as network_config")
    parser.add_argument("--isl-bandwidth-mbps", type=float, default=10_000.0, help="Default bandwidth for SAT-SAT links")
    parser.add_argument("--gsl-bandwidth-mbps", type=float, default=200.0, help="Default bandwidth for SAT-GS links")
    parser.add_argument(
        "--rs-sat-bandwidth-mbps",
        type=float,
        default=None,
        help="Optional default bandwidth for RS-SAT links; falls back to ISL bandwidth",
    )
    args = parser.parse_args()

    stk_dir = Path(args.stk_dir)
    args.chain2_access_data = args.chain2_access_data or str(stk_dir / "Chain2_Access_Data.txt")
    args.chain2_aer = args.chain2_aer or str(stk_dir / "Chain2_Access_AER.txt")
    args.chain4_access_data = args.chain4_access_data or str(stk_dir / "Chain4_Access_Data.txt")
    args.chain4_aer = args.chain4_aer or str(stk_dir / "Chain4_Access_AER.txt")
    args.chain5_access_data = args.chain5_access_data or str(stk_dir / "Chain5_Access_Data.txt")
    args.chain5_aer = args.chain5_aer or str(stk_dir / "Chain5_Access_AER.txt")

    windows, samples = load_stk_reports(
        chain2_access_data=args.chain2_access_data,
        chain2_aer=args.chain2_aer,
        chain4_access_data=args.chain4_access_data,
        chain4_aer=args.chain4_aer,
        chain5_access_data=args.chain5_access_data,
        chain5_aer=args.chain5_aer,
    )
    print(f"[STK] Loaded access windows: {len(windows)}")
    print(f"[STK] Loaded AER samples: {len(samples)}")

    paths = build_candidate_paths(
        windows=windows,
        samples=samples,
        time_start=args.time_start,
        time_stop=args.time_stop,
        source_node=args.source_node,
        ground_node=args.ground_node,
        max_hops=args.max_hops,
        exact_hops=args.exact_hops,
        max_paths=args.max_paths,
        max_neighbors_per_node=args.max_neighbors_per_node,
        sort_policy=args.sort_policy,
        beam_width_per_node=args.beam_width_per_node,
    )
    if not paths:
        raise SystemExit("[STK] No candidate paths found. Try a wider time window or larger max_hops.")

    outputs = write_stk_network_outputs(
        paths=paths,
        output_dir=args.output_dir,
        base_config_path=args.base_config,
        num_configs=args.num_configs,
        isl_bandwidth_mbps=args.isl_bandwidth_mbps,
        gsl_bandwidth_mbps=args.gsl_bandwidth_mbps,
        rs_sat_bandwidth_mbps=args.rs_sat_bandwidth_mbps,
    )

    print(f"[STK] Candidate paths exported: {outputs['candidate_paths']}")
    for path in paths[: min(args.max_paths, 10)]:
        print(
            "[STK] "
            f"rank={path.rank:03d} hops={path.hop_count} "
            f"duration={path.common_duration_s:.1f}s "
            f"delay={path.total_propagation_delay_ms:.3f}ms "
            f"window={format_stk_time(path.common_start)} -> {format_stk_time(path.common_stop)} "
            f"path={' -> '.join(path.path)}"
        )

    config_outputs = {key: value for key, value in outputs.items() if key.startswith("network_config")}
    print("[STK] Network config outputs:")
    print(json.dumps(config_outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
