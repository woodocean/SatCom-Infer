"""Build PMP network configs from STK visibility and AER reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.stk_parser import (
    AERSample,
    AccessWindow,
    derive_windows_from_aer_samples,
    filter_samples_by_time,
    filter_windows_by_time,
    format_stk_time,
    group_samples_by_pair,
    group_windows_by_pair,
    parse_access_aer_txt,
    parse_access_data_txt,
    parse_stk_time,
    propagation_delay_ms,
)


@dataclass(frozen=True)
class LinkMetric:
    from_id: str
    to_id: str
    link_type: str
    avg_range_km: float
    propagation_delay_ms: float
    sample_count: int
    window_count: int
    visibility_duration_s: float


@dataclass(frozen=True)
class CandidatePath:
    rank: int
    path: List[str]
    hop_count: int
    satellite_count: int
    common_start: datetime
    common_stop: datetime
    common_duration_s: float
    total_range_km: float
    total_propagation_delay_ms: float
    link_metrics: List[LinkMetric]


def load_stk_reports(
    chain2_access_data: Optional[str | Path] = None,
    chain2_aer: Optional[str | Path] = None,
    chain4_access_data: Optional[str | Path] = None,
    chain4_aer: Optional[str | Path] = None,
    chain5_access_data: Optional[str | Path] = None,
    chain5_aer: Optional[str | Path] = None,
    ignore_seed_satellite: bool = True,
) -> Tuple[List[AccessWindow], List[AERSample]]:
    """Load all currently supported STK report files.

    Chain mapping used by the current project:
        - Chain2: LEO -> GS
        - Chain4: LEO -> LEO
        - Chain5: RS -> LEO
    """

    windows: List[AccessWindow] = []
    samples: List[AERSample] = []

    if chain2_access_data:
        windows.extend(parse_access_data_txt(chain2_access_data, "SAT-GS", ignore_seed_satellite))
    if chain4_access_data:
        windows.extend(parse_access_data_txt(chain4_access_data, "SAT-SAT", ignore_seed_satellite))
    if chain5_access_data:
        windows.extend(parse_access_data_txt(chain5_access_data, "RS-SAT", ignore_seed_satellite))

    if chain2_aer:
        samples.extend(parse_access_aer_txt(chain2_aer, "SAT-GS", ignore_seed_satellite))
    if chain4_aer:
        samples.extend(parse_access_aer_txt(chain4_aer, "SAT-SAT", ignore_seed_satellite))
    if chain5_aer:
        samples.extend(parse_access_aer_txt(chain5_aer, "RS-SAT", ignore_seed_satellite))

    # AER text reports can be used as a fallback for visibility windows.  This
    # is useful when a Chain Access Data report is missing or too slow to export.
    existing_pairs = {(window.from_id, window.to_id) for window in windows}
    fallback_samples = [sample for sample in samples if (sample.from_id, sample.to_id) not in existing_pairs]
    if fallback_samples:
        windows.extend(derive_windows_from_aer_samples(fallback_samples))

    return windows, samples


def _infer_time_range(windows: Sequence[AccessWindow], samples: Sequence[AERSample]) -> Tuple[datetime, datetime]:
    starts: List[datetime] = []
    stops: List[datetime] = []
    if windows:
        starts.extend(window.start for window in windows)
        stops.extend(window.stop for window in windows)
    if samples:
        starts.extend(sample.time for sample in samples)
        stops.extend(sample.time for sample in samples)
    if not starts or not stops:
        raise ValueError("No STK access windows or AER samples were loaded.")
    return min(starts), max(stops)


def _intersect_intervals(
    current: Sequence[Tuple[datetime, datetime]],
    edge_windows: Sequence[AccessWindow],
) -> List[Tuple[datetime, datetime]]:
    overlaps: List[Tuple[datetime, datetime]] = []
    for current_start, current_stop in current:
        for window in edge_windows:
            start = max(current_start, window.start)
            stop = min(current_stop, window.stop)
            if stop > start:
                overlaps.append((start, stop))
    return _merge_intervals(overlaps)


def _merge_intervals(intervals: Sequence[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item[0])
    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        prev_start, prev_stop = merged[-1]
        if start <= prev_stop:
            merged[-1] = (prev_start, max(prev_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _total_interval_duration_s(intervals: Sequence[Tuple[datetime, datetime]]) -> float:
    return sum((stop - start).total_seconds() for start, stop in intervals)


def _average_range_for_pair(samples: Sequence[AERSample]) -> Tuple[float, int]:
    if not samples:
        return 0.0, 0
    return sum(sample.range_km for sample in samples) / len(samples), len(samples)


def _build_link_metrics(
    windows: Sequence[AccessWindow],
    samples: Sequence[AERSample],
    default_range_km: float = 1000.0,
) -> Dict[Tuple[str, str], LinkMetric]:
    by_window = group_windows_by_pair(windows)
    by_sample = group_samples_by_pair(samples)
    metrics: Dict[Tuple[str, str], LinkMetric] = {}

    for pair, pair_windows in by_window.items():
        pair_samples = by_sample.get(pair, [])
        avg_range_km, sample_count = _average_range_for_pair(pair_samples)
        if sample_count == 0:
            avg_range_km = default_range_km
        link_type = pair_windows[0].link_type if pair_windows else (pair_samples[0].link_type if pair_samples else "UNKNOWN")
        metrics[pair] = LinkMetric(
            from_id=pair[0],
            to_id=pair[1],
            link_type=link_type,
            avg_range_km=avg_range_km,
            propagation_delay_ms=propagation_delay_ms(avg_range_km),
            sample_count=sample_count,
            window_count=len(pair_windows),
            visibility_duration_s=sum(window.duration_s for window in pair_windows),
        )
    return metrics


def _edge_bandwidth_mbps(
    link_type: str,
    isl_bandwidth_mbps: float,
    gsl_bandwidth_mbps: float,
    rs_sat_bandwidth_mbps: Optional[float],
) -> float:
    if link_type == "SAT-GS":
        return gsl_bandwidth_mbps
    if link_type == "RS-SAT" and rs_sat_bandwidth_mbps is not None:
        return rs_sat_bandwidth_mbps
    return isl_bandwidth_mbps


def build_candidate_paths(
    windows: Sequence[AccessWindow],
    samples: Sequence[AERSample],
    time_start: Optional[str | datetime] = None,
    time_stop: Optional[str | datetime] = None,
    source_node: str = "RS",
    ground_node: str = "Shenzhen",
    max_hops: int = 4,
    exact_hops: Optional[int] = None,
    max_paths: int = 20,
    max_neighbors_per_node: int = 16,
    default_range_km: float = 1000.0,
    sort_policy: str = "shortest_delay",
    beam_width_per_node: int = 8,
) -> List[CandidatePath]:
    """Find RS -> SAT -> ... -> GS paths with common visibility intervals."""

    if isinstance(time_start, str):
        time_start = parse_stk_time(time_start)
    if isinstance(time_stop, str):
        time_stop = parse_stk_time(time_stop)
    if time_start is None or time_stop is None:
        inferred_start, inferred_stop = _infer_time_range(windows, samples)
        time_start = time_start or inferred_start
        time_stop = time_stop or inferred_stop
    if time_stop <= time_start:
        raise ValueError("time_stop must be later than time_start")

    filtered_windows = filter_windows_by_time(windows, time_start, time_stop)
    filtered_samples = filter_samples_by_time(samples, time_start, time_stop)
    windows_by_pair = group_windows_by_pair(filtered_windows)
    metrics = _build_link_metrics(filtered_windows, filtered_samples, default_range_km)

    adjacency: Dict[str, List[str]] = {}
    for from_id, to_id in windows_by_pair:
        adjacency.setdefault(from_id, []).append(to_id)

    for from_id, neighbors in adjacency.items():
        neighbors.sort(
            key=lambda neighbor: (
                metrics.get((from_id, neighbor), LinkMetric(from_id, neighbor, "", default_range_km, propagation_delay_ms(default_range_km), 0, 0, 0.0)).propagation_delay_ms,
                neighbor,
            )
        )
        if max_neighbors_per_node > 0:
            adjacency[from_id] = neighbors[:max_neighbors_per_node]

    found: List[CandidatePath] = []
    initial_intervals = [(time_start, time_stop)]

    def materialize(path: List[str], intervals: Sequence[Tuple[datetime, datetime]]) -> CandidatePath:
        link_metrics = [metrics[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        total_range = sum(item.avg_range_km for item in link_metrics)
        total_delay = sum(item.propagation_delay_ms for item in link_metrics)
        best_interval = max(intervals, key=lambda item: (item[1] - item[0]).total_seconds())
        return CandidatePath(
            rank=0,
            path=list(path),
            hop_count=len(path) - 1,
            satellite_count=max(0, len(path) - 2),
            common_start=best_interval[0],
            common_stop=best_interval[1],
            common_duration_s=(best_interval[1] - best_interval[0]).total_seconds(),
            total_range_km=total_range,
            total_propagation_delay_ms=total_delay,
            link_metrics=link_metrics,
        )

    def best_interval_duration_s(intervals: Sequence[Tuple[datetime, datetime]]) -> float:
        if not intervals:
            return 0.0
        return max((stop - start).total_seconds() for start, stop in intervals)

    def path_delay_ms(path: Sequence[str]) -> float:
        delay = 0.0
        for idx in range(len(path) - 1):
            metric = metrics.get((path[idx], path[idx + 1]))
            if metric is not None:
                delay += metric.propagation_delay_ms
        return delay

    def prune_states(states: Sequence[Tuple[str, List[str], Sequence[Tuple[datetime, datetime]]]]):
        grouped: Dict[str, List[Tuple[str, List[str], Sequence[Tuple[datetime, datetime]]]]] = {}
        for state in states:
            grouped.setdefault(state[0], []).append(state)

        pruned = []
        width = beam_width_per_node if beam_width_per_node > 0 else max_paths
        for node_states in grouped.values():
            node_states.sort(
                key=lambda item: (
                    -best_interval_duration_s(item[2]),
                    path_delay_ms(item[1]),
                    len(item[1]),
                    "->".join(item[1]),
                )
            )
            pruned.extend(node_states[:width])
        return pruned

    current_states: List[Tuple[str, List[str], Sequence[Tuple[datetime, datetime]]]] = [
        (source_node, [source_node], initial_intervals)
    ]

    for _ in range(max_hops):
        next_states: List[Tuple[str, List[str], Sequence[Tuple[datetime, datetime]]]] = []
        for node, path, intervals in current_states:
            hop_count = len(path) - 1
            if hop_count >= max_hops:
                continue
            for neighbor in adjacency.get(node, []):
                if neighbor in path:
                    continue
                next_hop_count = hop_count + 1
                if exact_hops is not None and next_hop_count > exact_hops:
                    continue
                edge_windows = windows_by_pair.get((node, neighbor), [])
                next_intervals = _intersect_intervals(intervals, edge_windows)
                if not next_intervals:
                    continue
                next_path = path + [neighbor]
                if neighbor == ground_node:
                    if exact_hops is None or next_hop_count == exact_hops:
                        found.append(materialize(next_path, next_intervals))
                else:
                    next_states.append((neighbor, next_path, next_intervals))
        if not next_states:
            break
        current_states = prune_states(next_states)

    if sort_policy == "longest_visibility":
        found.sort(
            key=lambda item: (
                -item.common_duration_s,
                item.hop_count,
                item.total_propagation_delay_ms,
                "->".join(item.path),
            )
        )
    elif sort_policy == "shortest_delay":
        found.sort(
            key=lambda item: (
                item.hop_count,
                item.total_propagation_delay_ms,
                -item.common_duration_s,
                "->".join(item.path),
            )
        )
    else:
        raise ValueError(f"Unsupported STK path sort_policy: {sort_policy}")
    ranked = []
    for idx, path in enumerate(found[:max_paths], start=1):
        ranked.append(
            CandidatePath(
                rank=idx,
                path=path.path,
                hop_count=path.hop_count,
                satellite_count=path.satellite_count,
                common_start=path.common_start,
                common_stop=path.common_stop,
                common_duration_s=path.common_duration_s,
                total_range_km=path.total_range_km,
                total_propagation_delay_ms=path.total_propagation_delay_ms,
                link_metrics=path.link_metrics,
            )
        )
    return ranked


def candidate_paths_to_jsonable(paths: Sequence[CandidatePath]) -> List[dict]:
    payload = []
    for path in paths:
        row = asdict(path)
        row["common_start"] = format_stk_time(path.common_start)
        row["common_stop"] = format_stk_time(path.common_stop)
        row["link_metrics"] = [asdict(item) for item in path.link_metrics]
        payload.append(row)
    return payload


def _load_base_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_node(base_nodes: Dict[str, dict], node_id: str, fallback: dict) -> dict:
    node = dict(base_nodes.get(node_id, fallback))
    node.pop("neighbors", None)
    return node


def build_network_config_for_path(
    path: CandidatePath,
    base_config_path: str | Path = "config/network_config.json",
    ground_alias: str = "GS",
    isl_bandwidth_mbps: float = 10_000.0,
    gsl_bandwidth_mbps: float = 200.0,
    rs_sat_bandwidth_mbps: Optional[float] = None,
) -> dict:
    """Convert one STK candidate path into the project's network_config schema."""

    base = _load_base_config(base_config_path)
    base_nodes = base.get("nodes", {})
    sat_templates = [node_id for node_id in base_nodes if node_id.startswith("SAT-")]
    sat_templates.sort()
    if not sat_templates:
        sat_templates = ["SAT-01"]

    rs_fallback = {
        "ip": "127.0.0.1",
        "port": 8000,
        "role": "remote_sensing",
        "device": "PC",
        "hardware": {"compute_speed_gflops_per_ms": 0.0, "memory_mb": 0, "compute_speed_tflops": 0.0},
    }
    sat_fallback = {
        "ip": "127.0.0.1",
        "port": 8001,
        "role": "leo_computing",
        "device": "Jetson_1",
        "hardware": {"compute_speed_gflops_per_ms": 4.0, "memory_mb": 4096, "compute_speed_tflops": 4.0},
    }
    gs_fallback = {
        "ip": "127.0.0.1",
        "port": 9000,
        "role": "ground_station",
        "device": "PC",
        "hardware": {"compute_speed_gflops_per_ms": 300.0, "memory_mb": 64000, "compute_speed_tflops": 300.0},
    }

    stk_path = path.path
    satellite_stk_ids = stk_path[1:-1]
    pipeline = ["RS"] + [f"SAT-{idx:02d}" for idx in range(1, len(satellite_stk_ids) + 1)] + [ground_alias]

    nodes: Dict[str, dict] = {}
    for idx, node_id in enumerate(pipeline):
        if node_id == "RS":
            nodes[node_id] = _copy_node(base_nodes, "RS", rs_fallback)
        elif node_id == ground_alias:
            nodes[node_id] = _copy_node(base_nodes, "GS", gs_fallback)
        else:
            template_id = sat_templates[(idx - 1) % len(sat_templates)]
            nodes[node_id] = _copy_node(base_nodes, template_id, sat_fallback)
            nodes[node_id]["stk_id"] = satellite_stk_ids[idx - 1]
        nodes[node_id]["neighbors"] = []

    for idx, node_id in enumerate(pipeline):
        if idx > 0:
            nodes[node_id]["neighbors"].append(pipeline[idx - 1])
        if idx < len(pipeline) - 1:
            nodes[node_id]["neighbors"].append(pipeline[idx + 1])

    links: Dict[str, dict] = {}
    for idx, metric in enumerate(path.link_metrics):
        src = pipeline[idx]
        dst = pipeline[idx + 1]
        link_name = f"{src}_to_{dst}"
        links[link_name] = {
            "bandwidth_mbps": _edge_bandwidth_mbps(
                metric.link_type,
                isl_bandwidth_mbps=isl_bandwidth_mbps,
                gsl_bandwidth_mbps=gsl_bandwidth_mbps,
                rs_sat_bandwidth_mbps=rs_sat_bandwidth_mbps,
            ),
            "propagation_delay_ms": round(metric.propagation_delay_ms, 6),
            "stk_from": metric.from_id,
            "stk_to": metric.to_id,
            "stk_link_type": metric.link_type,
            "avg_range_km": round(metric.avg_range_km, 6),
            "sample_count": metric.sample_count,
            "window_count": metric.window_count,
            "visibility_duration_s": round(metric.visibility_duration_s, 3),
        }

    config = {
        "global_settings": base.get("global_settings", {}),
        "nodes": nodes,
        "links": links,
        "simulation_paths": {
            "pipeline": pipeline,
            "parallel_candidates": pipeline[1:-1],
            "parallel_aggregator": pipeline[-2] if len(pipeline) > 2 else None,
            "stk_path": {
                "rank": path.rank,
                "original_path": stk_path,
                "common_start_utcg": format_stk_time(path.common_start),
                "common_stop_utcg": format_stk_time(path.common_stop),
                "common_duration_s": round(path.common_duration_s, 3),
                "total_range_km": round(path.total_range_km, 6),
                "total_propagation_delay_ms": round(path.total_propagation_delay_ms, 6),
            },
        },
    }
    if "reference_compute_speed" in base:
        config["reference_compute_speed"] = base["reference_compute_speed"]
    return config


def write_stk_network_outputs(
    paths: Sequence[CandidatePath],
    output_dir: str | Path,
    base_config_path: str | Path = "config/network_config.json",
    num_configs: int = 1,
    isl_bandwidth_mbps: float = 10_000.0,
    gsl_bandwidth_mbps: float = 200.0,
    rs_sat_bandwidth_mbps: Optional[float] = None,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / "candidate_paths.json"
    with candidates_path.open("w", encoding="utf-8") as f:
        json.dump(candidate_paths_to_jsonable(paths), f, ensure_ascii=False, indent=2)

    written = {"candidate_paths": str(candidates_path)}
    for path in paths[: max(0, num_configs)]:
        config = build_network_config_for_path(
            path,
            base_config_path=base_config_path,
            isl_bandwidth_mbps=isl_bandwidth_mbps,
            gsl_bandwidth_mbps=gsl_bandwidth_mbps,
            rs_sat_bandwidth_mbps=rs_sat_bandwidth_mbps,
        )
        config_path = output_dir / f"network_config_path{path.rank:03d}.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        written[f"network_config_path{path.rank:03d}"] = str(config_path)
    return written
