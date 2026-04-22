import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.pmp_solver import PMPSolver


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ordered_layers(raw_layers: Dict) -> List[Dict]:
    if isinstance(raw_layers, list):
        return list(raw_layers)

    return [raw_layers[key] for key in sorted(raw_layers.keys(), key=lambda item: int(item))]


def build_solver(
    model_name: str,
    config_key: str,
    batch_size: int,
    target_h: int,
    target_w: int,
) -> Tuple[PMPSolver, Dict]:
    net_config = load_json(PROJECT_ROOT / "config" / "network_config.json")
    pc_profiles = load_json(PROJECT_ROOT / "config" / "dnn_profiles_database_pc.json")
    jetson_profiles = load_json(PROJECT_ROOT / "config" / "dnn_profiles_database_jetson.json")

    if model_name not in pc_profiles:
        raise KeyError(f"PC 数据库中没有模型: {model_name}")
    if model_name not in jetson_profiles:
        raise KeyError(f"Jetson 数据库中没有模型: {model_name}")
    if config_key not in pc_profiles[model_name]:
        raise KeyError(f"PC 数据库中没有配置: {model_name} -> {config_key}")
    if config_key not in jetson_profiles[model_name]:
        raise KeyError(f"Jetson 数据库中没有配置: {model_name} -> {config_key}")

    model_profile = {
        "layers": {
            "pc": ordered_layers(pc_profiles[model_name][config_key]),
            "jetson": ordered_layers(jetson_profiles[model_name][config_key]),
        },
        "input_size_raw": (batch_size * 3 * target_h * target_w * 4) / (1024 ** 2),
    }

    if "simulation_paths" in net_config and "pipeline" in net_config["simulation_paths"]:
        compute_node_ids = net_config["simulation_paths"]["pipeline"][1:]
        current_source = net_config["simulation_paths"]["pipeline"][0]
    else:
        compute_node_ids = [node_id for node_id in net_config["nodes"].keys() if "RS" not in node_id]
        current_source = "RS"

    nodes: List[Dict] = []
    for node_id in compute_node_ids:
        node_info = dict(net_config["nodes"][node_id])
        node_info["id"] = node_id
        nodes.append(node_info)

    bandwidths: List[float] = []
    raw_links = net_config.get("links", {})
    for node in nodes:
        target_node = node["id"]
        forward_key = f"{current_source}_to_{target_node}"
        backward_key = f"{target_node}_to_{current_source}"
        bandwidth = 100.0

        if forward_key in raw_links:
            bandwidth = float(raw_links[forward_key].get("bandwidth_mbps", 100.0))
        elif backward_key in raw_links:
            bandwidth = float(raw_links[backward_key].get("bandwidth_mbps", 100.0))

        bandwidths.append(bandwidth)
        current_source = target_node

    env_status = {
        "nodes": nodes,
        "bandwidths": bandwidths,
        "reference_compute_speed": net_config.get("reference_compute_speed", 100.0),
    }

    return PMPSolver(model_profile, env_status), model_profile


def time_call(func: Callable[[], Tuple[float, Dict]], repeats: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        func()

    durations_ms: List[float] = []
    last_latency = float("nan")

    for _ in range(repeats):
        start = time.perf_counter()
        last_latency, _ = func()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.fmean(durations_ms),
        "std_ms": statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0,
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "last_latency_ms": last_latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PMPSolver search time with mean runtime reporting.")
    parser.add_argument("--model", default="yolov5", help="模型名称，默认 yolov5")
    parser.add_argument("--config-key", default="b16_640x640", help="模型配置键，默认 b16_640x640")
    parser.add_argument("--batch-size", type=int, default=16, help="输入 batch size")
    parser.add_argument("--height", type=int, default=640, help="输入高度")
    parser.add_argument("--width", type=int, default=640, help="输入宽度")
    parser.add_argument("--repeats", type=int, default=20, help="正式计时次数")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--pop-size", type=int, default=30, help="GA 种群大小")
    parser.add_argument("--generations", type=int, default=50, help="GA 迭代代数")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="GA 变异率")
    parser.add_argument("--output-csv", default="search_time_mean.csv", help="结果输出 CSV 文件")
    args = parser.parse_args()

    solver, model_profile = build_solver(
        model_name=args.model,
        config_key=args.config_key,
        batch_size=args.batch_size,
        target_h=args.height,
        target_w=args.width,
    )

    print("模型: {} | 配置: {}".format(args.model, args.config_key))
    print("输入大小: {:.2f} MB".format(model_profile["input_size_raw"]))
    print("重复次数: {} | 预热次数: {}".format(args.repeats, args.warmup))
    print("-" * 78)
    print("{:<28} {:>12} {:>12} {:>12} {:>12}".format("Algorithm", "Mean(ms)", "Std(ms)", "Min(ms)", "Max(ms)"))
    print("-" * 78)

    benchmarks = {
        "solve_la_dp": lambda: solver.solve_la_dp(),
        "solve_communication_greedy": lambda: solver.solve_communication_greedy(),
        "solve_ga": lambda: solver.solve_ga(
            pop_size=args.pop_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
        ),
    }

    results: List[Dict[str, float]] = []
    for name, func in benchmarks.items():
        stats = time_call(func, repeats=args.repeats, warmup=args.warmup)
        results.append({"algorithm": name, **stats})
        print(
            "{:<28} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(
                name,
                stats["mean_ms"],
                stats["std_ms"],
                stats["min_ms"],
                stats["max_ms"],
            )
        )

    csv_path = PROJECT_ROOT / args.output_csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "mean_ms", "std_ms", "min_ms", "max_ms", "last_latency_ms"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("-" * 78)
    print(f"结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()