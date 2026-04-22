import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cdp import CDPSolver


def build_mock_env(node_count: int) -> Dict:
    candidates = [f"SAT-{i:02d}" for i in range(1, node_count + 1)]
    aggregator = candidates[-1]

    nodes: Dict[str, Dict] = {}
    links: Dict[str, Dict] = {}

    for idx, node_id in enumerate(candidates):
        nodes[node_id] = {
            "hardware": {
                "compute_speed_gflops_per_ms": 15.0 + idx * 5.0,
            },
            "memory_mb": 4096,
        }

        links[f"RS_to_{node_id}"] = {
            "bandwidth_mbps": 2500.0 + idx * 700.0,
            "propagation_delay_ms": 10.0 + idx,
        }

        if node_id != aggregator:
            links[f"{node_id}_to_{aggregator}"] = {
                "bandwidth_mbps": 3000.0 + idx * 500.0,
                "propagation_delay_ms": 8.0 + idx,
            }

    env_status = {
        "nodes": nodes,
        "links": links,
        "simulation_paths": {
            "parallel_candidates": candidates,
            "parallel_aggregator": aggregator,
        },
    }
    return env_status


def build_model_profile() -> Dict:
    return {
        "input_size_mb": 150.0,
        "output_size_mb": 5.0,
        "compute_total_gflops": 120.0,
    }


def benchmark(func, repeats: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        func()

    durations_ms: List[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(durations_ms),
        "std_ms": statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0,
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "last_latency_ms": float(last_result[0]) if last_result is not None else float("nan"),
    }


def print_complexity_summary() -> None:
    print("理论复杂度对照")
    print("- LAWA: O(K)")
    print("- Greedy: O(K)")
    print("- GA: O(pop_size * generations * K)")
    print()


def run_once(node_count: int, repeats: int, warmup: int, pop_size: int, generations: int) -> Dict[str, Dict[str, float]]:
    solver = CDPSolver(build_model_profile(), build_mock_env(node_count))

    alg_map = {
        "LAWA": lambda: solver.solve_lawa(),
        "Greedy": lambda: solver.solve_greedy(),
        "GA": lambda: solver.solve_ga(pop_size=pop_size, gen=generations),
    }

    results = {}
    for name, func in alg_map.items():
        results[name] = benchmark(func, repeats=repeats, warmup=warmup)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CDP search time of LAWA, Greedy and GA.")
    parser.add_argument("--nodes", type=int, nargs="+", default=[3, 4, 5, 6], help="要测试的节点数列表")
    parser.add_argument("--repeats", type=int, default=20, help="正式计时重复次数")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--pop-size", type=int, default=20, help="GA 种群大小")
    parser.add_argument("--generations", type=int, default=30, help="GA 迭代代数")
    parser.add_argument("--output-csv", default="cdp_search_time_mean.csv", help="输出 CSV 文件名")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    print_complexity_summary()
    print(f"重复次数: {args.repeats} | 预热次数: {args.warmup} | GA(pop={args.pop_size}, gen={args.generations})")
    print("-" * 92)
    print("{:<8} {:<12} {:>12} {:>12} {:>12} {:>12}".format("K", "Algorithm", "Mean(ms)", "Std(ms)", "Min(ms)", "Max(ms)"))
    print("-" * 92)

    csv_rows: List[Dict[str, object]] = []
    for node_count in args.nodes:
        results = run_once(
            node_count=node_count,
            repeats=args.repeats,
            warmup=args.warmup,
            pop_size=args.pop_size,
            generations=args.generations,
        )

        for alg_name in ["LAWA", "Greedy", "GA"]:
            stat = results[alg_name]
            csv_rows.append({
                "K": node_count,
                "Algorithm": alg_name,
                "Mean_ms": stat["mean_ms"],
                "Std_ms": stat["std_ms"],
                "Min_ms": stat["min_ms"],
                "Max_ms": stat["max_ms"],
                "Last_Latency_ms": stat["last_latency_ms"],
            })
            print(
                "{:<8} {:<12} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(
                    node_count,
                    alg_name,
                    stat["mean_ms"],
                    stat["std_ms"],
                    stat["min_ms"],
                    stat["max_ms"],
                )
            )

    csv_path = PROJECT_ROOT / args.output_csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["K", "Algorithm", "Mean_ms", "Std_ms", "Min_ms", "Max_ms", "Last_Latency_ms"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("-" * 92)
    print(f"结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()