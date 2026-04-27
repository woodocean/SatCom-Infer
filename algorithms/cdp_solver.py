"""No-aggregator CDP solver.

CDP is modeled as data parallel inference:
    RS -> worker satellites for input distribution
    each worker runs the complete model on its assigned samples
    worker satellites -> GS for direct result return

There is no in-orbit aggregation node in this model.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
import statistics
import time
from typing import Dict, List, Tuple

import numpy as np


class CDPSolver:
    def __init__(self, model_profile: Dict, env_status: Dict):
        self.D_in = float(model_profile.get("input_size_mb", 100.0))
        self.D_out = float(model_profile.get("output_size_mb", 5.0))
        self.C_total = float(model_profile.get("compute_total_gflops", 50.0))
        self.batch_size = int(model_profile.get("batch_size", 32))

        self.nodes = env_status["nodes"]
        self.K = len(self.nodes)
        self.node_ids = [node["id"] for node in self.nodes]

        self.f = np.array(
            [float(node.get("compute_speed_gflops_per_s", 1.0)) for node in self.nodes],
            dtype=float,
        )
        self.compute_full_ms = np.array(
            [
                float(node.get("compute_full_model_ms", 0.0))
                for node in self.nodes
            ],
            dtype=float,
        )
        self.b_dist = np.array(
            [float(node.get("b_dist_mbps", 10.0)) for node in self.nodes],
            dtype=float,
        )
        self.b_return = np.array(
            [
                float(node.get("b_return_mbps", node.get("b_agg_mbps", 10.0)))
                for node in self.nodes
            ],
            dtype=float,
        )
        self.dist_prop_ms = np.array(
            [float(node.get("dist_prop_ms", 0.0)) for node in self.nodes],
            dtype=float,
        )
        self.return_prop_ms = np.array(
            [float(node.get("return_prop_ms", 0.0)) for node in self.nodes],
            dtype=float,
        )

        self.f = np.clip(self.f, a_min=1e-9, a_max=None)
        self.b_dist = np.clip(self.b_dist, a_min=1e-9, a_max=None)
        self.b_return = np.clip(self.b_return, a_min=1e-9, a_max=None)
        self.compute_full_ms = np.where(
            self.compute_full_ms > 0.0,
            self.compute_full_ms,
            (self.C_total / self.f) * 1000.0,
        )

    @staticmethod
    def _tx_ms(data_mb: np.ndarray | float, bandwidth_mbps: np.ndarray | float) -> np.ndarray | float:
        return (data_mb * 8.0 / bandwidth_mbps) * 1000.0

    def _compute_ms(self, data_alloc_mb: np.ndarray) -> np.ndarray:
        if self.D_in <= 0:
            return np.full(self.K, float("inf"))
        return self.compute_full_ms * (data_alloc_mb / self.D_in)

    def _evaluate_delay(self, data_alloc_mb: np.ndarray) -> Tuple[float, Dict[str, float]]:
        output_alloc_mb = data_alloc_mb * (self.D_out / self.D_in) if self.D_in > 0 else data_alloc_mb
        active = data_alloc_mb > 0.0

        t_dist = self._tx_ms(data_alloc_mb, self.b_dist) + np.where(active, self.dist_prop_ms, 0.0)
        t_comp = self._compute_ms(data_alloc_mb)
        t_return = self._tx_ms(output_alloc_mb, self.b_return) + np.where(active, self.return_prop_ms, 0.0)
        node_delays = t_dist + t_comp + t_return

        max_delay = float(np.max(node_delays)) if len(node_delays) else float("inf")
        plan = {self.node_ids[i]: round(float(data_alloc_mb[i]), 6) for i in range(self.K)}
        return max_delay, plan

    def _unit_cost_ms_per_mb(self) -> np.ndarray:
        if self.D_in <= 0:
            return np.full(self.K, float("inf"))
        term_dist = 8.0 * 1000.0 / self.b_dist
        term_comp = self.compute_full_ms / self.D_in
        term_return = (self.D_out / self.D_in) * 8.0 * 1000.0 / self.b_return
        return term_dist + term_comp + term_return

    def continuous_lawa_allocation(self) -> np.ndarray:
        unit_cost = self._unit_cost_ms_per_mb()
        gamma = 1.0 / np.clip(unit_cost, a_min=1e-12, a_max=None)
        gamma_total = float(np.sum(gamma))
        if gamma_total <= 0.0:
            return np.full(self.K, self.D_in / max(1, self.K), dtype=float)
        return self.D_in * (gamma / gamma_total)

    def solve_lawa(self) -> Tuple[float, Dict[str, float]]:
        return self._evaluate_delay(self.continuous_lawa_allocation())

    def _discretize_samples(self, continuous_alloc_mb: np.ndarray, batch_size: int) -> np.ndarray:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.K == 0:
            return np.array([], dtype=int)
        if self.K > batch_size:
            raise ValueError("worker count cannot exceed batch size for discrete CDP allocation")

        ideal = continuous_alloc_mb / max(self.D_in, 1e-12) * batch_size
        samples = np.floor(ideal).astype(int)
        samples = np.maximum(samples, 1)

        while int(np.sum(samples)) > batch_size:
            reducible = np.where(samples > 1)[0]
            if len(reducible) == 0:
                break
            idx = max(reducible, key=lambda item: (samples[item] - ideal[item], samples[item], -item))
            samples[idx] -= 1

        while int(np.sum(samples)) < batch_size:
            idx = int(np.argmax(ideal - samples))
            samples[idx] += 1

        return samples

    def solve_lawa_discrete(self, batch_size: int | None = None) -> Tuple[float, Dict]:
        batch_size = int(batch_size or self.batch_size)
        continuous_alloc = self.continuous_lawa_allocation()
        samples = self._discretize_samples(continuous_alloc, batch_size)
        sample_mb = self.D_in / batch_size
        discrete_alloc = samples.astype(float) * sample_mb
        latency_ms, alloc_mb = self._evaluate_delay(discrete_alloc)
        return latency_ms, {
            "alloc_mb": alloc_mb,
            "samples": {self.node_ids[i]: int(samples[i]) for i in range(self.K)},
            "split_ratio": {
                self.node_ids[i]: float(discrete_alloc[i] / self.D_in) if self.D_in > 0 else 0.0
                for i in range(self.K)
            },
            "continuous_alloc_mb": {
                self.node_ids[i]: float(continuous_alloc[i])
                for i in range(self.K)
            },
        }

    def solve_compute_greedy(self) -> Tuple[float, Dict[str, float]]:
        compute_rate = 1.0 / np.clip(self.compute_full_ms, a_min=1e-12, a_max=None)
        weights = compute_rate / np.sum(compute_rate)
        return self._evaluate_delay(self.D_in * weights)

    def solve_greedy(self) -> Tuple[float, Dict[str, float]]:
        return self.solve_compute_greedy()

    def solve_uniform(self) -> Tuple[float, Dict[str, float]]:
        return self._evaluate_delay(np.full(self.K, self.D_in / max(1, self.K), dtype=float))

    def solve_random_search(self, trials: int = 1000) -> Tuple[float, Dict[str, float]]:
        best_latency = float("inf")
        best_alloc = np.full(self.K, self.D_in / max(1, self.K), dtype=float)
        for _ in range(max(1, int(trials))):
            weights = np.random.rand(self.K)
            weights /= np.sum(weights)
            alloc = self.D_in * weights
            latency, _ = self._evaluate_delay(alloc)
            if latency < best_latency:
                best_latency = latency
                best_alloc = alloc
        return self._evaluate_delay(best_alloc)

    def solve_ga(self, pop_size: int = 50, generations: int = 150, mutation_rate: float = 0.3) -> Tuple[float, Dict[str, float]]:
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if generations < 1:
            raise ValueError("generations must be at least 1")

        def normalize(individual: np.ndarray) -> np.ndarray:
            candidate = np.clip(individual.astype(float), 0.0, None)
            total = float(np.sum(candidate))
            if total <= 0.0:
                return np.full(self.K, self.D_in / max(1, self.K), dtype=float)
            return candidate / total * self.D_in

        def fitness(individual: np.ndarray) -> float:
            latency, _ = self._evaluate_delay(normalize(individual))
            return -latency

        population = [np.random.dirichlet(np.ones(self.K)) * self.D_in for _ in range(pop_size)]
        best_alloc = normalize(population[0])
        best_score = -float("inf")

        for _ in range(generations):
            scores = [fitness(individual) for individual in population]
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_alloc = normalize(population[best_idx])

            ranked = np.argsort(scores)[::-1]
            selected = [population[index] for index in ranked[: max(2, pop_size // 2)]]
            next_population = [normalize(individual.copy()) for individual in selected[:2]]
            while len(next_population) < pop_size:
                parent_1, parent_2 = random.sample(selected, 2)
                cut = random.randint(1, self.K - 1) if self.K > 1 else 1
                child = np.concatenate([parent_1[:cut], parent_2[cut:]]) if self.K > 1 else parent_1.copy()
                if random.random() < mutation_rate:
                    child[random.randint(0, self.K - 1)] += random.uniform(-0.2, 0.2) * self.D_in
                next_population.append(normalize(child))
            population = next_population[:pop_size]

        return self._evaluate_delay(best_alloc)


def _build_mock_env(node_count: int) -> Dict:
    nodes = []
    for i in range(node_count):
        nodes.append(
            {
                "id": f"SAT-{i + 1:02d}",
                "compute_full_model_ms": 400.0 / (1.0 + i * 0.3),
                "b_dist_mbps": 500.0 + i * 200.0,
                "b_return_mbps": 80.0 + i * 30.0,
            }
        )
    return {"nodes": nodes}


def _build_mock_model_profile() -> Dict:
    return {
        "input_size_mb": 150.0,
        "output_size_mb": 5.0,
        "compute_total_gflops": 80.0,
        "batch_size": 64,
    }


def _benchmark(func, repeats: int, warmup: int) -> Dict[str, float]:
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


def run_benchmark(
    node_counts=None,
    repeats: int = 20,
    warmup: int = 3,
    pop_size: int = 20,
    generations: int = 30,
    output_csv: str = "cdp_search_time_mean.csv",
) -> None:
    if node_counts is None:
        node_counts = [3, 4, 5, 6]

    random.seed(42)
    np.random.seed(42)
    csv_rows = []
    for node_count in node_counts:
        solver = CDPSolver(_build_mock_model_profile(), _build_mock_env(node_count))
        algorithms = {
            "LAWA": lambda: solver.solve_lawa(),
            "LAWA-discrete": lambda: solver.solve_lawa_discrete(),
            "Greedy": lambda: solver.solve_greedy(),
            "GA": lambda: solver.solve_ga(pop_size=pop_size, generations=generations),
        }
        for name, func in algorithms.items():
            stat = _benchmark(func, repeats=repeats, warmup=warmup)
            csv_rows.append(
                {
                    "K": node_count,
                    "Algorithm": name,
                    "Mean_ms": stat["mean_ms"],
                    "Std_ms": stat["std_ms"],
                    "Min_ms": stat["min_ms"],
                    "Max_ms": stat["max_ms"],
                    "Last_Latency_ms": stat["last_latency_ms"],
                }
            )

    csv_path = Path(__file__).resolve().parent.parent / output_csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["K", "Algorithm", "Mean_ms", "Std_ms", "Min_ms", "Max_ms", "Last_Latency_ms"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    run_benchmark()
