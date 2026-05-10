# -*- coding: utf-8 -*-
"""Run the controlled PMP algorithm comparison used to redraw Fig. 01.

This script intentionally does not use experiments_runner.py because that
entrypoint randomizes the topology for algo_effectiveness experiments.  Here we
need a fixed three-LEO path, fixed bandwidths, homogeneous LEO compute, and
repeat-only variation for GA/Random.
"""

from __future__ import annotations

import csv
import json
import random
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.scheduler import Scheduler


SCENARIO = "homogeneous"
OUT_DIR = Path("result/paper_figures_controlled/fig01_pmp_algorithm")
BASE_CONFIG = Path("config/generated/stk_sat_inferv2_hop4/network_config_path001.json")
CONTROLLED_CONFIG = OUT_DIR / "controlled_three_leo_pmp_config.json"
RESULTS_LONG = OUT_DIR / "fig01_pmp_controlled_results_long.csv"
SUMMARY_CSV = OUT_DIR / "fig01_pmp_controlled_summary.csv"

RUN_ID = "controlled_fig01_pmp_b64_3leo"
EXP_TYPE = "controlled_pmp_algorithm_effectiveness"
REPEATS = 100

LEO_TFLOPS = 5.0
LEO_TFLOPS_BY_NODE = {"SAT-01": 5.0, "SAT-02": 5.0, "SAT-03": 5.0}
GS_TFLOPS = 500.0
LEO_MEMORY_MB = 4096
GS_MEMORY_MB = 64000
ISL_BW_MBPS = 1800.0
GSL_BW_MBPS = 100.0

MODELS = [
    ("yolov5", "YOLOv5", 64, 640, 640),
    ("vgg19", "VGG19", 64, 224, 224),
    ("swin_base", "Swin-Base", 64, 224, 224),
    ("vit_huge", "ViT-Huge", 64, 224, 224),
]

ALGORITHMS = ["LA-DP", "Greedy", "GA", "Random", "Uniform", "GS-Only"]
ALG_LABEL = {
    "LA-DP": "LADP",
    "Greedy": "贪心",
    "GA": "遗传算法",
    "Random": "随机",
    "Uniform": "均匀",
    "GS-Only": "GS-Only",
}
ALG_COLOR = {
    "LA-DP": "#244C85",
    "Greedy": "#E39D2D",
    "GA": "#2A8C88",
    "Random": "#E95B45",
    "Uniform": "#8272B2",
    "GS-Only": "#4A4A4A",
}


def configure_scenario(scenario: str) -> None:
    global SCENARIO, OUT_DIR, CONTROLLED_CONFIG, RESULTS_LONG, SUMMARY_CSV, RUN_ID
    global LEO_TFLOPS, LEO_TFLOPS_BY_NODE

    scenario = scenario.lower().strip()
    if scenario not in {"homogeneous", "heterogeneous"}:
        raise ValueError("scenario must be homogeneous or heterogeneous")

    SCENARIO = scenario
    if scenario == "homogeneous":
        OUT_DIR = Path("result/paper_figures_controlled/fig01_pmp_algorithm")
        RUN_ID = "controlled_fig01_pmp_b64_3leo"
        LEO_TFLOPS_BY_NODE = {"SAT-01": 5.0, "SAT-02": 5.0, "SAT-03": 5.0}
    else:
        OUT_DIR = Path("result/paper_figures_controlled/fig01_pmp_algorithm_heterogeneous")
        RUN_ID = "controlled_fig01_pmp_b64_3leo_heterogeneous"
        LEO_TFLOPS_BY_NODE = {"SAT-01": 3.0, "SAT-02": 8.0, "SAT-03": 5.0}

    LEO_TFLOPS = float(np.mean(list(LEO_TFLOPS_BY_NODE.values())))
    CONTROLLED_CONFIG = OUT_DIR / "controlled_three_leo_pmp_config.json"
    RESULTS_LONG = OUT_DIR / "fig01_pmp_controlled_results_long.csv"
    SUMMARY_CSV = OUT_DIR / "fig01_pmp_controlled_summary.csv"


def setup_font() -> None:
    candidates = [
        "SimSun",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11


def build_controlled_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config = deepcopy(config)
    for node_id, node in config["nodes"].items():
        hw = node.setdefault("hardware", {})
        if node_id.startswith("SAT-"):
            node_tflops = float(LEO_TFLOPS_BY_NODE.get(node_id, LEO_TFLOPS))
            hw["compute_speed_gflops_per_ms"] = node_tflops
            hw["compute_speed_tflops"] = node_tflops
            hw["memory_mb"] = LEO_MEMORY_MB
            node["role"] = "leo_computing"
        elif node_id == "GS":
            hw["compute_speed_gflops_per_ms"] = GS_TFLOPS
            hw["compute_speed_tflops"] = GS_TFLOPS
            hw["memory_mb"] = GS_MEMORY_MB
        elif node_id == "RS":
            hw["compute_speed_gflops_per_ms"] = 0.0
            hw["compute_speed_tflops"] = 0.0
            hw["memory_mb"] = 0

    for link_name, link in config["links"].items():
        if "GS" in link_name:
            link["bandwidth_mbps"] = GSL_BW_MBPS
            link["controlled_bandwidth_class"] = "GSL"
        else:
            link["bandwidth_mbps"] = ISL_BW_MBPS
            link["controlled_bandwidth_class"] = "ISL"

    config.setdefault("controlled_experiment", {})
    config["controlled_experiment"].update(
        {
            "description": "Fixed 3-hop LEO PMP algorithm comparison for Fig. 01",
            "scenario": SCENARIO,
            "pipeline": config.get("simulation_paths", {}).get("pipeline", []),
            "leo_compute_tflops_by_node": LEO_TFLOPS_BY_NODE,
            "gs_compute_tflops": GS_TFLOPS,
            "leo_memory_mb": LEO_MEMORY_MB,
            "isl_bandwidth_mbps": ISL_BW_MBPS,
            "gsl_bandwidth_mbps": GSL_BW_MBPS,
            "propagation_delay_source": str(BASE_CONFIG),
            "repeats": REPEATS,
        }
    )
    return config


def write_controlled_config() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_controlled_config()
    with CONTROLLED_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def run_experiment() -> pd.DataFrame:
    scheduler = Scheduler(
        net_config_path=str(CONTROLLED_CONFIG),
        pc_profiles_path="config/dnn_profiles_database_pc.json",
        jetson_profiles_path="config/dnn_profiles_database_jetson.json",
    )

    rows = []
    for model_name, model_label, batch, input_h, input_w in MODELS:
        print(f"[CONTROLLED-01] model={model_label}, batch={batch}, input={input_h}x{input_w}")
        for repeat in range(REPEATS):
            seed = 20260509 + repeat * 97 + len(rows)
            random.seed(seed)
            np.random.seed(seed)
            plans = scheduler.generate_task_and_schedule(
                task_id=f"{model_name}_rep{repeat:03d}",
                model_name=model_name,
                batch_size=batch,
                target_h=input_h,
                target_w=input_w,
                run_id=RUN_ID,
                exp_type=EXP_TYPE,
                mode="theory",
                algorithm_names=ALGORITHMS,
                persist_theory=False,
                return_full_plans=True,
                metadata_extra={
                    "sweep_param": "controlled_repeat",
                    "sweep_value": repeat,
                },
            )

            gs_latency = plans.get("GS-Only", {}).get("latency", float("inf"))
            gs_energy = plans.get("GS-Only", {}).get("satellite_energy_j", float("inf"))
            for alg in ALGORITHMS:
                data = plans.get(alg, {})
                latency = data.get("latency", float("inf"))
                energy = data.get("satellite_energy_j", float("inf"))
                norm_latency = latency / gs_latency if finite(latency) and finite(gs_latency) and gs_latency > 0 else np.nan
                norm_energy = energy / gs_energy if finite(energy) and finite(gs_energy) and gs_energy > 0 else np.nan
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "exp_type": EXP_TYPE,
                        "repeat": repeat,
                        "algorithm": alg,
                        "model_name": model_name,
                        "model_label": model_label,
                        "batch_size": batch,
                        "input_h": input_h,
                        "input_w": input_w,
                        "pipeline_path": "->".join(scheduler.net_config["simulation_paths"]["pipeline"]),
                        "pipeline_node_count": 3,
                        "pipeline_hop_count": 4,
                        "isl_avg_bw_mbps": ISL_BW_MBPS,
                        "gsl_avg_bw_mbps": GSL_BW_MBPS,
                        "leo_compute_tflops": LEO_TFLOPS,
                        "leo_compute_profile": json.dumps(LEO_TFLOPS_BY_NODE, ensure_ascii=False),
                        "sat01_compute_tflops": LEO_TFLOPS_BY_NODE.get("SAT-01"),
                        "sat02_compute_tflops": LEO_TFLOPS_BY_NODE.get("SAT-02"),
                        "sat03_compute_tflops": LEO_TFLOPS_BY_NODE.get("SAT-03"),
                        "gs_compute_tflops": GS_TFLOPS,
                        "leo_memory_mb": LEO_MEMORY_MB,
                        "latency_ms": latency,
                        "norm_latency_vs_gs": norm_latency,
                        "satellite_energy_j": energy,
                        "norm_energy_vs_gs": norm_energy,
                        "energy_compute_j": data.get("energy_compute_j", np.nan),
                        "energy_comm_j": data.get("energy_comm_j", np.nan),
                        "energy_model": data.get("energy_model", "satellite_only:P_compute=15W,P_tx=10W"),
                        "plan": json.dumps(data.get("plan"), ensure_ascii=False),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_LONG, index=False, encoding="utf-8-sig")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "norm_latency_vs_gs",
        "latency_ms",
        "satellite_energy_j",
        "norm_energy_vs_gs",
        "energy_compute_j",
        "energy_comm_j",
    ]
    df = df.copy()
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    summary = (
        df.groupby(["model_name", "model_label", "algorithm"], dropna=False)
        .agg(
            mean_norm_latency_vs_gs=("norm_latency_vs_gs", "mean"),
            std_norm_latency_vs_gs=("norm_latency_vs_gs", "std"),
            mean_latency_ms=("latency_ms", "mean"),
            std_latency_ms=("latency_ms", "std"),
            mean_satellite_energy_j=("satellite_energy_j", "mean"),
            std_satellite_energy_j=("satellite_energy_j", "std"),
            samples=("repeat", "count"),
        )
        .reset_index()
    )
    summary["algorithm"] = pd.Categorical(summary["algorithm"], ALGORITHMS, ordered=True)
    model_order = [m[1] for m in MODELS]
    summary["model_label"] = pd.Categorical(summary["model_label"], model_order, ordered=True)
    summary = summary.sort_values(["model_label", "algorithm"])
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return summary


def plot(summary: pd.DataFrame) -> None:
    setup_font()
    model_order = [m[1] for m in MODELS]
    pivot = (
        summary.pivot_table(
            index="model_label",
            columns="algorithm",
            values="mean_norm_latency_vs_gs",
            observed=False,
        )
        .reindex(index=model_order, columns=ALGORITHMS)
    )

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    x = np.arange(len(pivot.index))
    width = min(0.12, 0.78 / len(ALGORITHMS))
    for idx, alg in enumerate(ALGORITHMS):
        values = pivot[alg].astype(float).to_numpy()
        offset = (idx - (len(ALGORITHMS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=ALG_LABEL.get(alg, alg),
            color=ALG_COLOR.get(alg, "#666666"),
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.035,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#334155",
                )

    ax.set_title("PMP 模式下不同算法的归一化时延对比", pad=20)
    ax.set_ylabel("归一化时延（相对 GS-Only）")
    ax.set_xlabel("模型")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.grid(axis="y", linestyle="--", linewidth=0.9, color="#D5D5D5", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=6, frameon=False)
    ymax = max(1.15, float(np.nanmax(pivot.to_numpy())) * 1.18)
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.09, right=0.98)
    fig.savefig(OUT_DIR / "01_pmp_algorithm_latency_norm_controlled.png", dpi=300)
    fig.savefig(OUT_DIR / "01_pmp_algorithm_latency_norm_controlled.pdf")
    plt.close(fig)


def write_notes(summary: pd.DataFrame) -> None:
    best_rows = []
    for model_label in [m[1] for m in MODELS]:
        subset = summary[summary["model_label"].astype(str) == model_label].copy()
        subset["mean_norm_latency_vs_gs"] = pd.to_numeric(subset["mean_norm_latency_vs_gs"], errors="coerce")
        best = subset.loc[subset["mean_norm_latency_vs_gs"].idxmin()]
        best_rows.append(
            f"| {model_label} | {best['algorithm']} | {best['mean_norm_latency_vs_gs']:.3f} |"
        )

    notes = [
        "# Controlled PMP Fig. 01",
        "",
        "## Experiment setup",
        "",
        f"- Pipeline: `RS -> SAT-01 -> SAT-02 -> SAT-03 -> GS`.",
        f"- ISL bandwidth: `{ISL_BW_MBPS:.0f} Mbps`; GSL bandwidth: `{GSL_BW_MBPS:.0f} Mbps`.",
        f"- Scenario: `{SCENARIO}`.",
        f"- LEO compute: `{json.dumps(LEO_TFLOPS_BY_NODE, ensure_ascii=False)}` TFLOPS; GS compute: `{GS_TFLOPS:.1f} TFLOPS`.",
        f"- LEO memory: `{LEO_MEMORY_MB} MB`.",
        "- Propagation delays are inherited from the selected STK path.",
        f"- Repeats: `{REPEATS}` for each model, mainly to stabilize Random and GA.",
        "- Energy model: `P_compute=15 W`, `P_tx=10 W`.",
        "",
        "## Input profiles",
        "",
        "| Model | Batch | Input |",
        "|---|---:|---:|",
    ]
    notes.extend([f"| {label} | {batch} | {h}x{w} |" for _, label, batch, h, w in MODELS])
    notes.extend(
        [
            "",
            "## Best normalized latency",
            "",
            "| Model | Best algorithm | Mean normalized latency |",
            "|---|---|---:|",
            *best_rows,
            "",
            "## Files",
            "",
            f"- Config: `{CONTROLLED_CONFIG.as_posix()}`",
            f"- Long results: `{RESULTS_LONG.as_posix()}`",
            f"- Summary: `{SUMMARY_CSV.as_posix()}`",
            "- Figure: `01_pmp_algorithm_latency_norm_controlled.png/pdf`",
        ]
    )
    (OUT_DIR / "controlled_fig01_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled PMP Fig. 01 experiment")
    parser.add_argument(
        "--scenario",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Controlled compute scenario to run",
    )
    args = parser.parse_args()
    configure_scenario(args.scenario)
    write_controlled_config()
    df = run_experiment()
    summary = summarize(df)
    plot(summary)
    write_notes(summary)
    print(f"[CONTROLLED-01] outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
