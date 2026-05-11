# -*- coding: utf-8 -*-
"""Rerun one STK slot for a paper-ready PMP figure with stabilized Random/GA."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.scheduler import Scheduler


SLOT_CONFIG_BY_MODEL = {
    "yolov5": "result/stk_dynamic/stk_dynamic_yolo_001/configs/{slot_id}_network_config.json",
    "vgg19": "result/stk_dynamic/stk_dynamic_vgg19_001/configs/{slot_id}_network_config.json",
    "swin_base": "result/stk_dynamic/stk_dynamic_swin_base_001/configs/{slot_id}_network_config.json",
    "vit_huge": "result/stk_dynamic/stk_dynamic_vit_huge_001/configs/{slot_id}_network_config.json",
    "resnet101": "result/stk_dynamic/stk_dynamic_resnet101_001/configs/{slot_id}_network_config.json",
}

MODEL_TASKS = {
    "yolov5": ("YOLOv5", 32, 640, 640),
    "vgg19": ("VGG19", 32, 224, 224),
    "swin_base": ("Swin-Base", 32, 224, 224),
    "vit_huge": ("ViT-Huge", 32, 224, 224),
    "resnet101": ("ResNet101", 32, 224, 224),
}

ALGORITHMS = ["LA-DP", "Greedy", "GA", "Random", "Uniform", "GS-Only"]
DETERMINISTIC_ALGS = ["LA-DP", "Greedy", "Uniform", "GS-Only"]
STOCHASTIC_ALGS = ["GA", "Random"]

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
    "Uniform": "#9CA3AF",
    "GS-Only": "#4A4A4A",
}


def setup_style() -> None:
    candidates = [
        "SimSun",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in installed), "")
    if chosen:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [chosen, "Times New Roman", "DejaVu Serif"]
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.edgecolor": "#111827",
            "axes.linewidth": 1.0,
            "axes.labelsize": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.5,
            "grid.color": "#CFCFCF",
            "grid.linewidth": 0.9,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_scheduler(slot_id: str, model_name: str) -> Scheduler:
    config_path = SLOT_CONFIG_BY_MODEL[model_name].format(slot_id=slot_id)
    return Scheduler(
        net_config_path=config_path,
        pc_profiles_path="config/dnn_profiles_database_pc.json",
        jetson_profiles_path="config/dnn_profiles_database_jetson.json",
    )


def run_one_model(
    slot_id: str,
    model_name: str,
    repeats: int,
) -> list[dict]:
    model_label, batch_size, input_h, input_w = MODEL_TASKS[model_name]
    scheduler = build_scheduler(slot_id, model_name)
    isl_avg, gsl_avg = scheduler._extract_bw_metrics(scheduler.net_config.get("links", {}))
    pipeline = scheduler.net_config.get("simulation_paths", {}).get("pipeline", [])
    pipeline_path = "->".join(pipeline)
    pipeline_node_count = max(0, len(pipeline) - 2)
    pipeline_hop_count = max(0, len(pipeline) - 1)

    rows: list[dict] = []

    det = scheduler.generate_task_and_schedule(
        task_id=f"{slot_id}_{model_name}_det",
        model_name=model_name,
        batch_size=batch_size,
        target_h=input_h,
        target_w=input_w,
        run_id=f"slot_highlight_{slot_id}",
        exp_type="stk_slot_highlight_rerun",
        mode="theory",
        algorithm_names=DETERMINISTIC_ALGS,
        persist_theory=False,
        return_full_plans=True,
    )
    gs_latency = float(det["GS-Only"]["latency"])

    for alg in DETERMINISTIC_ALGS:
        latency = float(det[alg]["latency"])
        rows.append(
            {
                "slot_id": slot_id,
                "model_name": model_name,
                "model_label": model_label,
                "algorithm": alg,
                "repeat": 0,
                "batch_size": batch_size,
                "input_h": input_h,
                "input_w": input_w,
                "isl_avg_bw_mbps": isl_avg,
                "gsl_avg_bw_mbps": gsl_avg,
                "pipeline_node_count": pipeline_node_count,
                "pipeline_hop_count": pipeline_hop_count,
                "pipeline_path": pipeline_path,
                "latency_ms": latency,
                "norm_latency_vs_gs": latency / gs_latency if gs_latency > 0 else np.nan,
                "plan": json.dumps(det[alg].get("plan"), ensure_ascii=False),
                "sampling": "deterministic_once",
            }
        )

    for repeat in range(repeats):
        seed = 20260509 + repeat * 97 + len(model_name) * 13
        random.seed(seed)
        np.random.seed(seed)
        stc = scheduler.generate_task_and_schedule(
            task_id=f"{slot_id}_{model_name}_stc_{repeat:03d}",
            model_name=model_name,
            batch_size=batch_size,
            target_h=input_h,
            target_w=input_w,
            run_id=f"slot_highlight_{slot_id}",
            exp_type="stk_slot_highlight_rerun",
            mode="theory",
            algorithm_names=STOCHASTIC_ALGS,
            persist_theory=False,
            return_full_plans=True,
        )
        for alg in STOCHASTIC_ALGS:
            latency = float(stc[alg]["latency"])
            rows.append(
                {
                    "slot_id": slot_id,
                    "model_name": model_name,
                    "model_label": model_label,
                    "algorithm": alg,
                    "repeat": repeat,
                    "batch_size": batch_size,
                    "input_h": input_h,
                    "input_w": input_w,
                    "isl_avg_bw_mbps": isl_avg,
                    "gsl_avg_bw_mbps": gsl_avg,
                    "pipeline_node_count": pipeline_node_count,
                    "pipeline_hop_count": pipeline_hop_count,
                    "pipeline_path": pipeline_path,
                    "latency_ms": latency,
                    "norm_latency_vs_gs": latency / gs_latency if gs_latency > 0 else np.nan,
                    "plan": json.dumps(stc[alg].get("plan"), ensure_ascii=False),
                    "sampling": "stochastic_repeat",
                }
            )
    return rows


def summarize(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in ["latency_ms", "norm_latency_vs_gs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    summary = (
        df.groupby(["model_name", "model_label", "algorithm"], dropna=False)
        .agg(
            mean_norm_latency_vs_gs=("norm_latency_vs_gs", "mean"),
            std_norm_latency_vs_gs=("norm_latency_vs_gs", "std"),
            mean_latency_ms=("latency_ms", "mean"),
            std_latency_ms=("latency_ms", "std"),
            samples=("repeat", "count"),
        )
        .reset_index()
    )
    summary["algorithm"] = pd.Categorical(summary["algorithm"], ALGORITHMS, ordered=True)
    summary["model_label"] = pd.Categorical(
        summary["model_label"],
        [MODEL_TASKS[m][0] for m in model_order],
        ordered=True,
    )
    return summary.sort_values(["model_label", "algorithm"]).reset_index(drop=True)


def add_legend_below(fig: plt.Figure, ax: plt.Axes, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )


def annotate_bars(ax: plt.Axes, bars) -> None:
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * 0.012
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height) or height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + gap,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )


def plot(summary: pd.DataFrame, out_dir: Path, file_stem: str) -> None:
    setup_style()
    summary = summary.copy()
    for col in ["mean_norm_latency_vs_gs", "std_norm_latency_vs_gs"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    pivot = (
        summary.pivot_table(
            index="model_label",
            columns="algorithm",
            values="mean_norm_latency_vs_gs",
            aggfunc="mean",
            observed=False,
        )
        .reindex(columns=ALGORITHMS)
    )

    x_labels = pivot.index.astype(str).tolist()
    x = np.arange(len(x_labels))
    width = min(0.15, 0.78 / max(1, len(ALGORITHMS)))
    fig, ax = plt.subplots(figsize=(14.8, 7.2))
    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    max_value = float(np.max(finite_values)) if finite_values.size else 1.0
    ax.set_ylim(0, max(1.15, max_value * 1.30))

    for idx, alg in enumerate(ALGORITHMS):
        values = pivot[alg].astype(float).to_numpy()
        offset = (idx - (len(ALGORITHMS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            np.nan_to_num(values, nan=0.0),
            width=width,
            label=ALG_LABEL.get(alg, alg),
            color=ALG_COLOR.get(alg, "#64748B"),
            alpha=0.94,
        )
        annotate_bars(ax, bars)
        for xi, value in zip(x + offset, values):
            if not np.isfinite(value):
                ax.text(
                    xi,
                    ax.get_ylim()[1] * 0.03,
                    "不可行",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    rotation=90,
                    color="#B91C1C",
                )

    ax.set_title("PMP 模式下不同算法的归一化时延对比", pad=56)
    ax.set_ylabel("归一化时延（相对 GS-Only）")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_legend_below(fig, ax, 6)
    fig.subplots_adjust(top=0.72, bottom=0.13, left=0.08, right=0.98)
    fig.savefig(out_dir / f"{file_stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{file_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_notes(
    slot_id: str,
    repeats: int,
    model_order: list[str],
    summary: pd.DataFrame,
    out_dir: Path,
    file_stem: str,
) -> None:
    config_path = Path(SLOT_CONFIG_BY_MODEL[model_order[0]].format(slot_id=slot_id))
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    slot_meta = cfg.get("simulation_paths", {}).get("stk_dynamic_slot", {})
    path_meta = cfg.get("simulation_paths", {}).get("stk_path", {})
    notes = [
        f"# {slot_id} 单时间片 PMP 图",
        "",
        "## 时间片参数",
        "",
        f"- 时间片：`{slot_id}`",
        f"- 路径：`{' -> '.join(cfg.get('simulation_paths', {}).get('pipeline', []))}`",
        f"- 总跳数：`{max(0, len(cfg.get('simulation_paths', {}).get('pipeline', [])) - 1)}`",
        f"- 总传播时延：`{float(path_meta.get('total_propagation_delay_ms', 0.0)):.3f} ms`",
        f"- 路径公共持续时间：`{float(path_meta.get('common_duration_s', 0.0)):.3f} s`",
        f"- 带宽种子：`{slot_meta.get('bandwidth_seed', '')}`",
        "",
        "## 重跑口径",
        "",
        "- 确定性算法 `LADP / 贪心 / 均匀 / GS-Only` 运行 1 次。",
        f"- 随机算法 `GA / Random` 各重复 `{repeats}` 次并取均值。",
        "- 图标题沿用论文主图口径，不在图内写时间片编号。",
        "",
        "## 最终均值",
        "",
        "| 模型 | 算法 | 归一化时延均值 | 随机标准差 |",
        "|---|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        mean_value = "不可行" if pd.isna(row["mean_norm_latency_vs_gs"]) else f"{float(row['mean_norm_latency_vs_gs']):.3f}"
        std_value = "—" if pd.isna(row["std_norm_latency_vs_gs"]) else f"{float(row['std_norm_latency_vs_gs']):.3f}"
        notes.append(
            f"| {row['model_label']} | {ALG_LABEL.get(str(row['algorithm']), str(row['algorithm']))} | "
            f"{mean_value} | {std_value} |"
        )
    (out_dir / f"{file_stem}_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun one STK slot and redraw a paper-ready PMP figure")
    parser.add_argument("--slot-id", required=True, help="Slot id such as slot_033_064500_065000")
    parser.add_argument(
        "--models",
        default="yolov5,resnet101,vgg19,vit_huge",
        help="Comma-separated model ids",
    )
    parser.add_argument("--repeats", type=int, default=100, help="Repeat count for GA and Random")
    parser.add_argument(
        "--out-dir",
        default="result/paper_figures_controlled/stk_slot_highlight_rerun",
        help="Output directory",
    )
    args = parser.parse_args()

    model_order = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for model_name in model_order:
        print(f"[STK-SLOT] rerun {args.slot_id} | {model_name}")
        all_rows.extend(run_one_model(args.slot_id, model_name, args.repeats))

    df = pd.DataFrame(all_rows)
    summary = summarize(df, model_order)

    file_stem = "exp01_ladp_pmp_algorithm_effectiveness"
    df.to_csv(out_dir / f"{file_stem}_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / f"{file_stem}_summary.csv", index=False, encoding="utf-8-sig")
    plot(summary, out_dir, file_stem)
    write_notes(args.slot_id, args.repeats, model_order, summary, out_dir, file_stem)
    print(out_dir)


if __name__ == "__main__":
    main()
