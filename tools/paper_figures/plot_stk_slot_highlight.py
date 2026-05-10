# -*- coding: utf-8 -*-
"""Plot one highlighted STK dynamic slot across selected models.

This script only redraws from existing STK results and does not rerun any
experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ALG_ORDER = ["LA-DP", "Greedy", "GA", "Random", "Uniform", "GS-Only"]
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

MODEL_LABEL = {
    "yolov5": "YOLOv5",
    "vgg19": "VGG19",
    "swin_base": "Swin-Base",
    "vit_huge": "ViT-Huge",
    "resnet101": "ResNet101",
}

MODEL_RUN_DIR = {
    "yolov5": "stk_dynamic_yolo_001",
    "resnet101": "stk_dynamic_resnet101_001",
    "vgg19": "stk_dynamic_vgg19_001",
    "swin_base": "stk_dynamic_swin_base_001",
    "vit_huge": "stk_dynamic_vit_huge_001",
}


def setup_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "SimSun",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11


def load_slot_rows(slot_id: str, models: list[str]) -> pd.DataFrame:
    frames = []
    for model_name in models:
        run_dir = MODEL_RUN_DIR.get(model_name, f"stk_dynamic_{model_name}_001")
        path = Path(f"result/stk_dynamic/{run_dir}/results_long_stk_dynamic.csv")
        df = pd.read_csv(path)
        one = df[df["sweep_value"] == slot_id].copy()
        if one.empty:
            continue
        one["model_name"] = model_name
        one["model_label"] = MODEL_LABEL.get(model_name, model_name)
        frames.append(one)
    if not frames:
        raise ValueError(f"No rows found for slot {slot_id}")
    return pd.concat(frames, ignore_index=True)


def summarize_slot(slot_df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    keep = slot_df[
        [
            "model_name",
            "model_label",
            "algorithm",
            "batch_size",
            "input_h",
            "input_w",
            "isl_avg_bw_mbps",
            "gsl_avg_bw_mbps",
            "pipeline_node_count",
            "pipeline_hop_count",
            "pipeline_path",
            "latency_ms",
            "norm_latency_vs_gs",
            "satellite_energy_j",
        ]
    ].copy()
    keep["algorithm"] = pd.Categorical(keep["algorithm"], ALG_ORDER, ordered=True)
    keep["model_label"] = pd.Categorical(
        keep["model_label"],
        [MODEL_LABEL.get(m, m) for m in model_order],
        ordered=True,
    )
    return keep.sort_values(["model_label", "algorithm"])


def plot(summary: pd.DataFrame, slot_id: str, out_dir: Path) -> None:
    setup_font()
    models = list(summary["model_label"].dropna().unique())
    pivot = (
        summary.pivot_table(
            index="model_label",
            columns="algorithm",
            values="norm_latency_vs_gs",
            observed=False,
        )
        .reindex(index=models, columns=ALG_ORDER)
    )

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    x = np.arange(len(pivot.index))
    width = min(0.12, 0.78 / len(ALG_ORDER))

    for idx, alg in enumerate(ALG_ORDER):
        values = pivot[alg].astype(float).to_numpy()
        offset = (idx - (len(ALG_ORDER) - 1) / 2) * width
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
                    bar.get_height() + 0.04,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#334155",
                )

    ax.set_title(f"时间片 {slot_id} 下 PMP 算法归一化时延对比", pad=18)
    ax.set_ylabel("归一化时延（相对 GS-Only）")
    ax.set_xlabel("模型")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.grid(axis="y", linestyle="--", linewidth=0.9, color="#D5D5D5", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=6, frameon=False)
    ymax = max(1.2, float(np.nanmax(pivot.to_numpy())) * 1.16)
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.09, right=0.98)
    fig.savefig(out_dir / f"{slot_id}_pmp_latency_norm_no_resnet.png", dpi=300)
    fig.savefig(out_dir / f"{slot_id}_pmp_latency_norm_no_resnet.pdf")
    plt.close(fig)


def write_notes(summary: pd.DataFrame, slot_id: str, out_dir: Path) -> None:
    first = summary.iloc[0]
    notes = [
        f"# {slot_id} 单时间片图",
        "",
        "## Shared slot parameters",
        "",
        f"- ISL 平均带宽：`{float(first['isl_avg_bw_mbps']):.4f} Mbps`",
        f"- GSL 平均带宽：`{float(first['gsl_avg_bw_mbps']):.4f} Mbps`",
        f"- 流水线节点数：`{int(first['pipeline_node_count'])}`",
        f"- 跳数：`{int(first['pipeline_hop_count'])}`",
        f"- 路径：`{first['pipeline_path']}`",
        "",
        "## Model settings",
        "",
        "| 模型 | batch | 输入尺寸 | LADP归一化时延 |",
        "|---|---:|---:|---:|",
    ]
    for model_label, group in summary.groupby("model_label", observed=False):
        if pd.isna(model_label):
            continue
        ladp = group[group["algorithm"] == "LA-DP"].iloc[0]
        notes.append(
            f"| {model_label} | {int(ladp['batch_size'])} | "
            f"{int(ladp['input_h'])}x{int(ladp['input_w'])} | {float(ladp['norm_latency_vs_gs']):.3f} |"
        )
    (out_dir / f"{slot_id}_pmp_latency_norm_no_resnet_notes.md").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one highlighted STK slot")
    parser.add_argument("--slot-id", required=True, help="Slot id such as slot_033_064500_065000")
    parser.add_argument(
        "--models",
        default="yolov5,vgg19,swin_base,vit_huge",
        help="Comma-separated model ids",
    )
    parser.add_argument(
        "--out-dir",
        default="result/paper_figures_controlled/stk_slot_highlight",
        help="Output directory",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slot_df = load_slot_rows(args.slot_id, models)
    summary = summarize_slot(slot_df, models)
    summary.to_csv(out_dir / f"{args.slot_id}_pmp_latency_norm_no_resnet.csv", index=False, encoding="utf-8-sig")
    plot(summary, args.slot_id, out_dir)
    write_notes(summary, args.slot_id, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
