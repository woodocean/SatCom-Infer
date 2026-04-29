"""Plot cross-model summaries for STK mode-selection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MODEL_ORDER = ["yolov5", "resnet101", "vgg19", "swin_base", "vit_huge"]
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "resnet101": "ResNet101",
    "vgg19": "VGG19",
    "swin_base": "Swin-Base",
    "vit_huge": "ViT-Huge",
}
MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS"]
MODE_COLORS = {
    "PMP": "#1f77b4",
    "CDP": "#ff7f0e",
    "GS-Only": "#d62728",
    "Sat-Only": "#2ca02c",
    "FWMS": "#9467bd",
}
MODE_LABELS = {
    "PMP": "PMP（流水线）",
    "CDP": "CDP（数据并行）",
    "GS-Only": "GS-Only（地面站）",
    "Sat-Only": "Sat-Only（单星）",
    "FWMS": "FWMS（模式选择）",
}


def _model_name_from_metadata(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    effective_task = metadata.get("effective_task", {})
    return str(effective_task.get("model_name") or metadata.get("model_name_override") or run_dir.name)


def _load_runs(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    selection_frames = []
    for run_dir in sorted(input_root.glob("mode_selection_*_stage5_fwms_b64")):
        summary_path = run_dir / "data" / "summary_by_mode.csv"
        selection_path = run_dir / "data" / "fwms_selection_distribution.csv"
        if not summary_path.exists() or not selection_path.exists():
            continue
        model_name = _model_name_from_metadata(run_dir)
        summary = pd.read_csv(summary_path)
        summary["model_name"] = model_name
        summary["run_id"] = run_dir.name
        summary_frames.append(summary)

        selection = pd.read_csv(selection_path)
        selection["model_name"] = model_name
        selection["run_id"] = run_dir.name
        selection_frames.append(selection)

    if not summary_frames:
        raise FileNotFoundError(f"No mode-selection stage5 summaries found under {input_root}")
    return pd.concat(summary_frames, ignore_index=True), pd.concat(selection_frames, ignore_index=True)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    df["mode_family"] = pd.Categorical(df["mode_family"], categories=MODE_ORDER, ordered=True)
    df["model_label"] = df["model_name"].astype(str).map(MODEL_LABELS)
    return df.sort_values(["model_name", "mode_family"]).reset_index(drop=True)


def _style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("模型", fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_grouped_bars(df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    pivot = df.pivot(index="model_label", columns="mode_family", values=metric)
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODEL_ORDER])
    pivot = pivot[[mode for mode in MODE_ORDER if mode in pivot.columns]]

    x = np.arange(len(pivot.index))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    offsets = (np.arange(len(pivot.columns)) - (len(pivot.columns) - 1) / 2) * width
    for offset, mode in zip(offsets, pivot.columns):
        values = pivot[mode].astype(float).values
        valid = np.isfinite(values)
        ax.bar(
            x + offset,
            np.where(valid, values, 0.0),
            width=width,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, "#334155"),
            edgecolor="white",
            linewidth=0.7,
        )
        if metric != "feasible_rate" and not valid.all():
            y_top = np.nanmax(values) if np.isfinite(values).any() else 1.0
            y_text = max(y_top * 0.025, 1.0)
            for x_pos, is_valid in zip(x + offset, valid):
                if not is_valid:
                    ax.text(
                        x_pos,
                        y_text,
                        "不可行",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                        color="#475569",
                    )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    _style_axes(ax, ylabel)
    ax.legend(ncol=min(5, len(pivot.columns)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_fwms_distribution(selection: pd.DataFrame, output_path: Path) -> None:
    selection = selection.copy()
    selection["model_name"] = pd.Categorical(selection["model_name"], categories=MODEL_ORDER, ordered=True)
    selection["model_label"] = selection["model_name"].astype(str).map(MODEL_LABELS)
    pivot = (
        selection.pivot_table(
            index="model_label",
            columns="selected_mode",
            values="ratio",
            aggfunc="sum",
            fill_value=0.0,
            observed=False,
        )
        .reindex([MODEL_LABELS[m] for m in MODEL_ORDER])
        .fillna(0.0)
    )
    columns = [mode for mode in MODE_ORDER if mode in pivot.columns] + [
        mode for mode in pivot.columns if mode not in MODE_ORDER
    ]
    pivot = pivot[columns]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bottom = np.zeros(len(pivot.index))
    x = np.arange(len(pivot.index))
    for mode in pivot.columns:
        values = pivot[mode].astype(float).values
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, "#334155"),
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += values

    ax.set_title("FWMS 模式选择分布", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, "选择比例")
    ax.legend(ncol=min(5, len(pivot.columns)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-model mode-selection summaries.")
    parser.add_argument("--input-root", default="result/mode_selection", help="Directory containing mode-selection runs.")
    parser.add_argument(
        "--output-dir",
        default="result/mode_selection/cross_model_stage5",
        help="Output directory for summary tables and figures.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    summary, selection = _load_runs(input_root)
    summary = _ordered(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "mode_selection_cross_model_summary.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(output_dir / "fwms_selection_cross_model.csv", index=False, encoding="utf-8-sig")

    _plot_grouped_bars(
        summary,
        metric="avg_latency_ms",
        ylabel="平均端到端时延（ms）",
        title="跨模型平均时延对比",
        output_path=output_dir / "mode_selection_avg_latency.png",
    )
    _plot_grouped_bars(
        summary,
        metric="avg_satellite_energy_j",
        ylabel="平均卫星能耗（J）",
        title="跨模型平均卫星能耗对比",
        output_path=output_dir / "mode_selection_avg_energy.png",
    )
    _plot_grouped_bars(
        summary,
        metric="feasible_rate",
        ylabel="可行率",
        title="跨模型各模式可行率对比",
        output_path=output_dir / "mode_selection_feasible_rate.png",
    )
    _plot_fwms_distribution(selection, output_dir / "fwms_selection_distribution.png")

    print(f"[PLOT] Wrote summary and figures to {output_dir}")


if __name__ == "__main__":
    main()
