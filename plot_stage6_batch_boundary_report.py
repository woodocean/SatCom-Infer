"""Summarize Stage 6 mode-selection boundary changes across batch sizes."""

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
BATCH_ORDER = [32, 64]
MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS-Feature", "Oracle-Min-Latency"]
MODE_COLORS = {
    "PMP": "#2563EB",
    "CDP": "#F97316",
    "GS-Only": "#DC2626",
    "Sat-Only": "#16A34A",
    "FWMS-Feature": "#7C3AED",
    "Oracle-Min-Latency": "#475569",
}
BATCH_COLORS = {
    32: "#38BDF8",
    64: "#0F766E",
}


def _metadata(run_dir: Path) -> dict:
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_runs(input_root: Path, batches: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    selector_frames = []
    for batch in batches:
        for run_dir in sorted(input_root.glob(f"mode_selection_*_stage6_feature_oracle_b{batch}")):
            summary_path = run_dir / "data" / "summary_by_mode.csv"
            selector_path = run_dir / "data" / "fwms_selection_distribution.csv"
            metadata_path = run_dir / "metadata.json"
            if not summary_path.exists() or not selector_path.exists() or not metadata_path.exists():
                continue

            metadata = _metadata(run_dir)
            task = metadata.get("effective_task", {})
            model_name = str(task.get("model_name") or metadata.get("model_name_override") or run_dir.name)
            batch_size = int(task.get("batch_size") or metadata.get("batch_size_override") or batch)

            summary = pd.read_csv(summary_path)
            summary["model_name"] = model_name
            summary["model_label"] = MODEL_LABELS.get(model_name, model_name)
            summary["batch_size"] = batch_size
            summary["run_id"] = run_dir.name
            summary_frames.append(summary)

            selector = pd.read_csv(selector_path)
            selector["model_name"] = model_name
            selector["model_label"] = MODEL_LABELS.get(model_name, model_name)
            selector["batch_size"] = batch_size
            selector["run_id"] = run_dir.name
            selector_frames.append(selector)

    if not summary_frames:
        raise FileNotFoundError(f"No Stage 6 batch-boundary runs found under {input_root}")

    summary_all = pd.concat(summary_frames, ignore_index=True)
    selector_all = pd.concat(selector_frames, ignore_index=True)
    summary_all["model_name"] = pd.Categorical(summary_all["model_name"], MODEL_ORDER, ordered=True)
    summary_all["mode_family"] = pd.Categorical(summary_all["mode_family"], MODE_ORDER, ordered=True)
    summary_all["batch_size"] = pd.Categorical(summary_all["batch_size"], batches, ordered=True)
    summary_all = summary_all.sort_values(["batch_size", "model_name", "mode_family"]).reset_index(drop=True)
    return summary_all, selector_all


def _format_float(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_lines(df: pd.DataFrame) -> list[str]:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return lines


def _write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.write_text("\n".join(_markdown_lines(df)) + "\n", encoding="utf-8")


def _save(fig: plt.Figure, output_path: Path, rect: tuple[float, float, float, float] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rect:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_cdp_completion(summary: pd.DataFrame, output_path: Path) -> None:
    cdp = summary[summary["mode_family"].astype(str) == "CDP"]
    pivot = cdp.pivot(index="model_label", columns="batch_size", values="feasible_rate")
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODEL_ORDER])

    x = np.arange(len(pivot.index))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for idx, batch in enumerate(BATCH_ORDER):
        values = pivot[batch].astype(float).to_numpy()
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            color=BATCH_COLORS[batch],
            edgecolor="white",
            linewidth=0.8,
            label=f"batch={batch}",
        )
        for x_pos, value in zip(x + (idx - 0.5) * width, values):
            ax.text(x_pos, value + 0.025, f"{value * 100:.0f}%", ha="center", fontsize=9)

    ax.set_title("不同 batch 下 CDP 可行率变化", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 1.12)
    _style_axes(ax, "CDP 可行率")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    _save(fig, output_path)


def _plot_oracle_selection(selector: pd.DataFrame, output_path: Path) -> None:
    oracle = selector[selector["selector_family"] == "Oracle-Min-Latency"]
    fig, axes = plt.subplots(1, len(BATCH_ORDER), figsize=(12.2, 4.8), sharey=True)
    model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    for ax, batch in zip(axes, BATCH_ORDER):
        batch_df = oracle[oracle["batch_size"] == batch]
        pivot = batch_df.pivot_table(
            index="model_label",
            columns="selected_mode",
            values="ratio",
            aggfunc="sum",
            fill_value=0.0,
            observed=False,
        ).reindex(model_labels).fillna(0.0)
        columns = [mode for mode in MODE_ORDER if mode in pivot.columns]

        x = np.arange(len(model_labels))
        bottom = np.zeros(len(model_labels))
        for mode in columns:
            values = pivot[mode].astype(float).to_numpy()
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=MODE_COLORS.get(str(mode), "#334155"),
                edgecolor="white",
                linewidth=0.8,
                label=str(mode),
            )
            bottom += values

        ax.set_title(f"batch={batch}", fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20)
        ax.set_ylim(0, 1.05)
        _style_axes(ax, "选择比例" if batch == BATCH_ORDER[0] else None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("不同 batch 下 Oracle-Min-Latency 的模式选择边界", fontsize=14, weight="bold", y=0.98)
    _save(fig, output_path, rect=(0.0, 0.08, 1.0, 0.92))


def _plot_latency_ratio(summary: pd.DataFrame, output_path: Path) -> None:
    selector = summary[summary["mode_family"].astype(str).isin(["FWMS-Feature", "Oracle-Min-Latency"])]
    pivot = selector.pivot_table(
        index=["model_label", "batch_size"],
        columns="mode_family",
        values="avg_latency_ms",
        aggfunc="first",
        observed=False,
    ).reset_index()
    pivot["ratio"] = pivot["FWMS-Feature"] / pivot["Oracle-Min-Latency"]

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for idx, batch in enumerate(BATCH_ORDER):
        batch_df = pivot[pivot["batch_size"] == batch].set_index("model_label").reindex([MODEL_LABELS[m] for m in MODEL_ORDER])
        values = batch_df["ratio"].astype(float).to_numpy()
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            color=BATCH_COLORS[batch],
            edgecolor="white",
            linewidth=0.8,
            label=f"batch={batch}",
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{value:.2f}x", ha="center", fontsize=8)

    ax.axhline(1.0, color="#0F172A", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_title("不同 batch 下 FWMS-Feature 相对 Oracle 的时延比", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    _style_axes(ax, "FWMS / Oracle 时延比")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    _save(fig, output_path)


def _build_boundary_notes(summary: pd.DataFrame, selector: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for batch in BATCH_ORDER:
            model_label = MODEL_LABELS[model_name]
            model_summary = summary[
                (summary["model_name"].astype(str) == model_name) & (summary["batch_size"].astype(int) == batch)
            ].set_index("mode_family")
            cdp_rate = float(model_summary.loc["CDP", "feasible_rate"]) if "CDP" in model_summary.index else 0.0
            pmp_latency = model_summary.loc["PMP", "avg_latency_ms"] if "PMP" in model_summary.index else np.nan
            cdp_latency = model_summary.loc["CDP", "avg_latency_ms"] if "CDP" in model_summary.index else np.nan
            oracle_latency = (
                model_summary.loc["Oracle-Min-Latency", "avg_latency_ms"]
                if "Oracle-Min-Latency" in model_summary.index
                else np.nan
            )
            oracle_modes = selector[
                (selector["model_name"] == model_name)
                & (selector["batch_size"] == batch)
                & (selector["selector_family"] == "Oracle-Min-Latency")
            ]
            selected = ", ".join(
                f"{row.selected_mode}:{float(row.ratio) * 100:.0f}%" for row in oracle_modes.itertuples(index=False)
            )
            rows.append(
                {
                    "模型": model_label,
                    "batch": batch,
                    "CDP完成率": f"{cdp_rate * 100:.0f}%",
                    "PMP平均时延(ms)": _format_float(pmp_latency),
                    "CDP平均时延(ms)": _format_float(cdp_latency),
                    "Oracle平均时延(ms)": _format_float(oracle_latency),
                    "Oracle选择分布": selected,
                    "边界观察": _note_for(model_name, batch, cdp_rate),
                }
            )
    return pd.DataFrame(rows)


def _note_for(model_name: str, batch: int, cdp_rate: float) -> str:
    if cdp_rate <= 0:
        return "完整模型无法部署到可见 worker，CDP 不可行。"
    if model_name == "vit_huge" and batch == 32:
        return "batch=32 时 CDP 可行但不一定最低时延，说明可行性不等于最优性。"
    if model_name in {"yolov5", "resnet101", "swin_base"}:
        return "CDP 可行且通常成为时延上界选择，体现数据并行优势。"
    return "需要结合内存、通信和计算特征判断模式边界。"


def _write_notes(output_dir: Path, boundary_md: pd.DataFrame) -> None:
    lines = [
        "# Stage6 Batch 边界补充实验说明",
        "",
        "本实验补跑五个模型在 `batch=32` 下的 Stage6 模式选择，并与已有 `batch=64` 结果对比。目的不是替代主实验，而是补充说明任务规模变化时 CDP/PMP/GS-Only 的适用边界。",
        "",
        "## 新增实验",
        "",
        "- `mode_selection_yolo_stage6_feature_oracle_b32`",
        "- `mode_selection_resnet101_stage6_feature_oracle_b32`",
        "- `mode_selection_vgg19_stage6_feature_oracle_b32`",
        "- `mode_selection_swin_base_stage6_feature_oracle_b32`",
        "- `mode_selection_vit_huge_stage6_feature_oracle_b32`",
        "",
        "## 主要结论",
        "",
        "- YOLOv5、ResNet101、Swin-Base 在 batch=32/64 下 CDP 都可行，说明这些模型在当前资源条件下适合数据并行。",
        "- VGG19 在 batch=32/64 下 CDP 都不可行，说明其瓶颈主要来自完整模型部署约束，而不是 batch 规模。",
        "- ViT-Huge 在 batch=32 下 CDP 可行，但 batch=64 下 CDP 不可行，说明任务规模会改变模式可行边界。",
        "- batch=32 下 ViT-Huge 虽然 CDP 可行，但 Oracle 仍选择 GS-Only，说明“可行”不等于“最低时延”，这能支撑 FWMS 需要综合特征而不是只做内存筛选。",
        "",
        "## 输出文件",
        "",
        "- `batch_boundary_mode_summary.csv/md`：batch=32/64 的跨模型模式摘要。",
        "- `batch_boundary_selector_distribution.csv/md`：选择器分布。",
        "- `batch_boundary_notes.md`：每个模型和 batch 的边界观察表。",
        "- `batch_boundary_cdp_completion.png/pdf`：CDP 可行率随 batch 的变化。",
        "- `batch_boundary_oracle_selection.png/pdf`：Oracle 选择模式随 batch 的变化。",
        "- `batch_boundary_fwms_oracle_latency_ratio.png/pdf`：FWMS 相对 Oracle 的时延比变化。",
        "",
        "## 边界观察表",
        "",
    ]
    lines.extend(_markdown_lines(boundary_md))
    lines.append("")
    (output_dir / "batch_boundary_report_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build batch-boundary report for Stage 6 mode selection.")
    parser.add_argument("--input-root", default="result/mode_selection")
    parser.add_argument("--output-dir", default="result/mode_selection/batch_boundary_stage6")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, selector = _load_runs(Path(args.input_root), BATCH_ORDER)

    summary.to_csv(output_dir / "batch_boundary_mode_summary.csv", index=False, encoding="utf-8-sig")
    selector.to_csv(output_dir / "batch_boundary_selector_distribution.csv", index=False, encoding="utf-8-sig")

    summary_md = summary.copy()
    summary_md["模型"] = summary_md["model_label"]
    summary_md["batch"] = summary_md["batch_size"].astype(str)
    summary_md["模式"] = summary_md["mode_family"].astype(str)
    summary_md["完成率"] = summary_md["feasible_rate"].map(lambda x: f"{float(x) * 100:.0f}%")
    summary_md["平均时延(ms)"] = summary_md["avg_latency_ms"].map(_format_float)
    summary_md["平均卫星能耗(J)"] = summary_md["avg_satellite_energy_j"].map(_format_float)
    summary_md = summary_md[["模型", "batch", "模式", "完成率", "平均时延(ms)", "平均卫星能耗(J)"]]
    _write_markdown_table(summary_md, output_dir / "batch_boundary_mode_summary.md")

    selector_md = selector.copy()
    selector_md["模型"] = selector_md["model_label"]
    selector_md["batch"] = selector_md["batch_size"].astype(str)
    selector_md["选择器"] = selector_md["selector_family"]
    selector_md["被选模式"] = selector_md["selected_mode"]
    selector_md["比例"] = selector_md["ratio"].map(lambda x: f"{float(x) * 100:.1f}%")
    selector_md = selector_md[["模型", "batch", "选择器", "被选模式", "count", "比例"]]
    _write_markdown_table(selector_md, output_dir / "batch_boundary_selector_distribution.md")

    boundary_md = _build_boundary_notes(summary, selector)
    _write_markdown_table(boundary_md, output_dir / "batch_boundary_notes.md")

    _plot_cdp_completion(summary, output_dir / "batch_boundary_cdp_completion.png")
    _plot_oracle_selection(selector, output_dir / "batch_boundary_oracle_selection.png")
    _plot_latency_ratio(summary, output_dir / "batch_boundary_fwms_oracle_latency_ratio.png")
    _write_notes(output_dir, boundary_md)
    print(f"[BATCH] Wrote batch-boundary artifacts to {output_dir}")


if __name__ == "__main__":
    main()
