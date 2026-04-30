"""Build final Stage 6 mode-selection report tables and figures.

This script is intentionally a post-processing entrypoint. It does not rerun STK
or any scheduler logic; it only reads completed Stage 6 result directories and
generates paper/report-friendly summary artifacts.
"""

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

MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS-Feature", "Oracle-Min-Latency"]
MODE_LABELS = {
    "PMP": "PMP",
    "CDP": "CDP",
    "GS-Only": "GS-Only",
    "Sat-Only": "Sat-Only",
    "FWMS-Feature": "FWMS-Feature",
    "Oracle-Min-Latency": "Oracle-Min-Latency",
}
MODE_COLORS = {
    "PMP": "#2563EB",
    "CDP": "#F97316",
    "GS-Only": "#DC2626",
    "Sat-Only": "#16A34A",
    "FWMS-Feature": "#7C3AED",
    "Oracle-Min-Latency": "#475569",
}
SELECTOR_ORDER = ["FWMS-Feature", "Oracle-Min-Latency"]


def _metadata_model_name(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    effective_task = metadata.get("effective_task", {})
    return str(effective_task.get("model_name") or metadata.get("model_name_override") or run_dir.name)


def _load_stage6_runs(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    slot_frames = []
    distribution_frames = []

    for run_dir in sorted(input_root.glob("mode_selection_*_stage6_feature_oracle_b64")):
        summary_path = run_dir / "data" / "summary_by_mode.csv"
        slots_path = run_dir / "data" / "slot_mode_results.csv"
        distribution_path = run_dir / "data" / "fwms_selection_distribution.csv"
        metadata_path = run_dir / "metadata.json"
        if not summary_path.exists() or not slots_path.exists() or not distribution_path.exists() or not metadata_path.exists():
            continue

        model_name = _metadata_model_name(run_dir)

        summary = pd.read_csv(summary_path)
        summary["model_name"] = model_name
        summary["model_label"] = MODEL_LABELS.get(model_name, model_name)
        summary["run_id"] = run_dir.name
        summary_frames.append(summary)

        slots = pd.read_csv(slots_path)
        slots["model_name"] = model_name
        slots["model_label"] = MODEL_LABELS.get(model_name, model_name)
        slots["run_id"] = run_dir.name
        slot_frames.append(slots)

        distribution = pd.read_csv(distribution_path)
        distribution["model_name"] = model_name
        distribution["model_label"] = MODEL_LABELS.get(model_name, model_name)
        distribution["run_id"] = run_dir.name
        distribution_frames.append(distribution)

    if not summary_frames:
        raise FileNotFoundError(f"No Stage 6 mode-selection runs found under {input_root}")

    summary_all = pd.concat(summary_frames, ignore_index=True)
    slots_all = pd.concat(slot_frames, ignore_index=True)
    distribution_all = pd.concat(distribution_frames, ignore_index=True)

    summary_all["model_name"] = pd.Categorical(summary_all["model_name"], MODEL_ORDER, ordered=True)
    summary_all["mode_family"] = pd.Categorical(summary_all["mode_family"], MODE_ORDER, ordered=True)
    summary_all = summary_all.sort_values(["model_name", "mode_family"]).reset_index(drop=True)

    distribution_all["model_name"] = pd.Categorical(distribution_all["model_name"], MODEL_ORDER, ordered=True)
    distribution_all["selector_family"] = pd.Categorical(distribution_all["selector_family"], SELECTOR_ORDER, ordered=True)
    distribution_all["selected_mode"] = pd.Categorical(distribution_all["selected_mode"], MODE_ORDER, ordered=True)
    distribution_all = distribution_all.sort_values(["model_name", "selector_family", "selected_mode"]).reset_index(drop=True)

    return summary_all, slots_all, distribution_all


def _format_float(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.write_text("\n".join(_markdown_table_lines(df)) + "\n", encoding="utf-8")


def _markdown_table_lines(df: pd.DataFrame) -> list[str]:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return lines


def _style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, output_path: Path, rect: tuple[float, float, float, float] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_metric_bars(summary: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    pivot = summary.pivot(index="model_label", columns="mode_family", values=metric)
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODEL_ORDER])
    pivot = pivot[[mode for mode in MODE_ORDER if mode in pivot.columns]]

    x = np.arange(len(pivot.index))
    width = 0.125
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    offsets = (np.arange(len(pivot.columns)) - (len(pivot.columns) - 1) / 2) * width

    for offset, mode in zip(offsets, pivot.columns):
        values = pivot[mode].astype(float).to_numpy()
        valid = np.isfinite(values)
        ax.bar(
            x + offset,
            np.where(valid, values, 0.0),
            width=width,
            color=MODE_COLORS.get(str(mode), "#334155"),
            edgecolor="white",
            linewidth=0.7,
            label=MODE_LABELS.get(str(mode), str(mode)),
        )
        if metric != "feasible_rate":
            y_text = max(np.nanmax(np.where(valid, values, np.nan)) * 0.025, 0.02) if valid.any() else 0.02
            for x_pos, is_valid in zip(x + offset, valid):
                if not is_valid:
                    ax.text(x_pos, y_text, "不可行", ha="center", va="bottom", rotation=90, fontsize=8, color="#64748B")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    _style_axes(ax, ylabel)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _save_figure(fig, output_path)


def _plot_completion_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    pivot = summary.pivot(index="mode_family", columns="model_label", values="feasible_rate")
    pivot = pivot.reindex(MODE_ORDER)
    pivot = pivot[[MODEL_LABELS[m] for m in MODEL_ORDER]]

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    values = pivot.astype(float).to_numpy()
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("各模式在不同模型下的可行率/完成率", fontsize=14, weight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([MODE_LABELS.get(str(mode), str(mode)) for mode in pivot.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j] * 100:.0f}%", ha="center", va="center", color="#0F172A", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="完成率")
    _save_figure(fig, output_path)


def _plot_selector_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)
    model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

    for ax, selector in zip(axes, SELECTOR_ORDER):
        selector_df = distribution[distribution["selector_family"].astype(str) == selector]
        pivot = selector_df.pivot_table(
            index="model_label",
            columns="selected_mode",
            values="ratio",
            aggfunc="sum",
            fill_value=0.0,
            observed=False,
        ).reindex(model_labels).fillna(0.0)
        columns = [mode for mode in MODE_ORDER if mode in pivot.columns]

        bottom = np.zeros(len(model_labels))
        x = np.arange(len(model_labels))
        for mode in columns:
            values = pivot[mode].astype(float).to_numpy()
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=MODE_COLORS.get(str(mode), "#334155"),
                edgecolor="white",
                linewidth=0.8,
                label=MODE_LABELS.get(str(mode), str(mode)),
            )
            bottom += values

        ax.set_title(selector, fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20)
        ax.set_ylim(0, 1.05)
        _style_axes(ax, "选择比例" if selector == SELECTOR_ORDER[0] else None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FWMS-Feature 与 Oracle-Min-Latency 的模式选择分布", fontsize=14, weight="bold", y=0.98)
    _save_figure(fig, output_path, rect=(0.0, 0.08, 1.0, 0.92))


def _build_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        model_label = MODEL_LABELS[model_name]
        model_df = summary[summary["model_name"].astype(str) == model_name].set_index("mode_family")
        fwms = model_df.loc["FWMS-Feature"] if "FWMS-Feature" in model_df.index else None
        oracle = model_df.loc["Oracle-Min-Latency"] if "Oracle-Min-Latency" in model_df.index else None
        pmp = model_df.loc["PMP"] if "PMP" in model_df.index else None
        cdp = model_df.loc["CDP"] if "CDP" in model_df.index else None

        fwms_latency = float(fwms["avg_latency_ms"]) if fwms is not None and pd.notna(fwms["avg_latency_ms"]) else np.nan
        oracle_latency = (
            float(oracle["avg_latency_ms"]) if oracle is not None and pd.notna(oracle["avg_latency_ms"]) else np.nan
        )
        rows.append(
            {
                "model_name": model_name,
                "模型": model_label,
                "FWMS平均时延(ms)": fwms_latency,
                "Oracle平均时延(ms)": oracle_latency,
                "FWMS/Oracle时延比": fwms_latency / oracle_latency if np.isfinite(fwms_latency) and oracle_latency > 0 else np.nan,
                "PMP完成率": float(pmp["feasible_rate"]) if pmp is not None else np.nan,
                "CDP完成率": float(cdp["feasible_rate"]) if cdp is not None else np.nan,
                "边界解释": _boundary_note(model_name, pmp, cdp),
            }
        )
    return pd.DataFrame(rows)


def _boundary_note(model_name: str, pmp: pd.Series | None, cdp: pd.Series | None) -> str:
    cdp_rate = float(cdp["feasible_rate"]) if cdp is not None and pd.notna(cdp["feasible_rate"]) else 0.0
    if cdp_rate <= 0.0:
        return "CDP因完整模型部署约束不可行，PMP/GS-Only承担保底。"
    if model_name in {"yolov5", "resnet101", "swin_base"}:
        return "CDP可行且低时延优势明显，适合批量数据并行。"
    return "模式边界依赖资源约束，需要结合可行性与任务特征判别。"


def _plot_fwms_oracle_gap(gap: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(gap))
    values = gap["FWMS/Oracle时延比"].astype(float).to_numpy()
    bars = ax.bar(x, values, color="#7C3AED", edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color="#0F172A", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_title("FWMS-Feature 相对 Oracle-Min-Latency 的时延比", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(gap["模型"])
    _style_axes(ax, "时延比（越接近 1 越接近时延上界）")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{value:.2f}x", ha="center", fontsize=9)
    _save_figure(fig, output_path)


def _write_report_notes(output_dir: Path, summary_md: pd.DataFrame, gap_md: pd.DataFrame) -> None:
    lines = [
        "# Stage6 模式选择最终汇总说明",
        "",
        "本目录由 `plot_stage6_mode_selection_report.py` 生成，输入为五个 `mode_selection_*_stage6_feature_oracle_b64` 结果目录。",
        "",
        "## 关键口径",
        "",
        "- `FWMS-Feature`：论文算法叙事使用的特征加权模式边界判别器，先做可行性筛选，再根据任务/资源特征在 PMP 与 CDP 间判别。",
        "- `Oracle-Min-Latency`：离线理论上界基线，已知所有候选模式预测时延后选择最低者，不应表述为在线 FWMS。",
        "- `feasible_rate`：该模式在 42 个 STK 时间片中的可行率，也可作为完成率使用。",
        "",
        "## 可直接引用的结论",
        "",
        "- CDP 在 YOLOv5、ResNet101、Swin-Base 上可行且时延最低，说明批量数据并行在资源充足时有明显优势。",
        "- VGG19 与 ViT-Huge 下 CDP/Sat-Only 不可行，主要体现完整模型部署的内存边界，此时 PMP 或 GS-Only 是稳定完成任务的保底路径。",
        "- FWMS-Feature 的价值不等同于 Oracle 的最低时延，而是把固定模式扩展为基于可行性和特征的稳定模式判别。",
        "",
        "## 主要输出",
        "",
        "- `stage6_mode_summary.csv/md`：跨模型、跨模式平均时延、能耗和完成率。",
        "- `stage6_selector_distribution.csv/md`：FWMS-Feature 与 Oracle-Min-Latency 的选择分布。",
        "- `stage6_fwms_oracle_gap.csv/md`：FWMS-Feature 相对 Oracle-Min-Latency 的时延差距和边界解释。",
        "- `stage6_avg_latency_by_model.png/pdf`：平均时延对比图。",
        "- `stage6_avg_energy_by_model.png/pdf`：平均卫星能耗对比图。",
        "- `stage6_completion_heatmap.png/pdf`：可行率/完成率热力图。",
        "- `stage6_selector_distribution.png/pdf`：两类选择器的模式选择分布。",
        "- `stage6_fwms_oracle_latency_gap.png/pdf`：FWMS 与 Oracle 的时延比。",
        "",
        "## 表 1：跨模型模式摘要",
        "",
    ]
    lines.extend(_markdown_table_lines(summary_md))
    lines.extend(["", "## 表 2：FWMS 与 Oracle 差距", ""])
    lines.extend(_markdown_table_lines(gap_md))
    lines.append("")
    (output_dir / "stage6_final_report_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final Stage 6 mode-selection report artifacts.")
    parser.add_argument("--input-root", default="result/mode_selection", help="Directory containing Stage 6 run dirs.")
    parser.add_argument(
        "--output-dir",
        default="result/mode_selection/final_stage6_report",
        help="Output directory for final report tables and figures.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, _slots, distribution = _load_stage6_runs(input_root)
    gap = _build_gap_table(summary)

    summary.to_csv(output_dir / "stage6_mode_summary.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output_dir / "stage6_selector_distribution.csv", index=False, encoding="utf-8-sig")
    gap.to_csv(output_dir / "stage6_fwms_oracle_gap.csv", index=False, encoding="utf-8-sig")

    summary_md = summary.copy()
    summary_md["模型"] = summary_md["model_label"]
    summary_md["模式"] = summary_md["mode_family"].astype(str)
    summary_md["完成率"] = summary_md["feasible_rate"].map(lambda x: f"{float(x) * 100:.0f}%")
    summary_md["平均时延(ms)"] = summary_md["avg_latency_ms"].map(_format_float)
    summary_md["平均卫星能耗(J)"] = summary_md["avg_satellite_energy_j"].map(_format_float)
    summary_md = summary_md[["模型", "模式", "完成率", "平均时延(ms)", "平均卫星能耗(J)"]]
    _write_markdown_table(summary_md, output_dir / "stage6_mode_summary.md")

    distribution_md = distribution.copy()
    distribution_md["模型"] = distribution_md["model_label"]
    distribution_md["选择器"] = distribution_md["selector_family"].astype(str)
    distribution_md["被选模式"] = distribution_md["selected_mode"].astype(str)
    distribution_md["次数"] = distribution_md["count"].astype(int)
    distribution_md["比例"] = distribution_md["ratio"].map(lambda x: f"{float(x) * 100:.1f}%")
    distribution_md = distribution_md[["模型", "选择器", "被选模式", "次数", "比例"]]
    _write_markdown_table(distribution_md, output_dir / "stage6_selector_distribution.md")

    gap_md = gap.copy()
    gap_md["FWMS平均时延(ms)"] = gap_md["FWMS平均时延(ms)"].map(_format_float)
    gap_md["Oracle平均时延(ms)"] = gap_md["Oracle平均时延(ms)"].map(_format_float)
    gap_md["FWMS/Oracle时延比"] = gap_md["FWMS/Oracle时延比"].map(lambda x: _format_float(x, 2) + "x")
    gap_md["PMP完成率"] = gap_md["PMP完成率"].map(lambda x: f"{float(x) * 100:.0f}%")
    gap_md["CDP完成率"] = gap_md["CDP完成率"].map(lambda x: f"{float(x) * 100:.0f}%")
    gap_md = gap_md[["模型", "PMP完成率", "CDP完成率", "FWMS平均时延(ms)", "Oracle平均时延(ms)", "FWMS/Oracle时延比", "边界解释"]]
    _write_markdown_table(gap_md, output_dir / "stage6_fwms_oracle_gap.md")

    _plot_metric_bars(
        summary,
        metric="avg_latency_ms",
        ylabel="平均端到端时延（ms）",
        title="STK 动态拓扑下跨模型平均时延对比",
        output_path=output_dir / "stage6_avg_latency_by_model.png",
    )
    _plot_metric_bars(
        summary,
        metric="avg_satellite_energy_j",
        ylabel="平均卫星能耗（J）",
        title="STK 动态拓扑下跨模型卫星能耗对比",
        output_path=output_dir / "stage6_avg_energy_by_model.png",
    )
    _plot_completion_heatmap(summary, output_dir / "stage6_completion_heatmap.png")
    _plot_selector_distribution(distribution, output_dir / "stage6_selector_distribution.png")
    _plot_fwms_oracle_gap(gap, output_dir / "stage6_fwms_oracle_latency_gap.png")
    _write_report_notes(output_dir, summary_md, gap_md)

    print(f"[REPORT] Wrote Stage 6 report artifacts to {output_dir}")


if __name__ == "__main__":
    main()
