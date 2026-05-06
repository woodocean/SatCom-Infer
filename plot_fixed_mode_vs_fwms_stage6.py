"""Plot fixed-mode vs FWMS feasibility and latency comparisons for Stage 6.

The script is a pure post-processing utility. It reads completed Stage 6
mode-selection runs and builds the missing paper-facing figures for the claim:
fixed modes can fail or become suboptimal under different task/resource
conditions, so a mode-boundary selector is needed.
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
BATCH_ORDER = [32, 64]
MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS-Feature", "Oracle-Min-Latency"]
STRATEGY_LABELS = {
    "PMP": "Fixed-PMP",
    "CDP": "Fixed-CDP",
    "GS-Only": "Fixed-GS",
    "Sat-Only": "Fixed-Sat",
    "FWMS-Feature": "FWMS-Feature",
    "Oracle-Min-Latency": "Oracle",
}


def _read_metadata(run_dir: Path) -> dict:
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_stage6_summaries(input_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch in BATCH_ORDER:
        pattern = f"mode_selection_*_stage6_feature_oracle_b{batch}"
        for run_dir in sorted(input_root.glob(pattern)):
            summary_path = run_dir / "data" / "summary_by_mode.csv"
            metadata_path = run_dir / "metadata.json"
            if not summary_path.exists() or not metadata_path.exists():
                continue

            metadata = _read_metadata(run_dir)
            task = metadata.get("effective_task", {})
            model_name = str(task.get("model_name") or metadata.get("model_name_override") or run_dir.name)
            batch_size = int(task.get("batch_size") or metadata.get("batch_size_override") or batch)

            summary = pd.read_csv(summary_path)
            summary["model_name"] = model_name
            summary["model_label"] = MODEL_LABELS.get(model_name, model_name)
            summary["batch_size"] = batch_size
            summary["scenario"] = summary["model_label"] + "\nB=" + summary["batch_size"].astype(str)
            summary["strategy"] = summary["mode_family"].map(STRATEGY_LABELS)
            summary["run_id"] = run_dir.name
            frames.append(summary)

    if not frames:
        raise FileNotFoundError(f"No Stage 6 b32/b64 summaries found under {input_root}")

    df = pd.concat(frames, ignore_index=True)
    df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
    df["batch_size"] = pd.Categorical(df["batch_size"], BATCH_ORDER, ordered=True)
    df["mode_family"] = pd.Categorical(df["mode_family"], MODE_ORDER, ordered=True)
    return df.sort_values(["model_name", "batch_size", "mode_family"]).reset_index(drop=True)


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


def _write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    output_path.write_text("\n".join(_markdown_lines(df)) + "\n", encoding="utf-8")


def _scenario_order() -> list[str]:
    return [f"{MODEL_LABELS[m]}\nB={b}" for m in MODEL_ORDER for b in BATCH_ORDER]


def _save(fig: plt.Figure, output_path: Path, rect: tuple[float, float, float, float] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rect:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_completion_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot_table(
        index="scenario",
        columns="strategy",
        values="feasible_rate",
        aggfunc="first",
        observed=False,
    )
    columns = [STRATEGY_LABELS[m] for m in MODE_ORDER]
    pivot = pivot.reindex(_scenario_order())[columns]

    values = pivot.astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title("固定模式与 FWMS 的任务完成率对比", fontsize=14, weight="bold")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([idx.replace("\n", " ") for idx in pivot.index])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if value > 0.65 else "#0F172A"
            ax.text(j, i, f"{value * 100:.0f}%", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="完成率")
    _save(fig, output_path)


def _plot_latency_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot_table(
        index="scenario",
        columns="strategy",
        values="avg_latency_ms",
        aggfunc="first",
        observed=False,
    )
    columns = [STRATEGY_LABELS[m] for m in MODE_ORDER]
    pivot = pivot.reindex(_scenario_order())[columns]

    values = pivot.astype(float).to_numpy()
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#E5E7EB")

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    finite_values = values[np.isfinite(values)]
    vmax = np.percentile(finite_values, 90) if finite_values.size else 1.0
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_title("固定模式与 FWMS 的平均时延对比", fontsize=14, weight="bold")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([idx.replace("\n", " ") for idx in pivot.index])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "不可行" if not np.isfinite(value) else f"{value:.0f}"
            color = "#475569" if not np.isfinite(value) else ("white" if value > vmax * 0.62 else "#0F172A")
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="平均时延（ms）")
    _save(fig, output_path)


def _plot_energy_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot_table(
        index="scenario",
        columns="strategy",
        values="avg_satellite_energy_j",
        aggfunc="first",
        observed=False,
    )
    columns = [STRATEGY_LABELS[m] for m in MODE_ORDER]
    pivot = pivot.reindex(_scenario_order())[columns]

    values = pivot.astype(float).to_numpy()
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("PuBuGn").copy()
    cmap.set_bad("#E5E7EB")

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    finite_values = values[np.isfinite(values)]
    vmax = np.percentile(finite_values, 90) if finite_values.size else 1.0
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_title("固定模式与 FWMS 的平均星载能耗对比", fontsize=14, weight="bold")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([idx.replace("\n", " ") for idx in pivot.index])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "不可行" if not np.isfinite(value) else f"{value:.1f}"
            color = "#475569" if not np.isfinite(value) else ("white" if value > vmax * 0.62 else "#0F172A")
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="平均星载能耗（J）")
    _save(fig, output_path)


def _build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for batch in BATCH_ORDER:
            sub = df[(df["model_name"].astype(str) == model_name) & (df["batch_size"].astype(int) == batch)]
            row = {"模型": MODEL_LABELS[model_name], "batch": batch}
            for mode in MODE_ORDER:
                mode_df = sub[sub["mode_family"].astype(str) == mode]
                if mode_df.empty:
                    row[f"{STRATEGY_LABELS[mode]}完成率"] = "-"
                    row[f"{STRATEGY_LABELS[mode]}时延(ms)"] = "-"
                    continue
                item = mode_df.iloc[0]
                row[f"{STRATEGY_LABELS[mode]}完成率"] = f"{float(item['feasible_rate']) * 100:.0f}%"
                row[f"{STRATEGY_LABELS[mode]}时延(ms)"] = _format_float(item["avg_latency_ms"])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_boundary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for batch in BATCH_ORDER:
            sub = df[(df["model_name"].astype(str) == model_name) & (df["batch_size"].astype(int) == batch)]
            by_mode = {str(row.mode_family): row for row in sub.itertuples(index=False)}
            cdp_rate = float(getattr(by_mode["CDP"], "feasible_rate", 0.0)) if "CDP" in by_mode else 0.0
            pmp_latency = getattr(by_mode["PMP"], "avg_latency_ms", np.nan) if "PMP" in by_mode else np.nan
            cdp_latency = getattr(by_mode["CDP"], "avg_latency_ms", np.nan) if "CDP" in by_mode else np.nan
            gs_latency = getattr(by_mode["GS-Only"], "avg_latency_ms", np.nan) if "GS-Only" in by_mode else np.nan
            fwms_latency = getattr(by_mode["FWMS-Feature"], "avg_latency_ms", np.nan) if "FWMS-Feature" in by_mode else np.nan
            oracle_latency = (
                getattr(by_mode["Oracle-Min-Latency"], "avg_latency_ms", np.nan)
                if "Oracle-Min-Latency" in by_mode
                else np.nan
            )
            rows.append(
                {
                    "模型": MODEL_LABELS[model_name],
                    "batch": batch,
                    "CDP完成率": f"{cdp_rate * 100:.0f}%",
                    "PMP时延(ms)": _format_float(pmp_latency),
                    "CDP时延(ms)": _format_float(cdp_latency),
                    "GS-Only时延(ms)": _format_float(gs_latency),
                    "FWMS时延(ms)": _format_float(fwms_latency),
                    "Oracle时延(ms)": _format_float(oracle_latency),
                    "边界结论": _boundary_note(model_name, batch, cdp_rate, cdp_latency, gs_latency),
                }
            )
    return pd.DataFrame(rows)


def _boundary_note(
    model_name: str,
    batch: int,
    cdp_rate: float,
    cdp_latency: float | object,
    gs_latency: float | object,
) -> str:
    if cdp_rate <= 0:
        return "CDP 在该任务规模下不可行，PMP/GS-Only 提供保底。"
    if model_name == "vit_huge" and batch == 32:
        return "CDP 可行但时延高于 GS-Only，可行性不等于最优性。"
    if pd.notna(cdp_latency) and pd.notna(gs_latency) and float(cdp_latency) < float(gs_latency):
        return "CDP 可行且时延占优，适合数据并行。"
    return "需要结合任务特征和资源状态选择模式。"


def _write_report(output_dir: Path, summary_md: pd.DataFrame, boundary_md: pd.DataFrame) -> None:
    lines = [
        "# 固定模式与 FWMS 对比图说明",
        "",
        "本目录用于补充论文模式选择实验中最缺的一组图：固定使用某一种模式与采用 FWMS/Oracle 的可行率、时延和星载能耗对比。",
        "",
        "## 实验口径",
        "",
        "- `Fixed-PMP`、`Fixed-CDP`、`Fixed-GS`、`Fixed-Sat` 表示所有任务均固定采用对应模式。",
        "- `FWMS-Feature` 表示基于可行性与特征加权的模式边界判别方法。",
        "- `Oracle` 表示离线已知所有候选模式预测时延后的最小时延上界，不作为实际在线算法。",
        "- 完成率/可行率不是新的优化目标，而是能耗、内存、可见性等约束下能否产生可行解的统计指标。",
        "",
        "## 可直接写入论文的结论",
        "",
        "- Fixed-CDP 在 YOLOv5、ResNet101、Swin-Base 上可行且低时延优势明显，但在 VGG19 和部分 ViT-Huge 任务规模下会因完整模型部署约束不可行。",
        "- Fixed-PMP 完成率稳定，但在 CDP 可行且 batch 较大的场景下时延通常不占优。",
        "- GS-Only 在部分任务上时延较低，但不代表星上协作推理能力，且无法体现模型切分和数据并行的资源利用优势。",
        "- FWMS-Feature 的意义不是逼近 Oracle 的最低时延，而是在可行性和任务特征边界下避免固定模式失效。",
        "",
        "## 输出文件",
        "",
        "- `fixed_mode_completion_heatmap.png/pdf`：固定模式与 FWMS 的任务完成率对比。",
        "- `fixed_mode_latency_heatmap.png/pdf`：固定模式与 FWMS 的平均时延对比。",
        "- `fixed_mode_energy_heatmap.png/pdf`：固定模式与 FWMS 的平均星载能耗对比。",
        "- `fixed_mode_vs_fwms_summary.md/csv`：完整数值表。",
        "- `fixed_mode_boundary_notes.md/csv`：适合放进论文的边界结论表。",
        "",
        "## 边界结论表",
        "",
    ]
    lines.extend(_markdown_lines(boundary_md))
    lines.extend(["", "## 完整数值摘要", ""])
    lines.extend(_markdown_lines(summary_md))
    lines.append("")
    (output_dir / "fixed_mode_vs_fwms_report_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-mode vs FWMS Stage 6 figures.")
    parser.add_argument("--input-root", default="result/mode_selection")
    parser.add_argument("--output-dir", default="result/mode_selection/fixed_mode_vs_fwms_stage6")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_stage6_summaries(Path(args.input_root))
    df.to_csv(output_dir / "fixed_mode_vs_fwms_raw_summary.csv", index=False, encoding="utf-8-sig")

    summary_md = _build_summary_table(df)
    boundary_md = _build_boundary_table(df)
    summary_md.to_csv(output_dir / "fixed_mode_vs_fwms_summary.csv", index=False, encoding="utf-8-sig")
    boundary_md.to_csv(output_dir / "fixed_mode_boundary_notes.csv", index=False, encoding="utf-8-sig")
    _write_markdown(summary_md, output_dir / "fixed_mode_vs_fwms_summary.md")
    _write_markdown(boundary_md, output_dir / "fixed_mode_boundary_notes.md")

    _plot_completion_heatmap(df, output_dir / "fixed_mode_completion_heatmap.png")
    _plot_latency_heatmap(df, output_dir / "fixed_mode_latency_heatmap.png")
    _plot_energy_heatmap(df, output_dir / "fixed_mode_energy_heatmap.png")
    _write_report(output_dir, summary_md, boundary_md)

    print(f"[FIXED] Wrote fixed-mode comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
