# -*- coding: utf-8 -*-
"""Redraw sensitivity experiments from archived run CSV files.

This script only reads existing results under ``result/runs`` and writes
paper-ready Chinese figures to ``result/paper_figures``. It does not rerun any
simulation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


RUNS = Path("result/runs")
OUT = Path("result/paper_figures")

MODEL_ORDER = ["yolov5", "resnet101", "vgg19", "swin_base", "vit_huge"]
MODEL_LABEL = {
    "yolov5": "YOLOv5",
    "resnet101": "ResNet101",
    "vgg19": "VGG19",
    "swin_base": "Swin-Base",
    "vit_huge": "ViT-Huge",
}

ALGORITHM_ORDER = ["LA-DP", "Greedy", "GA", "Random", "Uniform", "GS-Only"]
DRAW_ORDER = ["GS-Only", "Uniform", "Random", "GA", "Greedy", "LA-DP"]
ALGORITHM_LABEL = {
    "LA-DP": "LADP",
    "Greedy": "贪心",
    "GA": "遗传算法",
    "Random": "随机",
    "Uniform": "均匀",
    "GS-Only": "GS-Only",
}
ALGORITHM_COLOR = {
    "LA-DP": "#244C85",
    "Greedy": "#E39D2D",
    "GA": "#2A8C88",
    "Random": "#E95B45",
    "Uniform": "#9CA3AF",
    "GS-Only": "#4A4A4A",
}
ALGORITHM_MARKER = {
    "LA-DP": "o",
    "Greedy": "s",
    "GA": "^",
    "Random": "D",
    "Uniform": "v",
    "GS-Only": "X",
}
ALGORITHM_LINESTYLE = {
    "LA-DP": "-",
    "Greedy": "--",
    "GA": "-.",
    "Random": ":",
    "Uniform": (0, (5, 2)),
    "GS-Only": (0, (1, 1)),
}
ALGORITHM_ZORDER = {
    "GS-Only": 2,
    "Uniform": 3,
    "Random": 4,
    "GA": 5,
    "Greedy": 6,
    "LA-DP": 8,
}


def setup_style() -> str:
    candidates = [
        "SimSun",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
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
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.2,
            "grid.color": "#D4D4D4",
            "grid.linewidth": 0.85,
            "grid.linestyle": "--",
        }
    )
    return chosen or "matplotlib default"


def latest_csv_files(summary_pattern: str, long_pattern: str) -> dict[str, Path]:
    """Return the latest summary CSV for each model, falling back to long CSV."""
    selected: dict[str, Path] = {}
    for pattern in [summary_pattern, long_pattern]:
        for path in sorted(RUNS.glob(pattern)):
            if not path.is_file():
                continue
            try:
                df = pd.read_csv(path, nrows=1)
            except Exception:
                continue
            if "model_name" not in df.columns or df.empty:
                continue
            model = str(df.loc[0, "model_name"])
            previous = selected.get(model)
            is_summary = path.name.startswith("summary_")
            previous_is_summary = previous is not None and previous.name.startswith("summary_")
            if previous is None:
                selected[model] = path
            elif is_summary and not previous_is_summary:
                selected[model] = path
            elif is_summary == previous_is_summary and str(path.parent.parent) > str(previous.parent.parent):
                selected[model] = path
    return selected


def read_grouped(files: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        path = files.get(model)
        if not path:
            continue
        df = pd.read_csv(path)
        if "mean_norm_latency_vs_gs" not in df.columns and "norm_latency_vs_gs" in df.columns:
            x_cols = [
                col
                for col in ["isl_avg_bw_mbps", "gsl_avg_bw_mbps", "pipeline_node_count"]
                if col in df.columns
            ]
            group_cols = ["run_id", "exp_type", "mode", *x_cols, "algorithm", "model_name", "batch_size", "input_h", "input_w"]
            df = (
                df.groupby(group_cols, dropna=False)
                .agg(
                    mean_latency_ms=("latency_ms", "mean"),
                    std_latency_ms=("latency_ms", "std"),
                    mean_norm_latency_vs_gs=("norm_latency_vs_gs", "mean"),
                    std_norm_latency_vs_gs=("norm_latency_vs_gs", "std"),
                    samples=("norm_latency_vs_gs", "count"),
                )
                .reset_index()
            )
        df["source_csv"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No sensitivity summary CSV files found.")
    return pd.concat(frames, ignore_index=True)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def nice_bandwidth_ticks(values: np.ndarray) -> list[float]:
    values = np.asarray(sorted(np.unique(values)), dtype=float)
    if len(values) <= 6:
        return values.tolist()
    idx = np.linspace(0, len(values) - 1, 6).round().astype(int)
    return values[idx].tolist()


def plot_sweep(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    stem: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    axes = axes.ravel()

    summary_rows: list[dict[str, object]] = []
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        sub = df[df["model_name"] == model].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x_values = np.asarray(sorted(sub[x_col].dropna().astype(float).unique()), dtype=float)
        if len(x_values) > 1:
            x_span = float(x_values.max() - x_values.min())
        else:
            x_span = 1.0

        for draw_idx, alg in enumerate(DRAW_ORDER):
            alg_df = sub[sub["algorithm"] == alg].sort_values(x_col)
            if alg_df.empty:
                continue
            x = alg_df[x_col].astype(float).to_numpy()
            y = alg_df["mean_norm_latency_vs_gs"].astype(float).to_numpy()
            offset_center = (len(DRAW_ORDER) - 1) / 2
            if x_col == "pipeline_node_count":
                x_offset = (draw_idx - offset_center) * 0.035
            else:
                x_offset = (draw_idx - offset_center) * x_span * 0.003
            marker_face = "white" if alg == "LA-DP" else ALGORITHM_COLOR.get(alg, "#64748B")
            ax.plot(
                x + x_offset,
                y,
                label=ALGORITHM_LABEL.get(alg, alg),
                color=ALGORITHM_COLOR.get(alg, "#64748B"),
                marker=ALGORITHM_MARKER.get(alg, "o"),
                linestyle=ALGORITHM_LINESTYLE.get(alg, "-"),
                linewidth=2.3 if alg == "LA-DP" else 1.75,
                markersize=6.0 if alg == "LA-DP" else 4.8,
                markerfacecolor=marker_face,
                markeredgewidth=1.35 if alg == "LA-DP" else 0.9,
                alpha=1.0 if alg == "LA-DP" else 0.90,
                zorder=ALGORITHM_ZORDER.get(alg, 3),
            )

        pivot = sub.pivot_table(
            index=x_col,
            columns="algorithm",
            values="mean_norm_latency_vs_gs",
            aggfunc="mean",
        )
        for alg in ALGORITHM_ORDER:
            if alg not in pivot.columns:
                continue
            values = pivot[alg].dropna()
            if values.empty:
                continue
            summary_rows.append(
                {
                    "experiment": stem,
                    "model": MODEL_LABEL.get(model, model),
                    "algorithm": ALGORITHM_LABEL.get(alg, alg),
                    "mean_norm_latency": float(values.mean()),
                    "best_norm_latency": float(values.min()),
                    "worst_norm_latency": float(values.max()),
                }
            )

        ax.axhline(1.0, color="#4A4A4A", linewidth=1.0, linestyle=":", alpha=0.75)
        ax.set_title(MODEL_LABEL.get(model, model))
        ax.set_xlabel(xlabel if idx >= 3 else "")
        ax.set_ylabel("归一化时延" if idx in [0, 3] else "")
        ax.grid(axis="y", alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.035)

        if "bw" in x_col:
            ax.set_xticks(nice_bandwidth_ticks(sub[x_col].to_numpy()))
            ax.ticklabel_format(axis="x", style="plain")
        else:
            xs = sorted(sub[x_col].dropna().astype(float).unique())
            ax.set_xticks(xs)
            ax.set_xticklabels([str(int(x)) for x in xs])

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.suptitle(title, fontsize=16.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.065, right=0.985, hspace=0.34, wspace=0.20)
    save(fig, stem)
    return pd.DataFrame(summary_rows)


def write_inventory(
    isl_files: dict[str, Path],
    gsl_files: dict[str, Path],
    node_files: dict[str, Path],
    summaries: list[pd.DataFrame],
    font_name: str,
) -> None:
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(OUT / "sensitivity_experiment_summary.csv", index=False, encoding="utf-8-sig")

    def best_table(experiment: str) -> str:
        sub = combined[combined["experiment"] == experiment].copy()
        rows = []
        for model in MODEL_LABEL.values():
            mdf = sub[sub["model"] == model]
            if mdf.empty:
                continue
            min_value = float(mdf["mean_norm_latency"].min())
            tied = mdf[np.isclose(mdf["mean_norm_latency"], min_value, atol=1e-6)]
            tied = tied.sort_values("algorithm")
            best = tied.iloc[0]
            best_names = " / ".join(tied["algorithm"].astype(str).tolist())
            rows.append(
                f"| {model} | {best_names} | {best['mean_norm_latency']:.3f} | "
                f"{best['best_norm_latency']:.3f} |"
            )
        return "\n".join(rows)

    lines = [
        "# 敏感性实验重绘说明",
        "",
        f"- 字体：`{font_name}`。",
        "- 图均由 `result/runs` 下已有 CSV 重绘，没有重新运行仿真。",
        "- 纵轴统一使用“相对 GS-Only 的归一化时延”，便于跨模型比较。",
        "- 虚线 `y=1` 表示 GS-Only 基线，低于 1 说明该算法优于 GS-Only。",
        "",
        "## 新增图表",
        "",
        "- `13_isl_bandwidth_sensitivity_norm.png/pdf`：ISL 带宽敏感性分析。",
        "- `14_gsl_bandwidth_sensitivity_norm.png/pdf`：GSL 带宽敏感性分析。",
        "- `15_node_count_sensitivity_norm.png/pdf`：节点数量敏感性分析。",
        "- `sensitivity_experiment_summary.csv`：上述三类实验的均值摘要。",
        "",
        "## 数据源",
        "",
    ]

    for title, files in [
        ("ISL 带宽敏感性", isl_files),
        ("GSL 带宽敏感性", gsl_files),
        ("节点数量敏感性", node_files),
    ]:
        lines += [f"### {title}", ""]
        for model in MODEL_ORDER:
            path = files.get(model)
            if path:
                lines.append(f"- {MODEL_LABEL[model]}：`{path}`")
        lines.append("")

    lines += [
        "## 平均表现最优算法",
        "",
        "### ISL 带宽敏感性",
        "",
        "| 模型 | 平均最优算法 | 平均归一化时延 | 最低归一化时延 |",
        "| --- | --- | ---: | ---: |",
        best_table("13_isl_bandwidth_sensitivity_norm"),
        "",
        "### GSL 带宽敏感性",
        "",
        "| 模型 | 平均最优算法 | 平均归一化时延 | 最低归一化时延 |",
        "| --- | --- | ---: | ---: |",
        best_table("14_gsl_bandwidth_sensitivity_norm"),
        "",
        "### 节点数量敏感性",
        "",
        "| 模型 | 平均最优算法 | 平均归一化时延 | 最低归一化时延 |",
        "| --- | --- | ---: | ---: |",
        best_table("15_node_count_sensitivity_norm"),
        "",
        "## 可写进论文的结论",
        "",
        "1. ISL 带宽提高后，PMP 模式中需要跨星传输中间特征的算法会受益，但收益不是无限增长；当通信不再是主要瓶颈后，推理计算和分层策略成为主导。",
        "2. GSL 带宽对输入上行和结果回传更敏感，尤其在输入较大的 YOLOv5 场景下更明显；这说明星地链路是端到端时延的重要约束。",
        "3. 节点数量增加会扩大 LADP 的模型切分搜索空间，通常能降低或稳定时延；随机和均匀分配容易引入不必要通信或负载不均，因此波动更大。",
        "4. LADP 在多数敏感性场景下保持较低归一化时延，说明它不是只在单一参数配置下有效，而是对带宽和节点数量变化具有一定鲁棒性。",
    ]
    (OUT / "sensitivity_experiment_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font_name = setup_style()

    isl_files = latest_csv_files("*/data/summary_isl_bw_*.csv", "*/data/results_long_isl_bw_*.csv")
    gsl_files = latest_csv_files("*/data/summary_gsl_bw_*.csv", "*/data/results_long_gsl_bw_*.csv")
    node_files = latest_csv_files(
        "*/data/summary_node_count_sensitivity_*.csv",
        "*/data/results_long_node_count_sensitivity_*.csv",
    )

    isl = read_grouped(isl_files)
    gsl = read_grouped(gsl_files)
    node = read_grouped(node_files)

    summaries = [
        plot_sweep(
            isl,
            "isl_avg_bw_mbps",
            "ISL 带宽敏感性分析",
            "ISL 平均带宽 / Mbps",
            "13_isl_bandwidth_sensitivity_norm",
        ),
        plot_sweep(
            gsl,
            "gsl_avg_bw_mbps",
            "GSL 带宽敏感性分析",
            "GSL 平均带宽 / Mbps",
            "14_gsl_bandwidth_sensitivity_norm",
        ),
        plot_sweep(
            node,
            "pipeline_node_count",
            "节点数量敏感性分析",
            "可用计算卫星数量",
            "15_node_count_sensitivity_norm",
        ),
    ]
    write_inventory(isl_files, gsl_files, node_files, summaries, font_name)

    print(f"[OK] sensitivity figures written to {OUT}")
    print(f"[OK] notes written to {OUT / 'sensitivity_experiment_notes.md'}")


if __name__ == "__main__":
    main()
