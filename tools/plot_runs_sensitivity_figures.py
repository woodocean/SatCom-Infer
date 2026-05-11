# -*- coding: utf-8 -*-
"""Redraw sensitivity experiments from archived run CSV files.

Default mode:
- Read the latest archived sensitivity runs under ``result/runs``.
- Regenerate the legacy paper-ready figures under ``result/paper_figures``.

Focused mode:
- Read explicitly provided long-table CSV files.
- Generate one or more node-count sensitivity figures in a caller-specified folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


RUNS = Path("result/runs")
DEFAULT_OUT = Path("result/paper_figures")

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


def explicit_csv_files(csv_paths: list[Path]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, nrows=1)
        if "model_name" not in df.columns or df.empty:
            raise ValueError(f"CSV does not contain model_name: {path}")
        model = str(df.loc[0, "model_name"])
        selected[model] = path
    return selected


def read_grouped(files: dict[str, Path], model_order: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    model_order = model_order or MODEL_ORDER
    for model in model_order:
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
            group_cols = [
                "run_id",
                "exp_type",
                "mode",
                *x_cols,
                "algorithm",
                "model_name",
                "batch_size",
                "input_h",
                "input_w",
            ]
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


def save(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
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
    out_dir: Path,
    model_order: list[str] | None = None,
    algorithm_order: list[str] | None = None,
) -> pd.DataFrame:
    model_order = model_order or MODEL_ORDER
    algorithm_order = algorithm_order or ALGORITHM_ORDER
    draw_order = [alg for alg in DRAW_ORDER if alg in algorithm_order]
    count = len(model_order)

    if count <= 2:
        nrows, ncols = 1, count
        figsize = (10.8 if count == 2 else 5.6, 4.6)
    elif count <= 4:
        nrows, ncols = 2, 2
        figsize = (11.8, 7.0)
    else:
        nrows, ncols = 2, 3
        figsize = (13.8, 7.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    summary_rows: list[dict[str, object]] = []
    for idx, model in enumerate(model_order):
        ax = axes[idx]
        sub = df[df["model_name"] == model].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x_values = np.asarray(sorted(sub[x_col].dropna().astype(float).unique()), dtype=float)
        x_span = float(x_values.max() - x_values.min()) if len(x_values) > 1 else 1.0
        visible_y_values: list[float] = []

        for draw_idx, alg in enumerate(draw_order):
            alg_df = sub[sub["algorithm"] == alg].sort_values(x_col)
            if alg_df.empty:
                continue
            x = alg_df[x_col].astype(float).to_numpy()
            y = alg_df["mean_norm_latency_vs_gs"].astype(float).to_numpy()
            visible_y_values.extend(y.tolist())
            offset_center = (len(draw_order) - 1) / 2
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
        for alg in algorithm_order:
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

        if "GS-Only" in algorithm_order:
            ax.axhline(1.0, color="#4A4A4A", linewidth=1.0, linestyle=":", alpha=0.75)
        if visible_y_values:
            y_min = min(visible_y_values)
            y_max = max(visible_y_values)
            if np.isclose(y_min, y_max):
                padding = max(0.01, abs(y_min) * 0.05)
            else:
                padding = max((y_max - y_min) * 0.20, 0.01)
            lower = max(0.0, y_min - padding)
            upper = y_max + padding
            if "GS-Only" in algorithm_order:
                upper = max(upper, 1.02)
            ax.set_ylim(lower, upper)
        ax.set_title(MODEL_LABEL.get(model, model))
        if nrows == 1:
            ax.set_xlabel(xlabel)
            ax.set_ylabel("归一化时延" if idx == 0 else "")
        else:
            ax.set_xlabel(xlabel if idx >= (nrows - 1) * ncols else "")
            ax.set_ylabel("归一化时延" if idx % ncols == 0 else "")
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

    for idx in range(count, len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, len(draw_order)),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.suptitle(title, fontsize=16.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.075, right=0.985, hspace=0.34, wspace=0.24)
    save(fig, stem, out_dir)
    return pd.DataFrame(summary_rows)


def plot_algorithm_trend(
    df: pd.DataFrame,
    algorithm: str,
    x_col: str,
    title: str,
    xlabel: str,
    stem: str,
    out_dir: Path,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    model_order = model_order or MODEL_ORDER
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    summary_rows: list[dict[str, object]] = []

    model_colors = {
        "yolov5": "#244C85",
        "vgg19": "#2A8C88",
        "resnet101": "#E39D2D",
        "swin_base": "#E95B45",
        "vit_huge": "#7C6FB0",
    }
    model_markers = {
        "yolov5": "o",
        "vgg19": "s",
        "resnet101": "^",
        "swin_base": "D",
        "vit_huge": "v",
    }

    for model in model_order:
        sub = df[(df["model_name"] == model) & (df["algorithm"] == algorithm)].sort_values(x_col).copy()
        if sub.empty:
            continue
        x = sub[x_col].astype(float).to_numpy()
        y = sub["mean_norm_latency_vs_gs"].astype(float).to_numpy()
        ax.plot(
            x,
            y,
            color=model_colors.get(model, "#64748B"),
            marker=model_markers.get(model, "o"),
            linewidth=2.2,
            markersize=6.0,
            label=MODEL_LABEL.get(model, model),
        )
        summary_rows.append(
            {
                "algorithm": ALGORITHM_LABEL.get(algorithm, algorithm),
                "model": MODEL_LABEL.get(model, model),
                "node_count_1": float(y[0]) if len(y) >= 1 else float("nan"),
                "node_count_last": float(y[-1]) if len(y) >= 1 else float("nan"),
                "trend_delta": float(y[-1] - y[0]) if len(y) >= 2 else 0.0,
            }
        )

    ax.axhline(1.0, color="#4A4A4A", linewidth=1.0, linestyle=":", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("归一化时延")
    ax.grid(axis="y", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_ticks = sorted(df[x_col].dropna().astype(float).unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks])
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, stem, out_dir)
    return pd.DataFrame(summary_rows)


def plot_node_count_scenarios(
    df: pd.DataFrame,
    title: str,
    stem: str,
    out_dir: Path,
    model_order: list[str] | None = None,
    y_col: str = "mean_norm_latency_vs_gs",
    y_label: str = "归一化时延",
) -> pd.DataFrame:
    model_order = model_order or MODEL_ORDER
    count = len(model_order)
    nrows, ncols = (2, 2) if count <= 4 else (2, 3)
    figsize = (11.8, 7.0) if count <= 4 else (13.8, 7.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    scenario_order = ["同构场景", "异构场景", "典型异构均值"]
    scenario_colors = {
        "同构场景": "#244C85",
        "异构场景": "#E39D2D",
        "典型异构均值": "#2A8C88",
    }
    scenario_markers = {
        "同构场景": "o",
        "异构场景": "s",
        "典型异构均值": "^",
    }

    summary_rows: list[dict[str, object]] = []
    for idx, model in enumerate(model_order):
        ax = axes[idx]
        sub = df[df["model_name"] == model].copy()
        if sub.empty:
            ax.axis("off")
            continue

        visible_y: list[float] = []
        for scenario in scenario_order:
            scenario_df = sub[sub["scenario_label"] == scenario].sort_values("pipeline_node_count")
            if scenario_df.empty:
                continue
            x = scenario_df["pipeline_node_count"].astype(float).to_numpy()
            y = scenario_df[y_col].astype(float).to_numpy()
            visible_y.extend(y.tolist())
            ax.plot(
                x,
                y,
                color=scenario_colors.get(scenario, "#64748B"),
                marker=scenario_markers.get(scenario, "o"),
                linewidth=2.2,
                markersize=5.8,
                label=scenario,
            )
            summary_rows.append(
                {
                    "model": MODEL_LABEL.get(model, model),
                    "scenario": scenario,
                    "node_count_1": float(y[0]) if len(y) >= 1 else float("nan"),
                    "node_count_last": float(y[-1]) if len(y) >= 1 else float("nan"),
                    "trend_delta": float(y[-1] - y[0]) if len(y) >= 2 else 0.0,
                }
            )

        if visible_y:
            y_min = min(visible_y)
            y_max = max(visible_y)
            padding = max((y_max - y_min) * 0.20, 0.01) if not np.isclose(y_min, y_max) else max(0.01, abs(y_min) * 0.05)
            ax.set_ylim(max(0.0, y_min - padding), y_max + padding)

        ax.set_title(MODEL_LABEL.get(model, model))
        ax.set_xlabel("中继 LEO 卫星数量" if idx >= (nrows - 1) * ncols else "")
        ax.set_ylabel(y_label if idx % ncols == 0 else "")
        xs = sorted(sub["pipeline_node_count"].dropna().astype(float).unique())
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.grid(axis="y", alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(count, len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle(title, fontsize=16.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.075, right=0.985, hspace=0.34, wspace=0.24)
    save(fig, stem, out_dir)
    return pd.DataFrame(summary_rows)


def write_inventory(
    out_dir: Path,
    isl_files: dict[str, Path],
    gsl_files: dict[str, Path],
    node_files: dict[str, Path],
    summaries: list[pd.DataFrame],
    font_name: str,
) -> None:
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(out_dir / "sensitivity_experiment_summary.csv", index=False, encoding="utf-8-sig")

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
                f"| {model} | {best_names} | {best['mean_norm_latency']:.3f} | {best['best_norm_latency']:.3f} |"
            )
        return "\n".join(rows)

    lines = [
        "# 敏感性实验重绘说明",
        "",
        f"- 字体：`{font_name}`",
        "- 图均由 `result/runs` 下已有 CSV 重绘，没有重新运行仿真。",
        "- 纵轴统一使用“相对 GS-Only 的归一化时延”，便于跨模型比较。",
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
    ]
    (out_dir / "sensitivity_experiment_notes.md").write_text("\n".join(lines), encoding="utf-8")


def parse_model_order(raw: str | None) -> list[str]:
    if not raw:
        return MODEL_ORDER
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return [model for model in MODEL_ORDER if model in models] or models


def parse_algorithm_order(raw: str | None) -> list[str]:
    if not raw:
        return ALGORITHM_ORDER
    algorithms = [item.strip() for item in raw.split(",") if item.strip()]
    return [alg for alg in ALGORITHM_ORDER if alg in algorithms] or algorithms


def parse_csv_list(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def run_default_mode() -> None:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
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
        plot_sweep(isl, "isl_avg_bw_mbps", "ISL 带宽敏感性分析", "ISL 平均带宽 / Mbps", "13_isl_bandwidth_sensitivity_norm", out_dir),
        plot_sweep(gsl, "gsl_avg_bw_mbps", "GSL 带宽敏感性分析", "GSL 平均带宽 / Mbps", "14_gsl_bandwidth_sensitivity_norm", out_dir),
        plot_sweep(node, "pipeline_node_count", "节点数量敏感性分析", "可用计算卫星数量", "15_node_count_sensitivity_norm", out_dir),
    ]
    write_inventory(out_dir, isl_files, gsl_files, node_files, summaries, font_name)

    print(f"[OK] sensitivity figures written to {out_dir}")
    print(f"[OK] notes written to {out_dir / 'sensitivity_experiment_notes.md'}")


def run_focused_node_plot(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    csv_paths = parse_csv_list(args.node_source_csvs)
    model_order = parse_model_order(args.models)
    algorithm_order = parse_algorithm_order(args.algorithms)
    files = explicit_csv_files(csv_paths)

    raw_frames = [pd.read_csv(path) for path in csv_paths]
    raw_df = pd.concat(raw_frames, ignore_index=True)
    raw_df.to_csv(out_dir / f"{args.stem}_long.csv", index=False, encoding="utf-8-sig")

    grouped = read_grouped(files, model_order=model_order)
    grouped.to_csv(out_dir / f"{args.stem}_grouped.csv", index=False, encoding="utf-8-sig")

    summary = plot_sweep(
        grouped,
        "pipeline_node_count",
        args.title,
        "中继 LEO 卫星数量",
        args.stem,
        out_dir,
        model_order=model_order,
        algorithm_order=algorithm_order,
    )
    summary.to_csv(out_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")

    trend_stem = None
    if args.trend_algorithm:
        trend_stem = f"{args.stem}_{args.trend_algorithm.lower().replace('-', '_')}_trend"
        trend_summary = plot_algorithm_trend(
            grouped,
            args.trend_algorithm,
            "pipeline_node_count",
            args.trend_title or f"{ALGORITHM_LABEL.get(args.trend_algorithm, args.trend_algorithm)} 节点数量变化趋势",
            "中继 LEO 卫星数量",
            trend_stem,
            out_dir,
            model_order=model_order,
        )
        trend_summary.to_csv(out_dir / f"{trend_stem}_summary.csv", index=False, encoding="utf-8-sig")

    note_lines = [
        f"# {args.title}",
        "",
        "- 图类型：折线图",
        "- 横坐标：中继 LEO 卫星数量",
        "- 纵坐标：归一化时延，GS-Only = 1",
        f"- 模型：{', '.join(MODEL_LABEL.get(model, model) for model in model_order)}",
        f"- 算法：{', '.join(ALGORITHM_LABEL.get(alg, alg) for alg in algorithm_order)}",
        "- 数据来源：",
    ]
    for path in csv_paths:
        note_lines.append(f"  - `{path}`")
    if trend_stem:
        note_lines += [
            "",
            f"- 额外趋势图：`{trend_stem}.png/pdf`",
            f"- 趋势图算法：`{ALGORITHM_LABEL.get(args.trend_algorithm, args.trend_algorithm)}`",
        ]
    (out_dir / f"{args.stem}_notes.md").write_text("\n".join(note_lines), encoding="utf-8")

    print(f"[OK] focused node-count figure written to {out_dir}")


def run_scenario_node_plot(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    scenario_csv = Path(args.scenario_csv)
    if not scenario_csv.is_file():
        raise FileNotFoundError(scenario_csv)

    df = pd.read_csv(scenario_csv)
    model_order = parse_model_order(args.models)
    summary = plot_node_count_scenarios(
        df=df,
        title=args.title,
        stem=args.stem,
        out_dir=out_dir,
        model_order=model_order,
    )
    summary.to_csv(out_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")

    active_summary = None
    if "mean_active_sat_count" in df.columns:
        active_summary = plot_node_count_scenarios(
            df=df,
            title="LADP 最优解实际启用计算星数量",
            stem=f"{args.stem}_active_sat_count",
            out_dir=out_dir,
            model_order=model_order,
            y_col="mean_active_sat_count",
            y_label="启用计算星数量",
        )
        active_summary.to_csv(
            out_dir / f"{args.stem}_active_sat_count_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    note_lines = [
        f"# {args.title}",
        "",
        "- 图类型：折线图",
        "- 横坐标：中继 LEO 卫星数量",
        "- 纵坐标：归一化时延，GS-Only = 1",
        f"- 模型：{', '.join(MODEL_LABEL.get(model, model) for model in model_order)}",
        "- 曲线：同构场景 / 异构场景 / 典型异构均值",
        f"- 数据来源：`{scenario_csv}`",
    ]
    if active_summary is not None:
        note_lines += [
            "- 额外图：`"
            f"{args.stem}_active_sat_count.png/pdf"
            "`，展示 LADP 最优解实际启用的计算卫星数量。",
        ]
    (out_dir / f"{args.stem}_notes.md").write_text("\n".join(note_lines), encoding="utf-8")
    print(f"[OK] scenario node-count figure written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw sensitivity figures from archived runs")
    parser.add_argument(
        "--node-source-csvs",
        type=str,
        default=None,
        help="Comma-separated explicit CSV paths for a focused node-count figure.",
    )
    parser.add_argument(
        "--scenario-csv",
        type=str,
        default=None,
        help="Prepared CSV for LADP-only scenario-based node-count plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names for focused plotting, e.g. yolov5,vgg19",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated algorithms for focused plotting, e.g. LA-DP,Greedy,GA,GS-Only",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="节点数量敏感性分析",
        help="Figure title used in focused plotting mode.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="15_node_count_sensitivity_norm",
        help="Output file stem used in focused plotting mode.",
    )
    parser.add_argument(
        "--trend-algorithm",
        type=str,
        default=None,
        help="Optional single algorithm used to generate an extra trend-only figure.",
    )
    parser.add_argument(
        "--trend-title",
        type=str,
        default=None,
        help="Optional title for the extra trend-only figure.",
    )
    args = parser.parse_args()

    if args.scenario_csv:
        run_scenario_node_plot(args)
        return
    if args.node_source_csvs:
        run_focused_node_plot(args)
        return
    run_default_mode()


if __name__ == "__main__":
    main()
