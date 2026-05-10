# -*- coding: utf-8 -*-
"""Generate paper-ready figures from existing experiment results.

The script only redraws figures and writes notes. It does not rerun any
simulation. The output follows the thesis storyline:

1. PMP optimization: LADP vs PMP baselines.
2. CDP optimization: LAWA vs data-allocation baselines.
3. Mode selection: fixed modes, FWMS-Feature, and Oracle-Min-Latency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


OUT = Path("result/paper_figures")
MODEL_ORDER = ["YOLOv5", "ResNet101", "VGG19", "Swin-Base", "ViT-Huge"]

PMP_ALG_ORDER = ["LA-DP", "Greedy", "GA", "Random", "Uniform", "GS-Only"]
CDP_ALG_ORDER = ["LAWA", "GA", "Random", "Greedy", "Uniform", "Pass-Through"]
MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS-Feature", "Oracle-Min-Latency"]
BASE_MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS-Feature"]
SELECTED_MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only"]

ALG_LABEL = {
    "LA-DP": "LADP",
    "LAWA": "LAWA",
    "Greedy": "贪心",
    "GA": "遗传算法",
    "Random": "随机",
    "Uniform": "均匀",
    "GS-Only": "GS-Only",
    "Pass-Through": "Sat-Only",
}

MODE_LABEL = {
    "PMP": "PMP",
    "CDP": "CDP",
    "GS-Only": "GS-Only",
    "Sat-Only": "Sat-Only",
    "FWMS-Feature": "FWMS",
    "Oracle-Min-Latency": "Oracle",
}

ALG_COLOR = {
    "LA-DP": "#244C85",
    "LAWA": "#244C85",
    "Greedy": "#E39D2D",
    "GA": "#2A8C88",
    "Random": "#E95B45",
    "Uniform": "#9CA3AF",
    "GS-Only": "#4A4A4A",
    "Pass-Through": "#8272B2",
}

MODE_COLOR = {
    "PMP": "#244C85",
    "CDP": "#E39D2D",
    "GS-Only": "#4A4A4A",
    "Sat-Only": "#8272B2",
    "FWMS-Feature": "#2A8C88",
    "Oracle-Min-Latency": "#E95B45",
}

MODEL_COLOR = {
    "YOLOv5": "#244C85",
    "ResNet101": "#E39D2D",
    "VGG19": "#2A8C88",
    "Swin-Base": "#8272B2",
    "ViT-Huge": "#E95B45",
}


def setup_style() -> str:
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
        }
    )
    return chosen or "matplotlib default"


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ordered(items: Iterable[str], preferred: list[str]) -> list[str]:
    values = list(dict.fromkeys(items))
    head = [item for item in preferred if item in values]
    tail = sorted(item for item in values if item not in head)
    return head + tail


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.align_labels()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


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


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", min_height: float = 0.0) -> None:
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * 0.012
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height) or height <= min_height:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + gap,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#334155",
            rotation=0,
        )


def grouped_bar(
    pivot: pd.DataFrame,
    title: str,
    ylabel: str,
    stem: str,
    colors: dict[str, str],
    labels: dict[str, str],
    ylim: tuple[float, float] | None = None,
    unavailable: bool = False,
    annotate: bool = True,
) -> None:
    pivot = pivot.copy()
    x_labels = pivot.index.astype(str).tolist()
    series = pivot.columns.astype(str).tolist()
    x = np.arange(len(x_labels))
    width = min(0.15, 0.78 / max(1, len(series)))

    fig, ax = plt.subplots(figsize=(14.8, 7.2))
    max_value = np.nanmax(pivot.to_numpy(dtype=float)) if pivot.size else 1.0
    if ylim is None:
        ylim = (0, max_value * 1.30 if max_value > 0 else 1)
    ax.set_ylim(*ylim)

    for idx, name in enumerate(series):
        values = pivot[name].astype(float).to_numpy()
        plot_values = np.nan_to_num(values, nan=0.0)
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            plot_values,
            width=width,
            label=labels.get(name, name),
            color=colors.get(name, "#64748B"),
            alpha=0.94,
        )
        if unavailable:
            for xi, value in zip(x + offset, values):
                if pd.isna(value):
                    ax.text(
                        xi,
                        ylim[1] * 0.03,
                        "不可行",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        rotation=90,
                        color="#B91C1C",
                    )
        if annotate:
            annotate_bars(ax, bars)

    ax.set_title(title, pad=56)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_legend_below(fig, ax, min(len(series), 6))
    fig.subplots_adjust(top=0.72, bottom=0.13, left=0.08, right=0.98)
    save_fig(fig, stem)


def plot_pmp_optimization() -> pd.DataFrame:
    df = read_csv("result/stk_dynamic/cross_model/stk_cross_model_summary_long.csv")
    models = ordered(df["model_label"].dropna().unique(), MODEL_ORDER)
    algs = ordered(df["algorithm"].dropna().unique(), PMP_ALG_ORDER)

    latency = (
        df.pivot_table(
            index="model_label",
            columns="algorithm",
            values="mean_norm_latency_vs_gs",
            aggfunc="mean",
        )
        .reindex(index=models, columns=algs)
    )
    grouped_bar(
        latency,
        "PMP 模式下不同算法的归一化时延对比",
        "归一化时延（相对 GS-Only）",
        "01_pmp_algorithm_latency_norm",
        ALG_COLOR,
        ALG_LABEL,
    )

    energy = (
        df.pivot_table(
            index="model_label",
            columns="algorithm",
            values="mean_satellite_energy_j",
            aggfunc="mean",
        )
        .reindex(index=models, columns=algs)
    )
    grouped_bar(
        energy,
        "PMP 模式下不同算法的平均星载能耗对比",
        "平均星载能耗 / J",
        "02_pmp_algorithm_energy",
        ALG_COLOR,
        ALG_LABEL,
    )

    best_rows = []
    for model in models:
        one = df[df["model_label"] == model].copy()
        one["mean_norm_latency_vs_gs"] = pd.to_numeric(one["mean_norm_latency_vs_gs"], errors="coerce")
        one["mean_satellite_energy_j"] = pd.to_numeric(one["mean_satellite_energy_j"], errors="coerce")
        lat_best = one.loc[one["mean_norm_latency_vs_gs"].idxmin()]
        ene_best = one.loc[one["mean_satellite_energy_j"].idxmin()]
        best_rows.append(
            {
                "模型": model,
                "PMP时延最优算法": ALG_LABEL.get(lat_best["algorithm"], lat_best["algorithm"]),
                "最小归一化时延": f"{lat_best['mean_norm_latency_vs_gs']:.3f}",
                "PMP能耗最优算法": ALG_LABEL.get(ene_best["algorithm"], ene_best["algorithm"]),
                "最小星载能耗(J)": f"{ene_best['mean_satellite_energy_j']:.2f}",
            }
        )
    return pd.DataFrame(best_rows)


def plot_cdp_optimization() -> pd.DataFrame:
    df = read_csv("result/v3.0/cdp_theoretical_results.csv")
    df["Latency"] = pd.to_numeric(df["Latency"], errors="coerce")
    algs = ordered(df["Algorithm"].dropna().unique(), CDP_ALG_ORDER)
    summary = (
        df.groupby("Algorithm", as_index=False)
        .agg(avg_latency_ms=("Latency", "mean"), std_latency_ms=("Latency", "std"))
        .set_index("Algorithm")
        .reindex(algs)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        summary["avg_latency_ms"],
        yerr=summary["std_latency_ms"],
        capsize=4,
        color=[ALG_COLOR.get(name, "#64748B") for name in summary["Algorithm"]],
        width=0.62,
        alpha=0.94,
    )
    ax.set_title("CDP 模式下不同数据分配算法的平均时延对比", pad=12)
    ax.set_ylabel("平均时延 / ms")
    ax.set_xticks(x)
    ax.set_xticklabels([ALG_LABEL.get(name, name) for name in summary["Algorithm"]], rotation=0)
    ax.grid(axis="y", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = float((summary["avg_latency_ms"] + summary["std_latency_ms"].fillna(0)).max())
    ax.set_ylim(0, ymax * 1.28)
    annotate_bars(ax, bars, fmt="{:.1f}")
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.10, right=0.98)
    save_fig(fig, "03_cdp_algorithm_latency")

    sat_only_baseline = float(summary.loc[summary["Algorithm"] == "Pass-Through", "avg_latency_ms"].iloc[0])
    summary["normalized_vs_sat_only"] = summary["avg_latency_ms"] / sat_only_baseline

    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    bars = ax.bar(
        x,
        summary["normalized_vs_sat_only"],
        color=[ALG_COLOR.get(name, "#64748B") for name in summary["Algorithm"]],
        width=0.62,
        alpha=0.94,
    )
    ax.set_title("CDP 模式下不同算法的归一化时延对比", pad=12)
    ax.set_ylabel("归一化时延（相对 Sat-Only）")
    ax.set_xticks(x)
    ax.set_xticklabels([ALG_LABEL.get(name, name) for name in summary["Algorithm"]], rotation=0)
    ax.set_ylim(0, max(1.08, summary["normalized_vs_sat_only"].max() * 1.28))
    ax.grid(axis="y", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_bars(ax, bars, fmt="{:.2f}")
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.10, right=0.98)
    save_fig(fig, "04_cdp_algorithm_latency_norm")

    best = summary.loc[summary["avg_latency_ms"].idxmin()]
    return pd.DataFrame(
        [
            {
                "CDP最优算法": ALG_LABEL.get(best["Algorithm"], best["Algorithm"]),
                "平均时延(ms)": f"{best['avg_latency_ms']:.2f}",
                "相对Sat-Only比例": f"{best['normalized_vs_sat_only']:.3f}",
            }
        ]
    )


def plot_mode_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    stage6 = read_csv("result/mode_selection/final_stage6_report/stage6_mode_summary.csv")
    models = ordered(stage6["model_label"].dropna().unique(), MODEL_ORDER)

    df = stage6[stage6["mode_family"].isin(BASE_MODE_ORDER)].copy()
    modes = [mode for mode in BASE_MODE_ORDER if mode in set(df["mode_family"])]

    latency = (
        df.pivot_table(
            index="model_label",
            columns="mode_family",
            values="avg_latency_ms",
            aggfunc="mean",
        )
        .reindex(index=models, columns=modes)
    )
    grouped_bar(
        latency,
        "不同推理模式的平均时延对比",
        "平均时延 / ms",
        "05_mode_latency_by_model",
        MODE_COLOR,
        MODE_LABEL,
        unavailable=True,
        annotate=False,
    )

    energy = (
        df.pivot_table(
            index="model_label",
            columns="mode_family",
            values="avg_satellite_energy_j",
            aggfunc="mean",
        )
        .reindex(index=models, columns=modes)
    )
    grouped_bar(
        energy,
        "不同推理模式的平均星载能耗对比",
        "平均星载能耗 / J",
        "06_mode_energy_by_model",
        MODE_COLOR,
        MODE_LABEL,
        unavailable=True,
        annotate=False,
    )

    completion = (
        df.pivot_table(
            index="model_label",
            columns="mode_family",
            values="feasible_rate",
            aggfunc="mean",
        )
        .reindex(index=models, columns=modes)
        * 100
    )
    grouped_bar(
        completion,
        "跨模式任务完成率对比",
        "任务完成率 / %",
        "07_mode_completion_by_model",
        MODE_COLOR,
        MODE_LABEL,
        ylim=(0, 112),
        annotate=True,
    )

    selector = read_csv("result/mode_selection/final_stage6_report/stage6_selector_distribution.csv")
    plot_selector_distribution(selector, models)

    batch_summary = read_csv("result/mode_selection/batch_boundary_stage6/batch_boundary_mode_summary.csv")
    batch_selector = read_csv("result/mode_selection/batch_boundary_stage6/batch_boundary_selector_distribution.csv")
    plot_batch_boundary(batch_summary, batch_selector)

    cdp_sensitivity = read_csv("result/mode_selection/cdp_sensitivity_yolo_stage6/cdp_sensitivity_summary.csv")
    plot_cdp_sensitivity(cdp_sensitivity)

    fixed = read_csv("result/mode_selection/fixed_mode_vs_fwms_stage6/fixed_mode_vs_fwms_summary.csv")
    plot_fixed_completion(fixed)

    boundary = []
    for model in models:
        one = stage6[stage6["model_label"] == model].copy()
        pmp = get_value(one, "PMP", "feasible_rate")
        cdp = get_value(one, "CDP", "feasible_rate")
        cdp_lat = get_value(one, "CDP", "avg_latency_ms")
        pmp_lat = get_value(one, "PMP", "avg_latency_ms")
        gs_lat = get_value(one, "GS-Only", "avg_latency_ms")
        if cdp == 0:
            conclusion = "CDP不可行，PMP/GS-Only承担保底"
        elif pd.notna(cdp_lat) and pd.notna(pmp_lat) and cdp_lat < pmp_lat:
            conclusion = "CDP可行时低时延占优"
        elif pd.notna(gs_lat) and pd.notna(pmp_lat) and gs_lat < pmp_lat:
            conclusion = "GS-Only时延较低，PMP主要作为资源保底"
        else:
            conclusion = "PMP在该模型下保持稳定可行"
        boundary.append(
            {
                "模型": model,
                "PMP完成率": pct(pmp),
                "CDP完成率": pct(cdp),
                "边界结论": conclusion,
            }
        )
    boundary_df = pd.DataFrame(boundary)

    return stage6, boundary_df


def get_value(df: pd.DataFrame, mode: str, col: str) -> float:
    rows = df[df["mode_family"] == mode]
    if rows.empty:
        return np.nan
    return pd.to_numeric(rows[col], errors="coerce").iloc[0]


def pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.0f}%"


def plot_selector_distribution(selector: pd.DataFrame, models: list[str]) -> None:
    selectors = ["FWMS-Feature", "Oracle-Min-Latency"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharey=True)
    for ax, selector_name in zip(axes, selectors):
        subset = selector[selector["selector_family"] == selector_name]
        x = np.arange(len(models))
        bottom = np.zeros(len(models))
        for mode in SELECTED_MODE_ORDER:
            values = []
            for model in models:
                rows = subset[(subset["model_label"] == model) & (subset["selected_mode"] == mode)]
                values.append(float(rows["ratio"].sum()) * 100 if not rows.empty else 0.0)
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.62,
                color=MODE_COLOR[mode],
                label=MODE_LABEL[mode],
                alpha=0.94,
            )
            bottom += np.array(values)
        ax.set_title(MODE_LABEL[selector_name], pad=12, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=12)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("选择比例 / %")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
    )
    fig.suptitle("FWMS 与最小时延 Oracle 的模式选择分布", fontsize=18, weight="bold")
    fig.subplots_adjust(bottom=0.13, top=0.73, wspace=0.16, left=0.08, right=0.98)
    save_fig(fig, "08_fwms_oracle_selection_distribution")


def plot_batch_boundary(batch_summary: pd.DataFrame, batch_selector: pd.DataFrame) -> None:
    cdp = batch_summary[batch_summary["mode_family"] == "CDP"].copy()
    cdp["batch_size"] = pd.to_numeric(cdp["batch_size"], errors="coerce").astype("Int64")
    cdp["feasible_rate"] = pd.to_numeric(cdp["feasible_rate"], errors="coerce")
    models = ordered(cdp["model_label"].dropna().unique(), MODEL_ORDER)
    batches = sorted(int(batch) for batch in cdp["batch_size"].dropna().unique())

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    x = np.arange(len(models))
    width = min(0.22, 0.72 / max(1, len(batches)))
    batch_colors = ["#244C85", "#E39D2D", "#2A8C88", "#8272B2"]
    for idx, batch in enumerate(batches):
        values = []
        for model in models:
            rows = cdp[(cdp["model_label"] == model) & (cdp["batch_size"] == batch)]
            values.append(float(rows["feasible_rate"].iloc[0]) * 100 if not rows.empty else np.nan)
        bars = ax.bar(
            x + (idx - (len(batches) - 1) / 2) * width,
            np.nan_to_num(values, nan=0.0),
            width=width,
            label=f"batch={batch}",
            color=batch_colors[idx % len(batch_colors)],
            alpha=0.94,
        )
        annotate_bars(ax, bars, fmt="{:.0f}")

    ax.set_title("不同模型与 batch 下的 CDP 可行性边界", pad=12)
    ax.set_ylabel("CDP 完成率 / %")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_legend_below(fig, ax, len(batches))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.98)
    save_fig(fig, "09_batch_cdp_feasibility")

    oracle = batch_selector[batch_selector["selector_family"] == "Oracle-Min-Latency"].copy()
    oracle["batch_size"] = pd.to_numeric(oracle["batch_size"], errors="coerce").astype("Int64")
    fig, axes = plt.subplots(1, len(batches), figsize=(15.8, 7.0), sharey=True)
    if len(batches) == 1:
        axes = [axes]
    for ax, batch in zip(axes, batches):
        subset = oracle[oracle["batch_size"] == batch]
        bottom = np.zeros(len(models))
        x = np.arange(len(models))
        for mode in SELECTED_MODE_ORDER:
            values = []
            for model in models:
                rows = subset[(subset["model_label"] == model) & (subset["selected_mode"] == mode)]
                values.append(float(rows["ratio"].sum()) * 100 if not rows.empty else 0.0)
            ax.bar(x, values, bottom=bottom, color=MODE_COLOR[mode], width=0.62, label=MODE_LABEL[mode])
            bottom += np.array(values)
        ax.set_title(f"batch={batch}", pad=12, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18)
        ax.grid(axis="y", alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Oracle 选择比例 / %")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
    )
    fig.suptitle("不同 batch 下最小时延模式的分布", fontsize=18, weight="bold")
    fig.subplots_adjust(bottom=0.18, top=0.72, wspace=0.18, left=0.08, right=0.98)
    save_fig(fig, "10_batch_oracle_selection_distribution")


def plot_cdp_sensitivity(summary: pd.DataFrame) -> None:
    batch = summary[summary["group"] == "batch"].copy()
    worker = summary[summary["group"] == "worker"].copy()
    batch["batch_size"] = pd.to_numeric(batch["batch_size"], errors="coerce")
    worker["cdp_max_workers"] = pd.to_numeric(worker["cdp_max_workers"], errors="coerce")
    for col in ["cdp_avg_latency_ms", "cdp_feasible_rate", "cdp_avg_energy_j"]:
        batch[col] = pd.to_numeric(batch[col], errors="coerce")
        worker[col] = pd.to_numeric(worker[col], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0))
    axes[0].plot(batch["batch_size"], batch["cdp_avg_latency_ms"], marker="o", color=MODE_COLOR["CDP"], linewidth=2.3)
    axes[0].set_title("CDP 对 batch 的时延敏感性", pad=10)
    axes[0].set_xlabel("batch 大小")
    axes[0].set_ylabel("平均时延 / ms")
    axes[0].grid(True, alpha=0.85)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    ax2 = axes[0].twinx()
    ax2.plot(batch["batch_size"], batch["cdp_feasible_rate"] * 100, marker="s", color="#64748B", linewidth=2.0)
    ax2.set_ylabel("完成率 / %")
    ax2.set_ylim(0, 110)

    axes[1].plot(worker["cdp_max_workers"], worker["cdp_avg_latency_ms"], marker="o", color=MODE_COLOR["CDP"], linewidth=2.3)
    axes[1].set_title("CDP 对 worker 数量的时延敏感性", pad=10)
    axes[1].set_xlabel("最大 worker 数")
    axes[1].set_ylabel("平均时延 / ms")
    axes[1].grid(True, alpha=0.85)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("CDP 模式边界敏感性分析", fontsize=18, weight="bold")
    fig.subplots_adjust(top=0.78, wspace=0.28, left=0.08, right=0.94, bottom=0.13)
    save_fig(fig, "11_cdp_boundary_sensitivity")


def parse_percent(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            return float(value[:-1])
        if value == "-":
            return np.nan
    return float(value)


def plot_fixed_completion(fixed: pd.DataFrame) -> None:
    fixed = fixed.copy()
    rows = []
    for _, row in fixed.iterrows():
        label = f"{row['模型']}\nB{int(row['batch'])}"
        for mode, col in [
            ("PMP", "Fixed-PMP完成率"),
            ("CDP", "Fixed-CDP完成率"),
            ("GS-Only", "Fixed-GS完成率"),
            ("Sat-Only", "Fixed-Sat完成率"),
            ("FWMS-Feature", "FWMS-Feature完成率"),
        ]:
            rows.append({"任务": label, "模式": mode, "完成率": parse_percent(row[col])})
    df = pd.DataFrame(rows)
    tasks = df["任务"].drop_duplicates().tolist()
    modes = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS-Feature"]
    pivot = df.pivot(index="任务", columns="模式", values="完成率").reindex(index=tasks, columns=modes)

    grouped_bar(
        pivot,
        "固定模式与 FWMS 的任务完成率对比",
        "任务完成率 / %",
        "12_fixed_mode_fwms_completion",
        MODE_COLOR,
        MODE_LABEL,
        ylim=(0, 112),
        annotate=False,
    )


def write_markdown_table(lines: list[str], df: pd.DataFrame) -> None:
    if df.empty:
        return
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    lines.append("")


def write_notes(pmp_best: pd.DataFrame, cdp_best: pd.DataFrame, boundary: pd.DataFrame, font_name: str) -> None:
    lines = [
        "# 论文实验图表索引与结论记录",
        "",
        f"- 绘图字体：`{font_name}`。",
        "- 这些图均由已有 CSV 重绘得到，没有重新运行仿真实验。",
        "- 图表顺序按论文叙事排列：PMP 优化、CDP 优化、模式选择与适用边界。",
        "- 全文建议固定使用下表色号，避免同一算法或模式在不同图里换颜色。",
        "",
        "## 0. 统一色号",
        "",
        "| 对象 | 色号 | 说明 |",
        "| --- | --- | --- |",
        "| LADP / LAWA / PMP | `#2563EB` | 本文核心优化算法或流水线模式 |",
        "| CDP / Greedy | `#E39D2D` | 数据并行模式或贪心算法 |",
        "| GS-Only | `#4A4A4A` | 地面站完整推理基线 |",
        "| Sat-Only | `#8272B2` | 单星完整推理基线 |",
        "| Uniform | `#9CA3AF` | 均匀分配基线 |",
        "| FWMS / GA | `#2A8C88` | 模式选择算法或遗传算法 |",
        "| Random / Oracle | `#E95B45` | 随机基线或最小时延上界 |",
        "| GA | `#059669` | 遗传算法基线 |",
        "| YOLOv5 | `#2563EB` | 模型作为图例对象时使用 |",
        "| ResNet101 | `#F59E0B` | 模型作为图例对象时使用 |",
        "| VGG19 | `#10B981` | 模型作为图例对象时使用 |",
        "| Swin-Base | `#7C3AED` | 模型作为图例对象时使用 |",
        "| ViT-Huge | `#EF4444` | 模型作为图例对象时使用 |",
        "",
        "## 1. PMP 优化实验",
        "",
        "- `01_pmp_algorithm_latency_norm.png/pdf`：PMP 模式下 LADP 与 Greedy、GA、Random、Uniform、GS-Only 的归一化时延对比。",
        "- `02_pmp_algorithm_energy.png/pdf`：PMP 模式下不同算法的星载能耗对比。",
        "",
    ]
    write_markdown_table(lines, pmp_best)

    lines += [
        "## 2. CDP 优化实验",
        "",
        "- `03_cdp_algorithm_latency.png/pdf`：CDP 模式下 LAWA 与数据分配基线算法的平均时延对比。",
        "- `04_cdp_algorithm_latency_norm.png/pdf`：CDP 模式下各算法相对 Sat-Only 基线的归一化时延。",
        "",
    ]
    write_markdown_table(lines, cdp_best)

    lines += [
        "## 3. 模式选择与适用边界",
        "",
        "- `05_mode_latency_by_model.png/pdf`：PMP、CDP、GS-Only、Sat-Only、FWMS 的跨模型时延对比。",
        "- `06_mode_energy_by_model.png/pdf`：不同模式的星载能耗对比。",
        "- `07_mode_completion_by_model.png/pdf`：不同模式在 STK 动态时间片下的任务完成率。",
        "- `08_fwms_oracle_selection_distribution.png/pdf`：FWMS 与最小时延 Oracle 的选择分布差异。",
        "- `09_batch_cdp_feasibility.png/pdf`：CDP 在不同模型和 batch 下的可行性边界。",
        "- `10_batch_oracle_selection_distribution.png/pdf`：不同 batch 下最小时延模式的分布。",
        "- `11_cdp_boundary_sensitivity.png/pdf`：CDP 对 batch 和 worker 数量的敏感性。",
        "- `12_fixed_mode_fwms_completion.png/pdf`：固定模式与 FWMS 的任务完成率对比。",
        "",
    ]
    write_markdown_table(lines, boundary)

    lines += [
        "## 4. 汇报用结论",
        "",
        "- PMP 内部优化结论：LADP 在 STK 动态拓扑下通常取得更低的归一化时延，并能降低星载能耗，说明模型切分优化是有效的。",
        "- CDP 内部优化结论：LAWA 在数据并行场景下明显优于 Sat-Only、均匀分配和贪心等基线，说明数据分配优化是有效的。",
        "- 模式边界结论：CDP 在可行且 batch 较大时低时延优势明显，但对单星内存和 worker 可见性更敏感；PMP 更稳定，适合作为大模型、资源受限或 CDP 不可行时的保底模式。",
        "- FWMS 叙事结论：FWMS 不应被表述为最小时延 Oracle，而应表述为结合模型特征、通信特征、内存约束和资源状态的模式边界判别方法。",
        "",
    ]
    (OUT / "paper_figures_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font_name = setup_style()
    pmp_best = plot_pmp_optimization()
    cdp_best = plot_cdp_optimization()
    _, boundary = plot_mode_selection()
    write_notes(pmp_best, cdp_best, boundary, font_name)
    print(f"[OK] paper-ready figures written to {OUT}")
    print(f"[OK] index written to {OUT / 'paper_figures_index.md'}")


if __name__ == "__main__":
    main()
