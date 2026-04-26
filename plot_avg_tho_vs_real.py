import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment_archive import (
    append_experiment_index,
    build_artifact_stem,
    create_run_archive,
    export_run_rows,
    find_run_archive,
    now_stamp,
    update_run_metadata,
)

warnings.filterwarnings("ignore")


ALGO_ORDER = ["LA-DP", "Greedy", "Random", "GA", "Uniform", "GS-Only"]
ALGO_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]
MODE_LABELS = {
    "theory": "理论",
    "physical": "实物",
    "hybrid": "混合",
}
EXP_TYPE_LABELS = {
    "algo_effectiveness": "算法有效性",
    "energy_comparison": "理论能耗对比",
    "isl_bandwidth_sensitivity": "ISL 带宽敏感性",
    "gsl_bandwidth_sensitivity": "GSL 带宽敏感性",
    "node_count_sensitivity": "节点数敏感性",
}


def _normalize_within_task_mode(df):
    """Fill missing normalized latency using GS-Only within the same task and mode."""
    work = df.copy()
    work["latency_ms"] = pd.to_numeric(work["latency_ms"], errors="coerce")
    work["norm_latency_vs_gs"] = pd.to_numeric(work["norm_latency_vs_gs"], errors="coerce")

    gs_rows = work[work["algorithm"] == "GS-Only"][["run_id", "task_id", "mode", "latency_ms"]].copy()
    gs_rows = gs_rows.rename(columns={"latency_ms": "gs_latency"})
    work = work.merge(gs_rows, on=["run_id", "task_id", "mode"], how="left")

    missing_norm = work["norm_latency_vs_gs"].isna()
    valid_gs = work["gs_latency"].notna() & (work["gs_latency"] != 0)
    valid_latency = work["latency_ms"].notna()
    fill_mask = missing_norm & valid_gs & valid_latency
    work.loc[fill_mask, "norm_latency_vs_gs"] = (
        work.loc[fill_mask, "latency_ms"] / work.loc[fill_mask, "gs_latency"]
    )

    gs_self = (work["algorithm"] == "GS-Only") & work["norm_latency_vs_gs"].isna()
    work.loc[gs_self, "norm_latency_vs_gs"] = 1.0

    return work.drop(columns=["gs_latency"])


def _normalize_energy_within_task_mode(df):
    """Fill missing normalized energy using GS-Only within the same task and mode."""
    required = {"satellite_energy_j", "norm_energy_vs_gs"}
    if not required.issubset(set(df.columns)):
        return df

    work = df.copy()
    work["satellite_energy_j"] = pd.to_numeric(work["satellite_energy_j"], errors="coerce")
    work["norm_energy_vs_gs"] = pd.to_numeric(work["norm_energy_vs_gs"], errors="coerce")

    gs_rows = work[work["algorithm"] == "GS-Only"][["run_id", "task_id", "mode", "satellite_energy_j"]].copy()
    gs_rows = gs_rows.rename(columns={"satellite_energy_j": "gs_energy"})
    work = work.merge(gs_rows, on=["run_id", "task_id", "mode"], how="left")

    missing_norm = work["norm_energy_vs_gs"].isna()
    valid_gs = work["gs_energy"].notna() & (work["gs_energy"] != 0)
    valid_energy = work["satellite_energy_j"].notna()
    fill_mask = missing_norm & valid_gs & valid_energy
    work.loc[fill_mask, "norm_energy_vs_gs"] = (
        work.loc[fill_mask, "satellite_energy_j"] / work.loc[fill_mask, "gs_energy"]
    )

    gs_self = (work["algorithm"] == "GS-Only") & work["norm_energy_vs_gs"].isna()
    work.loc[gs_self, "norm_energy_vs_gs"] = 1.0
    return work.drop(columns=["gs_energy"])


def load_long_results(csv_path="results_long.csv", run_id=None, exp_type=None):
    """Load the unified long table and optionally filter by run_id / exp_type."""
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    required_cols = {
        "run_id",
        "exp_type",
        "mode",
        "task_id",
        "algorithm",
        "model_name",
        "latency_ms",
        "norm_latency_vs_gs",
        "timestamp",
    }
    if not required_cols.issubset(set(df.columns)):
        missing = sorted(list(required_cols - set(df.columns)))
        print(f"[WARN] Missing required columns: {missing}")
        return None

    if run_id:
        df = df[df["run_id"] == run_id]
    elif not df.empty:
        latest_run_id = df.sort_values("timestamp").iloc[-1]["run_id"]
        df = df[df["run_id"] == latest_run_id]
        print(f"[INFO] No run_id specified, using latest run_id: {latest_run_id}")

    if exp_type and exp_type != "auto":
        df = df[df["exp_type"] == exp_type]

    if df.empty:
        print("[WARN] No data left after filtering.")
        return None

    df = _normalize_within_task_mode(df)
    df = _normalize_energy_within_task_mode(df)
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["run_id", "task_id", "mode", "algorithm"], keep="last")
        .reset_index(drop=True)
    )
    return df


def _infer_exp_type(df):
    unique_types = [x for x in df["exp_type"].dropna().unique().tolist() if str(x).strip()]
    if len(unique_types) == 1:
        return unique_types[0]
    if not df.empty:
        return df.sort_values("timestamp").iloc[-1]["exp_type"]
    return "algo_effectiveness"


def _preferred_algorithms(available):
    return [alg for alg in ALGO_ORDER if alg in available]


def _single_or_mixed(series):
    values = [v for v in series.dropna().unique().tolist()]
    if len(values) == 1:
        return values[0]
    return "mixed"


def _metadata_from_df(df, exp_type, run_id):
    first_timestamp = df["timestamp"].dropna().iloc[0] if "timestamp" in df and not df["timestamp"].dropna().empty else ""
    metadata = {
        "run_id": run_id,
        "started_at": first_timestamp,
        "started_at_compact": now_stamp(),
        "status": "plotted",
        "exp_type": exp_type,
        "exp_mode": _single_or_mixed(df["mode"]) if "mode" in df else "mixed",
        "fixed_model": _single_or_mixed(df["model_name"]) if "model_name" in df else "mixed",
        "fixed_batch_size": _single_or_mixed(df["batch_size"]) if "batch_size" in df else "mixed",
        "fixed_input_h": _single_or_mixed(df["input_h"]) if "input_h" in df else "mixed",
        "fixed_input_w": _single_or_mixed(df["input_w"]) if "input_w" in df else "mixed",
    }
    if exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"):
        x_col, _ = _bandwidth_x_column(exp_type)
        bw_values = sorted(pd.to_numeric(df[x_col], errors="coerce").dropna().unique().tolist())
        if bw_values:
            metadata["sweep_start"] = round(float(bw_values[0]), 6)
            metadata["sweep_stop"] = round(float(bw_values[-1]), 6)
            metadata["sweep_points"] = len(bw_values)
    if exp_type == "node_count_sensitivity":
        x_col, _ = _node_count_x_column(df)
        node_values = sorted(pd.to_numeric(df[x_col], errors="coerce").dropna().unique().tolist())
        if node_values:
            metadata["sweep_start"] = int(node_values[0])
            metadata["sweep_stop"] = int(node_values[-1])
            metadata["sweep_points"] = len(node_values)
    return metadata


def _resolve_archive(df, exp_type, run_id):
    archive_dir = find_run_archive(run_id)
    if archive_dir is not None:
        return archive_dir

    metadata = _metadata_from_df(df, exp_type, run_id)
    archive_dir = create_run_archive(metadata)
    append_experiment_index(metadata, archive_dir)
    return archive_dir


def _export_summary(df, exp_type, archive_dir, stem):
    archive_dir = os.path.abspath(archive_dir)
    summary_path = os.path.join(archive_dir, "data", f"summary_{stem}.csv")

    work = df.copy()
    work["latency_ms"] = pd.to_numeric(work["latency_ms"], errors="coerce")
    work["norm_latency_vs_gs"] = pd.to_numeric(work["norm_latency_vs_gs"], errors="coerce")
    for col in ["satellite_energy_j", "energy_compute_j", "energy_comm_j", "norm_energy_vs_gs"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"):
        x_col, _ = _bandwidth_x_column(exp_type)
        group_cols = ["run_id", "exp_type", "mode", x_col, "algorithm", "model_name", "batch_size", "input_h", "input_w"]
    elif exp_type == "node_count_sensitivity":
        x_col, _ = _node_count_x_column(work)
        group_cols = ["run_id", "exp_type", "mode", x_col, "algorithm", "model_name", "batch_size", "input_h", "input_w"]
    else:
        group_cols = ["run_id", "exp_type", "mode", "model_name", "algorithm"]

    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            mean_latency_ms=("latency_ms", "mean"),
            std_latency_ms=("latency_ms", "std"),
            mean_norm_latency_vs_gs=("norm_latency_vs_gs", "mean"),
            std_norm_latency_vs_gs=("norm_latency_vs_gs", "std"),
            mean_satellite_energy_j=("satellite_energy_j", "mean") if "satellite_energy_j" in work.columns else ("latency_ms", "count"),
            std_satellite_energy_j=("satellite_energy_j", "std") if "satellite_energy_j" in work.columns else ("latency_ms", "count"),
            mean_energy_compute_j=("energy_compute_j", "mean") if "energy_compute_j" in work.columns else ("latency_ms", "count"),
            mean_energy_comm_j=("energy_comm_j", "mean") if "energy_comm_j" in work.columns else ("latency_ms", "count"),
            mean_norm_energy_vs_gs=("norm_energy_vs_gs", "mean") if "norm_energy_vs_gs" in work.columns else ("latency_ms", "count"),
            std_norm_energy_vs_gs=("norm_energy_vs_gs", "std") if "norm_energy_vs_gs" in work.columns else ("latency_ms", "count"),
            samples=("norm_latency_vs_gs", "count"),
        )
        .reset_index()
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    return summary_path


def draw_bar_chart(ax, mode_df, title):
    """Draw the algorithm comparison bar chart."""
    if mode_df is None or mode_df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    labels = _preferred_algorithms(mode_df["algorithm"].unique())
    colors = ALGO_COLORS[: len(labels)]

    grouped = (
        mode_df.groupby(["model_name", "algorithm"])["norm_latency_vs_gs"]
        .mean()
        .reset_index()
        .dropna(subset=["norm_latency_vs_gs"])
    )
    active_models = sorted(grouped["model_name"].unique().tolist())
    if not active_models:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    means_dict = {lbl: [] for lbl in labels}
    for model_name in active_models:
        gm = grouped[grouped["model_name"] == model_name]
        gm_map = {row["algorithm"]: row["norm_latency_vs_gs"] for _, row in gm.iterrows()}
        for lbl in labels:
            means_dict[lbl].append(gm_map.get(lbl, np.nan))

    x = np.arange(len(active_models))
    width = 0.12
    n_algs = len(labels)

    bars_list = []
    for i, (lbl, color) in enumerate(zip(labels, colors)):
        offset = width * (i - (n_algs - 1) / 2)
        bars = ax.bar(
            x + offset,
            means_dict[lbl],
            width,
            label=lbl,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        bars_list.append(bars)

    ax.set_xlabel("模型", fontsize=10)
    ax.set_ylabel("归一化时延（GS-Only=1.0）", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(active_models, fontsize=10)
    ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="best")

    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def draw_energy_bar_chart(ax, mode_df, title):
    """Draw absolute satellite-side energy comparison in Joules."""
    if mode_df is None or mode_df.empty or "satellite_energy_j" not in mode_df.columns:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    labels = _preferred_algorithms(mode_df["algorithm"].unique())
    colors = ALGO_COLORS[: len(labels)]
    work = mode_df.copy()
    work["satellite_energy_j"] = pd.to_numeric(work["satellite_energy_j"], errors="coerce")
    grouped = (
        work.groupby("algorithm")["satellite_energy_j"]
        .mean()
        .reindex(labels)
        .dropna()
    )

    if grouped.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    x = np.arange(len(grouped.index))
    bars = ax.bar(
        x,
        grouped.values,
        width=0.55,
        color=[colors[labels.index(alg)] for alg in grouped.index],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xlabel("算法", fontsize=10)
    ax.set_ylabel("平均卫星能耗 (J)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _bandwidth_x_column(exp_type):
    if exp_type == "isl_bandwidth_sensitivity":
        return "isl_avg_bw_mbps", "ISL 平均带宽 (Mbps)"
    if exp_type == "gsl_bandwidth_sensitivity":
        return "gsl_avg_bw_mbps", "GSL 平均带宽 (Mbps)"
    raise ValueError(f"Unsupported bandwidth exp_type: {exp_type}")


def _node_count_x_column(df):
    if "pipeline_node_count" in df.columns:
        return "pipeline_node_count", "协作卫星数量"
    if "sweep_value" in df.columns:
        return "sweep_value", "协作卫星数量"
    raise ValueError("node_count_sensitivity requires pipeline_node_count or sweep_value column")


def draw_bandwidth_line_chart(ax, mode_df, exp_type, title):
    """Draw the bandwidth sensitivity line chart."""
    if mode_df is None or mode_df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    x_col, x_label = _bandwidth_x_column(exp_type)
    grouped = (
        mode_df.groupby([x_col, "algorithm"])["norm_latency_vs_gs"]
        .mean()
        .reset_index()
        .dropna(subset=[x_col, "norm_latency_vs_gs"])
        .sort_values(x_col)
    )

    available_algs = _preferred_algorithms(grouped["algorithm"].unique())
    if not available_algs:
        available_algs = sorted(grouped["algorithm"].unique().tolist())

    for idx, alg in enumerate(available_algs):
        alg_df = grouped[grouped["algorithm"] == alg].sort_values(x_col)
        if alg_df.empty:
            continue
        ax.plot(
            alg_df[x_col],
            alg_df["norm_latency_vs_gs"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=alg,
            color=ALGO_COLORS[idx % len(ALGO_COLORS)],
        )

        for _, row in alg_df.iterrows():
            ax.text(
                row[x_col],
                row["norm_latency_vs_gs"] + 0.02,
                f"{row['norm_latency_vs_gs']:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("归一化时延（GS-Only=1.0）", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--")
    ax.legend(fontsize=8, loc="best")


def draw_node_count_line_chart(ax, mode_df, title):
    """Draw the pipeline node-count sensitivity line chart."""
    if mode_df is None or mode_df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    x_col, x_label = _node_count_x_column(mode_df)
    work = mode_df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    grouped = (
        work.groupby([x_col, "algorithm"])["norm_latency_vs_gs"]
        .mean()
        .reset_index()
        .dropna(subset=[x_col, "norm_latency_vs_gs"])
        .sort_values(x_col)
    )

    available_algs = _preferred_algorithms(grouped["algorithm"].unique())
    if not available_algs:
        available_algs = sorted(grouped["algorithm"].unique().tolist())

    for idx, alg in enumerate(available_algs):
        alg_df = grouped[grouped["algorithm"] == alg].sort_values(x_col)
        if alg_df.empty:
            continue
        ax.plot(
            alg_df[x_col],
            alg_df["norm_latency_vs_gs"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=alg,
            color=ALGO_COLORS[idx % len(ALGO_COLORS)],
        )

        for _, row in alg_df.iterrows():
            ax.text(
                row[x_col],
                row["norm_latency_vs_gs"] + 0.02,
                f"{row['norm_latency_vs_gs']:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("归一化时延（GS-Only=1.0）", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(sorted(grouped[x_col].dropna().unique().tolist()))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--")
    ax.legend(fontsize=8, loc="best")


def _plot_algo_effectiveness(df, run_id, output_path, show):
    theory_df = df[df["mode"] == "theory"]
    physical_df = df[df["mode"] == "physical"]

    mode_panels = []
    if not theory_df.empty:
        mode_panels.append(("theory", theory_df, "理论算法对比"))
    if not physical_df.empty:
        mode_panels.append(("physical", physical_df, "实物算法对比"))

    if not mode_panels:
        print("[WARN] No theory/physical data available for algo_effectiveness.")
        return None

    fig, axes = plt.subplots(1, len(mode_panels), figsize=(7 * len(mode_panels), 6))
    if len(mode_panels) == 1:
        axes = [axes]

    for ax, (_, mode_df, title) in zip(axes, mode_panels):
        draw_bar_chart(ax, mode_df, title)

    suptitle = f"实验批次：{run_id} | 实验类型：算法有效性" if run_id else "实验类型：算法有效性"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    output_path = output_path or "theory_vs_experiment_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_energy_comparison(df, run_id, output_path, show):
    theory_df = df[df["mode"] == "theory"]
    physical_df = df[df["mode"] == "physical"]

    mode_panels = []
    if not theory_df.empty:
        mode_panels.append(("theory", theory_df, "理论卫星能耗对比"))
    if not physical_df.empty:
        mode_panels.append(("physical", physical_df, "实物卫星能耗对比"))

    if not mode_panels:
        print("[WARN] No theory/physical data available for energy_comparison.")
        return None

    fig, axes = plt.subplots(1, len(mode_panels), figsize=(7 * len(mode_panels), 6))
    if len(mode_panels) == 1:
        axes = [axes]

    for ax, (_, mode_df, title) in zip(axes, mode_panels):
        draw_energy_bar_chart(ax, mode_df, title)

    suptitle = f"实验批次：{run_id} | 实验类型：理论能耗对比" if run_id else "实验类型：理论能耗对比"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    output_path = output_path or "energy_comparison_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_bandwidth_sensitivity(df, exp_type, run_id, output_path, show):
    x_col, x_label = _bandwidth_x_column(exp_type)
    modes = [m for m in ["theory", "physical"] if m in df["mode"].unique().tolist()]
    if not modes:
        modes = sorted(df["mode"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]

    pretty_title = "ISL 带宽敏感性" if exp_type == "isl_bandwidth_sensitivity" else "GSL 带宽敏感性"
    for ax, mode in zip(axes, modes):
        mode_df = df[df["mode"] == mode]
        mode_label = MODE_LABELS.get(mode, mode)
        draw_bandwidth_line_chart(ax, mode_df, exp_type, f"{pretty_title}（{mode_label}）")

    exp_label = EXP_TYPE_LABELS.get(exp_type, exp_type)
    suptitle = f"实验批次：{run_id} | 实验类型：{exp_label}" if run_id else f"实验类型：{exp_label}"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    default_name = f"{exp_type}_analysis.png"
    output_path = output_path or default_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_node_count_sensitivity(df, run_id, output_path, show):
    modes = [m for m in ["theory", "physical"] if m in df["mode"].unique().tolist()]
    if not modes:
        modes = sorted(df["mode"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        mode_df = df[df["mode"] == mode]
        mode_label = MODE_LABELS.get(mode, mode)
        draw_node_count_line_chart(ax, mode_df, f"节点数敏感性（{mode_label}）")

    suptitle = f"实验批次：{run_id} | 实验类型：节点数敏感性" if run_id else "实验类型：节点数敏感性"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    output_path = output_path or "node_count_sensitivity_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_experiment_results(results_csv="results_long.csv", run_id=None, exp_type="auto", output_path=None, show=True):
    """Auto-dispatch plot type based on exp_type."""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    df = load_long_results(results_csv, run_id=run_id, exp_type=None if exp_type == "auto" else exp_type)
    if df is None or df.empty:
        return None

    detected_exp_type = exp_type
    if detected_exp_type == "auto":
        detected_exp_type = _infer_exp_type(df)
        print(f"[INFO] Auto-detected exp_type: {detected_exp_type}")

    actual_run_id = run_id or df["run_id"].iloc[0]
    archive_dir = _resolve_archive(df, detected_exp_type, actual_run_id)
    metadata = _metadata_from_df(df, detected_exp_type, actual_run_id)
    stem = build_artifact_stem(metadata)
    if output_path is None:
        output_path = os.path.join(archive_dir, "figures", f"{stem}.png")
    exported_data_path = os.path.join(archive_dir, "data", f"results_long_{stem}.csv")
    exported_rows = export_run_rows(results_csv, actual_run_id, exported_data_path)

    if detected_exp_type == "algo_effectiveness":
        figure_path = _plot_algo_effectiveness(df, actual_run_id, output_path, show)
    elif detected_exp_type == "energy_comparison":
        figure_path = _plot_energy_comparison(df, actual_run_id, output_path, show)
    elif detected_exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"):
        figure_path = _plot_bandwidth_sensitivity(df, detected_exp_type, actual_run_id, output_path, show)
    elif detected_exp_type == "node_count_sensitivity":
        figure_path = _plot_node_count_sensitivity(df, actual_run_id, output_path, show)
    else:
        raise ValueError(f"Unsupported exp_type: {detected_exp_type}")

    summary_path = _export_summary(df, detected_exp_type, archive_dir, stem)
    update_run_metadata(
        archive_dir,
        {
            "plot_status": "plotted",
            "last_plotted_at": now_stamp(),
            "figure_path": str(figure_path),
            "summary_csv": str(summary_path),
            "plot_exported_results_csv": str(exported_data_path),
            "plot_exported_rows": exported_rows,
        },
    )
    print(f"[OK] Saved summary to {summary_path}")
    return figure_path


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from results_long.csv")
    parser.add_argument("--results-csv", type=str, default="results_long.csv", help="Unified long-table CSV path")
    parser.add_argument("--run-id", type=str, default=None, help="Filter by run_id")
    parser.add_argument(
        "--exp-type",
        type=str,
        default="auto",
        choices=[
            "auto",
            "algo_effectiveness",
            "energy_comparison",
            "isl_bandwidth_sensitivity",
            "gsl_bandwidth_sensitivity",
            "node_count_sensitivity",
        ],
        help="Plot type selector",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--no-show", action="store_true", help="Save only, do not open a window")
    args = parser.parse_args()

    plot_experiment_results(
        results_csv=args.results_csv,
        run_id=args.run_id,
        exp_type=args.exp_type,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
