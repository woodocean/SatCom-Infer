import argparse
import glob
import os
import re
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
    "stk_dynamic_pmp": "STK 动态拓扑（理论）",
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


def _same_path(left, right):
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _find_latest_stk_results_csv():
    pattern = os.path.join("result", "stk_dynamic", "*", "results_long_stk_dynamic.csv")
    candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_results_csv_path(results_csv):
    candidate = results_csv or "results_long.csv"
    if os.path.isdir(candidate):
        stk_csv = os.path.join(candidate, "results_long_stk_dynamic.csv")
        if os.path.exists(stk_csv):
            return stk_csv
        generic_csv = os.path.join(candidate, "results_long.csv")
        if os.path.exists(generic_csv):
            return generic_csv
    return candidate


def _default_output_path(results_csv, exp_type, archive_dir, stem):
    resolved_csv = os.path.abspath(results_csv)
    if exp_type == "stk_dynamic_pmp" and os.path.basename(resolved_csv) == "results_long_stk_dynamic.csv":
        return os.path.join(os.path.dirname(resolved_csv), "stk_dynamic_plot.png")
    return os.path.join(archive_dir, "figures", f"{stem}.png")


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
    if exp_type == "stk_dynamic_pmp":
        slot_col = "sweep_value" if "sweep_value" in df.columns else "task_id"
        slot_values = [v for v in df[slot_col].dropna().astype(str).unique().tolist() if v]
        if slot_values:
            metadata["sweep_param"] = "time_slot"
            metadata["sweep_points"] = len(slot_values)
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
    elif exp_type == "stk_dynamic_pmp":
        slot_col = "sweep_value" if "sweep_value" in work.columns else "task_id"
        group_cols = [
            "run_id",
            "exp_type",
            "mode",
            slot_col,
            "algorithm",
            "model_name",
            "batch_size",
            "input_h",
            "input_w",
        ]
        for extra_col in ["pipeline_node_count", "pipeline_hop_count", "isl_avg_bw_mbps", "gsl_avg_bw_mbps"]:
            if extra_col in work.columns:
                group_cols.append(extra_col)
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


def _slot_index_from_label(value):
    match = re.search(r"slot_(\d+)", str(value))
    if not match:
        return np.nan
    return int(match.group(1))


def _resolve_stk_slot_column(df):
    if "sweep_value" in df.columns and not df["sweep_value"].dropna().empty:
        return "sweep_value", "时间片"
    if "task_id" in df.columns and not df["task_id"].dropna().empty:
        return "task_id", "任务"
    raise ValueError("stk_dynamic_pmp requires sweep_value or task_id column")


def _build_stk_slot_frame(mode_df):
    slot_col, slot_label = _resolve_stk_slot_column(mode_df)
    work = mode_df.copy()
    work["_slot_label"] = work[slot_col].astype(str)
    work["_slot_idx"] = work["_slot_label"].map(_slot_index_from_label)

    if work["_slot_idx"].isna().all():
        unique_labels = work["_slot_label"].drop_duplicates().tolist()
        fallback_map = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        work["_slot_idx"] = work["_slot_label"].map(fallback_map).astype(float)
    elif work["_slot_idx"].isna().any():
        start_idx = int(np.nanmax(work["_slot_idx"])) + 1
        missing_labels = work.loc[work["_slot_idx"].isna(), "_slot_label"].drop_duplicates().tolist()
        missing_map = {label: start_idx + idx for idx, label in enumerate(missing_labels)}
        work.loc[work["_slot_idx"].isna(), "_slot_idx"] = (
            work.loc[work["_slot_idx"].isna(), "_slot_label"].map(missing_map)
        )

    return work, slot_label


def draw_stk_dynamic_line_chart(ax, mode_df, metric_col, y_label, title, baseline=None):
    """Draw STK dynamic trend chart across time slots."""
    if mode_df is None or mode_df.empty or metric_col not in mode_df.columns:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    work, x_label = _build_stk_slot_frame(mode_df)
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    grouped = (
        work.groupby(["_slot_idx", "_slot_label", "algorithm"])[metric_col]
        .mean()
        .reset_index()
        .dropna(subset=[metric_col])
        .sort_values(["_slot_idx", "_slot_label"])
    )

    if grouped.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    available_algs = _preferred_algorithms(grouped["algorithm"].unique())
    if not available_algs:
        available_algs = sorted(grouped["algorithm"].unique().tolist())

    for idx, alg in enumerate(available_algs):
        alg_df = grouped[grouped["algorithm"] == alg].sort_values("_slot_idx")
        if alg_df.empty:
            continue
        ax.plot(
            alg_df["_slot_idx"],
            alg_df[metric_col],
            marker="o",
            linewidth=2,
            markersize=4,
            label=alg,
            color=ALGO_COLORS[idx % len(ALGO_COLORS)],
        )

    slot_ticks_df = grouped[["_slot_idx", "_slot_label"]].drop_duplicates().sort_values("_slot_idx")
    tick_values = slot_ticks_df["_slot_idx"].tolist()
    tick_labels = slot_ticks_df["_slot_label"].astype(str).tolist()
    tick_step = max(1, int(np.ceil(len(tick_values) / 10)))

    ax.set_xticks(tick_values[::tick_step])
    ax.set_xticklabels(tick_labels[::tick_step], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    if baseline is not None:
        ax.axhline(y=baseline, color="black", linewidth=1, linestyle="--")
    ax.legend(fontsize=8, loc="best")


def draw_stk_dynamic_mean_bar_chart(ax, mode_df, metric_col, y_label, title, baseline=None):
    """Draw STK mean bar chart aggregated over all time slots."""
    if mode_df is None or mode_df.empty or metric_col not in mode_df.columns:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    work = mode_df.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    grouped = work.groupby("algorithm")[metric_col].mean().dropna()

    if grouped.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return

    ordered_algs = _preferred_algorithms(grouped.index.tolist())
    if not ordered_algs:
        ordered_algs = sorted(grouped.index.tolist())
    grouped = grouped.reindex(ordered_algs).dropna()

    x = np.arange(len(grouped.index))
    bars = ax.bar(
        x,
        grouped.values,
        width=0.6,
        color=[ALGO_COLORS[i % len(ALGO_COLORS)] for i in range(len(grouped.index))],
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if baseline is not None:
        ax.axhline(y=baseline, color="black", linewidth=1, linestyle="--")

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


def _plot_stk_dynamic_pmp(df, run_id, output_path, show):
    modes = [m for m in ["theory", "physical"] if m in df["mode"].unique().tolist()]
    if not modes:
        modes = sorted(df["mode"].dropna().unique().tolist())

    if not modes:
        print("[WARN] No data available for stk_dynamic_pmp.")
        return None

    energy_ready = (
        "satellite_energy_j" in df.columns
        and pd.to_numeric(df["satellite_energy_j"], errors="coerce").notna().any()
    )

    right_metric = "satellite_energy_j" if energy_ready else "latency_ms"
    right_ylabel = "卫星能耗 (J)" if energy_ready else "端到端时延 (ms)"
    right_title = "STK 动态卫星能耗趋势" if energy_ready else "STK 动态端到端时延趋势"
    right_mean_title = "STK 全时片平均卫星能耗" if energy_ready else "STK 全时片平均端到端时延"

    fig, axes = plt.subplots(len(modes), 4, figsize=(25, 4.5 * len(modes)), squeeze=False)

    for row_idx, mode in enumerate(modes):
        mode_df = df[df["mode"] == mode]
        mode_label = MODE_LABELS.get(mode, mode)

        draw_stk_dynamic_line_chart(
            axes[row_idx][0],
            mode_df,
            "norm_latency_vs_gs",
            "归一化时延（GS-Only=1.0）",
            f"STK 动态归一化时延（{mode_label}）",
            baseline=1.0,
        )
        draw_stk_dynamic_line_chart(
            axes[row_idx][1],
            mode_df,
            right_metric,
            right_ylabel,
            f"{right_title}（{mode_label}）",
        )
        draw_stk_dynamic_mean_bar_chart(
            axes[row_idx][2],
            mode_df,
            "norm_latency_vs_gs",
            "平均归一化时延（GS-Only=1.0）",
            f"STK 全时片平均归一化时延（{mode_label}）",
            baseline=1.0,
        )
        draw_stk_dynamic_mean_bar_chart(
            axes[row_idx][3],
            mode_df,
            right_metric,
            f"平均{right_ylabel}",
            f"{right_mean_title}（{mode_label}）",
        )

    suptitle = f"实验批次：{run_id} | 实验类型：STK 动态拓扑" if run_id else "实验类型：STK 动态拓扑"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    output_path = output_path or "stk_dynamic_pmp_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_stk_dynamic_pmp_split(df, run_id, output_path, show):
    modes = [m for m in ["theory", "physical"] if m in df["mode"].unique().tolist()]
    if not modes:
        modes = sorted(df["mode"].dropna().unique().tolist())

    if not modes:
        print("[WARN] No data available for stk_dynamic_pmp.")
        return None

    energy_ready = (
        "satellite_energy_j" in df.columns
        and pd.to_numeric(df["satellite_energy_j"], errors="coerce").notna().any()
    )

    right_metric = "satellite_energy_j" if energy_ready else "latency_ms"
    right_ylabel = "Satellite Energy (J)" if energy_ready else "End-to-End Latency (ms)"
    right_title = "STK Dynamic Satellite Energy" if energy_ready else "STK Dynamic End-to-End Latency"
    right_mean_title = "STK Mean Satellite Energy" if energy_ready else "STK Mean End-to-End Latency"

    base_output_path = output_path or "stk_dynamic_pmp_analysis.png"
    output_root, output_ext = os.path.splitext(base_output_path)
    if not output_ext:
        output_ext = ".png"
    line_output_path = f"{output_root}_lines{output_ext}"
    bar_output_path = f"{output_root}_bars{output_ext}"

    line_fig, line_axes = plt.subplots(len(modes), 2, figsize=(14, 4.5 * len(modes)), squeeze=False)
    bar_fig, bar_axes = plt.subplots(len(modes), 2, figsize=(14, 4.5 * len(modes)), squeeze=False)

    for row_idx, mode in enumerate(modes):
        mode_df = df[df["mode"] == mode]
        mode_label = MODE_LABELS.get(mode, mode)

        draw_stk_dynamic_line_chart(
            line_axes[row_idx][0],
            mode_df,
            "norm_latency_vs_gs",
            "Normalized Latency (GS-Only=1.0)",
            f"STK Dynamic Normalized Latency ({mode_label})",
            baseline=1.0,
        )
        draw_stk_dynamic_line_chart(
            line_axes[row_idx][1],
            mode_df,
            right_metric,
            right_ylabel,
            f"{right_title} ({mode_label})",
        )
        draw_stk_dynamic_mean_bar_chart(
            bar_axes[row_idx][0],
            mode_df,
            "norm_latency_vs_gs",
            "Mean Normalized Latency (GS-Only=1.0)",
            f"STK Mean Normalized Latency ({mode_label})",
            baseline=1.0,
        )
        draw_stk_dynamic_mean_bar_chart(
            bar_axes[row_idx][1],
            mode_df,
            right_metric,
            f"Mean {right_ylabel}",
            f"{right_mean_title} ({mode_label})",
        )

    title_prefix = f"Run: {run_id}" if run_id else "STK Dynamic Topology"
    line_fig.suptitle(f"{title_prefix} | Trend Views", fontsize=11)
    bar_fig.suptitle(f"{title_prefix} | Summary Bars", fontsize=11)
    line_fig.tight_layout(rect=[0, 0, 1, 0.96])
    bar_fig.tight_layout(rect=[0, 0, 1, 0.96])

    line_fig.savefig(line_output_path, dpi=300, bbox_inches="tight")
    bar_fig.savefig(bar_output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to {line_output_path}")
    print(f"[OK] Saved figure to {bar_output_path}")

    if show:
        plt.show()

    plt.close(line_fig)
    plt.close(bar_fig)
    return {"line": line_output_path, "bar": bar_output_path}


def plot_experiment_results(results_csv="results_long.csv", run_id=None, exp_type="auto", output_path=None, show=True):
    """Auto-dispatch plot type based on exp_type."""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    resolved_results_csv = _resolve_results_csv_path(results_csv)
    df = load_long_results(resolved_results_csv, run_id=run_id, exp_type=None if exp_type == "auto" else exp_type)

    if (
        (df is None or df.empty)
        and exp_type == "stk_dynamic_pmp"
        and _same_path(resolved_results_csv, _resolve_results_csv_path("results_long.csv"))
    ):
        fallback_csv = _find_latest_stk_results_csv()
        if fallback_csv and not _same_path(fallback_csv, resolved_results_csv):
            print(f"[INFO] Retrying with latest STK results CSV: {fallback_csv}")
            resolved_results_csv = fallback_csv
            df = load_long_results(
                resolved_results_csv,
                run_id=run_id,
                exp_type=None if exp_type == "auto" else exp_type,
            )

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
        output_path = _default_output_path(resolved_results_csv, detected_exp_type, archive_dir, stem)
    exported_data_path = os.path.join(archive_dir, "data", f"results_long_{stem}.csv")
    exported_rows = export_run_rows(resolved_results_csv, actual_run_id, exported_data_path)

    if detected_exp_type == "algo_effectiveness":
        figure_path = _plot_algo_effectiveness(df, actual_run_id, output_path, show)
    elif detected_exp_type == "energy_comparison":
        figure_path = _plot_energy_comparison(df, actual_run_id, output_path, show)
    elif detected_exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"):
        figure_path = _plot_bandwidth_sensitivity(df, detected_exp_type, actual_run_id, output_path, show)
    elif detected_exp_type == "node_count_sensitivity":
        figure_path = _plot_node_count_sensitivity(df, actual_run_id, output_path, show)
    elif detected_exp_type == "stk_dynamic_pmp":
        figure_path = _plot_stk_dynamic_pmp_split(df, actual_run_id, output_path, show)
    else:
        raise ValueError(f"Unsupported exp_type: {detected_exp_type}")

    summary_path = _export_summary(df, detected_exp_type, archive_dir, stem)
    metadata_updates = {
        "plot_status": "plotted",
        "last_plotted_at": now_stamp(),
        "summary_csv": str(summary_path),
        "plot_exported_results_csv": str(exported_data_path),
        "plot_exported_rows": exported_rows,
    }
    if isinstance(figure_path, dict):
        normalized_paths = {key: str(value) for key, value in figure_path.items()}
        metadata_updates["figure_path"] = normalized_paths.get("line") or next(iter(normalized_paths.values()))
        metadata_updates["figure_paths"] = normalized_paths
    else:
        metadata_updates["figure_path"] = str(figure_path)
    update_run_metadata(
        archive_dir,
        metadata_updates,
    )
    print(f"[OK] Saved summary to {summary_path}")
    return figure_path


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from results_long.csv")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results_long.csv",
        help="Unified long-table CSV path, or a result directory that contains results_long*.csv",
    )
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
            "stk_dynamic_pmp",
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
