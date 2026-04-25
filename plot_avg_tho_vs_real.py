import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


ALGO_ORDER = ["LA-DP", "Greedy", "Random", "GA", "Uniform", "GS-Only"]
ALGO_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]


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


def draw_bar_chart(ax, mode_df, title):
    """Draw the algorithm comparison bar chart."""
    if mode_df is None or mode_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
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
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
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

    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Normalized latency (GS-Only=1.0)", fontsize=10)
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


def _bandwidth_x_column(exp_type):
    if exp_type == "isl_bandwidth_sensitivity":
        return "isl_avg_bw_mbps", "ISL bandwidth (Mbps)"
    if exp_type == "gsl_bandwidth_sensitivity":
        return "gsl_avg_bw_mbps", "GSL bandwidth (Mbps)"
    raise ValueError(f"Unsupported bandwidth exp_type: {exp_type}")


def draw_bandwidth_line_chart(ax, mode_df, exp_type, title):
    """Draw the bandwidth sensitivity line chart."""
    if mode_df is None or mode_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
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
    ax.set_ylabel("Normalized latency (GS-Only=1.0)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--")
    ax.legend(fontsize=8, loc="best")


def _plot_algo_effectiveness(df, run_id, output_path, show):
    theory_df = df[df["mode"] == "theory"]
    physical_df = df[df["mode"] == "physical"]

    mode_panels = []
    if not theory_df.empty:
        mode_panels.append(("theory", theory_df, "Theoretical comparison"))
    if not physical_df.empty:
        mode_panels.append(("physical", physical_df, "Physical comparison"))

    if not mode_panels:
        print("[WARN] No theory/physical data available for algo_effectiveness.")
        return None

    fig, axes = plt.subplots(1, len(mode_panels), figsize=(7 * len(mode_panels), 6))
    if len(mode_panels) == 1:
        axes = [axes]

    for ax, (_, mode_df, title) in zip(axes, mode_panels):
        draw_bar_chart(ax, mode_df, title)

    suptitle = f"run_id: {run_id} | exp_type: algo_effectiveness" if run_id else "exp_type: algo_effectiveness"
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


def _plot_bandwidth_sensitivity(df, exp_type, run_id, output_path, show):
    x_col, x_label = _bandwidth_x_column(exp_type)
    modes = [m for m in ["theory", "physical"] if m in df["mode"].unique().tolist()]
    if not modes:
        modes = sorted(df["mode"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]

    pretty_title = "ISL bandwidth sensitivity" if exp_type == "isl_bandwidth_sensitivity" else "GSL bandwidth sensitivity"
    for ax, mode in zip(axes, modes):
        mode_df = df[df["mode"] == mode]
        draw_bandwidth_line_chart(ax, mode_df, exp_type, f"{pretty_title} ({mode})")

    suptitle = f"run_id: {run_id} | exp_type: {exp_type}" if run_id else f"exp_type: {exp_type}"
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

    if detected_exp_type == "algo_effectiveness":
        return _plot_algo_effectiveness(df, run_id, output_path, show)

    if detected_exp_type in ("isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"):
        return _plot_bandwidth_sensitivity(df, detected_exp_type, run_id, output_path, show)

    raise ValueError(f"Unsupported exp_type: {detected_exp_type}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from results_long.csv")
    parser.add_argument("--results-csv", type=str, default="results_long.csv", help="Unified long-table CSV path")
    parser.add_argument("--run-id", type=str, default=None, help="Filter by run_id")
    parser.add_argument(
        "--exp-type",
        type=str,
        default="auto",
        choices=["auto", "algo_effectiveness", "isl_bandwidth_sensitivity", "gsl_bandwidth_sensitivity"],
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
