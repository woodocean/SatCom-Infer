import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGO_ORDER = ["LA-DP", "Greedy", "Random", "GA", "Uniform", "GS-Only"]
ALGO_COLORS = {
    "LA-DP": "#1f77b4",
    "Greedy": "#ff7f0e",
    "Random": "#2ca02c",
    "GA": "#9467bd",
    "Uniform": "#8c564b",
    "GS-Only": "#d62728",
}
MODEL_ORDER = ["yolov5", "resnet101", "vgg19", "swin_base", "vit_huge", "convnext_xxl"]
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "resnet101": "ResNet101",
    "vgg19": "VGG19",
    "swin_base": "Swin-Base",
    "vit_huge": "ViT-Huge",
    "convnext_xxl": "ConvNeXt-XXL",
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_existing_path(path_text):
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    candidate = Path.cwd() / path_text
    if candidate.exists():
        return candidate
    return None


def discover_latest_stk_runs(runs_dir, model_order):
    runs_dir = Path(runs_dir)
    latest_by_model = {}

    for metadata_path in runs_dir.glob("*/metadata.json"):
        try:
            metadata = _load_json(metadata_path)
        except Exception:
            continue

        if metadata.get("exp_type") != "stk_dynamic_pmp":
            continue

        model_name = metadata.get("fixed_model") or metadata.get("model_name")
        if model_name not in model_order:
            continue

        summary_path = _resolve_existing_path(metadata.get("summary_csv"))
        if summary_path is None:
            continue

        sort_key = (
            metadata.get("last_plotted_at")
            or metadata.get("started_at_compact")
            or metadata.get("started_at")
            or ""
        )
        current = {
            "model_name": model_name,
            "run_id": metadata.get("run_id", ""),
            "summary_csv": summary_path,
            "metadata_path": metadata_path,
            "sort_key": str(sort_key),
        }
        previous = latest_by_model.get(model_name)
        if previous is None or current["sort_key"] > previous["sort_key"]:
            latest_by_model[model_name] = current

    missing = [model for model in model_order if model not in latest_by_model]
    if missing:
        raise FileNotFoundError(f"Missing STK summary runs for models: {', '.join(missing)}")

    return [latest_by_model[model] for model in model_order]


def load_cross_model_summary(run_records):
    frames = []

    for record in run_records:
        df = pd.read_csv(record["summary_csv"])
        df["source_run_id"] = record["run_id"]
        df["source_summary_csv"] = str(record["summary_csv"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "mean_norm_latency_vs_gs",
        "std_norm_latency_vs_gs",
        "mean_satellite_energy_j",
        "std_satellite_energy_j",
        "mean_latency_ms",
        "std_latency_ms",
        "pipeline_node_count",
        "pipeline_hop_count",
        "samples",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    aggregated = (
        combined.groupby(["model_name", "algorithm"], dropna=False, observed=False)
        .agg(
            mean_norm_latency_vs_gs=("mean_norm_latency_vs_gs", "mean"),
            std_norm_latency_vs_gs=("mean_norm_latency_vs_gs", "std"),
            mean_satellite_energy_j=("mean_satellite_energy_j", "mean"),
            std_satellite_energy_j=("mean_satellite_energy_j", "std"),
            mean_latency_ms=("mean_latency_ms", "mean"),
            slot_count=("sweep_value", "nunique"),
        )
        .reset_index()
    )

    aggregated["model_name"] = pd.Categorical(aggregated["model_name"], categories=MODEL_ORDER, ordered=True)
    aggregated["algorithm"] = pd.Categorical(aggregated["algorithm"], categories=ALGO_ORDER, ordered=True)
    aggregated = aggregated.sort_values(["model_name", "algorithm"]).reset_index(drop=True)

    aggregated["latency_rank"] = (
        aggregated.groupby("model_name", observed=False)["mean_norm_latency_vs_gs"].rank(
            method="min", ascending=True
        )
    )
    aggregated["energy_rank"] = (
        aggregated.groupby("model_name", observed=False)["mean_satellite_energy_j"].rank(
            method="min", ascending=True
        )
    )
    aggregated["model_label"] = aggregated["model_name"].map(MODEL_LABELS)
    return aggregated


def build_best_by_model(aggregated):
    records = []
    for model_name in MODEL_ORDER:
        model_df = aggregated[aggregated["model_name"] == model_name].copy()
        if model_df.empty:
            continue

        latency_row = model_df.sort_values(
            ["mean_norm_latency_vs_gs", "mean_satellite_energy_j", "algorithm"]
        ).iloc[0]
        energy_row = model_df.sort_values(
            ["mean_satellite_energy_j", "mean_norm_latency_vs_gs", "algorithm"]
        ).iloc[0]

        records.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, str(model_name)),
                "best_latency_algorithm": latency_row["algorithm"],
                "best_latency_value": latency_row["mean_norm_latency_vs_gs"],
                "best_energy_algorithm": energy_row["algorithm"],
                "best_energy_value_j": energy_row["mean_satellite_energy_j"],
            }
        )

    return pd.DataFrame(records)


def build_algorithm_stability(aggregated):
    stability = (
        aggregated.groupby("algorithm", dropna=False, observed=False)
        .agg(
            avg_norm_latency_vs_gs=("mean_norm_latency_vs_gs", "mean"),
            std_norm_latency_vs_gs_across_models=("mean_norm_latency_vs_gs", "std"),
            avg_satellite_energy_j=("mean_satellite_energy_j", "mean"),
            std_satellite_energy_j_across_models=("mean_satellite_energy_j", "std"),
        )
        .reset_index()
    )
    stability["algorithm"] = pd.Categorical(stability["algorithm"], categories=ALGO_ORDER, ordered=True)
    stability = stability.sort_values("algorithm").reset_index(drop=True)
    return stability


def _pivot_metric_table(aggregated, metric_col):
    pivot = aggregated.pivot(index="model_name", columns="algorithm", values=metric_col)
    pivot = pivot.reindex(index=MODEL_ORDER, columns=ALGO_ORDER)
    pivot.index = [MODEL_LABELS.get(model, str(model)) for model in pivot.index]
    return pivot


def _write_markdown_table(f, title, pivot, decimals):
    f.write(f"## {title}\n\n")
    header = ["Model"] + ALGO_ORDER
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join(["---"] * len(header)) + "|\n")

    for model_label, row in pivot.iterrows():
        numeric_row = pd.to_numeric(row, errors="coerce")
        valid_values = numeric_row.dropna()
        best_value = valid_values.min() if not valid_values.empty else np.nan

        formatted_cells = [model_label]
        for alg in ALGO_ORDER:
            value = numeric_row.get(alg, np.nan)
            if pd.isna(value):
                formatted_cells.append("")
                continue
            cell = f"{value:.{decimals}f}"
            if not pd.isna(best_value) and np.isclose(value, best_value, rtol=1e-9, atol=1e-12):
                cell = f"**{cell}**"
            formatted_cells.append(cell)

        f.write("| " + " | ".join(formatted_cells) + " |\n")
    f.write("\n")


def write_paper_tables(aggregated, best_by_model, output_path):
    latency_pivot = _pivot_metric_table(aggregated, "mean_norm_latency_vs_gs")
    energy_pivot = _pivot_metric_table(aggregated, "mean_satellite_energy_j")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# STK Cross-Model Summary\n\n")
        f.write("Lower is better for both normalized latency and satellite energy.\n\n")
        _write_markdown_table(f, "Average Normalized Latency (GS-Only=1.0)", latency_pivot, decimals=3)
        _write_markdown_table(f, "Average Satellite Energy (J)", energy_pivot, decimals=3)

        f.write("## Best Algorithm Per Model\n\n")
        f.write("| Model | Best Latency Algo | Best Latency | Best Energy Algo | Best Energy (J) |\n")
        f.write("|---|---|---:|---|---:|\n")
        for _, row in best_by_model.iterrows():
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row["model_label"]),
                        str(row["best_latency_algorithm"]),
                        f"{row['best_latency_value']:.3f}",
                        str(row["best_energy_algorithm"]),
                        f"{row['best_energy_value_j']:.3f}",
                    ]
                )
                + " |\n"
            )


def export_wide_csvs(aggregated, output_dir):
    latency_pivot = _pivot_metric_table(aggregated, "mean_norm_latency_vs_gs").reset_index().rename(
        columns={"index": "model_label"}
    )
    energy_pivot = _pivot_metric_table(aggregated, "mean_satellite_energy_j").reset_index().rename(
        columns={"index": "model_label"}
    )

    latency_path = output_dir / "stk_cross_model_latency_wide.csv"
    energy_path = output_dir / "stk_cross_model_energy_wide.csv"
    latency_pivot.to_csv(latency_path, index=False, encoding="utf-8")
    energy_pivot.to_csv(energy_path, index=False, encoding="utf-8")
    return latency_path, energy_path


def draw_cross_model_plot(aggregated, output_path, show=False):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(MODEL_ORDER))
    model_labels = [MODEL_LABELS.get(model, model) for model in MODEL_ORDER]

    metric_specs = [
        (
            axes[0],
            "mean_norm_latency_vs_gs",
            "std_norm_latency_vs_gs",
            "Average Normalized Latency (GS-Only=1.0)",
            1.0,
        ),
        (
            axes[1],
            "mean_satellite_energy_j",
            "std_satellite_energy_j",
            "Average Satellite Energy (J)",
            None,
        ),
    ]

    for ax, mean_col, std_col, title, baseline in metric_specs:
        for alg in ALGO_ORDER:
            alg_df = aggregated[aggregated["algorithm"] == alg].copy()
            alg_df = alg_df.set_index("model_name").reindex(MODEL_ORDER)
            y = pd.to_numeric(alg_df[mean_col], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(alg_df[std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=2,
                markersize=5,
                capsize=3,
                label=alg,
                color=ALGO_COLORS.get(alg, "#333333"),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if baseline is not None:
            ax.axhline(y=baseline, color="black", linewidth=1, linestyle="--")

    axes[0].set_ylabel("Normalized Latency")
    axes[1].set_ylabel("Satellite Energy (J)")
    axes[1].legend(fontsize=8, loc="best")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("STK Dynamic Topology Cross-Model Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build cross-model STK summary tables and comparison plot")
    parser.add_argument("--runs-dir", type=str, default="result/runs", help="Directory containing run archives")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/stk_dynamic/cross_model",
        help="Directory for the cross-model summary outputs",
    )
    parser.add_argument("--show", action="store_true", help="Display the generated figure")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records = discover_latest_stk_runs(args.runs_dir, MODEL_ORDER)
    aggregated = load_cross_model_summary(run_records)
    best_by_model = build_best_by_model(aggregated)
    stability = build_algorithm_stability(aggregated)

    long_csv_path = output_dir / "stk_cross_model_summary_long.csv"
    best_csv_path = output_dir / "stk_cross_model_best_by_model.csv"
    stability_csv_path = output_dir / "stk_cross_model_algorithm_stability.csv"
    markdown_path = output_dir / "stk_cross_model_summary_table.md"
    figure_path = output_dir / "stk_cross_model_comparison.png"

    aggregated.to_csv(long_csv_path, index=False, encoding="utf-8")
    best_by_model.to_csv(best_csv_path, index=False, encoding="utf-8")
    stability.to_csv(stability_csv_path, index=False, encoding="utf-8")
    latency_wide_path, energy_wide_path = export_wide_csvs(aggregated, output_dir)
    write_paper_tables(aggregated, best_by_model, markdown_path)
    draw_cross_model_plot(aggregated, figure_path, show=args.show)

    run_manifest = pd.DataFrame(run_records)[["model_name", "run_id", "summary_csv"]]
    manifest_path = output_dir / "stk_cross_model_run_manifest.csv"
    run_manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    print(f"[OK] Saved long summary to {long_csv_path}")
    print(f"[OK] Saved latency wide table to {latency_wide_path}")
    print(f"[OK] Saved energy wide table to {energy_wide_path}")
    print(f"[OK] Saved best-by-model table to {best_csv_path}")
    print(f"[OK] Saved algorithm stability table to {stability_csv_path}")
    print(f"[OK] Saved markdown table to {markdown_path}")
    print(f"[OK] Saved figure to {figure_path}")
    print(f"[OK] Saved run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
