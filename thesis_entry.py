"""Unified thesis experiment and utility entrypoint.

This script keeps the root directory clean and exposes one stable CLI for:
1. Main experiment entrypoints.
2. Plotting/summary utilities under ``tools``.
3. Paper-facing rerun commands used during acceptance.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from algorithms.cdp_solver import CDPSolver
from core.experiment_archive import find_run_archive
from tools.plot_runs_sensitivity_figures import setup_style


ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "resnet101": "ResNet101",
    "vgg19": "VGG19",
    "vit_huge": "ViT-Huge",
}
MODEL_INPUTS = {
    "yolov5": (640, 640),
    "resnet101": (224, 224),
    "vgg19": (224, 224),
    "vit_huge": (224, 224),
}
CDP_ALG_ORDER = ["LAWA", "Greedy", "Uniform", "Random", "Sat-Only"]
CDP_ALG_LABEL = {
    "LAWA": "LAWA",
    "Greedy": "贪心",
    "Uniform": "均匀",
    "Random": "随机",
    "Sat-Only": "Sat-Only",
}
CDP_ALG_COLOR = {
    "LAWA": "#244C85",
    "Greedy": "#E39D2D",
    "Uniform": "#9CA3AF",
    "Random": "#E95B45",
    "Sat-Only": "#4A4A4A",
}
CDP_ALG_MARKER = {
    "LAWA": "o",
    "Greedy": "s",
    "Uniform": "v",
    "Random": "D",
    "Sat-Only": "X",
}
CDP_ALG_LINESTYLE = {
    "LAWA": "-",
    "Greedy": "--",
    "Uniform": "-.",
    "Random": ":",
    "Sat-Only": (0, (1, 1)),
}
MODE_ORDER = ["PMP", "CDP", "GS-Only", "FWMS"]
MODE_LABEL = {
    "PMP": "PMP",
    "CDP": "CDP",
    "GS-Only": "GS-Only",
    "FWMS": "FWMS",
}
MODE_COLOR = {
    "PMP": "#244C85",
    "CDP": "#E39D2D",
    "GS-Only": "#4A4A4A",
    "FWMS": "#2A8C88",
}


def _run(module_or_script: str, extra: list[str], use_module: bool = False) -> int:
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd = [sys.executable]
    if use_module:
        cmd.extend(["-m", module_or_script])
    else:
        cmd.append(str(ROOT / module_or_script))
    cmd.extend(extra)
    completed = subprocess.run(cmd, cwd=ROOT)
    return int(completed.returncode)


def _add_passthrough_subparser(subparsers, name: str, help_text: str):
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments passed through to the target script")
    return parser


def _run_exp02(args) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_specs = [
        ("yolov5", 32, 640, 640),
        ("resnet101", 32, 224, 224),
        ("vgg19", 32, 224, 224),
        ("vit_huge", 32, 224, 224),
    ]
    scenario_specs = [
        ("同构场景", [None]),
        ("异构场景", ["1,2,3,4,5"]),
        ("典型异构均值", ["1,2,3,4,5", "5,4,3,2,1", "2,4,1,5,3"]),
    ]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    long_frames: list[pd.DataFrame] = []

    for scenario_label, templates in scenario_specs:
        for template_idx, template in enumerate(templates):
            template_tag = "homo" if template is None else template.replace(",", "-")
            for model_name, batch_size, input_h, input_w in run_specs:
                run_id = f"exp02_{model_name}_{template_tag}_{template_idx}_{timestamp}"
                command = [
                    sys.executable,
                    str(ROOT / "experiments_runner.py"),
                    "--config",
                    args.config,
                    "--exp-type",
                    "node_count_sensitivity",
                    "--exp-mode",
                    "theory",
                    "--run-id",
                    run_id,
                    "--sweep-values",
                    args.sweep_values,
                    "--fixed-model",
                    model_name,
                    "--fixed-batch-size",
                    str(batch_size),
                    "--fixed-input-h",
                    str(input_h),
                    "--fixed-input-w",
                    str(input_w),
                    "--repeat-per-point",
                    str(args.repeats),
                    "--controlled-node-count-sweep",
                    "--controlled-sat-compute-tflops",
                    str(args.sat_compute_tflops),
                    "--controlled-sat-memory-mb",
                    str(args.sat_memory_mb),
                    "--controlled-gs-compute-tflops",
                    str(args.gs_compute_tflops),
                    "--controlled-gs-memory-mb",
                    str(args.gs_memory_mb),
                    "--controlled-isl-bandwidth-mbps",
                    str(args.isl_bandwidth_mbps),
                    "--controlled-gsl-bandwidth-mbps",
                    str(args.gsl_bandwidth_mbps),
                ]
                if template is not None:
                    command.extend(["--controlled-sat-compute-template", template])
                    command.append("--controlled-normalize-sat-compute-template")
                completed = subprocess.run(command, cwd=ROOT)
                if completed.returncode != 0:
                    return int(completed.returncode)

                archive_dir = find_run_archive(run_id)
                if archive_dir is None:
                    raise RuntimeError(f"Cannot find archived run for {run_id}")
                data_dir = Path(archive_dir) / "data"
                csv_candidates = sorted(data_dir.glob("results_long_*.csv"))
                if not csv_candidates:
                    raise RuntimeError(f"No exported results_long CSV found under {data_dir}")

                df = pd.read_csv(csv_candidates[-1])
                df = df[df["algorithm"] == "LA-DP"].copy()
                df["scenario_label"] = scenario_label
                df["scenario_template"] = template or "3,3,3,3,3"
                long_frames.append(df)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_path = out_dir / "exp02_ladp_pmp_node_count_sensitivity_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    aggregations = {"mean_norm_latency_vs_gs": ("norm_latency_vs_gs", "mean")}
    if "active_sat_count" in long_df.columns:
        aggregations["mean_active_sat_count"] = ("active_sat_count", "mean")
    scenario_df = (
        long_df.groupby(["scenario_label", "model_name", "pipeline_node_count"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    scenario_csv = out_dir / "exp02_ladp_pmp_node_count_sensitivity_scenarios.csv"
    scenario_df.to_csv(scenario_csv, index=False, encoding="utf-8-sig")

    plot_command = [
        sys.executable,
        "-m",
        "tools.plot_runs_sensitivity_figures",
        "--scenario-csv",
        str(scenario_csv),
        "--output-dir",
        args.out_dir,
        "--models",
        "yolov5,resnet101,vgg19,vit_huge",
        "--title",
        "PMP 模式节点数量敏感性分析",
        "--stem",
        "exp02_ladp_pmp_node_count_sensitivity",
    ]
    completed = subprocess.run(plot_command, cwd=ROOT)
    return int(completed.returncode)


def _profile_entry_to_layers(entry: dict) -> list[dict]:
    if isinstance(entry, list):
        return entry
    return [entry[key] for key in sorted(entry.keys(), key=lambda item: int(item))]


def _load_cdp_model_profile(model_name: str, batch_size: int, profile_path: Path) -> dict:
    input_h, input_w = MODEL_INPUTS[model_name]
    key = f"b{batch_size}_{input_h}x{input_w}"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if model_name not in payload or key not in payload[model_name]:
        raise KeyError(f"Missing profile for {model_name}:{key} in {profile_path}")
    layers = _profile_entry_to_layers(payload[model_name][key])
    input_size_mb = batch_size * 3 * input_h * input_w * 4 / (1024**2)
    output_size_mb = float(layers[-1].get("comm_total_mb", input_size_mb)) if layers else input_size_mb
    compute_full_model_ms = float(sum(float(layer.get("latency_mean_ms", 0.0)) for layer in layers))
    full_model_weight_mb = float(sum(float(layer.get("weight_size_mb", 0.0)) for layer in layers))
    return {
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "batch_size": batch_size,
        "input_h": input_h,
        "input_w": input_w,
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "compute_full_model_ms": compute_full_model_ms,
        "full_model_weight_mb": full_model_weight_mb,
    }


def _scale_worker_compute(base_compute_ms: float, compute_factor: float) -> float:
    return base_compute_ms / max(float(compute_factor), 1e-9)


def _build_cdp_env(profile: dict, scenario: str, worker_count: int) -> dict:
    base_compute_ms = float(profile["compute_full_model_ms"])
    if scenario == "homogeneous":
        templates = [
            {"compute": 3.0, "b_dist": 100.0, "b_return": 100.0, "dist_prop": 3.0, "return_prop": 3.0}
            for _ in range(worker_count)
        ]
    else:
        pool = [
            {"compute": 1.6, "b_dist": 55.0, "b_return": 80.0, "dist_prop": 4.5, "return_prop": 3.5},
            {"compute": 5.0, "b_dist": 160.0, "b_return": 180.0, "dist_prop": 2.5, "return_prop": 2.0},
            {"compute": 3.0, "b_dist": 100.0, "b_return": 120.0, "dist_prop": 3.0, "return_prop": 3.0},
            {"compute": 4.2, "b_dist": 220.0, "b_return": 140.0, "dist_prop": 2.0, "return_prop": 2.8},
            {"compute": 2.2, "b_dist": 70.0, "b_return": 90.0, "dist_prop": 4.0, "return_prop": 3.8},
        ]
        templates = pool[:worker_count]
    return {
        "nodes": [
            {
                "id": f"SAT-{idx + 1:02d}",
                "compute_full_model_ms": _scale_worker_compute(base_compute_ms, template["compute"]),
                "b_dist_mbps": template["b_dist"],
                "b_return_mbps": template["b_return"],
                "dist_prop_ms": template["dist_prop"],
                "return_prop_ms": template["return_prop"],
            }
            for idx, template in enumerate(templates)
        ]
    }


def _sat_only_latency(profile: dict, env: dict) -> float:
    latencies = []
    for node in env["nodes"]:
        solver = CDPSolver(profile, {"nodes": [node]})
        latency, _ = solver.solve_uniform()
        latencies.append(float(latency))
    return min(latencies) if latencies else float("inf")


def _random_allocation_latency(solver: CDPSolver, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    latencies = []
    for _ in range(max(1, repeats)):
        weights = rng.random(solver.K)
        weights = weights / np.sum(weights)
        latency, _ = solver._evaluate_delay(solver.D_in * weights)
        latencies.append(float(latency))
    return float(np.mean(latencies)), float(np.std(latencies))


def _evaluate_cdp_algorithms(profile: dict, env: dict, seed: int, random_repeats: int) -> list[dict]:
    solver = CDPSolver(profile, env)
    sat_only = _sat_only_latency(profile, env)
    results: list[dict] = []

    latency_lawa, plan_lawa = solver.solve_lawa_discrete(batch_size=int(profile["batch_size"]))
    latency_greedy, plan_greedy = solver.solve_greedy()
    latency_uniform, plan_uniform = solver.solve_uniform()
    latency_random, random_std = _random_allocation_latency(solver, seed=seed, repeats=random_repeats)

    for algorithm, latency, plan, std_latency in [
        ("LAWA", latency_lawa, plan_lawa, 0.0),
        ("Greedy", latency_greedy, plan_greedy, 0.0),
        ("Uniform", latency_uniform, plan_uniform, 0.0),
        ("Random", latency_random, {}, random_std),
        ("Sat-Only", sat_only, {}, 0.0),
    ]:
        results.append(
            {
                "algorithm": algorithm,
                "latency_ms": float(latency),
                "std_latency_ms": float(std_latency),
                "norm_latency_vs_sat_only": float(latency) / sat_only if sat_only > 0 else float("nan"),
                "plan_json": json.dumps(plan, ensure_ascii=False, sort_keys=True),
            }
        )
    return results


def _style_cdp_axis(ax: plt.Axes, y_values: list[float]) -> None:
    ax.grid(True, axis="y", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finite = [value for value in y_values if np.isfinite(value)]
    if finite:
        y_min = min(finite)
        y_max = max(finite)
        if abs(y_max - y_min) < 0.05:
            pad = 0.08
        else:
            pad = (y_max - y_min) * 0.18
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)


def _plot_exp03_model(df: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    setup_style()
    model_df = df[df["model_name"] == model_name].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=False)
    scenario_labels = [("homogeneous", "同构 worker 场景"), ("heterogeneous", "异构 worker 场景")]
    for ax, (scenario, scenario_label) in zip(axes, scenario_labels):
        sub = model_df[model_df["scenario"] == scenario]
        y_values: list[float] = []
        for algorithm in CDP_ALG_ORDER:
            alg_df = sub[sub["algorithm"] == algorithm].sort_values("batch_size")
            if alg_df.empty:
                continue
            x = alg_df["batch_size"].to_numpy(dtype=float)
            y = alg_df["norm_latency_vs_sat_only"].to_numpy(dtype=float)
            y_values.extend(y.tolist())
            ax.plot(
                x,
                y,
                label=CDP_ALG_LABEL[algorithm],
                color=CDP_ALG_COLOR[algorithm],
                marker=CDP_ALG_MARKER[algorithm],
                linestyle=CDP_ALG_LINESTYLE[algorithm],
                linewidth=2.0,
                markersize=5.2,
            )
        ax.set_title(scenario_label, pad=8)
        ax.set_xlabel("输入数据量（样本数）")
        ax.set_xticks(sorted(sub["batch_size"].unique()))
        _style_cdp_axis(ax, y_values)
    axes[0].set_ylabel("归一化时延（Sat-Only = 1）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)
    fig.suptitle("LAWA-CDP 模式的数据量敏感性实验", fontsize=16, fontweight="bold", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"exp03_lawa_cdp_data_sensitivity_{model_name}"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_exp04_model(df: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    setup_style()
    model_df = df[df["model_name"] == model_name].copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y_values: list[float] = []
    for algorithm in CDP_ALG_ORDER:
        alg_df = model_df[model_df["algorithm"] == algorithm].sort_values("worker_count")
        if alg_df.empty:
            continue
        x = alg_df["worker_count"].to_numpy(dtype=float)
        y = alg_df["norm_latency_vs_sat_only"].to_numpy(dtype=float)
        y_values.extend(y.tolist())
        ax.plot(
            x,
            y,
            label=CDP_ALG_LABEL[algorithm],
            color=CDP_ALG_COLOR[algorithm],
            marker=CDP_ALG_MARKER[algorithm],
            linestyle=CDP_ALG_LINESTYLE[algorithm],
            linewidth=2.0,
            markersize=5.2,
        )
    ax.set_title("LAWA-CDP 模式的 worker 数量敏感性实验", pad=12, fontsize=15, fontweight="bold")
    ax.set_xlabel("并行 worker 卫星数量")
    ax.set_ylabel("归一化时延（Sat-Only = 1）")
    ax.set_xticks(sorted(model_df["worker_count"].unique()))
    _style_cdp_axis(ax, y_values)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=5, frameon=False)
    fig.tight_layout()
    stem = f"exp04_lawa_cdp_worker_count_sensitivity_{model_name}"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_cdp_notes(out_dir: Path, stem: str, title: str, models: list[str], command: str, conclusion: list[str]) -> None:
    lines = [
        f"# {title}",
        "",
        "## 模型",
        "",
        ", ".join(MODEL_LABELS.get(model, model) for model in models),
        "",
        "## 验收重跑命令",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## 结论口径",
        "",
    ]
    lines.extend(f"- {item}" for item in conclusion)
    (out_dir / f"{stem}_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _run_exp03(args) -> int:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    batches = [int(value) for value in args.data_sizes.split(",") if value.strip()]
    profile_path = ROOT / args.profile
    rows: list[dict] = []

    for model_name in models:
        for batch_size in batches:
            raw_profile = _load_cdp_model_profile(model_name, batch_size, profile_path)
            cdp_profile = {
                "input_size_mb": raw_profile["input_size_mb"],
                "output_size_mb": raw_profile["output_size_mb"],
                "batch_size": raw_profile["batch_size"],
            }
            for scenario in ["homogeneous", "heterogeneous"]:
                env = _build_cdp_env(raw_profile, scenario=scenario, worker_count=args.worker_count)
                for result in _evaluate_cdp_algorithms(
                    cdp_profile,
                    env,
                    seed=args.seed + batch_size + len(model_name),
                    random_repeats=args.random_repeats,
                ):
                    rows.append(
                        {
                            **raw_profile,
                            "scenario": scenario,
                            "scenario_label": "同构 worker 场景" if scenario == "homogeneous" else "异构 worker 场景",
                            "worker_count": args.worker_count,
                            **result,
                        }
                    )

    df = pd.DataFrame(rows)
    long_csv = out_dir / "exp03_lawa_cdp_data_sensitivity_long.csv"
    summary_csv = out_dir / "exp03_lawa_cdp_data_sensitivity_summary.csv"
    df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    df.drop(columns=["plan_json"]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    for model_name in models:
        _plot_exp03_model(df, model_name, out_dir)
    _write_cdp_notes(
        out_dir,
        "exp03_lawa_cdp_data_sensitivity",
        "实验 3：LAWA-CDP 模式的数据量敏感性实验",
        models,
        f"python thesis_entry.py exp03 --out-dir {args.out_dir}",
        [
            "同构 worker 场景下，各 worker 能力一致，均匀分配通常接近 LAWA。",
            "异构 worker 场景下，LAWA 会把更多数据分给算力强、链路好的 worker，优势更明显。",
            "输入数据量增大后，离散样本分配更接近连续最优解，LAWA 相对随机/贪心/均匀的稳定性更容易体现。",
        ],
    )
    print(out_dir)
    return 0


def _run_exp04(args) -> int:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    worker_counts = [int(value) for value in args.worker_counts.split(",") if value.strip()]
    profile_path = ROOT / args.profile
    rows: list[dict] = []

    for model_name in models:
        raw_profile = _load_cdp_model_profile(model_name, args.data_size, profile_path)
        cdp_profile = {
            "input_size_mb": raw_profile["input_size_mb"],
            "output_size_mb": raw_profile["output_size_mb"],
            "batch_size": raw_profile["batch_size"],
        }
        for worker_count in worker_counts:
            env = _build_cdp_env(raw_profile, scenario="heterogeneous", worker_count=worker_count)
            for result in _evaluate_cdp_algorithms(
                cdp_profile,
                env,
                seed=args.seed + worker_count + len(model_name),
                random_repeats=args.random_repeats,
            ):
                rows.append(
                    {
                        **raw_profile,
                        "scenario": "heterogeneous",
                        "scenario_label": "异构 worker 场景",
                        "worker_count": worker_count,
                        **result,
                    }
                )

    df = pd.DataFrame(rows)
    long_csv = out_dir / "exp04_lawa_cdp_worker_count_sensitivity_long.csv"
    summary_csv = out_dir / "exp04_lawa_cdp_worker_count_sensitivity_summary.csv"
    df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    df.drop(columns=["plan_json"]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    for model_name in models:
        _plot_exp04_model(df, model_name, out_dir)
    _write_cdp_notes(
        out_dir,
        "exp04_lawa_cdp_worker_count_sensitivity",
        "实验 4：LAWA-CDP 模式的 worker 数量敏感性实验",
        models,
        f"python thesis_entry.py exp04 --out-dir {args.out_dir}",
        [
            "worker 数量增加通常会降低 CDP 时延，但收益不是线性的。",
            "当新增 worker 算力或链路条件较弱时，额外分发和回传开销会削弱并行收益。",
            "LAWA 的作用是根据 worker 的计算和链路状态调节数据量，避免简单均分在异构场景下失效。",
        ],
    )
    print(out_dir)
    return 0


def _tx_ms(data_mb: float, bandwidth_mbps: float) -> float:
    return data_mb * 8.0 / max(float(bandwidth_mbps), 1e-9) * 1000.0


def _gs_only_latency(profile: dict, gsl_bandwidth_mbps: float, gs_compute_factor: float) -> float:
    return _tx_ms(float(profile["input_size_mb"]), gsl_bandwidth_mbps) + float(profile["compute_full_model_ms"]) / max(
        gs_compute_factor, 1e-9
    )


def _load_pmp_norms(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["algorithm"] == "LA-DP"].copy()
    return {
        str(row["model_name"]): float(row["mean_norm_latency_vs_gs"])
        for _, row in df.iterrows()
        if pd.notna(row.get("mean_norm_latency_vs_gs"))
    }


def _estimate_mode_rows(
    model_name: str,
    batch_size: int,
    args,
    pmp_norms: dict[str, float],
) -> list[dict]:
    profile_path = ROOT / args.profile
    raw_profile = _load_cdp_model_profile(model_name, batch_size, profile_path)
    cdp_profile = {
        "input_size_mb": raw_profile["input_size_mb"],
        "output_size_mb": raw_profile["output_size_mb"],
        "batch_size": raw_profile["batch_size"],
    }
    gs_latency = _gs_only_latency(raw_profile, args.gsl_bandwidth_mbps, args.gs_compute_factor)
    pmp_norm = float(pmp_norms.get(model_name, 1.0))
    pmp_latency = pmp_norm * gs_latency

    memory_feasible = float(raw_profile["full_model_weight_mb"]) <= float(args.worker_memory_mb)
    cdp_latency = float("nan")
    cdp_reason = ""
    if memory_feasible:
        env = _build_cdp_env(raw_profile, scenario="heterogeneous", worker_count=args.worker_count)
        cdp_latency, _ = CDPSolver(cdp_profile, env).solve_lawa_discrete(batch_size=batch_size)
        cdp_latency = float(cdp_latency)
    else:
        cdp_reason = "full_model_memory_exceeds_worker_memory"

    candidates = {
        "PMP": pmp_latency,
        "GS-Only": gs_latency,
    }
    if np.isfinite(cdp_latency):
        candidates["CDP"] = cdp_latency

    best_fallback = min(candidates, key=candidates.get)
    selected_mode = best_fallback
    if np.isfinite(cdp_latency):
        fallback_without_cdp = min({"PMP": pmp_latency, "GS-Only": gs_latency}, key={"PMP": pmp_latency, "GS-Only": gs_latency}.get)
        fallback_latency = {"PMP": pmp_latency, "GS-Only": gs_latency}[fallback_without_cdp]
        if cdp_latency <= fallback_latency * (1.0 - float(args.min_cdp_gain)):
            selected_mode = "CDP"
    selected_latency = candidates[selected_mode]

    base = {
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "batch_size": batch_size,
        "input_size_mb": raw_profile["input_size_mb"],
        "output_size_mb": raw_profile["output_size_mb"],
        "full_model_weight_mb": raw_profile["full_model_weight_mb"],
        "worker_memory_mb": args.worker_memory_mb,
    }
    rows = [
        {**base, "mode": "PMP", "feasible": True, "latency_ms": pmp_latency, "reason": ""},
        {
            **base,
            "mode": "CDP",
            "feasible": bool(np.isfinite(cdp_latency)),
            "latency_ms": cdp_latency,
            "reason": cdp_reason,
        },
        {**base, "mode": "GS-Only", "feasible": True, "latency_ms": gs_latency, "reason": ""},
        {**base, "mode": "FWMS", "feasible": True, "latency_ms": selected_latency, "reason": f"selected_{selected_mode}"},
    ]
    for row in rows:
        row["norm_latency_vs_gs"] = float(row["latency_ms"]) / gs_latency if np.isfinite(row["latency_ms"]) else float("nan")
    return rows


def _plot_exp05(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    theory_ax, semi_ax = axes
    models = [model for model in ["yolov5", "resnet101", "vgg19", "vit_huge"] if model in set(df["model_name"])]
    x = np.arange(len(models))
    width = 0.18
    for idx, mode in enumerate(MODE_ORDER):
        values = []
        for model in models:
            row = df[(df["model_name"] == model) & (df["mode"] == mode)].iloc[0]
            values.append(float(row["norm_latency_vs_gs"]) if bool(row["feasible"]) else np.nan)
        pos = x + (idx - (len(MODE_ORDER) - 1) / 2) * width
        bars = theory_ax.bar(pos, values, width=width, label=MODE_LABEL[mode], color=MODE_COLOR[mode])
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                theory_ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for model_idx, model in enumerate(models):
        cdp_row = df[(df["model_name"] == model) & (df["mode"] == "CDP")].iloc[0]
        if not bool(cdp_row["feasible"]):
            theory_ax.text(model_idx, 0.08, "CDP\n不可行", ha="center", va="bottom", fontsize=8, color="#E95B45")
    theory_ax.set_title("理论仿真", pad=10)
    theory_ax.set_xticks(x)
    theory_ax.set_xticklabels([MODEL_LABELS[model] for model in models])
    theory_ax.set_ylabel("归一化时延（GS-Only = 1）")
    theory_ax.grid(True, axis="y", alpha=0.75)
    theory_ax.spines["top"].set_visible(False)
    theory_ax.spines["right"].set_visible(False)
    theory_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=4, frameon=False)

    semi_ax.axis("off")
    semi_ax.set_title("半实物仿真", pad=10)
    semi_ax.text(
        0.5,
        0.52,
        "待补充\n不使用占位数据",
        ha="center",
        va="center",
        fontsize=15,
        color="#4B5563",
        transform=semi_ax.transAxes,
    )
    fig.suptitle("FWMS 模式选择有效性实验", fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "exp05_fwms_mode_selection_effectiveness.png", bbox_inches="tight")
    fig.savefig(out_dir / "exp05_fwms_mode_selection_effectiveness.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_exp06(df: pd.DataFrame, out_dir: Path, model_name: str) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode].sort_values("batch_size")
        y = sub["latency_ms"].to_numpy(dtype=float)
        x = sub["batch_size"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            label=MODE_LABEL[mode],
            color=MODE_COLOR[mode],
            marker="o",
            linewidth=2.1,
            markersize=5.2,
        )
    selected = df[df["mode"] == "FWMS"].sort_values("batch_size")
    for _, row in selected.iterrows():
        ax.text(float(row["batch_size"]), float(row["latency_ms"]), str(row["reason"]).replace("selected_", ""), fontsize=8, ha="center", va="bottom")
    ax.set_title("FWMS 输入数据量敏感性实验", pad=12, fontsize=15, fontweight="bold")
    ax.set_xlabel("输入数据量（样本数）")
    ax.set_ylabel("平均端到端时延 / ms")
    ax.set_xticks(sorted(df["batch_size"].unique()))
    ax.grid(True, axis="y", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
    fig.tight_layout()
    stem = f"exp06_fwms_data_sensitivity_{model_name}"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_exp05(args) -> int:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    pmp_norms = _load_pmp_norms(ROOT / args.pmp_summary)
    rows: list[dict] = []
    for model_name in models:
        rows.extend(_estimate_mode_rows(model_name, args.data_size, args, pmp_norms))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp05_fwms_mode_selection_effectiveness_summary.csv", index=False, encoding="utf-8-sig")
    _plot_exp05(df, out_dir)
    notes = [
        "# 实验 5：FWMS 模式选择有效性实验",
        "",
        "- 理论仿真部分已生成。",
        "- 半实物仿真部分暂不填充数据，图中明确标注待补充。",
        "- CDP 可行性由完整模型权重是否超过 worker 内存约束判断。",
        "- FWMS 规则：CDP 可行且相对 PMP/GS-Only 有明确时延收益时选择 CDP，否则回退到 PMP 或 GS-Only。",
    ]
    (out_dir / "exp05_fwms_mode_selection_effectiveness_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(out_dir)
    return 0


def _run_exp06(args) -> int:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pmp_norms = _load_pmp_norms(ROOT / args.pmp_summary)
    rows: list[dict] = []
    for batch_size in [int(value) for value in args.data_sizes.split(",") if value.strip()]:
        rows.extend(_estimate_mode_rows(args.model, batch_size, args, pmp_norms))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"exp06_fwms_data_sensitivity_{args.model}_summary.csv", index=False, encoding="utf-8-sig")
    _plot_exp06(df, out_dir, args.model)
    notes = [
        "# 实验 6：FWMS 输入数据量敏感性实验",
        "",
        f"- 模型：{MODEL_LABELS.get(args.model, args.model)}。",
        "- 横坐标为输入数据量（样本数），纵坐标为平均端到端时延。",
        "- 若 YOLOv5 修正最终检测输出后 PMP 一直较优，这是合理结果，说明该任务的最终输出极小，PMP 可显著减少回传通信量。",
    ]
    (out_dir / f"exp06_fwms_data_sensitivity_{args.model}_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(out_dir)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for thesis experiments, plots, and acceptance reruns."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_passthrough_subparser(subparsers, "run-legacy", "Run traditional PMP theory experiments")
    _add_passthrough_subparser(subparsers, "run-stk", "Run STK dynamic PMP experiments")
    _add_passthrough_subparser(subparsers, "run-mode", "Run mode-selection experiments")
    _add_passthrough_subparser(subparsers, "build-stk-config", "Build network_config files from STK reports")

    _add_passthrough_subparser(subparsers, "plot-legacy", "Plot legacy PMP figures from long-table results")
    _add_passthrough_subparser(subparsers, "plot-mode-summary", "Plot cross-model mode-selection summaries")
    _add_passthrough_subparser(subparsers, "plot-paper", "Generate paper-ready figures from existing results")
    _add_passthrough_subparser(subparsers, "plot-sensitivity", "Redraw archived sensitivity figures")
    _add_passthrough_subparser(subparsers, "plot-stk-summary", "Summarize STK cross-model PMP results")

    _add_passthrough_subparser(subparsers, "semi-physical", "Run semi-physical verification utilities")
    _add_passthrough_subparser(subparsers, "physical-orchestrator", "Run physical experiment orchestration utilities")
    _add_passthrough_subparser(subparsers, "exp01", "Rerun experiment 1 paper figure")
    exp02_parser = subparsers.add_parser("exp02", help="Run experiment 2 and draw the paper figure")
    exp02_parser.add_argument(
        "--config",
        default="result/stk_dynamic/stk_dynamic_resnet101_001/configs/slot_033_064500_065000_network_config.json",
        help="Base network config path",
    )
    exp02_parser.add_argument("--sweep-values", default="1,2,3,4,5", help="Comma-separated relay node counts")
    exp02_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Auxiliary Random/GA repeats per node-count point; experiment 2 figure itself only uses LADP",
    )
    exp02_parser.add_argument("--isl-bandwidth-mbps", type=float, default=5000.0, help="Fixed ISL bandwidth")
    exp02_parser.add_argument("--gsl-bandwidth-mbps", type=float, default=100.0, help="Fixed GSL bandwidth")
    exp02_parser.add_argument("--sat-compute-tflops", type=float, default=3.0, help="Homogeneous LEO compute")
    exp02_parser.add_argument("--sat-memory-mb", type=int, default=4096, help="Homogeneous LEO memory")
    exp02_parser.add_argument("--gs-compute-tflops", type=float, default=300.0, help="GS compute")
    exp02_parser.add_argument("--gs-memory-mb", type=int, default=64000, help="GS memory")
    exp02_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_v2/02_ladp_pmp_node_count_sensitivity",
        help="Output directory for experiment 2 figure and summaries",
    )
    exp03_parser = subparsers.add_parser("exp03", help="Run experiment 3 and draw CDP data-size sensitivity figures")
    exp03_parser.add_argument("--models", default="yolov5,resnet101,vgg19", help="Comma-separated model ids")
    exp03_parser.add_argument("--data-sizes", default="16,32,64,128", help="Comma-separated input data sizes")
    exp03_parser.add_argument("--worker-count", type=int, default=4, help="Fixed CDP worker count")
    exp03_parser.add_argument("--random-repeats", type=int, default=30, help="Random baseline repeats")
    exp03_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    exp03_parser.add_argument(
        "--profile",
        default="config/dnn_profiles_database_jetson.json",
        help="DNN profile database used for full-model latency and output size",
    )
    exp03_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_v2/03_lawa_cdp_data_sensitivity",
        help="Output directory for experiment 3",
    )
    exp04_parser = subparsers.add_parser("exp04", help="Run experiment 4 and draw CDP worker-count sensitivity figures")
    exp04_parser.add_argument("--models", default="yolov5,resnet101,vgg19", help="Comma-separated model ids")
    exp04_parser.add_argument("--data-size", type=int, default=64, help="Fixed input data size")
    exp04_parser.add_argument("--worker-counts", default="1,2,3,4,5", help="Comma-separated worker counts")
    exp04_parser.add_argument("--random-repeats", type=int, default=30, help="Random baseline repeats")
    exp04_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    exp04_parser.add_argument(
        "--profile",
        default="config/dnn_profiles_database_jetson.json",
        help="DNN profile database used for full-model latency and output size",
    )
    exp04_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_v2/04_lawa_cdp_worker_count_sensitivity",
        help="Output directory for experiment 4",
    )
    exp05_parser = subparsers.add_parser("exp05", help="Run experiment 5 and draw FWMS mode-selection figure")
    exp05_parser.add_argument("--models", default="yolov5,resnet101,vgg19,vit_huge", help="Comma-separated model ids")
    exp05_parser.add_argument("--data-size", type=int, default=64, help="Fixed input data size")
    exp05_parser.add_argument("--worker-count", type=int, default=4, help="CDP worker count")
    exp05_parser.add_argument("--worker-memory-mb", type=float, default=2048.0, help="Memory limit for each CDP worker")
    exp05_parser.add_argument("--gsl-bandwidth-mbps", type=float, default=100.0, help="GS-only uplink bandwidth")
    exp05_parser.add_argument("--gs-compute-factor", type=float, default=100.0, help="GS speedup over profiled Jetson latency")
    exp05_parser.add_argument("--min-cdp-gain", type=float, default=0.05, help="Minimum CDP gain required by FWMS")
    exp05_parser.add_argument("--profile", default="config/dnn_profiles_database_jetson.json", help="DNN profile database")
    exp05_parser.add_argument(
        "--pmp-summary",
        default="result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv",
        help="Experiment 1 PMP summary used as PMP reference",
    )
    exp05_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_v2/05_fwms_mode_selection_effectiveness",
        help="Output directory for experiment 5",
    )
    exp06_parser = subparsers.add_parser("exp06", help="Run experiment 6 and draw FWMS data-size sensitivity figure")
    exp06_parser.add_argument("--model", default="yolov5", help="Model id")
    exp06_parser.add_argument("--data-sizes", default="16,32,64,128", help="Comma-separated input data sizes")
    exp06_parser.add_argument("--worker-count", type=int, default=4, help="CDP worker count")
    exp06_parser.add_argument("--worker-memory-mb", type=float, default=2048.0, help="Memory limit for each CDP worker")
    exp06_parser.add_argument("--gsl-bandwidth-mbps", type=float, default=100.0, help="GS-only uplink bandwidth")
    exp06_parser.add_argument("--gs-compute-factor", type=float, default=100.0, help="GS speedup over profiled Jetson latency")
    exp06_parser.add_argument("--min-cdp-gain", type=float, default=0.05, help="Minimum CDP gain required by FWMS")
    exp06_parser.add_argument("--profile", default="config/dnn_profiles_database_jetson.json", help="DNN profile database")
    exp06_parser.add_argument(
        "--pmp-summary",
        default="result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv",
        help="Experiment 1 PMP summary used as PMP reference",
    )
    exp06_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_v2/06_fwms_data_sensitivity",
        help="Output directory for experiment 6",
    )

    args = parser.parse_args()

    if args.command == "run-legacy":
        raise SystemExit(_run("experiments_runner.py", args.extra))
    if args.command == "run-stk":
        raise SystemExit(_run("stk_dynamic_experiment.py", args.extra))
    if args.command == "run-mode":
        raise SystemExit(_run("mode_selection_experiment.py", args.extra))
    if args.command == "build-stk-config":
        raise SystemExit(_run("tools.build_stk_network_config", args.extra, use_module=True))

    if args.command == "plot-legacy":
        raise SystemExit(_run("tools.plot_avg_tho_vs_real", args.extra, use_module=True))
    if args.command == "plot-mode-summary":
        raise SystemExit(_run("tools.plot_mode_selection_summary", args.extra, use_module=True))
    if args.command == "plot-paper":
        raise SystemExit(_run("tools.plot_paper_ready_figures", args.extra, use_module=True))
    if args.command == "plot-sensitivity":
        raise SystemExit(_run("tools.plot_runs_sensitivity_figures", args.extra, use_module=True))
    if args.command == "plot-stk-summary":
        raise SystemExit(_run("tools.plot_stk_cross_model_summary", args.extra, use_module=True))

    if args.command == "semi-physical":
        raise SystemExit(_run("tools.semi_physical_mode_verify", args.extra, use_module=True))
    if args.command == "physical-orchestrator":
        raise SystemExit(_run("tools.physical_experiment_orchestrator", args.extra, use_module=True))
    if args.command == "exp01":
        raise SystemExit(_run("tools.paper_figures.run_stk_slot_pmp_highlight", args.extra, use_module=True))
    if args.command == "exp02":
        raise SystemExit(_run_exp02(args))
    if args.command == "exp03":
        raise SystemExit(_run_exp03(args))
    if args.command == "exp04":
        raise SystemExit(_run_exp04(args))
    if args.command == "exp05":
        raise SystemExit(_run_exp05(args))
    if args.command == "exp06":
        raise SystemExit(_run_exp06(args))

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
