"""Unified thesis experiment and utility entrypoint.

This script keeps the root directory clean and exposes one stable CLI for:
1. Main experiment entrypoints.
2. Plotting/summary utilities under ``tools``.
3. Paper-facing rerun commands used during acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import pandas as pd
import numpy as np
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

from algorithms.cdp_solver import CDPSolver
from algorithms.pmp_solver import PMPSolver
from core.experiment_archive import find_run_archive
from core.mode_evaluators import (
    StkPathResolver,
    _build_env_status,
    _build_model_profile,
    _network_config_for_candidate,
)
from core.mode_scene_builder import TaskSpec, load_stk_slot_scenes
from core.scheduler import Scheduler
from tools.plot_runs_sensitivity_figures import setup_style


ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "resnet101": "ResNet101",
    "vgg19": "VGG19",
    "vit_huge": "ViT-Huge",
}
MODEL_FIG_LABELS = {
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
CDP_ALG_ORDER_EXP04 = ["LAWA", "Greedy", "Sat-Only"]
CDP_ALG_LABEL = {
    "LAWA": "LAWA",
    "Greedy": "贪心",
    "Uniform": "均匀",
    "Random": "随机",
    "Sat-Only": "Single-LEO",
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
MODE_ORDER = ["PMP", "CDP", "GS-Only", "Sat-Only", "FWMS"]
MODE_LABEL = {
    "PMP": "PMP",
    "CDP": "CDP",
    "GS-Only": "GS-Only",
    "Sat-Only": "Single-LEO",
    "FWMS": "FWMS",
}
MODE_COLOR = {
    "PMP": "#244C85",
    "CDP": "#E39D2D",
    "GS-Only": "#4A4A4A",
    "Sat-Only": "#8A63D2",
    "FWMS": "#2A8C88",
}
MODE_MARKER = {
    "PMP": "s",
    "CDP": "o",
    "GS-Only": "D",
    "Sat-Only": "X",
    "FWMS": "^",
}
MODE_LINESTYLE = {
    "PMP": "-",
    "CDP": "--",
    "GS-Only": "-",
    "Sat-Only": ":",
    "FWMS": "-.",
}
MODE_ZORDER = {
    "PMP": 3,
    "CDP": 4,
    "GS-Only": 2,
    "Sat-Only": 3,
    "FWMS": 5,
}
MODE_SELECTION_STK_RUNS = {
    "yolov5": "result/stk_dynamic/stk_dynamic_yolo_001",
    "resnet101": "result/stk_dynamic/stk_dynamic_resnet101_001",
    "vgg19": "result/stk_dynamic/stk_dynamic_vgg19_001",
    "vit_huge": "result/stk_dynamic/stk_dynamic_vit_huge_001",
}
MODE_SELECTION_ALIASES = {
    "PMP": "PMP",
    "CDP": "CDP",
    "GS-Only": "GS-Only",
    "Sat-Only": "Sat-Only",
    "FWMS-Feature": "FWMS",
    "FWMS": "FWMS",
}
EXP02_MODEL = "yolov5"
EXP02_MODELS = ["yolov5", "resnet101", "vgg19", "vit_huge"]
EXP02_BLOCK_BATCH = 64
EXP02_BLOCK_COUNT = 50
EXP02_RAW_IMAGE_GIB = (4096 * 4096 * 3) / float(1024 ** 3)
EXP02_EQ_IMAGES_PER_BLOCK = 1.0


def _is_show_only(args) -> bool:
    return bool(getattr(args, "show_only", False))


def _prepare_runtime_out_dir(requested_out_dir: str, show_only: bool, prefix: str) -> tuple[Path, Path | None]:
    if show_only:
        base = ROOT / ".codex_tmp"
        base.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=base))
        return temp_dir, temp_dir
    out_dir = ROOT / requested_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, None


def _cleanup_runtime_dir(cleanup_dir: Path | None) -> None:
    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)


def _finalize_figure(
    fig: plt.Figure,
    png_path: Path | None,
    pdf_path: Path | None,
    *,
    show_only: bool = False,
    dpi: int | None = None,
) -> None:
    if show_only:
        plt.show()
        plt.close(fig)
        return
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(png_path, **save_kwargs)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


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


def _parse_profile_key(key: str) -> tuple[int, int, int]:
    batch_part, size_part = key.split("_", 1)
    batch_size = int(batch_part.lstrip("b"))
    input_h, input_w = [int(item) for item in size_part.split("x")]
    return batch_size, input_h, input_w


def _profile_key_sort_key(key: str) -> tuple[int, int, int, str]:
    try:
        batch_size, input_h, input_w = _parse_profile_key(key)
        return input_h, input_w, batch_size, key
    except Exception:
        return 0, 0, 0, key


def _select_profile_key(
    entries: dict,
    target_batch: int | None = None,
    input_h: int | None = None,
    input_w: int | None = None,
) -> str:
    parsed = []
    for key in entries:
        if not key.startswith("b") or "_" not in key:
            continue
        batch_size, key_h, key_w = _parse_profile_key(key)
        if input_h is not None and input_w is not None and (key_h != input_h or key_w != input_w):
            continue
        parsed.append((batch_size, key))
    if not parsed:
        raise ValueError("No matching batch profile keys found.")
    if target_batch is not None:
        for batch_size, key in parsed:
            if batch_size == target_batch:
                return key
    return min(parsed, key=lambda item: item[0])[1]


def _profile_layers_from_entry(entry: dict | list) -> list[dict]:
    if isinstance(entry, list):
        return entry
    return [entry[key] for key in sorted(entry.keys(), key=lambda item: int(item))]


def _profile_consistency_report(primary: dict, reference: dict, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for model_name in models:
        primary_model = primary.get(model_name)
        reference_model = reference.get(model_name)
        if primary_model is None or reference_model is None:
            rows.append(
                {
                    "model_name": model_name,
                    "profile_key": "",
                    "check": "model_present",
                    "primary_layer_count": np.nan,
                    "reference_layer_count": np.nan,
                    "max_comm_total_mb_abs_diff": np.nan,
                    "ok": False,
                    "detail": "missing model in primary or reference profile",
                }
            )
            continue

        primary_keys = set(primary_model.keys())
        reference_keys = set(reference_model.keys())
        key_sets_match = primary_keys == reference_keys
        rows.append(
            {
                "model_name": model_name,
                "profile_key": "",
                "check": "profile_keys",
                "primary_layer_count": len(primary_keys),
                "reference_layer_count": len(reference_keys),
                "max_comm_total_mb_abs_diff": np.nan,
                "ok": key_sets_match,
                "detail": "" if key_sets_match else f"primary-only={sorted(primary_keys-reference_keys)}, reference-only={sorted(reference_keys-primary_keys)}",
            }
        )

        for profile_key in sorted(primary_keys & reference_keys, key=_profile_key_sort_key):
            primary_layers = _profile_layers_from_entry(primary_model[profile_key])
            reference_layers = _profile_layers_from_entry(reference_model[profile_key])
            layer_count_match = len(primary_layers) == len(reference_layers)
            diffs = []
            for left, right in zip(primary_layers, reference_layers):
                diffs.append(abs(float(left.get("comm_total_mb", 0.0)) - float(right.get("comm_total_mb", 0.0))))
            max_diff = max(diffs) if diffs else 0.0
            ok = layer_count_match and max_diff <= 1e-9
            rows.append(
                {
                    "model_name": model_name,
                    "profile_key": profile_key,
                    "check": "comm_total_mb",
                    "primary_layer_count": len(primary_layers),
                    "reference_layer_count": len(reference_layers),
                    "max_comm_total_mb_abs_diff": max_diff,
                    "ok": ok,
                    "detail": "" if ok else "layer count or feature-size mismatch",
                }
            )
    return rows


def _plot_model_layer_outputs(
    model_name: str,
    profile_key: str,
    layers: list[dict],
    plot_batch_size: int,
    out_dir: Path,
    show_only: bool = False,
) -> list[dict]:
    setup_style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    source_batch, input_h, input_w = _parse_profile_key(profile_key)
    batch_scale = float(plot_batch_size) / float(source_batch)
    input_mb = plot_batch_size * 3 * input_h * input_w * 4 / (1024**2)
    layer_values = [float(layer.get("comm_total_mb", 0.0)) * batch_scale for layer in layers]

    labels = ["Input"] + [str(idx) for idx in range(len(layer_values))]
    values = [input_mb] + layer_values
    input_color = "#2F8F64"
    low_color = "#3B6EA8"
    high_color = "#C2413B"
    edge_color = "#20242A"
    colors = [input_color] + [high_color if value > input_mb else low_color for value in layer_values]

    fig_width = max(12.0, min(18.0, 0.27 * len(labels) + 7.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor=edge_color, linewidth=0.42, width=0.76, zorder=2)
    ax.axhline(input_mb, color=input_color, linestyle=(0, (6, 3)), linewidth=2.2, zorder=1.5)

    y_max = max(max(values), input_mb) if values else input_mb
    y_pad = max(y_max * 0.28, input_mb * 0.60, 0.16)
    ax.set_ylim(0, y_max + y_pad)
    ax.set_xlim(-0.75, len(labels) - 0.25)

    input_bar = bars[0]
    input_text_y = min(input_bar.get_height() + y_pad * 0.12, y_max + y_pad * 0.66)
    ax.text(
        input_bar.get_x() + input_bar.get_width() / 2,
        input_text_y,
        "input",
        ha="center",
        va="bottom",
        fontsize=10.8,
        color=input_color,
        fontweight="bold",
        linespacing=1.08,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
        clip_on=False,
    )

    result_bar = bars[-1]
    result_value = layer_values[-1] if layer_values else 0.0
    if result_value <= input_mb:
        result_text_y = input_mb + y_pad * 0.10
    else:
        result_text_y = result_bar.get_height() + y_pad * 0.12
    result_text_y = min(result_text_y, y_max + y_pad * 0.66)
    ax.text(
        result_bar.get_x() + result_bar.get_width() / 2,
        result_text_y,
        "result",
        ha="center",
        va="bottom",
        fontsize=10.8,
        color=edge_color,
        fontweight="bold",
        linespacing=1.08,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
        clip_on=False,
    )

    tick_step = max(1, int(np.ceil(len(labels) / 22)))
    tick_positions = [0] + [idx for idx in range(1, len(labels), tick_step)]
    if len(labels) - 1 not in tick_positions:
        if tick_positions and len(labels) - 1 - tick_positions[-1] < max(2, tick_step):
            tick_positions = tick_positions[:-1]
        tick_positions.append(len(labels) - 1)
    tick_labels = [labels[idx] for idx in tick_positions]
    tick_labels[-1] = "Result"
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9.8)
    ax.set_ylabel(f"输出数据量 / MB（batch={plot_batch_size}）", fontsize=12.5, labelpad=8)
    ax.set_xlabel("层编号", fontsize=12.5, labelpad=8)
    ax.set_title(
        f"{MODEL_FIG_LABELS.get(model_name, model_name)} 层级输出特征图数据量分布",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, axis="y", alpha=0.58, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_items = [
        Patch(facecolor=input_color, edgecolor=edge_color, label="输入 input"),
        Line2D([0], [0], color=input_color, lw=2.2, linestyle=(0, (6, 3)), label="输入大小虚线"),
        Patch(facecolor=low_color, edgecolor=edge_color, label="层输出 ≤ input"),
        Patch(facecolor=high_color, edgecolor=edge_color, label="层输出 > input"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        fontsize=9.8,
        handlelength=1.8,
        columnspacing=1.0,
    )

    stem = f"exp00_layer_output_distribution_{model_name}"
    fig.tight_layout()
    _finalize_figure(
        fig,
        out_dir / f"{stem}.png",
        out_dir / f"{stem}.pdf",
        show_only=show_only,
        dpi=300,
    )

    rows = [
        {
            "model_name": model_name,
            "model_label": MODEL_FIG_LABELS.get(model_name, model_name),
            "source_profile_key": profile_key,
            "source_batch_size": source_batch,
            "plot_batch_size": plot_batch_size,
            "input_h": input_h,
            "input_w": input_w,
            "layer": "input",
            "comm_total_mb": input_mb,
            "relative_to_input": 1.0,
            "category": "input",
        }
    ]
    for idx, value in enumerate(layer_values):
        category = "result" if idx == len(layer_values) - 1 else ("larger_than_input" if value > input_mb else "smaller_or_equal_input")
        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_FIG_LABELS.get(model_name, model_name),
                "source_profile_key": profile_key,
                "source_batch_size": source_batch,
                "plot_batch_size": plot_batch_size,
                "input_h": input_h,
                "input_w": input_w,
                "layer": idx,
                "comm_total_mb": value,
                "relative_to_input": value / input_mb if input_mb > 0 else np.nan,
                "category": category,
            }
        )
    return rows


def _run_exp00(args) -> int:
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp00_show")
    profile_path = ROOT / args.profile
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    try:
        consistency_ok = None
        if args.reference_profile:
            reference_path = ROOT / args.reference_profile
            reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
            report_rows = _profile_consistency_report(payload, reference_payload, models)
            report = pd.DataFrame(report_rows)
            if not _is_show_only(args):
                report.to_csv(out_dir / "exp00_profile_consistency_report.csv", index=False, encoding="utf-8-sig")
            consistency_ok = bool(report["ok"].all()) if not report.empty else True
            if not consistency_ok:
                failed = report.loc[~report["ok"], ["model_name", "profile_key", "check", "detail"]]
                raise ValueError(f"Profile consistency check failed:\n{failed.to_string(index=False)}")

        all_rows: list[dict] = []
        source_keys: dict[str, str] = {}
        for model_name in models:
            if model_name not in payload:
                raise KeyError(f"Model not found in profile: {model_name}")
            input_h, input_w = MODEL_INPUTS[model_name]
            profile_key = _select_profile_key(payload[model_name], target_batch=args.batch_size, input_h=input_h, input_w=input_w)
            source_keys[model_name] = profile_key
            layers = _profile_layers_from_entry(payload[model_name][profile_key])
            all_rows.extend(_plot_model_layer_outputs(model_name, profile_key, layers, args.batch_size, out_dir, show_only=_is_show_only(args)))

        if not _is_show_only(args):
            summary = pd.DataFrame(all_rows)
            summary.to_csv(out_dir / "exp00_layer_output_distribution_summary.csv", index=False, encoding="utf-8-sig")
            notes = [
                "# 实验 00：模型层级输出特征图数据量分布",
                "",
                f"- profile：`{args.profile}`",
                f"- reference profile：`{args.reference_profile}`" if args.reference_profile else "- reference profile：未启用",
                f"- PC/Jetson 特征图数据量一致性检查：{'通过' if consistency_ok else '未启用'}",
                f"- 展示批次：batch={args.batch_size}；若 profile 未包含该批次，则按最小可用 batch 线性折算。",
                "- 绿色柱表示输入 input，绿色虚线表示输入大小。",
                "- 红色柱表示该层输出大于 input，蓝色柱表示该层输出不大于 input。",
                "- 最后一层输出标注为 result。",
                "",
                "## 使用的源 profile key",
                "",
            ]
            notes.extend(f"- {MODEL_FIG_LABELS.get(model, model)}：`{profile_key}`" for model, profile_key in source_keys.items())
            (out_dir / "exp00_layer_output_distribution_notes.md").write_text("\n".join(notes), encoding="utf-8")
            print(out_dir)
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


def _run_exp02(args) -> int:
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp02_show")
    try:
        controlled_long_df, controlled_summary_df = _run_exp02_controlled(args, out_dir)
        engineering_long_df, engineering_summary_df = _run_exp02_engineering(args, out_dir)

        if not _is_show_only(args):
            controlled_long_df.to_csv(
                out_dir / "exp02a_control_node_count_total_latency_long.csv",
                index=False,
                encoding="utf-8-sig",
            )
            controlled_summary_df.to_csv(
                out_dir / "exp02a_control_node_count_total_latency_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            engineering_long_df.to_csv(
                out_dir / "exp02b_engineering_node_count_capacity_long.csv",
                index=False,
                encoding="utf-8-sig",
            )
            engineering_summary_df.to_csv(
                out_dir / "exp02b_engineering_node_count_capacity_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

        _plot_exp02a_control_total_latency(controlled_summary_df, out_dir, show_only=_is_show_only(args))
        _plot_exp02b_engineering_capacity_gb(engineering_summary_df, out_dir, show_only=_is_show_only(args))
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


def _exp02_models(args) -> list[str]:
    raw_value = str(getattr(args, "engineering_model", "") or "")
    models = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not models:
        models = list(EXP02_MODELS)
    valid = [model for model in models if model in MODEL_INPUTS]
    if not valid:
        raise ValueError(f"No valid exp02 models found in '{raw_value}'")
    return valid


def _run_exp02_controlled(args, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = int(getattr(args, "task_block_batch", EXP02_BLOCK_BATCH))
    templates = ["1,2,3,4,5", "5,4,3,2,1", "2,4,1,5,3"]
    long_frames: list[pd.DataFrame] = []

    for model_name in _exp02_models(args):
        input_h, input_w = MODEL_INPUTS[model_name]
        for template_idx, template in enumerate(templates):
            template_tag = template.replace(",", "-")
            run_id = f"exp02a_control_{model_name}_{template_tag}_{template_idx}_{timestamp}"
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
                "--controlled-sat-compute-template",
                template,
                "--controlled-normalize-sat-compute-template",
            ]
            completed = subprocess.run(command, cwd=ROOT)
            if completed.returncode != 0:
                raise RuntimeError(f"Controlled exp02 runner failed for {model_name} template {template}")

            archive_dir = find_run_archive(run_id)
            if archive_dir is None:
                raise RuntimeError(f"Cannot find archived run for {run_id}")
            data_dir = Path(archive_dir) / "data"
            csv_candidates = sorted(data_dir.glob("results_long_*.csv"))
            if not csv_candidates:
                raise RuntimeError(f"No exported results_long CSV found under {data_dir}")

            df = pd.read_csv(csv_candidates[-1])
            df = df[df["algorithm"].isin(["LA-DP", "GS-Only"])].copy()
            df["scenario_template"] = template
            long_frames.append(df)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["block_count"] = int(args.task_block_count)
    long_df["total_latency_ms_50_blocks"] = long_df["latency_ms"].astype(float) * int(args.task_block_count)
    long_df["total_latency_s_50_blocks"] = long_df["total_latency_ms_50_blocks"] / 1000.0
    long_df["total_latency_min_50_blocks"] = long_df["total_latency_s_50_blocks"] / 60.0
    long_df["algorithm_label"] = long_df["algorithm"].map({"LA-DP": "LADP-PMP", "GS-Only": "GS-Only"})

    summary_df = (
        long_df.groupby(["model_name", "pipeline_node_count", "algorithm_label"], dropna=False)
        .agg(
            mean_latency_ms=("latency_ms", "mean"),
            mean_total_latency_s_50_blocks=("total_latency_s_50_blocks", "mean"),
            mean_total_latency_min_50_blocks=("total_latency_min_50_blocks", "mean"),
        )
        .reset_index()
    )
    summary_df["experiment"] = "exp02A_control"
    return long_df, summary_df


def _select_best_candidate_for_satellite_count(candidates, satellite_count: int, min_common_duration_s: float):
    filtered = [
        candidate
        for candidate in candidates
        if int(candidate.satellite_count) == int(satellite_count) and float(candidate.common_duration_s) >= float(min_common_duration_s)
    ]
    if not filtered:
        return None
    filtered.sort(key=lambda item: (-float(item.common_duration_s), int(item.rank)))
    return filtered[0]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_shared_route_pmp(scene, config_path: Path, run_id: str) -> tuple[bool, float, dict]:
    scheduler = Scheduler(net_config_path=str(config_path))
    plans = scheduler.generate_task_and_schedule(
        task_id=f"{scene.slot_id}_exp02b_pmp",
        model_name=scene.task.model_name,
        batch_size=scene.task.batch_size,
        target_h=scene.task.input_h,
        target_w=scene.task.input_w,
        run_id=run_id,
        exp_type="mode_selection",
        mode="theory",
        persist_theory=False,
        algorithm_names=["LA-DP"],
        return_full_plans=True,
        metadata_extra={"sweep_param": "time_slot", "sweep_value": scene.slot_id},
    )
    data = plans.get("LA-DP", {})
    latency = float(data.get("latency", float("inf")))
    plan = data.get("plan") or {}
    feasible = bool(plan) and np.isfinite(latency)
    return feasible, latency, data


def _evaluate_shared_route_gs_only(scene, config: dict) -> tuple[bool, float, dict]:
    model_profile = _build_model_profile(scene)
    env_status = _build_env_status(config)
    solver = PMPSolver(model_profile, env_status)
    latency, plan = solver.solve_bent_pipe()
    feasible = bool(plan) and np.isfinite(float(latency))
    return feasible, float(latency), plan


def _run_exp02_engineering(args, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolver = StkPathResolver()
    config_dir = out_dir / "_exp02b_route_configs"
    rows: list[dict] = []
    batch_size = int(getattr(args, "task_block_batch", EXP02_BLOCK_BATCH))
    min_common_duration_s = float(getattr(args, "min_common_duration_s", 0.0))

    for model_name in _exp02_models(args):
        run_dir = ROOT / MODE_SELECTION_STK_RUNS[model_name]
        scenes = load_stk_slot_scenes(run_dir)
        input_h, input_w = MODEL_INPUTS[model_name]

        for scene in scenes:
            scene_for_eval = replace(
                scene,
                task=TaskSpec(
                    model_name=model_name,
                    batch_size=batch_size,
                    input_h=input_h,
                    input_w=input_w,
                ),
            )
            candidates = resolver.candidate_paths(scene_for_eval)
            for satellite_count in range(1, 6):
                candidate = _select_best_candidate_for_satellite_count(
                    candidates,
                    satellite_count=satellite_count,
                    min_common_duration_s=min_common_duration_s,
                )
                if candidate is None:
                    continue

                config = _network_config_for_candidate(scene_for_eval, candidate)
                config["simulation_paths"]["mode_selection"] = {
                    "slot_id": scene_for_eval.slot_id,
                    "mode_family": "PMP",
                    "route_policy": f"exp02b_shared_route_sat{satellite_count}",
                    "source_stk_run_id": scene_for_eval.source_run_id,
                }
                config_path = config_dir / f"{model_name}_{scene_for_eval.slot_id}_sat{satellite_count}_shared_route.json"
                _write_json(config_path, config)

                pmp_feasible, pmp_latency_ms, _ = _evaluate_shared_route_pmp(
                    scene_for_eval,
                    config_path,
                    run_id=f"exp02b_{model_name}_{scene_for_eval.slot_id}_sat{satellite_count}",
                )
                gs_feasible, gs_latency_ms, _ = _evaluate_shared_route_gs_only(scene_for_eval, config)

                for mode, feasible, latency_ms in (
                    ("LADP-PMP", pmp_feasible, pmp_latency_ms),
                    ("GS-Only", gs_feasible, gs_latency_ms),
                ):
                    row = {
                        "model_name": model_name,
                        "slot_id": scene_for_eval.slot_id,
                        "pipeline_node_count": satellite_count,
                        "mode": mode,
                        "feasible": bool(feasible),
                        "latency_ms": float(latency_ms) if feasible else np.nan,
                        "common_duration_s": float(candidate.common_duration_s),
                        "candidate_rank": int(candidate.rank),
                        "route_hop_count": int(candidate.hop_count),
                        "task_block_batch": batch_size,
                        "task_block_count": int(args.task_block_count),
                    }
                    if feasible:
                        total_ms = float(latency_ms) * int(args.task_block_count)
                        max_blocks = math.floor((float(candidate.common_duration_s) * 1000.0) / float(latency_ms))
                        row.update(
                            {
                                "total_latency_ms_50_blocks": total_ms,
                                "total_latency_s_50_blocks": total_ms / 1000.0,
                                "total_latency_min_50_blocks": total_ms / 60000.0,
                                "max_blocks_in_visibility": max_blocks,
                                "max_raw_images_in_visibility": max_blocks * EXP02_EQ_IMAGES_PER_BLOCK,
                                "max_raw_gib_in_visibility": max_blocks * EXP02_RAW_IMAGE_GIB,
                            }
                        )
                    rows.append(row)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No engineering exp02 rows were produced.")
    feasible_df = long_df[long_df["feasible"]].copy()
    if feasible_df.empty:
        raise RuntimeError("No feasible engineering exp02 rows were produced.")

    summary_df = (
        feasible_df.groupby(["model_name", "pipeline_node_count", "mode"], dropna=False)
        .agg(
            mean_total_latency_s_50_blocks=("total_latency_s_50_blocks", "mean"),
            mean_total_latency_min_50_blocks=("total_latency_min_50_blocks", "mean"),
            mean_max_blocks_in_visibility=("max_blocks_in_visibility", "mean"),
            mean_max_raw_images_in_visibility=("max_raw_images_in_visibility", "mean"),
            mean_max_raw_gib_in_visibility=("max_raw_gib_in_visibility", "mean"),
            mean_common_duration_s=("common_duration_s", "mean"),
            slot_count=("slot_id", "nunique"),
        )
        .reset_index()
    )
    summary_df["experiment"] = "exp02B_engineering"
    return long_df, summary_df


def _plot_exp02a_control_total_latency(df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True)
    style_map = {
        "LADP-PMP": ("#244C85", "o", "-"),
        "GS-Only": ("#4A4A4A", "D", "--"),
    }
    models = [model for model in ["yolov5", "resnet101", "vgg19"] if model in set(df["model_name"])]
    handles = []
    labels = []
    for ax, model_name in zip(axes, models):
        model_df = df[df["model_name"] == model_name]
        for mode in ["LADP-PMP", "GS-Only"]:
            sub = model_df[model_df["algorithm_label"] == mode].sort_values("pipeline_node_count")
            x = sub["pipeline_node_count"].astype(float).to_numpy()
            y = sub["mean_total_latency_min_50_blocks"].astype(float).to_numpy()
            color, marker, linestyle = style_map[mode]
            line, = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=2.0, markersize=5.4, label=mode)
            if mode not in labels:
                handles.append(line)
                labels.append(mode)
        ax.set_title(MODEL_LABELS[model_name], fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(sorted(model_df["pipeline_node_count"].astype(int).unique()))
        ax.grid(axis="y", which="major", alpha=0.85)
        ax.grid(axis="y", which="minor", alpha=0.3, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(9))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel("LEO节点数")
    axes[0].set_ylabel("总时延 / min")
    fig.suptitle("固定资源条件下中继节点数对总时延的影响", fontsize=15, fontweight="bold", y=0.975)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _finalize_figure(
        fig,
        out_dir / "exp02a_control_node_count_total_latency.png",
        out_dir / "exp02a_control_node_count_total_latency.pdf",
        show_only=show_only,
    )


def _plot_exp02b_engineering_total_latency(df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)
    axes = axes.flatten()
    style_map = {
        "LADP-PMP": ("#244C85", "o", "-"),
        "GS-Only": ("#4A4A4A", "D", "--"),
    }
    models = [model for model in EXP02_MODELS if model in set(df["model_name"])]
    handles = []
    labels = []
    for ax, model_name in zip(axes, models):
        model_df = df[df["model_name"] == model_name]
        slot_count = int(model_df["slot_count"].max()) if not model_df.empty else 0
        for mode in ["LADP-PMP", "GS-Only"]:
            sub = model_df[model_df["mode"] == mode].sort_values("pipeline_node_count")
            x = sub["pipeline_node_count"].astype(float).to_numpy()
            y = sub["mean_total_latency_min_50_blocks"].astype(float).to_numpy()
            color, marker, linestyle = style_map[mode]
            line, = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=2.0, markersize=5.4, label=mode)
            if mode not in labels:
                handles.append(line)
                labels.append(mode)
        ax.set_title(f"{MODEL_LABELS[model_name]}（{slot_count}个时间片）", fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(sorted(model_df["pipeline_node_count"].astype(int).unique()))
        ax.grid(axis="y", alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx, ax in enumerate(axes):
        if idx >= len(models):
            ax.axis("off")
            continue
        if idx % 2 == 0:
            ax.set_ylabel("总时延 / min")
        if idx >= 2:
            ax.set_xlabel("参与协作的 LEO 节点数")
    fig.suptitle("实验2B：时间窗口场景下 50 个任务块总时延", fontsize=16, fontweight="bold", y=0.98)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.945))
    _finalize_figure(
        fig,
        out_dir / "exp02b_engineering_total_latency_50_blocks.png",
        out_dir / "exp02b_engineering_total_latency_50_blocks.pdf",
        show_only=show_only,
    )


def _plot_exp02b_engineering_capacity_gb(df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True)
    colors = {"LADP-PMP": "#244C85", "GS-Only": "#9A9A9A"}
    width = 0.36
    models = [model for model in ["yolov5", "resnet101", "vgg19"] if model in set(df["model_name"])]
    handles = []
    labels = []
    for ax, model_name in zip(axes, models):
        model_df = df[df["model_name"] == model_name].sort_values("pipeline_node_count")
        node_counts = sorted(model_df["pipeline_node_count"].astype(int).unique())
        x = np.arange(len(node_counts), dtype=float)
        for offset, mode in [(-width / 2, "LADP-PMP"), (width / 2, "GS-Only")]:
            sub = model_df[model_df["mode"] == mode].sort_values("pipeline_node_count")
            y = sub["mean_max_raw_gib_in_visibility"].astype(float).to_numpy()
            bars = ax.bar(x + offset, y, width=width, color=colors[mode], label=mode)
            if mode not in labels:
                handles.append(bars[0])
                labels.append(mode)
        ax.set_title(MODEL_LABELS[model_name], fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(node_counts)
        ax.yaxis.set_major_locator(plt.MaxNLocator(9))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="y", which="major", alpha=0.85)
        ax.grid(axis="y", which="minor", alpha=0.3, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("LEO节点数")
    axes[0].set_ylabel("可处理数据量 / GiB")
    fig.suptitle("时间窗口内可处理数据量", fontsize=15, fontweight="bold", y=0.975)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _finalize_figure(
        fig,
        out_dir / "exp02b_engineering_visibility_capacity_gib.png",
        out_dir / "exp02b_engineering_visibility_capacity_gib.pdf",
        show_only=show_only,
    )


def _plot_exp02b_engineering_capacity_images(df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)
    axes = axes.flatten()
    colors = {"LADP-PMP": "#244C85", "GS-Only": "#9A9A9A"}
    width = 0.36
    models = [model for model in EXP02_MODELS if model in set(df["model_name"])]
    handles = []
    labels = []
    for ax, model_name in zip(axes, models):
        model_df = df[df["model_name"] == model_name].sort_values("pipeline_node_count")
        node_counts = sorted(model_df["pipeline_node_count"].astype(int).unique())
        x = np.arange(len(node_counts), dtype=float)
        for offset, mode in [(-width / 2, "LADP-PMP"), (width / 2, "GS-Only")]:
            sub = model_df[model_df["mode"] == mode].sort_values("pipeline_node_count")
            y = sub["mean_max_raw_images_in_visibility"].astype(float).to_numpy()
            bars = ax.bar(x + offset, y, width=width, color=colors[mode], label=mode)
            if mode not in labels:
                handles.append(bars[0])
                labels.append(mode)
        ax.set_title(MODEL_LABELS[model_name], fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(node_counts)
        ax.grid(axis="y", alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx, ax in enumerate(axes):
        if idx >= len(models):
            ax.axis("off")
            continue
        if idx % 2 == 0:
            ax.set_ylabel("最大可处理等效原始图像数 / 张")
        if idx >= 2:
            ax.set_xlabel("参与协作的 LEO 节点数")
    fig.suptitle("实验2B：时间窗口内最大可处理等效 4096×4096 遥感图像数", fontsize=16, fontweight="bold", y=0.98)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.945))
    _finalize_figure(
        fig,
        out_dir / "exp02b_engineering_visibility_capacity_images.png",
        out_dir / "exp02b_engineering_visibility_capacity_images.pdf",
        show_only=show_only,
    )


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


def _build_cdp_env_exp04(profile: dict, worker_count: int) -> dict:
    base_compute_ms = float(profile["compute_full_model_ms"])
    pool = [
        {"compute": 3.0, "b_dist": 5000.0, "b_return": 100.0, "dist_prop": 3.0, "return_prop": 3.0},
        {"compute": 3.8, "b_dist": 5000.0, "b_return": 210.0, "dist_prop": 2.1, "return_prop": 2.0},
        {"compute": 2.4, "b_dist": 5000.0, "b_return": 80.0, "dist_prop": 4.0, "return_prop": 3.5},
        {"compute": 3.4, "b_dist": 5000.0, "b_return": 165.0, "dist_prop": 2.4, "return_prop": 2.6},
        {"compute": 2.8, "b_dist": 5000.0, "b_return": 95.0, "dist_prop": 3.7, "return_prop": 3.2},
    ]
    templates = [dict(item) for item in pool[:worker_count]]
    if templates:
        mean_compute = sum(item["compute"] for item in templates) / len(templates)
        if mean_compute > 0:
            scale = 3.0 / mean_compute
            for item in templates:
                item["compute"] *= scale
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


def _single_leo_mean_latency(profile: dict) -> float:
    mean_node = {
        "id": "SAT-MEAN",
        "compute_full_model_ms": _scale_worker_compute(float(profile["compute_full_model_ms"]), 3.0),
        "b_dist_mbps": 5000.0,
        "b_return_mbps": 100.0,
        "dist_prop_ms": 3.0,
        "return_prop_ms": 3.0,
    }
    solver = CDPSolver(profile, {"nodes": [mean_node]})
    latency, _ = solver.solve_uniform()
    return float(latency)


def _random_allocation_latency(solver: CDPSolver, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    latencies = []
    for _ in range(max(1, repeats)):
        weights = rng.random(solver.K)
        weights = weights / np.sum(weights)
        latency, _ = solver._evaluate_delay(solver.D_in * weights)
        latencies.append(float(latency))
    return float(np.mean(latencies)), float(np.std(latencies))


def _evaluate_cdp_algorithms(
    profile: dict,
    env: dict,
    seed: int,
    random_repeats: int,
    sat_only_latency_override: float | None = None,
) -> list[dict]:
    solver = CDPSolver(profile, env)
    sat_only = float(sat_only_latency_override) if sat_only_latency_override is not None else _sat_only_latency(profile, env)
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


def _plot_exp03_model(df: pd.DataFrame, model_name: str, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    model_df = df[df["model_name"] == model_name].copy()
    sub = model_df[model_df["scenario"] == "heterogeneous"].copy()
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    batch_order = sorted(sub["batch_size"].unique())
    x = np.arange(len(batch_order), dtype=float) * 1.35
    width = 0.18
    offsets = np.linspace(-2 * width, 2 * width, num=len(CDP_ALG_ORDER))
    y_values: list[float] = []
    handles = []
    labels = []
    for offset, algorithm in zip(offsets, CDP_ALG_ORDER):
        alg_df = sub[sub["algorithm"] == algorithm].sort_values("batch_size")
        if alg_df.empty:
            continue
        y = alg_df["norm_latency_vs_sat_only"].to_numpy(dtype=float)
        y_values.extend(y.tolist())
        bars = ax.bar(
            x + offset,
            y,
            width=width,
            color=CDP_ALG_COLOR[algorithm],
            edgecolor="white",
            linewidth=0.7,
            label=CDP_ALG_LABEL[algorithm],
        )
        handles.append(bars[0])
        labels.append(CDP_ALG_LABEL[algorithm])
        for bar, value in zip(bars, y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.6,
                color=CDP_ALG_COLOR[algorithm],
                rotation=0,
                clip_on=False,
            )
    ax.set_title("CDP模式下不同算法的归一化时延对比", pad=10, fontsize=15, fontweight="bold")
    ax.set_xlabel("输入数据量（样本数）")
    ax.set_ylabel("归一化时延（Single-LEO = 1）")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(batch)) for batch in batch_order])
    _style_cdp_axis(ax, y_values)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    stem = f"exp03_lawa_cdp_data_sensitivity_{model_name}"
    _finalize_figure(fig, out_dir / f"{stem}.png", out_dir / f"{stem}.pdf", show_only=show_only)


def _plot_exp04_model(df: pd.DataFrame, model_name: str, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    model_df = df[df["model_name"] == model_name].copy()
    style_map = {
        "LAWA": ("#244C85", "o", "-"),
        "Sat-Only": ("#4A4A4A", "D", "--"),
    }

    def _draw(y_field: str, title: str, ylabel: str, stem_suffix: str) -> None:
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        y_values: list[float] = []
        for algorithm in ["LAWA", "Sat-Only"]:
            alg_df = model_df[model_df["algorithm"] == algorithm].sort_values("worker_count")
            if alg_df.empty:
                continue
            x = alg_df["worker_count"].to_numpy(dtype=float)
            if y_field == "latency_total_min":
                y = (alg_df["latency_ms"].to_numpy(dtype=float) * 50.0) / 1000.0 / 60.0
            else:
                y = alg_df["norm_latency_vs_sat_only"].to_numpy(dtype=float)
            y_values.extend(y.tolist())
            color, marker, linestyle = style_map[algorithm]
            ax.plot(
                x,
                y,
                label=CDP_ALG_LABEL[algorithm],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=5.2,
            )
        ax.set_title(title, pad=12, fontsize=15, fontweight="bold")
        ax.set_xlabel("参与 LEO 节点数")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(model_df["worker_count"].unique()))
        _style_cdp_axis(ax, y_values)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
        fig.tight_layout()
        stem = f"exp04_lawa_cdp_worker_count_sensitivity_{model_name}{stem_suffix}"
        _finalize_figure(fig, out_dir / f"{stem}.png", out_dir / f"{stem}.pdf", show_only=show_only)

    _draw("latency_total_min", "CDP 节点数与总时延", "总时延 / min", "")
    _draw("norm_latency_vs_sat_only", "CDP 节点数与归一化时延", "归一化时延（Single-LEO = 1）", "_normalized")

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
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp03_show")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    batches = [int(value) for value in args.data_sizes.split(",") if value.strip()]
    profile_path = ROOT / args.profile
    rows: list[dict] = []
    try:
        for model_name in models:
            for batch_size in batches:
                raw_profile = _load_cdp_model_profile(model_name, batch_size, profile_path)
                cdp_profile = {
                    "input_size_mb": raw_profile["input_size_mb"],
                    "output_size_mb": raw_profile["output_size_mb"],
                    "batch_size": raw_profile["batch_size"],
                }
                for scenario in ["heterogeneous"]:
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
        if not _is_show_only(args):
            long_csv = out_dir / "exp03_lawa_cdp_data_sensitivity_long.csv"
            summary_csv = out_dir / "exp03_lawa_cdp_data_sensitivity_summary.csv"
            df.to_csv(long_csv, index=False, encoding="utf-8-sig")
            df.drop(columns=["plan_json"]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        for model_name in models:
            _plot_exp03_model(df, model_name, out_dir, show_only=_is_show_only(args))
        if not _is_show_only(args):
            _write_cdp_notes(
                out_dir,
                "exp03_lawa_cdp_data_sensitivity",
                "实验 3：LAWA-CDP 模式的数据量敏感性实验",
                models,
                f"python thesis_entry.py exp03 --out-dir {args.out_dir}",
                  [
                      "实验 3 使用 PC profile，并仅保留异构 worker 场景。",
                      "横轴 batch 从 64 扩展到 512，用于观察 CDP 在更大任务输入下的相对时延变化。",
                      "LAWA 的优势主要体现在异构节点间的数据分配能力。",
                  ],
              )
            print(out_dir)
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


def _run_exp04(args) -> int:
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp04_show")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    worker_counts = [int(value) for value in args.worker_counts.split(",") if value.strip()]
    profile_path = ROOT / args.profile
    rows: list[dict] = []
    try:
        for model_name in models:
            raw_profile = _load_cdp_model_profile(model_name, args.data_size, profile_path)
            cdp_profile = {
                "input_size_mb": raw_profile["input_size_mb"],
                "output_size_mb": raw_profile["output_size_mb"],
                "batch_size": raw_profile["batch_size"],
            }
            single_leo_latency = _single_leo_mean_latency(raw_profile)
            for worker_count in worker_counts:
                env = _build_cdp_env_exp04(raw_profile, worker_count=worker_count)
                for result in _evaluate_cdp_algorithms(
                    cdp_profile,
                    env,
                    seed=args.seed + worker_count + len(model_name),
                    random_repeats=args.random_repeats,
                    sat_only_latency_override=single_leo_latency,
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
        if not _is_show_only(args):
            long_csv = out_dir / "exp04_lawa_cdp_worker_count_sensitivity_long.csv"
            summary_csv = out_dir / "exp04_lawa_cdp_worker_count_sensitivity_summary.csv"
            df.to_csv(long_csv, index=False, encoding="utf-8-sig")
            df.drop(columns=["plan_json"]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        for model_name in models:
            _plot_exp04_model(df, model_name, out_dir, show_only=_is_show_only(args))
        if not _is_show_only(args):
            _write_cdp_notes(
                out_dir,
                "exp04_lawa_cdp_worker_count_sensitivity",
                "实验 4：LAWA-CDP 模式的 worker 数量敏感性实验",
                models,
                f"python thesis_entry.py exp04 --out-dir {args.out_dir}",
                  [
                      "实验 4 使用异构 worker 场景、batch=64，并将任务总量固定为 50 个任务块。",
                      "图中仅保留 LAWA-CDP 与 Single-LEO 两种方式，比较分钟级总时延。",
                      "随着参与 LEO 节点数增加，LAWA 更能把新增节点资源转化为总时延收益。",
                  ],
              )
            print(out_dir)
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


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


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _finite_float(value, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _route_leo_count(route: str) -> int:
    nodes = [item.strip().upper() for item in str(route or "").split("->") if item.strip()]
    return sum(1 for node in nodes if node.startswith("LEO") or node.startswith("SAT-"))


def _canonical_mode(mode_family: str) -> str | None:
    return MODE_SELECTION_ALIASES.get(str(mode_family))


def _mode_selection_source_dir(model_name: str) -> Path:
    if model_name not in MODE_SELECTION_STK_RUNS:
        raise KeyError(f"No STK dynamic source directory configured for model {model_name!r}.")
    return ROOT / MODE_SELECTION_STK_RUNS[model_name]


def _ensure_mode_selection_results(
    model_name: str,
    batch_size: int,
    args,
    out_dir: Path,
    tag: str,
    worker_count: int | None = None,
    profile_device: str | None = None,
    sat_memory_range_mb: str | None = None,
    sat_memory_values_mb: str | None = None,
    sat_compute_scale: float | None = None,
    isl_bandwidth_scale: float | None = None,
    gsl_bandwidth_scale: float | None = None,
) -> tuple[pd.DataFrame, Path]:
    source_dir = _mode_selection_source_dir(model_name)
    source_out_dir = out_dir / "_mode_selection_sources" / f"{tag}_{model_name}_b{batch_size}"
    csv_path = source_out_dir / "data" / "slot_mode_results.csv"
    metadata_path = source_out_dir / "metadata.json"

    reuse = bool(getattr(args, "reuse_mode_results", False))
    if reuse and profile_device and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reuse = str(existing_metadata.get("profile_device", "mixed")) == str(profile_device)
        except (OSError, json.JSONDecodeError):
            reuse = False
    if reuse and sat_memory_range_mb and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reuse = str(existing_metadata.get("sat_memory_range_mb", "")) == str(sat_memory_range_mb)
        except (OSError, json.JSONDecodeError):
            reuse = False
    if reuse and sat_memory_values_mb and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reuse = str(existing_metadata.get("sat_memory_values_mb", "")) == str(sat_memory_values_mb)
        except (OSError, json.JSONDecodeError):
            reuse = False
    if reuse and sat_compute_scale is not None and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reuse = str(existing_metadata.get("sat_compute_scale", "")) == str(sat_compute_scale)
        except (OSError, json.JSONDecodeError):
            reuse = False
    if reuse and isl_bandwidth_scale is not None and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reuse = str(existing_metadata.get("isl_bandwidth_scale", "")) == str(isl_bandwidth_scale)
        except (OSError, json.JSONDecodeError):
            reuse = False
    if reuse and gsl_bandwidth_scale is not None and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reuse = str(existing_metadata.get("gsl_bandwidth_scale", "")) == str(gsl_bandwidth_scale)
        except (OSError, json.JSONDecodeError):
            reuse = False
    effective_worker_count = int(worker_count if worker_count is not None else getattr(args, "worker_count", 4))
    if not reuse or not csv_path.exists() or not metadata_path.exists():
        command = [
            sys.executable,
            str(ROOT / "mode_selection_experiment.py"),
            "--stk-run-dir",
            str(source_dir),
            "--run-id",
            f"{tag}_{model_name}_b{batch_size}",
            "--output-dir",
            str(source_out_dir),
            "--batch-size-override",
            str(batch_size),
            "--cdp-max-workers",
            str(effective_worker_count),
            "--shared-pmp-min-hops",
            "3",
        ]
        if profile_device:
            command.extend(["--profile-device", str(profile_device)])
        if sat_memory_range_mb:
            command.extend(["--sat-memory-range-mb", str(sat_memory_range_mb)])
        if sat_memory_values_mb:
            command.extend(["--sat-memory-values-mb", str(sat_memory_values_mb)])
        if sat_compute_scale is not None:
            command.extend(["--sat-compute-scale", str(sat_compute_scale)])
        if isl_bandwidth_scale is not None:
            command.extend(["--isl-bandwidth-scale", str(isl_bandwidth_scale)])
        if gsl_bandwidth_scale is not None:
            command.extend(["--gsl-bandwidth-scale", str(gsl_bandwidth_scale)])
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"mode_selection_experiment.py failed for {model_name} batch={batch_size}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task = metadata.get("effective_task", {})
    df = pd.read_csv(csv_path)
    df["model_name"] = str(task.get("model_name", model_name))
    df["model_label"] = df["model_name"].map(lambda value: MODEL_LABELS.get(value, value))
    df["batch_size"] = int(task.get("batch_size", batch_size))
    df["input_h"] = int(task.get("input_h", MODEL_INPUTS.get(model_name, (0, 0))[0]))
    df["input_w"] = int(task.get("input_w", MODEL_INPUTS.get(model_name, (0, 0))[1]))
    df["mode"] = df["mode_family"].map(_canonical_mode)
    df["feasible_bool"] = df["feasible"].map(_truthy)
    df["latency_ms_num"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["active_sat_count_num"] = pd.to_numeric(df["active_sat_count"], errors="coerce").fillna(0)
    df["pmp_route_leo_count"] = df["route"].map(_route_leo_count)
    df["source_results_csv"] = str(csv_path)
    return df, source_out_dir


def _mode_row_by_plot_mode(slot_df: pd.DataFrame, mode: str) -> pd.Series | None:
    sub = slot_df[slot_df["mode"] == mode]
    if sub.empty:
        return None
    return sub.iloc[0]


def _strict_representative_slots(df: pd.DataFrame, min_pmp_route_leo: int, min_cdp_active_sats: int) -> list[str]:
    slots: list[str] = []
    for slot_id, slot_df in df.groupby("slot_id", sort=True):
        pmp = _mode_row_by_plot_mode(slot_df, "PMP")
        cdp = _mode_row_by_plot_mode(slot_df, "CDP")
        gs = _mode_row_by_plot_mode(slot_df, "GS-Only")
        fwms = _mode_row_by_plot_mode(slot_df, "FWMS")
        if pmp is None or cdp is None or gs is None or fwms is None:
            continue
        if not (_truthy(pmp["feasible_bool"]) and _truthy(cdp["feasible_bool"]) and _truthy(gs["feasible_bool"]) and _truthy(fwms["feasible_bool"])):
            continue
        if int(pmp["pmp_route_leo_count"]) < int(min_pmp_route_leo):
            continue
        if int(cdp["active_sat_count_num"]) < int(min_cdp_active_sats):
            continue
        slots.append(str(slot_id))
    return slots


def _fallback_representative_slots(df: pd.DataFrame, min_pmp_route_leo: int) -> list[str]:
    slots: list[str] = []
    for slot_id, slot_df in df.groupby("slot_id", sort=True):
        pmp = _mode_row_by_plot_mode(slot_df, "PMP")
        gs = _mode_row_by_plot_mode(slot_df, "GS-Only")
        fwms = _mode_row_by_plot_mode(slot_df, "FWMS")
        if pmp is None or gs is None or fwms is None:
            continue
        if not (_truthy(pmp["feasible_bool"]) and _truthy(gs["feasible_bool"]) and _truthy(fwms["feasible_bool"])):
            continue
        if int(pmp["pmp_route_leo_count"]) < int(min_pmp_route_leo):
            continue
        slots.append(str(slot_id))
    return slots


def _choose_median_pmp_slot(df: pd.DataFrame, slots: list[str]) -> str:
    candidates = []
    for slot_id in sorted(slots):
        slot_df = df[df["slot_id"] == slot_id]
        pmp = _mode_row_by_plot_mode(slot_df, "PMP")
        latency = _finite_float(pmp["latency_ms_num"] if pmp is not None else np.nan)
        if np.isfinite(latency):
            candidates.append((slot_id, latency))
    if not candidates:
        raise ValueError("No representative slot has finite PMP latency.")
    median_latency = float(np.median([item[1] for item in candidates]))
    return min(candidates, key=lambda item: (abs(item[1] - median_latency), item[0]))[0]


def _selected_mode_from_plan(row: pd.Series | None) -> str:
    if row is None:
        return ""
    try:
        payload = json.loads(str(row.get("plan_json", "{}")))
    except json.JSONDecodeError:
        return ""
    return str(payload.get("selected_mode", ""))


def _rows_for_representative_slot(
    df: pd.DataFrame,
    slot_id: str,
    selection_status: str,
    strict_slot_count: int,
    fallback_slot_count: int,
    source_dir: Path,
) -> list[dict]:
    slot_df = df[df["slot_id"] == slot_id]
    gs_row = _mode_row_by_plot_mode(slot_df, "GS-Only")
    pmp_row = _mode_row_by_plot_mode(slot_df, "PMP")
    gs_latency = _finite_float(gs_row["latency_ms_num"] if gs_row is not None else np.nan)
    model_name = str(slot_df["model_name"].iloc[0])
    rows = []
    for mode in MODE_ORDER:
        row = _mode_row_by_plot_mode(slot_df, mode)
        feasible = bool(row is not None and _truthy(row["feasible_bool"]))
        latency = _finite_float(row["latency_ms_num"] if row is not None else np.nan)
        reason = str(row.get("reason", "")) if row is not None and pd.notna(row.get("reason", "")) else ""
        if mode == "FWMS":
            selected_mode = _selected_mode_from_plan(row)
            reason = f"selected_{selected_mode}" if selected_mode else reason
        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "batch_size": int(slot_df["batch_size"].iloc[0]),
                "input_h": int(slot_df["input_h"].iloc[0]),
                "input_w": int(slot_df["input_w"].iloc[0]),
                "slot_id": slot_id,
                "mode": mode,
                "mode_family_source": str(row.get("mode_family", "")) if row is not None else "",
                "mode_algo": str(row.get("mode_algo", "")) if row is not None else "",
                "feasible": feasible,
                "reason": reason,
                "latency_ms": latency,
                "norm_latency_vs_gs": latency / gs_latency if feasible and np.isfinite(latency) and gs_latency > 0 else float("nan"),
                "active_sat_count": int(_finite_float(row.get("active_sat_count_num", 0), 0.0)) if row is not None else 0,
                "hop_count": int(_finite_float(row.get("hop_count", 0), 0.0)) if row is not None else 0,
                "route": str(row.get("route", "")) if row is not None else "",
                "pmp_route_leo_count": int(_finite_float(pmp_row.get("pmp_route_leo_count", 0), 0.0)) if pmp_row is not None else 0,
                "strict_slot_count": strict_slot_count,
                "fallback_slot_count": fallback_slot_count,
                "selection_status": selection_status,
                "source_results_csv": str(row.get("source_results_csv", "")) if row is not None else "",
                "source_mode_selection_dir": str(source_dir),
            }
        )
    return rows


def _build_representative_mode_rows(
    df: pd.DataFrame,
    source_dir: Path,
    min_pmp_route_leo: int,
    min_cdp_active_sats: int,
    preferred_slot: str | None = None,
) -> tuple[list[dict], dict]:
    strict_slots = _strict_representative_slots(df, min_pmp_route_leo, min_cdp_active_sats)
    fallback_slots = _fallback_representative_slots(df, min_pmp_route_leo)
    if preferred_slot and preferred_slot in strict_slots:
        slot_id = preferred_slot
        status = "strict_common_slot"
    elif strict_slots:
        slot_id = _choose_median_pmp_slot(df, strict_slots)
        status = "strict_median_pmp_latency"
    elif fallback_slots:
        slot_id = _choose_median_pmp_slot(df, fallback_slots)
        status = "fallback_cdp_not_comparable"
    else:
        raise ValueError("No slot satisfies the minimum PMP route length filter.")
    rows = _rows_for_representative_slot(df, slot_id, status, len(strict_slots), len(fallback_slots), source_dir)
    audit = {
        "model_name": rows[0]["model_name"],
        "batch_size": rows[0]["batch_size"],
        "representative_slot_id": slot_id,
        "selection_status": status,
        "strict_slot_count": len(strict_slots),
        "fallback_slot_count": len(fallback_slots),
        "min_pmp_route_leo": min_pmp_route_leo,
        "min_cdp_active_sats": min_cdp_active_sats,
        "source_mode_selection_dir": str(source_dir),
    }
    return rows, audit


def _common_representative_slot(loaded: list[tuple[pd.DataFrame, Path]], min_pmp_route_leo: int, min_cdp_active_sats: int) -> str | None:
    strict_sets = [
        set(_strict_representative_slots(df, min_pmp_route_leo, min_cdp_active_sats))
        for df, _ in loaded
    ]
    strict_sets = [item for item in strict_sets if item]
    if not strict_sets:
        return None
    common = set.intersection(*strict_sets)
    if not common:
        return None
    scores = []
    for slot_id in sorted(common):
        latencies = []
        for df, _ in loaded:
            pmp = _mode_row_by_plot_mode(df[df["slot_id"] == slot_id], "PMP")
            latency = _finite_float(pmp["latency_ms_num"] if pmp is not None else np.nan)
            if np.isfinite(latency):
                latencies.append(latency)
        if latencies:
            scores.append((slot_id, float(np.mean(latencies))))
    if not scores:
        return None
    median_score = float(np.median([item[1] for item in scores]))
    return min(scores, key=lambda item: (abs(item[1] - median_score), item[0]))[0]


def _baseline_average_slots(df: pd.DataFrame, min_pmp_route_leo: int) -> list[str]:
    slots: list[str] = []
    for slot_id, slot_df in df.groupby("slot_id", sort=True):
        pmp = _mode_row_by_plot_mode(slot_df, "PMP")
        gs = _mode_row_by_plot_mode(slot_df, "GS-Only")
        fwms = _mode_row_by_plot_mode(slot_df, "FWMS")
        if pmp is None or gs is None or fwms is None:
            continue
        if not (_truthy(pmp["feasible_bool"]) and _truthy(gs["feasible_bool"]) and _truthy(fwms["feasible_bool"])):
            continue
        if int(pmp["pmp_route_leo_count"]) < int(min_pmp_route_leo):
            continue
        slots.append(str(slot_id))
    return slots


def _common_average_slots(loaded: list[tuple[pd.DataFrame, Path]], min_pmp_route_leo: int) -> list[str]:
    slot_sets = [set(_baseline_average_slots(df, min_pmp_route_leo)) for df, _ in loaded]
    slot_sets = [item for item in slot_sets if item]
    if not slot_sets:
        return []
    return sorted(set.intersection(*slot_sets))


def _slot_mode_eligible(row: pd.Series | None, mode: str, min_cdp_active_sats: int) -> bool:
    if row is None or not _truthy(row.get("feasible_bool", False)):
        return False
    if mode == "CDP":
        return int(_finite_float(row.get("active_sat_count_num", 0), 0.0)) >= int(min_cdp_active_sats)
    return True


def _aggregate_mode_rows_over_slots(
    df: pd.DataFrame,
    slots: list[str],
    source_dir: Path,
    min_cdp_active_sats: int,
    selection_status: str,
) -> list[dict]:
    if not slots:
        raise ValueError("No common slots available for averaging.")
    model_name = str(df["model_name"].iloc[0])
    batch_size = int(df["batch_size"].iloc[0])
    input_h = int(df["input_h"].iloc[0])
    input_w = int(df["input_w"].iloc[0])
    min_route_leo = min(
        int(_finite_float(_mode_row_by_plot_mode(df[df["slot_id"] == slot_id], "PMP").get("pmp_route_leo_count", 0), 0.0))
        for slot_id in slots
        if _mode_row_by_plot_mode(df[df["slot_id"] == slot_id], "PMP") is not None
    )
    rows = []
    for mode in MODE_ORDER:
        latencies = []
        norm_latencies = []
        active_counts = []
        hop_counts = []
        selected_modes = []
        source_csv = ""
        first_row = None
        for slot_id in slots:
            slot_df = df[df["slot_id"] == slot_id]
            row = _mode_row_by_plot_mode(slot_df, mode)
            gs_row = _mode_row_by_plot_mode(slot_df, "GS-Only")
            if row is None or gs_row is None:
                continue
            if first_row is None:
                first_row = row
            if not source_csv:
                source_csv = str(row.get("source_results_csv", ""))
            if mode == "FWMS":
                selected_mode = _selected_mode_from_plan(row)
                if selected_mode:
                    selected_modes.append(selected_mode)
            if not _slot_mode_eligible(row, mode, min_cdp_active_sats):
                continue
            latency = _finite_float(row.get("latency_ms_num", np.nan))
            gs_latency = _finite_float(gs_row.get("latency_ms_num", np.nan))
            if not np.isfinite(latency):
                continue
            latencies.append(latency)
            if np.isfinite(gs_latency) and gs_latency > 0:
                norm_latencies.append(latency / gs_latency)
            active_counts.append(int(_finite_float(row.get("active_sat_count_num", 0), 0.0)))
            hop_counts.append(int(_finite_float(row.get("hop_count", 0), 0.0)))
        feasible_slot_count = len(latencies)
        reason = ""
        if mode == "FWMS" and selected_modes:
            counts = pd.Series(selected_modes).value_counts()
            reason = ",".join(f"selected_{idx}:{int(val)}" for idx, val in counts.items())
        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "batch_size": batch_size,
                "input_h": input_h,
                "input_w": input_w,
                "slot_id": f"AVG[{len(slots)}_common_slots]",
                "mode": mode,
                "mode_family_source": str(first_row.get("mode_family", "")) if first_row is not None else "",
                "mode_algo": "mean_over_common_slots",
                "feasible": feasible_slot_count > 0,
                "reason": reason,
                "latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
                "norm_latency_vs_gs": float(np.mean(norm_latencies)) if norm_latencies else float("nan"),
                "active_sat_count": float(np.mean(active_counts)) if active_counts else 0,
                "hop_count": float(np.mean(hop_counts)) if hop_counts else 0,
                "route": "avg_over_common_slots",
                "pmp_route_leo_count": min_route_leo,
                "strict_slot_count": len(slots),
                "fallback_slot_count": len(slots),
                "selection_status": selection_status,
                "source_results_csv": source_csv,
                "source_mode_selection_dir": str(source_dir),
                "common_slot_count": len(slots),
                "feasible_slot_count": feasible_slot_count,
            }
        )
    return rows

def _plot_exp05(df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    fig, theory_ax = plt.subplots(figsize=(11.6, 5.4))
    models = [model for model in ["yolov5", "resnet101", "vgg19", "vit_huge"] if model in set(df["model_name"])]
    x = np.arange(len(models))
    width = 0.145
    finite_values = df.loc[df["feasible"].apply(_truthy), "norm_latency_vs_gs"].astype(float).tolist()
    y_max = max(finite_values) if finite_values else 1.0
    infeasible_y = max(0.05, y_max * 0.07)
    for idx, mode in enumerate(MODE_ORDER):
        values = []
        for model in models:
            row = df[(df["model_name"] == model) & (df["mode"] == mode)].iloc[0]
            values.append(float(row["norm_latency_vs_gs"]) if _truthy(row["feasible"]) else np.nan)
        pos = x + (idx - (len(MODE_ORDER) - 1) / 2) * width
        bars = theory_ax.bar(
            pos,
            values,
            width=width,
            label=MODE_LABEL[mode],
            color=MODE_COLOR[mode],
            edgecolor="white",
            linewidth=0.8,
            hatch="//" if mode == "FWMS" else None,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                theory_ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for model_idx, model in enumerate(models):
        for mode_idx, mode in enumerate(MODE_ORDER):
            row = df[(df["model_name"] == model) & (df["mode"] == mode)].iloc[0]
            if _truthy(row["feasible"]):
                continue
            xpos = model_idx + (mode_idx - (len(MODE_ORDER) - 1) / 2) * width
            theory_ax.scatter(
                [xpos],
                [infeasible_y],
                marker="x",
                s=46,
                linewidths=1.2,
                color=MODE_COLOR[mode],
                zorder=6,
            )
            theory_ax.text(
                xpos,
                infeasible_y + y_max * 0.035,
                "不可行",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=MODE_COLOR[mode],
                rotation=90,
            )
    theory_ax.set_title("实验5：模式选择有效性", pad=12, fontsize=15, fontweight="bold")
    theory_ax.set_xticks(x)
    theory_ax.set_xticklabels([MODEL_LABELS[model] for model in models])
    theory_ax.set_ylabel("归一化时延（GS-Only = 1）")
    theory_ax.grid(True, axis="y", alpha=0.75)
    theory_ax.spines["top"].set_visible(False)
    theory_ax.spines["right"].set_visible(False)
    theory_ax.set_ylim(0.0, y_max * 1.16)
    theory_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=5, frameon=False)
    fig.tight_layout()
    _finalize_figure(
        fig,
        out_dir / "exp05_fwms_mode_selection_effectiveness.png",
        out_dir / "exp05_fwms_mode_selection_effectiveness.pdf",
        show_only=show_only,
    )


def _plot_exp06(df: pd.DataFrame, out_dir: Path, model_name: str, show_only: bool = False) -> None:
    setup_style()
    stem = f"exp06_fwms_data_sensitivity_{model_name}"
    mode_label_map = dict(MODE_LABEL)
    mode_label_map["Sat-Only"] = "Single-LEO"

    def _plot_mode_bars(value_col: str, ylabel: str, title: str, suffix: str) -> None:
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        finite_values: list[float] = []
        batches = sorted(df["batch_size"].unique())
        x = np.arange(len(batches), dtype=float)
        width = 0.14
        offsets = np.linspace(-2 * width, 2 * width, num=len(MODE_ORDER))
        mode_to_idx = {mode: idx for idx, mode in enumerate(MODE_ORDER)}

        for offset, mode in zip(offsets, MODE_ORDER):
            sub = df[df["mode"] == mode].sort_values("batch_size")
            batch_to_row = {int(row["batch_size"]): row for _, row in sub.iterrows()}
            y_values = []
            feasible_flags = []
            for batch in batches:
                row = batch_to_row.get(int(batch))
                if row is None:
                    y_values.append(np.nan)
                    feasible_flags.append(False)
                else:
                    value = float(row[value_col]) if pd.notna(row[value_col]) else float("nan")
                    y_values.append(value)
                    feasible_flags.append(_truthy(row.get("feasible", False)))

            bar_heights = [value if np.isfinite(value) else 0.0 for value in y_values]
            finite_values.extend([float(value) for value in y_values if np.isfinite(float(value))])
            bars = ax.bar(
                x + offset,
                bar_heights,
                width=width,
                color=MODE_COLOR[mode],
                edgecolor="white",
                linewidth=0.7,
                label=mode_label_map[mode],
                zorder=MODE_ZORDER[mode],
            )

            for bar, value in zip(bars, y_values):
                if not np.isfinite(float(value)):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(0.01, bar.get_height() * 0.02),
                    f"{value:.2f}" if value_col == "norm_latency_vs_gs" else f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    color=MODE_COLOR[mode],
                    rotation=90,
                    clip_on=False,
                )

            for batch_x, value, feasible in zip(x + offset, y_values, feasible_flags):
                if np.isfinite(float(value)):
                    continue
                if not feasible:
                    max_value = max(finite_values) if finite_values else 1.0
                    anchor = max_value * 0.03
                    y_step = max_value * 0.028
                    ax.text(
                        batch_x,
                        anchor + mode_to_idx[mode] * y_step,
                        "不可行",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color=MODE_COLOR[mode],
                        rotation=90,
                    )
        ax.set_title(title, pad=12, fontsize=15, fontweight="bold")
        ax.set_xlabel("输入数据量（样本数）")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(batch)) for batch in batches])
        ax.grid(True, axis="y", alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if finite_values:
            ax.set_ylim(0.0, max(finite_values) * 1.16)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)
        fig.tight_layout()
        _finalize_figure(
            fig,
            out_dir / f"{stem}{suffix}.png",
            out_dir / f"{stem}{suffix}.pdf",
            show_only=show_only,
        )

    _plot_mode_bars(
        "latency_ms",
        "平均端到端时延 / ms",
        "实验6：输入数据量敏感性",
        "",
    )
    _plot_mode_bars(
        "norm_latency_vs_gs",
        "归一化时延（GS-Only = 1）",
        "实验6：输入数据量敏感性",
        "_normalized",
    )


EXP07_SCENARIOS = [
    {
        "scenario_id": "tight_memory",
        "scenario_label": "紧内存",
        "sat_memory_range_mb": "4096,6144",
        "sat_compute_scale": 0.85,
        "isl_bandwidth_scale": 0.90,
        "gsl_bandwidth_scale": 0.90,
        "description": "内存偏紧，星上算力略弱，重点考察可行性边界。",
    },
    {
        "scenario_id": "balanced",
        "scenario_label": "均衡",
        "sat_memory_range_mb": "6144,10240",
        "sat_compute_scale": 1.00,
        "isl_bandwidth_scale": 1.00,
        "gsl_bandwidth_scale": 1.00,
        "description": "作为默认资源基线，观察不同模型的自然模式偏好。",
    },
    {
        "scenario_id": "pipeline_friendly",
        "scenario_label": "高ISL",
        "sat_memory_range_mb": "6144,10240",
        "sat_compute_scale": 1.10,
        "isl_bandwidth_scale": 1.60,
        "gsl_bandwidth_scale": 0.70,
        "description": "放大星间链路优势，压低星地回传，考察 PMP 的适用场景。",
    },
    {
        "scenario_id": "parallel_friendly",
        "scenario_label": "高并行",
        "sat_memory_range_mb": "12288,16384",
        "sat_compute_scale": 1.70,
        "isl_bandwidth_scale": 1.00,
        "gsl_bandwidth_scale": 1.35,
        "description": "给足内存和并行算力，考察 CDP 何时成为最优模式。",
    },
    {
        "scenario_id": "return_limited",
        "scenario_label": "低GSL",
        "sat_memory_range_mb": "12288,16384",
        "sat_compute_scale": 1.50,
        "isl_bandwidth_scale": 1.00,
        "gsl_bandwidth_scale": 0.35,
        "description": "保持星上资源充足，但压低回传带宽，考察并行模式是否受限。",
    },
    {
        "scenario_id": "weak_sat",
        "scenario_label": "弱星侧",
        "sat_memory_range_mb": "4096,8192",
        "sat_compute_scale": 0.55,
        "isl_bandwidth_scale": 0.65,
        "gsl_bandwidth_scale": 1.00,
        "description": "压低星上计算和链路能力，检验 FWMS 是否能主动退回保守模式。",
    },
]
EXP07_SCENARIO_BY_ID = {item["scenario_id"]: item for item in EXP07_SCENARIOS}
EXP07_MODE_CODE = {"GS-Only": 0, "PMP": 1, "Sat-Only": 2, "CDP": 3, "": 4}
EXP07_MODE_COLOR = {
    "GS-Only": "#4A4A4A",
    "PMP": "#244C85",
    "Sat-Only": "#8A63D2",
    "CDP": "#E39D2D",
    "": "#D1D5DB",
}
EXP07_MODE_SHORT = {
    "GS-Only": "GS",
    "PMP": "PMP",
    "Sat-Only": "S-LEO",
    "CDP": "CDP",
    "": "N/A",
}


def _parse_exp07_scenarios(raw_value: str) -> list[dict]:
    requested = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not requested:
        return list(EXP07_SCENARIOS)
    scenarios = []
    for scenario_id in requested:
        if scenario_id not in EXP07_SCENARIO_BY_ID:
            raise KeyError(f"Unknown exp07 scenario: {scenario_id}")
        scenarios.append(EXP07_SCENARIO_BY_ID[scenario_id])
    return scenarios


def _exp07_selected_mode(row: pd.Series | None) -> str:
    if row is None:
        return ""
    try:
        payload = json.loads(str(row.get("plan_json", "") or "{}"))
    except json.JSONDecodeError:
        return ""
    return _canonical_mode(str(payload.get("selected_mode", ""))) or ""


def _mode_mix_string(values: pd.Series) -> str:
    counts = values[values.astype(str) != ""].value_counts()
    if counts.empty:
        return ""
    return "|".join(f"{MODE_LABEL.get(mode, mode)}:{int(count)}" for mode, count in counts.items())


def _dominant_mode(values: pd.Series) -> str:
    counts = values[values.astype(str) != ""].value_counts()
    return str(counts.index[0]) if not counts.empty else ""


def _build_exp07_resource_rows(df: pd.DataFrame, model_name: str, batch_size: int, scenario: dict) -> list[dict]:
    rows: list[dict] = []
    for slot_id, slot_df in df.groupby("slot_id", sort=True):
        fwms_row = slot_df[slot_df["mode_family"] == "FWMS-Feature"]
        oracle_row = slot_df[slot_df["mode_family"] == "Oracle-Min-Latency"]
        fwms = fwms_row.iloc[0] if not fwms_row.empty else None
        oracle = oracle_row.iloc[0] if not oracle_row.empty else None

        fwms_mode = _exp07_selected_mode(fwms)
        oracle_mode = _exp07_selected_mode(oracle)
        fwms_latency = _finite_float(fwms.get("latency_ms_num", np.nan) if fwms is not None else np.nan)
        oracle_latency = _finite_float(oracle.get("latency_ms_num", np.nan) if oracle is not None else np.nan)
        regret_pct = float("nan")
        if np.isfinite(fwms_latency) and np.isfinite(oracle_latency) and oracle_latency > 0:
            regret_pct = max(0.0, (fwms_latency - oracle_latency) / oracle_latency * 100.0)

        try:
            fwms_payload = json.loads(str(fwms.get("plan_json", "") or "{}")) if fwms is not None else {}
        except json.JSONDecodeError:
            fwms_payload = {}

        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "batch_size": batch_size,
                "slot_id": str(slot_id),
                "scenario_id": scenario["scenario_id"],
                "scenario_label": scenario["scenario_label"],
                "sat_memory_range_mb": scenario["sat_memory_range_mb"],
                "sat_compute_scale": scenario["sat_compute_scale"],
                "isl_bandwidth_scale": scenario["isl_bandwidth_scale"],
                "gsl_bandwidth_scale": scenario["gsl_bandwidth_scale"],
                "fwms_selected_mode": fwms_mode,
                "oracle_selected_mode": oracle_mode,
                "selection_correct": bool(fwms_mode != "" and fwms_mode == oracle_mode),
                "fwms_latency_ms": fwms_latency,
                "oracle_latency_ms": oracle_latency,
                "regret_pct": regret_pct,
                "fwms_feasible": bool(fwms is not None and _truthy(fwms.get("feasible_bool", False))),
                "oracle_feasible": bool(oracle is not None and _truthy(oracle.get("feasible_bool", False))),
                "fwms_reason": str(fwms_payload.get("decision_reason", "")),
                "source_results_csv": str(fwms.get("source_results_csv", "")) if fwms is not None else "",
            }
        )
    return rows


def _summarize_exp07(summary_source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model_name, scenario_id), sub in summary_source.groupby(["model_name", "scenario_id"], sort=True):
        valid_regret = pd.to_numeric(sub["regret_pct"], errors="coerce")
        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "batch_size": int(sub["batch_size"].iloc[0]),
                "scenario_id": scenario_id,
                "scenario_label": str(sub["scenario_label"].iloc[0]),
                "sat_memory_range_mb": str(sub["sat_memory_range_mb"].iloc[0]),
                "sat_compute_scale": float(sub["sat_compute_scale"].iloc[0]),
                "isl_bandwidth_scale": float(sub["isl_bandwidth_scale"].iloc[0]),
                "gsl_bandwidth_scale": float(sub["gsl_bandwidth_scale"].iloc[0]),
                "slot_count": len(sub),
                "selection_accuracy_pct": float(sub["selection_correct"].mean() * 100.0),
                "mean_regret_pct": float(valid_regret.dropna().mean()) if valid_regret.notna().any() else float("nan"),
                "median_regret_pct": float(valid_regret.dropna().median()) if valid_regret.notna().any() else float("nan"),
                "dominant_oracle_mode": _dominant_mode(sub["oracle_selected_mode"]),
                "dominant_fwms_mode": _dominant_mode(sub["fwms_selected_mode"]),
                "oracle_mode_mix": _mode_mix_string(sub["oracle_selected_mode"]),
                "fwms_mode_mix": _mode_mix_string(sub["fwms_selected_mode"]),
            }
        )
    return pd.DataFrame(rows)


def _summarize_exp07_overall(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario_id, sub in summary_df.groupby("scenario_id", sort=False):
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_label": str(sub["scenario_label"].iloc[0]),
                "mean_accuracy_pct": float(sub["selection_accuracy_pct"].mean()),
                "mean_regret_pct": float(pd.to_numeric(sub["mean_regret_pct"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_exp07_mode_map(summary_df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    models = [model for model in ["yolov5", "resnet101", "vgg19", "vit_huge"] if model in set(summary_df["model_name"])]
    scenarios = [item for item in EXP07_SCENARIOS if item["scenario_id"] in set(summary_df["scenario_id"])]
    code_matrix = np.full((len(models), len(scenarios)), EXP07_MODE_CODE[""], dtype=float)
    text_matrix: list[list[str]] = [["" for _ in scenarios] for _ in models]

    for row_idx, model_name in enumerate(models):
        for col_idx, scenario in enumerate(scenarios):
            sub = summary_df[
                (summary_df["model_name"] == model_name)
                & (summary_df["scenario_id"] == scenario["scenario_id"])
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            oracle_mode = str(row["dominant_oracle_mode"])
            fwms_mode = str(row["dominant_fwms_mode"])
            code_matrix[row_idx, col_idx] = EXP07_MODE_CODE.get(oracle_mode, EXP07_MODE_CODE[""])
            text_matrix[row_idx][col_idx] = (
                f"O:{EXP07_MODE_SHORT.get(oracle_mode, oracle_mode)}\n"
                f"F:{EXP07_MODE_SHORT.get(fwms_mode, fwms_mode)}\n"
                f"{float(row['selection_accuracy_pct']):.0f}% / {float(row['mean_regret_pct']):.1f}%"
            )

    cmap = ListedColormap(
        [
            EXP07_MODE_COLOR["GS-Only"],
            EXP07_MODE_COLOR["PMP"],
            EXP07_MODE_COLOR["Sat-Only"],
            EXP07_MODE_COLOR["CDP"],
            EXP07_MODE_COLOR[""],
        ]
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.imshow(code_matrix, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([item["scenario_label"] for item in scenarios], fontsize=10)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in models], fontsize=10)
    ax.set_title("资源场景下的最优模式与 FWMS 选择", pad=10, fontsize=15, fontweight="bold")

    for row_idx in range(len(models)):
        for col_idx in range(len(scenarios)):
            text = text_matrix[row_idx][col_idx]
            if not text:
                continue
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.6, color="white")

    ax.set_xticks(np.arange(-0.5, len(scenarios), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor=EXP07_MODE_COLOR[mode], label=MODE_LABEL.get(mode, mode))
        for mode in ["GS-Only", "PMP", "Sat-Only", "CDP"]
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _finalize_figure(
        fig,
        out_dir / "exp07_fwms_resource_mode_map.png",
        out_dir / "exp07_fwms_resource_mode_map.pdf",
        show_only=show_only,
    )


def _plot_exp07_metrics(summary_df: pd.DataFrame, out_dir: Path, show_only: bool = False) -> None:
    setup_style()
    models = [model for model in ["yolov5", "resnet101", "vgg19", "vit_huge"] if model in set(summary_df["model_name"])]
    scenarios = [item for item in EXP07_SCENARIOS if item["scenario_id"] in set(summary_df["scenario_id"])]
    accuracy_matrix = np.full((len(models), len(scenarios)), np.nan, dtype=float)
    regret_matrix = np.full((len(models), len(scenarios)), np.nan, dtype=float)

    for row_idx, model_name in enumerate(models):
        for col_idx, scenario in enumerate(scenarios):
            sub = summary_df[
                (summary_df["model_name"] == model_name)
                & (summary_df["scenario_id"] == scenario["scenario_id"])
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            accuracy_matrix[row_idx, col_idx] = float(row["selection_accuracy_pct"])
            regret_matrix[row_idx, col_idx] = float(row["mean_regret_pct"])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1))
    metric_specs = [
        (accuracy_matrix, axes[0], "匹配率 / %", "YlGn", lambda value: f"{value:.1f}"),
        (regret_matrix, axes[1], "平均 regret / %", "YlOrRd", lambda value: f"{value:.1f}"),
    ]
    for matrix, ax, title, cmap_name, formatter in metric_specs:
        image = ax.imshow(matrix, cmap=plt.get_cmap(cmap_name), aspect="auto")
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels([item["scenario_label"] for item in scenarios], fontsize=10)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in models], fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        for row_idx in range(len(models)):
            for col_idx in range(len(scenarios)):
                value = matrix[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                ax.text(col_idx, row_idx, formatter(value), ha="center", va="center", fontsize=9, color="#111827")
        ax.set_xticks(np.arange(-0.5, len(scenarios), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("FWMS 资源场景实验指标", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _finalize_figure(
        fig,
        out_dir / "exp07_fwms_resource_metrics.png",
        out_dir / "exp07_fwms_resource_metrics.pdf",
        show_only=show_only,
    )


def _run_exp07(args) -> int:
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp07_show")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    scenarios = _parse_exp07_scenarios(args.scenarios)
    try:
        rows: list[dict] = []
        audits: list[dict] = []
        for scenario in scenarios:
            for model_name in models:
                tag = f"exp07_{scenario['scenario_id']}"
                df_source, source_dir = _ensure_mode_selection_results(
                    model_name,
                    args.batch_size,
                    args,
                    out_dir,
                    tag,
                    worker_count=args.worker_count,
                    profile_device=args.profile_device,
                    sat_memory_range_mb=scenario["sat_memory_range_mb"],
                    sat_compute_scale=scenario["sat_compute_scale"],
                    isl_bandwidth_scale=scenario["isl_bandwidth_scale"],
                    gsl_bandwidth_scale=scenario["gsl_bandwidth_scale"],
                )
                scenario_rows = _build_exp07_resource_rows(df_source, model_name, args.batch_size, scenario)
                rows.extend(scenario_rows)
                audits.append(
                    {
                        "model_name": model_name,
                        "batch_size": args.batch_size,
                        "worker_count": args.worker_count,
                        "scenario_id": scenario["scenario_id"],
                        "scenario_label": scenario["scenario_label"],
                        "slot_count": len({row["slot_id"] for row in scenario_rows}),
                        "source_mode_selection_dir": str(source_dir),
                    }
                )

        long_df = pd.DataFrame(rows)
        summary_df = _summarize_exp07(long_df)
        overall_df = _summarize_exp07_overall(summary_df)
        if not _is_show_only(args):
            long_df.to_csv(out_dir / "exp07_fwms_resource_scenarios_long.csv", index=False, encoding="utf-8-sig")
            summary_df.to_csv(out_dir / "exp07_fwms_resource_scenarios_summary.csv", index=False, encoding="utf-8-sig")
            overall_df.to_csv(out_dir / "exp07_fwms_resource_scenarios_overall.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(audits).to_csv(out_dir / "exp07_fwms_resource_scenarios_audit.csv", index=False, encoding="utf-8-sig")
        _plot_exp07_mode_map(summary_df, out_dir, show_only=_is_show_only(args))
        _plot_exp07_metrics(summary_df, out_dir, show_only=_is_show_only(args))
        if not _is_show_only(args):
            notes = [
                "# 实验 7：资源场景下的 FWMS 模式选择验证",
                "",
                f"- 模型：{', '.join(MODEL_LABELS.get(model, model) for model in models)}。",
                f"- batch 固定为 {args.batch_size}。",
                f"- CDP 最大并行卫星数：{args.worker_count}。",
                f"- Profile 口径：{str(args.profile_device).upper()}。",
                "- 对每个资源场景分别重跑 mode_selection_experiment.py，并统计 FWMS 与 Oracle-Min-Latency 的模式一致率和 regret。",
                "",
                "## 资源场景",
                "",
            ]
            for scenario in scenarios:
                notes.append(
                    f"- {scenario['scenario_label']}：内存 {scenario['sat_memory_range_mb']} MB，"
                    f"算力缩放 {scenario['sat_compute_scale']:.2f}，"
                    f"ISL 缩放 {scenario['isl_bandwidth_scale']:.2f}，"
                    f"GSL 缩放 {scenario['gsl_bandwidth_scale']:.2f}。"
                )
            (out_dir / "exp07_fwms_resource_scenarios_notes.md").write_text("\n".join(notes), encoding="utf-8")
            print(out_dir)
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


def _run_exp05(args) -> int:
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp05_show")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    try:
        loaded: list[tuple[pd.DataFrame, Path]] = []
        for model_name in models:
            loaded.append(
                _ensure_mode_selection_results(
                    model_name,
                    args.data_size,
                    args,
                    out_dir,
                    "exp05_fair",
                    sat_compute_scale=args.sat_compute_scale,
                )
            )
        common_slots = _common_average_slots(loaded, args.min_pmp_route_leo)
        if not common_slots:
            raise ValueError("No common slots satisfy the shared-route baseline filter for exp05.")
        rows: list[dict] = []
        audits: list[dict] = []
        for df_source, source_dir in loaded:
            model_rows = _aggregate_mode_rows_over_slots(
                df_source,
                common_slots,
                source_dir,
                args.min_cdp_active_sats,
                "common_slot_mean",
            )
            cdp_row = next(row for row in model_rows if row["mode"] == "CDP")
            audit = {
                "model_name": model_rows[0]["model_name"],
                "batch_size": model_rows[0]["batch_size"],
                "selection_status": "common_slot_mean",
                "common_slot_count": len(common_slots),
                "common_slot_ids": "|".join(common_slots),
                "min_pmp_route_leo": args.min_pmp_route_leo,
                "min_cdp_active_sats": args.min_cdp_active_sats,
                "cdp_feasible_slot_count": int(cdp_row["feasible_slot_count"]),
                "source_mode_selection_dir": str(source_dir),
            }
            rows.extend(model_rows)
            audits.append(audit)
        df = pd.DataFrame(rows)
        if not _is_show_only(args):
            df.to_csv(out_dir / "exp05_fwms_mode_selection_effectiveness_summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(audits).to_csv(out_dir / "exp05_fwms_mode_selection_effectiveness_slot_filter.csv", index=False, encoding="utf-8-sig")
        _plot_exp05(df, out_dir, show_only=_is_show_only(args))
        if not _is_show_only(args):
            notes = [
                "# 实验 5：FWMS 模式选择有效性实验",
                "",
                "- 本实验直接读取 mode_selection_experiment.py 生成的 slot_mode_results.csv。",
                f"- 共同时间片集合：所有模型共享同一批 slot，并要求 PMP/GS-Only/FWMS 可行且 PMP 路由至少包含 {args.min_pmp_route_leo} 颗 LEO。",
                f"- 统计口径：各模式在共同 slot 集合上取均值；CDP 仅统计 active_sat_count 至少为 {args.min_cdp_active_sats} 的可行 slot。",
                "- 路由口径：PMP、GS-Only、Sat-Only 在每个 slot 上共用同一路由；Sat-Only 只在该共享路由上选择单星执行位置。",
                f"- 星上算力缩放：{args.sat_compute_scale:.2f}。",
            ]
            (out_dir / "exp05_fwms_mode_selection_effectiveness_notes.md").write_text("\n".join(notes), encoding="utf-8")
            print(out_dir)
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


def _run_exp06(args) -> int:
    out_dir, cleanup_dir = _prepare_runtime_out_dir(args.out_dir, _is_show_only(args), "exp06_show")
    try:
        loaded: list[tuple[pd.DataFrame, Path]] = []
        batches = [int(value) for value in args.data_sizes.split(",") if value.strip()]
        memory_range_arg = None if getattr(args, "sat_memory_values_mb", None) else args.sat_memory_range_mb
        for batch_size in batches:
            loaded.append(
                _ensure_mode_selection_results(
                    args.model,
                    batch_size,
                    args,
                    out_dir,
                    "exp06_fair",
                    profile_device=args.profile_device,
                    sat_memory_range_mb=memory_range_arg,
                    sat_memory_values_mb=getattr(args, "sat_memory_values_mb", None),
                )
            )
        common_slots = _common_average_slots(loaded, args.min_pmp_route_leo)
        if not common_slots:
            raise ValueError("No common slots satisfy the shared-route baseline filter for exp06.")
        rows: list[dict] = []
        audits: list[dict] = []
        for df_source, source_dir in loaded:
            model_rows = _aggregate_mode_rows_over_slots(
                df_source,
                common_slots,
                source_dir,
                args.min_cdp_active_sats,
                "common_slot_mean",
            )
            cdp_row = next(row for row in model_rows if row["mode"] == "CDP")
            audit = {
                "model_name": model_rows[0]["model_name"],
                "batch_size": model_rows[0]["batch_size"],
                "selection_status": "common_slot_mean",
                "common_slot_count": len(common_slots),
                "common_slot_ids": "|".join(common_slots),
                "min_pmp_route_leo": args.min_pmp_route_leo,
                "min_cdp_active_sats": args.min_cdp_active_sats,
                "cdp_feasible_slot_count": int(cdp_row["feasible_slot_count"]),
                "source_mode_selection_dir": str(source_dir),
            }
            rows.extend(model_rows)
            audits.append(audit)
        df = pd.DataFrame(rows)
        if not _is_show_only(args):
            df.to_csv(out_dir / f"exp06_fwms_data_sensitivity_{args.model}_summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(audits).to_csv(out_dir / f"exp06_fwms_data_sensitivity_{args.model}_slot_filter.csv", index=False, encoding="utf-8-sig")
        _plot_exp06(df, out_dir, args.model, show_only=_is_show_only(args))
        if not _is_show_only(args):
            notes = [
                "# 实验 6：FWMS 输入数据量敏感性实验",
                "",
                f"- 模型：{MODEL_LABELS.get(args.model, args.model)}。",
                f"- Profile 口径：所有计算节点统一使用 {str(args.profile_device).upper()} profile。",
                f"- LEO 内存：{getattr(args, 'sat_memory_values_mb', None) or args.sat_memory_range_mb} MB，按 STK 卫星 ID 稳定映射。",
                "- 每个 batch 都读取 mode_selection_experiment.py 的 slot_mode_results.csv，不使用实验 1 的 PMP 比例外推。",
                f"- 共同时间片集合：所有 batch 共享同一批 slot，并要求 PMP/GS-Only/FWMS 可行且 PMP 路由至少包含 {args.min_pmp_route_leo} 颗 LEO。",
                f"- 统计口径：各模式在共同 slot 集合上取均值；CDP 仅统计 active_sat_count 至少为 {args.min_cdp_active_sats} 的可行 slot。",
                "- 路由口径：PMP、GS-Only、Sat-Only 在每个 slot 上共用同一路由；Sat-Only 只在该共享路由上选择单星执行位置。",
            ]
            (out_dir / f"exp06_fwms_data_sensitivity_{args.model}_notes.md").write_text("\n".join(notes), encoding="utf-8")
            print(out_dir)
        return 0
    finally:
        if _is_show_only(args):
            _cleanup_runtime_dir(cleanup_dir)


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
    exp00_parser = subparsers.add_parser("exp00", help="Draw layer output size distribution figures")
    exp00_parser.add_argument("--profile", default="config/dnn_profiles_database_pc.json", help="Profile database JSON")
    exp00_parser.add_argument(
        "--reference-profile",
        default="config/dnn_profiles_database_jetson.json",
        help="Optional second profile JSON used to validate layer output sizes before drawing",
    )
    exp00_parser.add_argument("--models", default="yolov5,resnet101,vgg19,vit_huge", help="Comma-separated model ids")
    exp00_parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for plotting (auto-scale if profile is missing)")
    exp00_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_v2/00_layer_output_distribution",
        help="Output directory for exp00 figures",
    )
    exp00_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")
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
    exp02_parser.add_argument("--engineering-model", default="yolov5,resnet101,vgg19,vit_huge", help="Comma-separated model ids used by exp02A/exp02B")
    exp02_parser.add_argument("--task-block-batch", type=int, default=EXP02_BLOCK_BATCH, help="Batch size of each task block")
    exp02_parser.add_argument("--task-block-count", type=int, default=EXP02_BLOCK_COUNT, help="Number of task blocks in the fixed-load experiment")
    exp02_parser.add_argument("--min-common-duration-s", type=float, default=0.0, help="Optional minimum STK common visibility duration for engineering exp02; default keeps all routes")
    exp02_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_final/02_ladp_pmp_node_count_sensitivity",
        help="Output directory for experiment 2 figure and summaries",
    )
    exp02_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")
    exp03_parser = subparsers.add_parser("exp03", help="Run experiment 3 and draw CDP data-size sensitivity figures")
    exp03_parser.add_argument("--models", default="resnet101", help="Comma-separated model ids")
    exp03_parser.add_argument("--data-sizes", default="64,128,256,512", help="Comma-separated input data sizes")
    exp03_parser.add_argument("--worker-count", type=int, default=4, help="Fixed CDP worker count")
    exp03_parser.add_argument("--random-repeats", type=int, default=30, help="Random baseline repeats")
    exp03_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    exp03_parser.add_argument(
        "--profile",
        default="config/dnn_profiles_database_pc.json",
        help="DNN profile database used for full-model latency and output size",
    )
    exp03_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_final/03_lawa_cdp_data_sensitivity",
        help="Output directory for experiment 3",
    )
    exp03_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")
    exp04_parser = subparsers.add_parser("exp04", help="Run experiment 4 and draw CDP worker-count sensitivity figures")
    exp04_parser.add_argument("--models", default="yolov5", help="Comma-separated model ids")
    exp04_parser.add_argument("--data-size", type=int, default=64, help="Fixed input data size")
    exp04_parser.add_argument("--worker-counts", default="1,2,3,4,5", help="Comma-separated worker counts")
    exp04_parser.add_argument("--random-repeats", type=int, default=30, help="Random baseline repeats")
    exp04_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    exp04_parser.add_argument(
        "--profile",
        default="config/dnn_profiles_database_pc.json",
        help="DNN profile database used for full-model latency and output size",
    )
    exp04_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_final/04_lawa_cdp_worker_count_sensitivity",
        help="Output directory for experiment 4",
    )
    exp04_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")
    exp05_parser = subparsers.add_parser("exp05", help="Run experiment 5 and draw FWMS mode-selection figure")
    exp05_parser.add_argument("--models", default="yolov5,resnet101,vgg19,vit_huge", help="Comma-separated model ids")
    exp05_parser.add_argument("--data-size", type=int, default=64, help="Fixed input data size")
    exp05_parser.add_argument("--worker-count", type=int, default=4, help="CDP worker count")
    exp05_parser.add_argument("--worker-memory-mb", type=float, default=2048.0, help="Memory limit for each CDP worker")
    exp05_parser.add_argument("--gsl-bandwidth-mbps", type=float, default=100.0, help="GS-only uplink bandwidth")
    exp05_parser.add_argument("--gs-compute-factor", type=float, default=100.0, help="GS speedup over profiled Jetson latency")
    exp05_parser.add_argument("--min-cdp-gain", type=float, default=0.05, help="Minimum CDP gain required by FWMS")
    exp05_parser.add_argument(
        "--sat-compute-scale",
        type=float,
        default=1.2,
        help="Scale factor applied to all satellite TFLOPS in experiment 5",
    )
    exp05_parser.add_argument("--profile", default="config/dnn_profiles_database_jetson.json", help="DNN profile database")
    exp05_parser.add_argument("--min-pmp-route-leo", type=int, default=3, help="Minimum LEO satellites on the PMP route for representative-slot filtering")
    exp05_parser.add_argument("--min-cdp-active-sats", type=int, default=3, help="Minimum active CDP worker satellites for representative-slot filtering")
    exp05_parser.add_argument(
        "--reuse-mode-results",
        action="store_true",
        help="Reuse existing mode_selection sources under the output directory instead of rerunning them",
    )
    exp05_parser.add_argument(
        "--pmp-summary",
        default="result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv",
        help="Deprecated: kept for backward compatibility; exp05 now reads mode_selection slot_mode_results.csv",
    )
    exp05_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_final/05_fwms_mode_selection_effectiveness",
        help="Output directory for experiment 5",
    )
    exp05_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")
    exp06_parser = subparsers.add_parser("exp06", help="Run experiment 6 and draw FWMS data-size sensitivity figure")
    exp06_parser.add_argument("--model", default="yolov5", help="Model id")
    exp06_parser.add_argument("--data-sizes", default="64,128,256,512", help="Comma-separated input data sizes")
    exp06_parser.add_argument("--worker-count", type=int, default=4, help="CDP worker count")
    exp06_parser.add_argument("--worker-memory-mb", type=float, default=2048.0, help="Memory limit for each CDP worker")
    exp06_parser.add_argument("--gsl-bandwidth-mbps", type=float, default=100.0, help="GS-only uplink bandwidth")
    exp06_parser.add_argument("--gs-compute-factor", type=float, default=100.0, help="GS speedup over profiled latency")
    exp06_parser.add_argument("--min-cdp-gain", type=float, default=0.05, help="Minimum CDP gain required by FWMS")
    exp06_parser.add_argument("--profile", default="config/dnn_profiles_database_pc.json", help="DNN profile database")
    exp06_parser.add_argument(
        "--profile-device",
        choices=["pc", "jetson", "mixed"],
        default="pc",
        help="Profile device family used by the mode-selection rerun for experiment 6",
    )
    exp06_parser.add_argument(
        "--sat-memory-range-mb",
        default="4096,16384",
        help="Stable LEO memory override range for experiment 6, in MB",
    )
    exp06_parser.add_argument(
        "--sat-memory-values-mb",
        default=None,
        help="Optional stable discrete LEO memory set for experiment 6, in MB, for example 2048,4096,8192",
    )
    exp06_parser.add_argument("--min-pmp-route-leo", type=int, default=3, help="Minimum LEO satellites on the PMP route for representative-slot filtering")
    exp06_parser.add_argument("--min-cdp-active-sats", type=int, default=3, help="Minimum active CDP worker satellites for representative-slot filtering")
    exp06_parser.add_argument(
        "--reuse-mode-results",
        action="store_true",
        help="Reuse existing mode_selection sources under the output directory instead of rerunning them",
    )
    exp06_parser.add_argument(
        "--pmp-summary",
        default="result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv",
        help="Deprecated: kept for backward compatibility; exp06 now reads mode_selection slot_mode_results.csv",
    )
    exp06_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_final/06_fwms_data_sensitivity",
        help="Output directory for experiment 6",
    )
    exp06_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")
    exp07_parser = subparsers.add_parser("exp07", help="Run PMP/CDP flip-boundary verification across models")
    exp07_parser.add_argument("--models", default="yolov5,resnet101,vgg19,vit_huge", help="Comma-separated model ids")
    exp07_parser.add_argument("--batch-size", type=int, default=16, help="Fixed batch size used for all models")
    exp07_parser.add_argument("--worker-counts", default="2,3,4", help="Comma-separated CDP worker counts")
    exp07_parser.add_argument(
        "--reuse-mode-results",
        action="store_true",
        help="Reuse existing mode_selection sources under the output directory instead of rerunning them",
    )
    exp07_parser.add_argument(
        "--out-dir",
        default="result/paper_figures_final/07_pmp_cdp_flip_boundary",
        help="Output directory for experiment 7",
    )
    exp07_parser.add_argument("--show-only", action="store_true", help="Display figures only and do not keep outputs")

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
    if args.command == "exp00":
        raise SystemExit(_run_exp00(args))
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
    if args.command == "exp07":
        raise SystemExit(_run_exp07(args))

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()





