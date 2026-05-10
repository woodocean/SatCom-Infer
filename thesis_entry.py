"""Unified thesis experiment and utility entrypoint.

This script keeps the root directory clean and exposes one stable CLI for:
1. Main experiment entrypoints.
2. Plotting/summary utilities under ``tools``.
3. Paper-facing rerun commands used during acceptance.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


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

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
