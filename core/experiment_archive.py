import csv
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


RESULT_ROOT = Path("result")
RUNS_DIR = RESULT_ROOT / "runs"
INDEX_PATH = RESULT_ROOT / "EXPERIMENT_INDEX.md"


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_token(value, default="na", max_len=80):
    text = str(value) if value is not None else default
    text = text.strip()
    if not text:
        text = default
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or default


def short_exp_type(exp_type):
    mapping = {
        "algo_effectiveness": "algo",
        "isl_bandwidth_sensitivity": "isl_bw",
        "gsl_bandwidth_sensitivity": "gsl_bw",
    }
    return mapping.get(exp_type, sanitize_token(exp_type))


def _sweep_label(metadata):
    if metadata.get("sweep_start") is not None and metadata.get("sweep_stop") is not None:
        points = metadata.get("sweep_points", "na")
        return f"{metadata['sweep_start']}to{metadata['sweep_stop']}_p{points}"
    values = metadata.get("sweep_values")
    if values:
        if isinstance(values, str):
            count = len([v for v in values.split(",") if v.strip()])
        else:
            count = len(values)
        return f"values_p{count}"
    return "nosweep"


def build_artifact_stem(metadata):
    exp = short_exp_type(metadata.get("exp_type"))
    model = metadata.get("fixed_model") or metadata.get("model_name") or "mixed"
    batch = metadata.get("fixed_batch_size") or metadata.get("batch_size") or "mixed"
    input_h = metadata.get("fixed_input_h") or metadata.get("input_h") or "mixed"
    input_w = metadata.get("fixed_input_w") or metadata.get("input_w") or "mixed"
    mode = metadata.get("exp_mode") or metadata.get("mode") or "unknown"
    repeat = metadata.get("repeat_per_point")
    seed = metadata.get("seed")
    sweep = _sweep_label(metadata)

    parts = [
        exp,
        sanitize_token(model),
        f"b{sanitize_token(batch)}",
        f"{sanitize_token(input_h)}x{sanitize_token(input_w)}",
        sanitize_token(mode),
        sanitize_token(sweep),
    ]
    if repeat is not None:
        parts.append(f"r{sanitize_token(repeat)}")
    if seed is not None:
        parts.append(f"seed{sanitize_token(seed)}")
    return "_".join(parts)


def build_run_folder_name(metadata):
    started = metadata.get("started_at_compact") or now_stamp()
    run_id = sanitize_token(metadata.get("run_id"), "run", max_len=64)
    stem = build_artifact_stem(metadata)
    return sanitize_token(f"{started}__{run_id}__{stem}", max_len=180)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_readme(archive_dir, metadata):
    lines = [
        f"# Experiment {metadata.get('run_id', '')}",
        "",
        f"- started_at: {metadata.get('started_at', '')}",
        f"- completed_at: {metadata.get('completed_at', '')}",
        f"- status: {metadata.get('status', '')}",
        f"- exp_type: {metadata.get('exp_type', '')}",
        f"- exp_mode: {metadata.get('exp_mode', '')}",
        f"- preset: {metadata.get('preset', '')}",
        f"- model: {metadata.get('fixed_model', 'mixed')}",
        f"- batch: {metadata.get('fixed_batch_size', 'mixed')}",
        f"- input: {metadata.get('fixed_input_h', 'mixed')}x{metadata.get('fixed_input_w', 'mixed')}",
        f"- sweep: {_sweep_label(metadata)}",
        f"- repeat_per_point: {metadata.get('repeat_per_point', '')}",
        f"- seed: {metadata.get('seed', '')}",
        "",
        "## Files",
        "",
        "- metadata: metadata.json",
        "- data: data/",
        "- figures: figures/",
        "- config snapshots: config/",
        "",
    ]
    with open(archive_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def create_run_archive(metadata, snapshot_paths=None, result_root=RESULT_ROOT):
    result_root = Path(result_root)
    archive_dir = result_root / "runs" / build_run_folder_name(metadata)
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "data").mkdir(exist_ok=True)
    (archive_dir / "figures").mkdir(exist_ok=True)
    (archive_dir / "config").mkdir(exist_ok=True)

    metadata = dict(metadata)
    metadata["archive_dir"] = str(archive_dir)
    _write_json(archive_dir / "metadata.json", metadata)
    _write_readme(archive_dir, metadata)

    for name, source in (snapshot_paths or {}).items():
        if source and os.path.exists(source):
            target = archive_dir / "config" / f"{sanitize_token(name)}_snapshot{Path(source).suffix}"
            shutil.copyfile(source, target)

    return archive_dir


def update_run_metadata(archive_dir, updates):
    archive_dir = Path(archive_dir)
    metadata_path = archive_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(updates)
    _write_json(metadata_path, metadata)
    _write_readme(archive_dir, metadata)
    return metadata


def find_run_archive(run_id, result_root=RESULT_ROOT):
    if not run_id:
        return None
    runs_dir = Path(result_root) / "runs"
    if not runs_dir.exists():
        return None

    matches = []
    for metadata_path in runs_dir.glob("*/metadata.json"):
        try:
            metadata = _read_json(metadata_path)
        except Exception:
            continue
        if metadata.get("run_id") == run_id:
            matches.append(metadata_path.parent)

    if not matches:
        return None
    return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]


def export_run_rows(source_csv, run_id, output_path):
    source_csv = Path(source_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_csv.exists():
        return 0

    count = 0
    with open(source_csv, "r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        with open(output_path, "w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("run_id") == run_id:
                    writer.writerow(row)
                    count += 1
    return count


def append_experiment_index(metadata, archive_dir):
    archive_dir = Path(archive_dir)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            f.write("# Experiment Index\n\n")
            f.write("| time | run_id | exp_type | model | batch | input | mode | sweep | repeat | seed | status | folder |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    rel_folder = archive_dir.as_posix()
    row = [
        metadata.get("started_at", ""),
        metadata.get("run_id", ""),
        metadata.get("exp_type", ""),
        metadata.get("fixed_model", "mixed"),
        metadata.get("fixed_batch_size", "mixed"),
        f"{metadata.get('fixed_input_h', 'mixed')}x{metadata.get('fixed_input_w', 'mixed')}",
        metadata.get("exp_mode", ""),
        _sweep_label(metadata),
        metadata.get("repeat_per_point", ""),
        metadata.get("seed", ""),
        metadata.get("status", ""),
        rel_folder,
    ]
    escaped = [str(item).replace("|", "/") for item in row]
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write("| " + " | ".join(escaped) + " |\n")
