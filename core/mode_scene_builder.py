"""Scene loading helpers for mode-selection experiments.

The mode-selection pipeline treats each STK time slot as one scene.  This
module keeps that scene format independent from the older PMP-only runner so
new mode evaluators can share the same inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class TaskSpec:
    model_name: str
    batch_size: int
    input_h: int
    input_w: int


@dataclass(frozen=True)
class SlotScene:
    source_run_id: str
    source_run_dir: Path
    slot_id: str
    slot_start: str
    slot_stop: str
    task: TaskSpec
    config_path: Path
    candidate_path: Optional[Path]
    selected_stk_path: List[str]
    pipeline_path: List[str]
    candidate_paths: List[dict]
    slot_row: Dict[str, str]
    metadata: Dict
    network_config: Dict

    @property
    def candidate_id(self) -> str:
        rank = self.selected_candidate_rank()
        return f"selected_path_rank_{rank:03d}" if rank is not None else "selected_path"

    def selected_candidate_rank(self) -> Optional[int]:
        selected = "->".join(self.selected_stk_path)
        for candidate in self.candidate_paths:
            if "->".join(candidate.get("path", [])) == selected:
                rank = candidate.get("rank")
                return int(rank) if rank not in (None, "") else None
        return None

    def to_jsonable(self) -> Dict:
        config_links = self.network_config.get("links", {})
        link_summaries = []
        for link_name, info in config_links.items():
            link_summaries.append(
                {
                    "name": link_name,
                    "bandwidth_mbps": info.get("bandwidth_mbps"),
                    "propagation_delay_ms": info.get("propagation_delay_ms"),
                    "stk_from": info.get("stk_from"),
                    "stk_to": info.get("stk_to"),
                    "stk_link_type": info.get("stk_link_type"),
                }
            )

        return {
            "schema_version": 1,
            "source_run_id": self.source_run_id,
            "source_run_dir": str(self.source_run_dir),
            "slot_id": self.slot_id,
            "slot_start": self.slot_start,
            "slot_stop": self.slot_stop,
            "task": asdict(self.task),
            "config_path": str(self.config_path),
            "candidate_path": str(self.candidate_path) if self.candidate_path else "",
            "selected_candidate_id": self.candidate_id,
            "selected_stk_path": self.selected_stk_path,
            "pipeline_path": self.pipeline_path,
            "candidate_count": len(self.candidate_paths),
            "slot_row": self.slot_row,
            "link_summaries": link_summaries,
        }


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_route(value: str) -> List[str]:
    return [item for item in str(value or "").split("->") if item]


def _resolve_existing_path(path_value: str, run_dir: Path) -> Path:
    path = Path(path_value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, run_dir / path, run_dir / path.name])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _candidate_path_for_slot(run_dir: Path, slot_id: str) -> Optional[Path]:
    candidate = run_dir / "candidates" / f"{slot_id}_candidate_paths.json"
    return candidate if candidate.exists() else None


def load_stk_slot_scenes(
    run_dir: str | Path,
    slot_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[SlotScene]:
    """Load completed STK dynamic slots as mode-selection scenes."""

    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in STK run dir: {run_dir}")
    metadata = _load_json(metadata_path)

    slots_csv = _resolve_existing_path(metadata.get("slots_csv", str(run_dir / "stk_dynamic_slots.csv")), run_dir)
    if not slots_csv.exists():
        raise FileNotFoundError(f"Missing STK slots CSV: {slots_csv}")

    wanted = set(slot_ids or [])
    scenes: List[SlotScene] = []
    with slots_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "completed":
                continue
            slot_id = row.get("slot_id", "")
            if wanted and slot_id not in wanted:
                continue

            config_path = _resolve_existing_path(row.get("config_path", ""), run_dir)
            if not config_path.exists():
                raise FileNotFoundError(f"Missing slot network config: {config_path}")
            network_config = _load_json(config_path)

            candidate_path = _candidate_path_for_slot(run_dir, slot_id)
            candidate_payload = _load_json(candidate_path) if candidate_path else {}
            candidate_paths = candidate_payload.get("paths", [])

            task = TaskSpec(
                model_name=str(metadata.get("model_name", "yolov5")),
                batch_size=int(metadata.get("batch_size", 32)),
                input_h=int(metadata.get("input_h", 640)),
                input_w=int(metadata.get("input_w", 640)),
            )
            scene = SlotScene(
                source_run_id=str(metadata.get("run_id", row.get("run_id", ""))),
                source_run_dir=run_dir,
                slot_id=slot_id,
                slot_start=row.get("slot_start", ""),
                slot_stop=row.get("slot_stop", ""),
                task=task,
                config_path=config_path,
                candidate_path=candidate_path,
                selected_stk_path=_split_route(row.get("selected_path", "")),
                pipeline_path=_split_route(row.get("pipeline_path", "")),
                candidate_paths=candidate_paths,
                slot_row=dict(row),
                metadata=metadata,
                network_config=network_config,
            )
            scenes.append(scene)
            if limit is not None and len(scenes) >= limit:
                break

    return scenes


def write_slot_scene(scene: SlotScene, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scene.to_jsonable(), f, ensure_ascii=False, indent=2)


def write_slot_scenes(scenes: Iterable[SlotScene], output_dir: str | Path) -> List[Path]:
    output_dir = Path(output_dir)
    written = []
    for scene in scenes:
        path = output_dir / f"{scene.slot_id}_scene.json"
        write_slot_scene(scene, path)
        written.append(path)
    return written
