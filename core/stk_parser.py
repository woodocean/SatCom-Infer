"""Utilities for parsing STK Chain Access reports.

The STK text report keeps per-link section titles such as
``LEO011 to Shenzhen``.  Those titles are often lost when exporting to CSV, so
the parser intentionally supports the original TXT format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STK_TIME_FORMAT = "%d %b %Y %H:%M:%S.%f"
LIGHT_SPEED_KM_PER_S = 299_792.458

_DATE_PREFIX_RE = re.compile(r"^\s*\d{1,2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2}\.\d+")
_SECTION_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+to\s+(.+?)\s*$")
_DASH_RE = re.compile(r"^-{3,}$")
_SATELLITE_RE = re.compile(r"^LEO\d{3}$")


@dataclass(frozen=True)
class AccessWindow:
    from_id: str
    to_id: str
    link_type: str
    start: datetime
    stop: datetime
    duration_s: float


@dataclass(frozen=True)
class AERSample:
    from_id: str
    to_id: str
    link_type: str
    time: datetime
    azimuth_deg: float
    elevation_deg: float
    range_km: float


def parse_stk_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), STK_TIME_FORMAT)


def format_stk_time(value: datetime) -> str:
    return value.strftime(STK_TIME_FORMAT)[:-3]


def propagation_delay_ms(range_km: float) -> float:
    return (float(range_km) / LIGHT_SPEED_KM_PER_S) * 1000.0


def normalize_object_name(raw: str) -> str:
    """Normalize STK object labels into compact node IDs.

    Examples:
        ``From Satellite LEO011`` -> ``LEO011``
        ``To Place Shenzhen`` -> ``Shenzhen``
    """

    text = " ".join(str(raw).strip().split())
    prefixes = (
        "From Satellite ",
        "To Satellite ",
        "From Place ",
        "To Place ",
        "From Facility ",
        "To Facility ",
        "From Target ",
        "To Target ",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def is_seed_satellite(node_id: str) -> bool:
    return node_id == "LEO"


def is_walker_satellite(node_id: str) -> bool:
    return bool(_SATELLITE_RE.match(node_id))


def keep_stk_node(node_id: str, ignore_seed_satellite: bool = True) -> bool:
    if ignore_seed_satellite and is_seed_satellite(node_id):
        return False
    return True


def parse_access_data_txt(
    path: str | Path,
    link_type: str,
    ignore_seed_satellite: bool = True,
) -> List[AccessWindow]:
    """Parse a STK ``Chain Access Data`` TXT report.

    The report rows are fixed-width-ish, but columns are reliably separated by
    two or more spaces.  Timestamps contain single spaces, so splitting on
    ``\\s{2,}`` preserves each timestamp as one token.
    """

    path = Path(path)
    windows: List[AccessWindow] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not _DATE_PREFIX_RE.match(line):
                continue
            cols = re.split(r"\s{2,}", line.strip())
            if len(cols) < 6:
                continue
            try:
                start = parse_stk_time(cols[0])
                stop = parse_stk_time(cols[1])
                duration_s = float(cols[2])
            except ValueError:
                continue

            # The last four fields are To Object / From Object / To Parent /
            # From Parent in STK's Access Data report.
            to_id = normalize_object_name(cols[-4])
            from_id = normalize_object_name(cols[-3])
            if not (keep_stk_node(from_id, ignore_seed_satellite) and keep_stk_node(to_id, ignore_seed_satellite)):
                continue
            if stop <= start:
                continue
            windows.append(
                AccessWindow(
                    from_id=from_id,
                    to_id=to_id,
                    link_type=link_type,
                    start=start,
                    stop=stop,
                    duration_s=duration_s,
                )
            )
    return windows


def parse_access_aer_txt(
    path: str | Path,
    link_type: str,
    ignore_seed_satellite: bool = True,
) -> List[AERSample]:
    """Parse a STK ``Chain Access AER Data`` TXT report.

    Section titles carry the link endpoints, for example:

    ``LEO011 to LEO021``
    """

    path = Path(path)
    samples: List[AERSample] = []
    current_pair: Optional[Tuple[str, str]] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            section = _SECTION_RE.match(stripped)
            if section and not _DASH_RE.match(stripped):
                from_id = normalize_object_name(section.group(1))
                to_id = normalize_object_name(section.group(2))
                if keep_stk_node(from_id, ignore_seed_satellite) and keep_stk_node(to_id, ignore_seed_satellite):
                    current_pair = (from_id, to_id)
                else:
                    current_pair = None
                continue

            if current_pair is None or not _DATE_PREFIX_RE.match(stripped):
                continue

            cols = re.split(r"\s{2,}", stripped)
            if len(cols) < 4:
                continue
            try:
                samples.append(
                    AERSample(
                        from_id=current_pair[0],
                        to_id=current_pair[1],
                        link_type=link_type,
                        time=parse_stk_time(cols[0]),
                        azimuth_deg=float(cols[1]),
                        elevation_deg=float(cols[2]),
                        range_km=float(cols[3]),
                    )
                )
            except ValueError:
                continue

    return samples


def group_windows_by_pair(windows: Iterable[AccessWindow]) -> Dict[Tuple[str, str], List[AccessWindow]]:
    grouped: Dict[Tuple[str, str], List[AccessWindow]] = {}
    for window in windows:
        grouped.setdefault((window.from_id, window.to_id), []).append(window)
    for values in grouped.values():
        values.sort(key=lambda item: item.start)
    return grouped


def group_samples_by_pair(samples: Iterable[AERSample]) -> Dict[Tuple[str, str], List[AERSample]]:
    grouped: Dict[Tuple[str, str], List[AERSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.from_id, sample.to_id), []).append(sample)
    for values in grouped.values():
        values.sort(key=lambda item: item.time)
    return grouped


def derive_windows_from_aer_samples(
    samples: Sequence[AERSample],
    gap_threshold_s: float = 90.0,
) -> List[AccessWindow]:
    """Derive approximate visibility windows from AER samples.

    Use this as a fallback when STK Access Data is not available for a link
    class.  Consecutive samples separated by more than ``gap_threshold_s`` are
    treated as different windows.
    """

    windows: List[AccessWindow] = []
    by_pair = group_samples_by_pair(samples)
    for (from_id, to_id), pair_samples in by_pair.items():
        if not pair_samples:
            continue
        segment_start = pair_samples[0].time
        previous = pair_samples[0].time
        link_type = pair_samples[0].link_type
        for sample in pair_samples[1:]:
            gap_s = (sample.time - previous).total_seconds()
            if gap_s > gap_threshold_s:
                if previous > segment_start:
                    windows.append(
                        AccessWindow(
                            from_id=from_id,
                            to_id=to_id,
                            link_type=link_type,
                            start=segment_start,
                            stop=previous,
                            duration_s=(previous - segment_start).total_seconds(),
                        )
                    )
                segment_start = sample.time
            previous = sample.time
        if previous > segment_start:
            windows.append(
                AccessWindow(
                    from_id=from_id,
                    to_id=to_id,
                    link_type=link_type,
                    start=segment_start,
                    stop=previous,
                    duration_s=(previous - segment_start).total_seconds(),
                )
            )
    return windows


def filter_windows_by_time(
    windows: Iterable[AccessWindow],
    start: datetime,
    stop: datetime,
) -> List[AccessWindow]:
    filtered: List[AccessWindow] = []
    for window in windows:
        clipped_start = max(window.start, start)
        clipped_stop = min(window.stop, stop)
        if clipped_stop <= clipped_start:
            continue
        filtered.append(
            AccessWindow(
                from_id=window.from_id,
                to_id=window.to_id,
                link_type=window.link_type,
                start=clipped_start,
                stop=clipped_stop,
                duration_s=(clipped_stop - clipped_start).total_seconds(),
            )
        )
    return filtered


def filter_samples_by_time(
    samples: Iterable[AERSample],
    start: datetime,
    stop: datetime,
) -> List[AERSample]:
    return [sample for sample in samples if start <= sample.time <= stop]
