"""Build a segment-based track model from a structured telemetry CSV."""

from __future__ import annotations

import csv
from pathlib import Path

_KMH_TO_MS = 1.0 / 3.6
_MIN_SEGMENT_LENGTH_M = 10.0


def _merge_consecutive_rows(rows: list[dict]) -> list[dict]:
    """Merge consecutive rows with the same Corner_Type into segments."""
    segments: list[dict] = []
    cur_type = rows[0]["Corner_Type"]
    cur_rows: list[dict] = [rows[0]]

    for row in rows[1:]:
        if row["Corner_Type"] == cur_type:
            cur_rows.append(row)
        else:
            segments.append(_summarise_run(len(segments), cur_type, cur_rows))
            cur_type = row["Corner_Type"]
            cur_rows = [row]
    segments.append(_summarise_run(len(segments), cur_type, cur_rows))
    return segments


def _summarise_run(seg_id: int, seg_type: str, rows: list[dict]) -> dict:
    speeds = [float(r["Speed"]) for r in rows]
    d_start = float(rows[0]["Distance"])
    d_end = float(rows[-1]["Distance"])

    if seg_type == "Straight":
        v_ref_kmh = max(speeds)
    else:
        v_ref_kmh = min(speeds)

    return {
        "seg_id": seg_id,
        "d_start": d_start,
        "d_end": d_end,
        "length": d_end - d_start,
        "seg_type": seg_type,
        "v_ref_kmh": v_ref_kmh,
        "v_ref": v_ref_kmh * _KMH_TO_MS,
    }


def _close_gaps(segments: list[dict]) -> list[dict]:
    """Extend each segment's d_end to the next segment's d_start so there are no gaps."""
    for i in range(len(segments) - 1):
        segments[i]["d_end"] = segments[i + 1]["d_start"]
        segments[i]["length"] = segments[i]["d_end"] - segments[i]["d_start"]
    return segments


def _absorb_micro_segments(segments: list[dict]) -> list[dict]:
    """Merge segments shorter than the threshold into their predecessor."""
    cleaned: list[dict] = []
    for seg in segments:
        if seg["length"] < _MIN_SEGMENT_LENGTH_M and cleaned:
            prev = cleaned[-1]
            prev["d_end"] = seg["d_end"]
            prev["length"] = prev["d_end"] - prev["d_start"]
            prev["v_ref"] = min(prev["v_ref"], seg["v_ref"])
            prev["v_ref_kmh"] = prev["v_ref"] / _KMH_TO_MS
        else:
            cleaned.append(seg)

    for i, seg in enumerate(cleaned):
        seg["seg_id"] = i
    return cleaned


def build_track_segments(csv_path: str | Path) -> list[dict]:
    """Return a list of segment dicts from a structured telemetry CSV.

    Each dict has: seg_id, d_start, d_end, length, seg_type, v_ref (m/s), v_ref_kmh.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    segments = _merge_consecutive_rows(rows)
    segments = _close_gaps(segments)
    segments = _absorb_micro_segments(segments)
    return segments


def print_segments(segments: list[dict]) -> None:
    """Pretty-print a segment table for debugging."""
    print(f"{'seg':>3}  {'type':>22}  {'d_start':>8}  {'d_end':>8}  {'length':>7}  {'v_ref km/h':>10}")
    print("-" * 72)
    total_len = 0.0
    for s in segments:
        total_len += s["length"]
        print(
            f"{s['seg_id']:3d}  {s['seg_type']:>22}  {s['d_start']:8.1f}  "
            f"{s['d_end']:8.1f}  {s['length']:7.1f}  {s['v_ref_kmh']:10.1f}"
        )
    print(f"\nTotal segments: {len(segments)}  |  Total distance: {total_len:.1f} m")
    v_refs = [s["v_ref_kmh"] for s in segments]
    print(f"v_ref range: {min(v_refs):.1f} – {max(v_refs):.1f} km/h")
    micro = [s for s in segments if s["length"] < _MIN_SEGMENT_LENGTH_M]
    print(f"Micro-segments (<{_MIN_SEGMENT_LENGTH_M}m): {len(micro)}")
