"""
Pull race-level lap timing from FastF1 (Japan 2026 Race) — not assignment telemetry CSVs.

Exports a compact CSV for Piastri vs Antonelli (adjust ABBREVS if your session uses different codes).

Usage:
  uv run python scripts/pull_race_laps.py
"""

from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "fastf1"
OUT_DIR = ROOT / "data"
OUTPUT = OUT_DIR / "race_laps_japan_2026_pia_vs_ant.csv"

ABBREVS = ("PIA", "ANT")


def ensure_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))


def load_session_laps(year: int, event: str, session: str) -> pd.DataFrame:
    sess = fastf1.get_session(year, event, session)
    sess.load(laps=True, telemetry=False, weather=False, messages=False)
    return sess.laps.copy()


def main() -> None:
    ensure_cache()
    laps = load_session_laps(2026, "Japan", "R")

    # Driver abbreviations row filter
    sub = laps[laps["Driver"].isin(ABBREVS)]
    if sub.empty:
        avail = sorted(laps["Driver"].dropna().unique().tolist())
        raise SystemExit(
            "No laps for requested drivers. "
            f"Try editing ABBREVS. Available Driver codes: {avail}"
        )

    cols = [
        c
        for c in (
            "Driver",
            "DriverNumber",
            "LapNumber",
            "LapTime",
            "Compound",
            "TyreLife",
            "Stint",
            "FreshTyre",
            "IsPersonalBest",
            "TrackStatus",
        )
        if c in sub.columns
    ]
    out = sub[cols].sort_values(["Driver", "LapNumber"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({len(out)} rows).")


if __name__ == "__main__":
    main()
