#!/usr/bin/env python3
"""Quick row-count / path sanity between fallback draft CSV vs teammate-B export."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_FALLBACK = _ROOT / "data" / "race_laps_japan_2026_pia_vs_ant.csv"


def _md5_tail(path: Path, n_lines: int = 5) -> str:
    text = Path(path).read_bytes()
    return hashlib.md5(text).hexdigest()[:12]


def main() -> None:
    fb = Path(os.environ.get("F1_TEAM_B_FALLBACK", _FALLBACK))
    team = os.environ.get("F1_TEAM_B_LAPTIMES_CSV")

    df_fb = pd.read_csv(fb)
    print(f"Fallback: {fb}")
    print(f"  rows: {len(df_fb)}")
    print(f"  drivers: {sorted(df_fb['Driver'].dropna().unique().tolist())}")
    print(f"  md5-sample: {_md5_tail(fb)}")

    if team:
        p = Path(team)
        df_t = pd.read_csv(p)
        print(f"\nTeam B CSV: {p}")
        print(f"  rows: {len(df_t)}")
        print(f"  drivers: {sorted(df_t['Driver'].dropna().unique().tolist()) if 'Driver' in df_t.columns else 'N/A'}")
        print(f"  md5-sample: {_md5_tail(p)}")
        print("\nAction: reconcile any row mismatch with teammate B audit before slides.")
    else:
        print("\nF1_TEAM_B_LAPTIMES_CSV unset — parity vs team artefact deferred.")


if __name__ == "__main__":
    main()
