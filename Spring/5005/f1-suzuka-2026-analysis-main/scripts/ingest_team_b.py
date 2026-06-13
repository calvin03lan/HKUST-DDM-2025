"""Optional ingestion of teammate B lap-timing CSV → means for What-If inputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

DEFAULT_FALLBACK = (
    Path(__file__).resolve().parents[1] / "data" / "race_laps_japan_2026_pia_vs_ant.csv"
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from what_if_model import pace_delta_seconds_per_lap


def _laptime_to_seconds(series: pd.Series) -> pd.Series:
    """Coerce timedelta-like lap times to float seconds."""
    return pd.to_timedelta(series).dt.total_seconds()


def mean_lap_seconds_by_driver(
    df: pd.DataFrame,
    *,
    driver_col: str = "Driver",
    laptime_col: str = "LapTime",
    driver_value: str,
    lap_min: int | None = None,
    lap_max: int | None = None,
) -> float:
    """Mean lap time in seconds for optional LapNumber window."""
    sub = df[df[driver_col] == driver_value]
    if lap_min is not None:
        sub = sub[sub["LapNumber"] >= lap_min]
    if lap_max is not None:
        sub = sub[sub["LapNumber"] <= lap_max]
    if sub.empty:
        raise ValueError(f"No rows for {driver_value=} with lap filters")
    return float(_laptime_to_seconds(sub[laptime_col]).mean())


def load_csv_path() -> Path:
    p = os.environ.get("F1_TEAM_B_LAPTIMES_CSV")
    if p:
        return Path(p)
    return DEFAULT_FALLBACK


def summarize_pace_delta(
    path: Path | None = None,
    *,
    stint_pia: tuple[int | None, int | None],
    stint_opp: tuple[int | None, int | None],
    opp_abbr: str = "ANT",
) -> dict[str, float]:
    """
    Compute mean stint lap times & pace delta seconds/lap.

    stint_* are (lap_min, lap_max) inclusive on LapNumber, or (None, None) for whole race.
    """
    path = path or load_csv_path()
    df = pd.read_csv(path)

    lap_min_p, lap_max_p = stint_pia
    lap_min_o, lap_max_o = stint_opp

    m_pia = mean_lap_seconds_by_driver(
        df, driver_value="PIA", lap_min=lap_min_p, lap_max=lap_max_p
    )
    m_opp = mean_lap_seconds_by_driver(
        df, driver_value=opp_abbr, lap_min=lap_min_o, lap_max=lap_max_o
    )
    dp = pace_delta_seconds_per_lap(m_opp, m_pia)

    return {
        "mean_pia_s": m_pia,
        "mean_opp_s": m_opp,
        "delta_p_s_per_lap": dp,
    }


if __name__ == "__main__":
    csv = load_csv_path()
    print(f"Using CSV: {csv}")
    stats = summarize_pace_delta(csv, stint_pia=(None, None), stint_opp=(None, None))
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
