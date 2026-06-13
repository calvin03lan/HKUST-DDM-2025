#!/usr/bin/env python3
"""Export simulation data as CSV for teammates' visualization work.

Run:  uv run python scripts/export_race_log.py

Outputs:
  - outputs/gap_summary.csv       Per-lap gap for baseline + all sensitivity scenarios
  - outputs/sensitivity_results.csv  One-row-per-scenario summary (for bar charts)
  - outputs/race_log.csv          Per-segment telemetry (baseline only)
  - outputs/track_xy.csv          (generated separately, not by this script)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from sim.track_model import build_track_segments
from sim.car_model import (
    CarModel,
    estimate_car_params,
    calibrate_car_params,
    calibrate_car_from_race_pace,
)
from sim.race_sim import simulate_from_fork

_STRUCTURED_CSV = _REPO / "data" / "piastri_2026_japan_structured.csv"
_RACE_LAPS_CSV = _REPO / "data" / "race_laps_japan_2026_pia_vs_ant.csv"
_OUT_DIR = _REPO / "outputs"
_GAP_CSV = _OUT_DIR / "gap_summary.csv"
_SENS_CSV = _OUT_DIR / "sensitivity_results.csv"
_LOG_CSV = _OUT_DIR / "race_log.csv"

N_LAPS = 53
FORK_LAP = 22
REAL_FASTEST_LAP_S = 92.996
ANT_REAL_HARD_MEAN = 93.003

PIA_CUMUL_L21 = 2017.2
ANT_CUMUL_L21 = 1998.9

BASELINE_ANT_PIT_LAP = 22
BASELINE_PIT_DELTA = 22.0
BASELINE_DEG_HARD = 0.0001


def _parse_lap_time(t: str) -> float:
    hms = t.strip().split()[-1]
    h, m, s = hms.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def _load_real_cumul() -> dict[str, dict[int, float]]:
    with open(_RACE_LAPS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    cumul: dict[str, dict[int, float]] = {"PIA": {}, "ANT": {}}
    for driver in ("PIA", "ANT"):
        dr = sorted(
            [r for r in rows if r["Driver"] == driver],
            key=lambda r: int(float(r["LapNumber"])),
        )
        total = 0.0
        for r in dr:
            lap = int(float(r["LapNumber"]))
            raw = r["LapTime"].strip()
            if not raw:
                continue
            t = _parse_lap_time(raw)
            total += t
            cumul[driver][lap] = total
    return cumul


def _run_scenario(
    scenario: str,
    segments: list[dict],
    pia_base: CarModel,
    ant_base: CarModel,
    ant_pit_lap: int,
    pit_delta: float,
    deg_hard: float,
    real_cumul: dict[str, dict[int, float]],
) -> tuple[list[dict], dict]:
    """Run one scenario. Returns (gap_rows, summary_row)."""
    pia_tyre_age = ant_pit_lap - 18
    ant_fork_cumul = ANT_CUMUL_L21 + pit_delta

    pia_car = CarModel(
        name="PIA",
        a_accel_low=pia_base.a_accel_low,
        a_accel_high=pia_base.a_accel_high,
        a_brake=pia_base.a_brake,
        v_max=pia_base.v_max,
        deg_pct_per_lap=deg_hard,
        tyre_compound="HARD",
        tyre_age=pia_tyre_age,
    )
    ant_car = CarModel(
        name="ANT",
        a_accel_low=ant_base.a_accel_low,
        a_accel_high=ant_base.a_accel_high,
        a_brake=ant_base.a_brake,
        v_max=ant_base.v_max,
        deg_pct_per_lap=deg_hard,
        tyre_compound="HARD",
        tyre_age=0,
    )

    pia_sim = simulate_from_fork(
        segments, pia_car,
        start_lap=ant_pit_lap, end_lap=N_LAPS,
        t_race_cumul_init=PIA_CUMUL_L21,
    )
    ant_sim = simulate_from_fork(
        segments, ant_car,
        start_lap=ant_pit_lap, end_lap=N_LAPS,
        t_race_cumul_init=ant_fork_cumul,
    )

    pia_sim_cumul = pia_sim.groupby("lap")["t_race_cumul"].max()
    ant_sim_cumul = ant_sim.groupby("lap")["t_race_cumul"].max()

    gap_rows: list[dict] = []

    for lap in range(1, FORK_LAP):
        pia_c = real_cumul["PIA"].get(lap)
        ant_c = real_cumul["ANT"].get(lap)
        if pia_c is not None and ant_c is not None:
            gap_rows.append({
                "scenario": scenario,
                "ant_pit_lap": ant_pit_lap,
                "pit_delta_s": pit_delta,
                "deg_rate": deg_hard,
                "lap": lap,
                "pia_cumul_s": round(pia_c, 3),
                "ant_cumul_s": round(ant_c, 3),
                "gap_s": round(ant_c - pia_c, 3),
                "source": "real",
            })

    for lap in range(ant_pit_lap, N_LAPS + 1):
        pia_c = float(pia_sim_cumul.loc[lap])
        ant_c = float(ant_sim_cumul.loc[lap])
        gap_rows.append({
            "scenario": scenario,
            "ant_pit_lap": ant_pit_lap,
            "pit_delta_s": pit_delta,
            "deg_rate": deg_hard,
            "lap": lap,
            "pia_cumul_s": round(pia_c, 3),
            "ant_cumul_s": round(ant_c, 3),
            "gap_s": round(ant_c - pia_c, 3),
            "source": "simulated",
        })

    final_gap = gap_rows[-1]["gap_s"]
    summary = {
        "scenario": scenario,
        "ant_pit_lap": ant_pit_lap,
        "pit_delta_s": pit_delta,
        "deg_rate": deg_hard,
        "final_gap_s": round(final_gap, 3),
        "winner": "PIA" if final_gap > 0 else "ANT",
        "is_baseline": scenario == "baseline",
    }

    return gap_rows, summary, pia_sim, ant_sim


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building models and calibrating...")
    segments = build_track_segments(str(_STRUCTURED_CSV))
    pia_initial = estimate_car_params(str(_STRUCTURED_CSV))
    pia_base = calibrate_car_params(
        pia_initial, segments, REAL_FASTEST_LAP_S, verbose=False,
    )
    ant_base = calibrate_car_from_race_pace(
        pia_base, segments, ANT_REAL_HARD_MEAN, name="ANT",
    )

    real_cumul = _load_real_cumul()

    # ── Define all scenarios ──────────────────────────────────
    scenarios: list[tuple[str, int, float, float]] = []

    # Baseline
    scenarios.append(("baseline", BASELINE_ANT_PIT_LAP, BASELINE_PIT_DELTA, BASELINE_DEG_HARD))

    # Sweep 1: ANT pit lap
    for pl in [20, 21, 23, 24]:
        scenarios.append((f"pit_lap_{pl}", pl, BASELINE_PIT_DELTA, BASELINE_DEG_HARD))

    # Sweep 2: Pit delta
    for pit_d in [20.0, 21.0, 23.0, 24.0]:
        scenarios.append((f"pit_delta_{int(pit_d)}", BASELINE_ANT_PIT_LAP, pit_d, BASELINE_DEG_HARD))

    # Sweep 3: Degradation rate
    for dr in [0.00005, 0.0002, 0.0005]:
        scenarios.append((f"deg_{dr:.5f}", BASELINE_ANT_PIT_LAP, BASELINE_PIT_DELTA, dr))

    # ── Run all scenarios ─────────────────────────────────────
    all_gap_rows: list[dict] = []
    all_summaries: list[dict] = []
    baseline_pia_sim = None
    baseline_ant_sim = None

    for name, pit_lap, pit_delta, deg in scenarios:
        print(f"  Running scenario: {name}...")
        gap_rows, summary, pia_sim, ant_sim = _run_scenario(
            name, segments, pia_base, ant_base,
            ant_pit_lap=pit_lap, pit_delta=pit_delta, deg_hard=deg,
            real_cumul=real_cumul,
        )
        all_gap_rows.extend(gap_rows)
        all_summaries.append(summary)

        if name == "baseline":
            baseline_pia_sim = pia_sim
            baseline_ant_sim = ant_sim

    # ── gap_summary.csv ──────────────────────────────────────
    gap_df = pd.DataFrame(all_gap_rows)
    gap_df.to_csv(_GAP_CSV, index=False)
    n_scenarios = gap_df["scenario"].nunique()
    print(f"\nWrote {_GAP_CSV} ({len(gap_df)} rows, {n_scenarios} scenarios)")

    # ── sensitivity_results.csv ──────────────────────────────
    sens_df = pd.DataFrame(all_summaries)
    sens_df.to_csv(_SENS_CSV, index=False)
    print(f"Wrote {_SENS_CSV} ({len(sens_df)} rows)")

    # ── race_log.csv (baseline only) ─────────────────────────
    race_log = pd.concat([baseline_pia_sim, baseline_ant_sim], ignore_index=True)
    race_log.to_csv(_LOG_CSV, index=False)
    print(f"Wrote {_LOG_CSV} ({len(race_log)} rows)")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'Scenario':<20} {'Pit Lap':>8} {'Delta':>6} {'Deg Rate':>10} {'Gap':>8} {'Winner':>6}")
    print("-" * 65)
    for s in all_summaries:
        marker = " ◄" if s["is_baseline"] else ""
        print(f"{s['scenario']:<20} L{s['ant_pit_lap']:<7} {s['pit_delta_s']:5.0f}s "
              f"{s['deg_rate']:10.5f} {s['final_gap_s']:+7.2f}s {s['winner']:>6}{marker}")


if __name__ == "__main__":
    main()
