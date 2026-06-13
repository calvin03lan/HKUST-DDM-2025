#!/usr/bin/env python3
"""Sensitivity analysis: how robust is the PIA +1.8s conclusion?

Sweeps three variables independently:
  1. ANT pit lap (L20–L24)
  2. Pit time delta (20–24s)
  3. HARD degradation rate (0.00005–0.0005)

Run:  uv run python scripts/sensitivity.py

Outputs:
  - Console summary table
  - outputs/sensitivity.png (3-panel chart)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from sim.track_model import build_track_segments
from sim.car_model import (
    CarModel,
    estimate_car_params,
    calibrate_car_params,
    calibrate_car_from_race_pace,
)
from sim.lap_solver import solve_lap
from sim.race_sim import simulate_from_fork

_STRUCTURED_CSV = _REPO / "data" / "piastri_2026_japan_structured.csv"
_RACE_LAPS_CSV = _REPO / "data" / "race_laps_japan_2026_pia_vs_ant.csv"
_OUTPUT = _REPO / "outputs" / "sensitivity.png"

N_LAPS = 53
REAL_FASTEST_LAP_S = 92.996

PIA_CUMUL_L21 = 2017.2
ANT_CUMUL_L21 = 1998.9

# Baselines
BASELINE_ANT_PIT_LAP = 22
BASELINE_PIT_DELTA = 22.0
BASELINE_DEG_HARD = 0.0001


def _run_fork_sim(
    segments: list[dict],
    pia_base: CarModel,
    ant_base: CarModel,
    ant_pit_lap: int,
    pit_delta: float,
    deg_hard: float,
) -> float:
    """Run fork-point sim and return final gap at L53 (positive = PIA ahead)."""
    pia_tyre_age = ant_pit_lap - 18  # PIA pitted L18
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

    pia_final = pia_sim.groupby("lap")["t_race_cumul"].max().loc[N_LAPS]
    ant_final = ant_sim.groupby("lap")["t_race_cumul"].max().loc[N_LAPS]
    return ant_final - pia_final


def main() -> None:
    print("Building models...")
    segments = build_track_segments(str(_STRUCTURED_CSV))
    pia_initial = estimate_car_params(str(_STRUCTURED_CSV))
    pia_base = calibrate_car_params(
        pia_initial, segments, REAL_FASTEST_LAP_S, verbose=False,
    )
    ant_base = calibrate_car_from_race_pace(
        pia_base, segments, 93.003, name="ANT",
    )

    # ── Sweep 1: ANT pit lap ─────────────────────────────────
    pit_laps = [20, 21, 22, 23, 24]
    gaps_pit_lap: list[float] = []
    print("\n=== Sweep 1: ANT Pit Lap ===")
    print(f"{'Pit Lap':>8}  {'Final Gap':>10}  {'Result':>12}")
    print("-" * 35)
    for pl in pit_laps:
        g = _run_fork_sim(segments, pia_base, ant_base,
                          ant_pit_lap=pl, pit_delta=BASELINE_PIT_DELTA,
                          deg_hard=BASELINE_DEG_HARD)
        gaps_pit_lap.append(g)
        result = "PIA wins" if g > 0 else "ANT wins"
        marker = " ◄ baseline" if pl == BASELINE_ANT_PIT_LAP else ""
        print(f"   L{pl:2d}     {g:+10.2f}s  {result}{marker}")

    # ── Sweep 2: Pit delta ────────────────────────────────────
    pit_deltas = [20.0, 21.0, 22.0, 23.0, 24.0]
    gaps_pit_delta: list[float] = []
    print("\n=== Sweep 2: Pit Time Delta ===")
    print(f"{'Delta':>8}  {'Final Gap':>10}  {'Result':>12}")
    print("-" * 35)
    for pd in pit_deltas:
        g = _run_fork_sim(segments, pia_base, ant_base,
                          ant_pit_lap=BASELINE_ANT_PIT_LAP, pit_delta=pd,
                          deg_hard=BASELINE_DEG_HARD)
        gaps_pit_delta.append(g)
        result = "PIA wins" if g > 0 else "ANT wins"
        marker = " ◄ baseline" if pd == BASELINE_PIT_DELTA else ""
        print(f"   {pd:4.0f}s    {g:+10.2f}s  {result}{marker}")

    # ── Sweep 3: Degradation rate ─────────────────────────────
    deg_rates = [0.00005, 0.0001, 0.0002, 0.0005]
    gaps_deg: list[float] = []
    print("\n=== Sweep 3: HARD Degradation Rate ===")
    print(f"{'Rate':>10}  {'Final Gap':>10}  {'Result':>12}")
    print("-" * 40)
    for dr in deg_rates:
        g = _run_fork_sim(segments, pia_base, ant_base,
                          ant_pit_lap=BASELINE_ANT_PIT_LAP,
                          pit_delta=BASELINE_PIT_DELTA, deg_hard=dr)
        gaps_deg.append(g)
        result = "PIA wins" if g > 0 else "ANT wins"
        marker = " ◄ baseline" if dr == BASELINE_DEG_HARD else ""
        print(f"  {dr:.5f}   {g:+10.2f}s  {result}{marker}")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: ANT pit lap
    ax = axes[0]
    colors = ["steelblue" if pl != BASELINE_ANT_PIT_LAP else "navy" for pl in pit_laps]
    labels = [f"L{pl}" for pl in pit_laps]
    ax.bar(labels, gaps_pit_lap, color=colors, alpha=0.8, zorder=3)
    ax.axhline(y=0, color="red", linewidth=1.5, label="Gap = 0 (PIA loses)")
    ax.axhline(y=1.5, color="orange", linewidth=1, linestyle="--",
               label="DRS threshold (1.5s)")
    ax.set_xlabel("ANT Pit Lap")
    ax.set_ylabel("Final Gap at L53 (seconds)")
    ax.set_title("ANT Pit Lap Sensitivity", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Pit delta
    ax = axes[1]
    colors = ["steelblue" if pd != BASELINE_PIT_DELTA else "navy" for pd in pit_deltas]
    labels = [f"{pd:.0f}s" for pd in pit_deltas]
    ax.bar(labels, gaps_pit_delta, color=colors, alpha=0.8, zorder=3)
    ax.axhline(y=0, color="red", linewidth=1.5, label="Gap = 0")
    ax.axhline(y=1.5, color="orange", linewidth=1, linestyle="--",
               label="DRS threshold")
    ax.set_xlabel("Pit Stop Delta (seconds)")
    ax.set_title("Pit Delta Sensitivity", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Degradation rate
    ax = axes[2]
    colors = ["steelblue" if dr != BASELINE_DEG_HARD else "navy" for dr in deg_rates]
    labels = [f"{dr*10000:.1f}e-4" for dr in deg_rates]
    ax.bar(labels, gaps_deg, color=colors, alpha=0.8, zorder=3)
    ax.axhline(y=0, color="red", linewidth=1.5, label="Gap = 0")
    ax.axhline(y=1.5, color="orange", linewidth=1, linestyle="--",
               label="DRS threshold")
    ax.set_xlabel("deg_pct_per_lap")
    ax.set_title("Tyre Degradation Sensitivity", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Sensitivity Analysis: How Robust Is the PIA +1.8s Conclusion?",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_OUTPUT), dpi=150)
    print(f"\nChart saved to {_OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
