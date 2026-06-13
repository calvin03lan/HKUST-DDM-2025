#!/usr/bin/env python3
"""Generate fork-point gap chart: real data L1-21, simulated What-If L22-53.

Run:  uv run python scripts/plot_gap.py

Outputs:  outputs/gap_vs_lap.png
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
    print_car_params,
)
from sim.lap_solver import solve_lap
from sim.race_sim import simulate_from_fork

_STRUCTURED_CSV = _REPO / "data" / "piastri_2026_japan_structured.csv"
_RACE_LAPS_CSV = _REPO / "data" / "race_laps_japan_2026_pia_vs_ant.csv"
_OUTPUT = _REPO / "outputs" / "gap_vs_lap.png"

N_LAPS = 53
FORK_LAP = 22

# Real race events
PIA_PIT_LAP = 18
ANT_PIT_LAP = 22
SC_START_LAP = 22
SC_END_LAP = 27

# Degradation rates
DEG_MEDIUM = 0.0008
DEG_HARD = 0.0001

# Initial conditions at end of L21 (from real data)
PIA_CUMUL_L21 = 2017.2
ANT_CUMUL_L21 = 1998.9
PIT_DELTA = 22.0


def _parse_lap_time(t: str) -> float:
    hms = t.strip().split()[-1]
    h, m, s = hms.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def _load_real_cumul() -> dict[str, dict[int, float]]:
    """Load real race cumulative times per driver up to L21."""
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


def _real_gaps(cumul: dict[str, dict[int, float]], max_lap: int) -> dict[int, float]:
    """Gap = ANT_cumul - PIA_cumul (positive = PIA ahead) for laps up to max_lap."""
    gaps: dict[int, float] = {}
    for lap in range(1, max_lap + 1):
        if lap in cumul["PIA"] and lap in cumul["ANT"]:
            gaps[lap] = cumul["ANT"][lap] - cumul["PIA"][lap]
    return gaps


def main() -> None:
    # ── Track & car models ───────────────────────────────────
    print("Building track and car models...")
    segments = build_track_segments(str(_STRUCTURED_CSV))

    pia_initial = estimate_car_params(str(_STRUCTURED_CSV))
    REAL_FASTEST_LAP_S = 92.996

    print("\n--- PIA Calibration ---")
    pia_base = calibrate_car_params(
        pia_initial, segments, target_lap_s=REAL_FASTEST_LAP_S,
    )
    pia_sim_lap = sum(r.t_segment for r in solve_lap(segments, pia_base))

    ant_real_hard_mean = 93.003
    ant_base = calibrate_car_from_race_pace(
        pia_base, segments, ant_real_hard_mean, name="ANT"
    )

    print("\n=== PIA (calibrated) ===")
    print_car_params(pia_base)
    print(f"Sim one-lap: {pia_sim_lap:.3f}s")

    ant_sim_lap = sum(r.t_segment for r in solve_lap(segments, ant_base))
    print("\n=== ANT (calibrated from PIA baseline) ===")
    print_car_params(ant_base)
    print(f"Sim one-lap: {ant_sim_lap:.3f}s")
    print(f"Delta PIA-ANT: {pia_sim_lap - ant_sim_lap:+.3f}s\n")

    # ── Phase 1: Real data L1–L21 ────────────────────────────
    real_cumul = _load_real_cumul()
    real_gap = _real_gaps(real_cumul, FORK_LAP - 1)

    print(f"Real data L1–{FORK_LAP - 1}:")
    for lap in [1, 10, PIA_PIT_LAP, FORK_LAP - 1]:
        if lap in real_gap:
            print(f"  L{lap:02d}  gap = {real_gap[lap]:+.1f}s")

    # ── Phase 2: Fork-point simulation L22–L53 ───────────────
    # What-If: ANT pits normally at L22 (loses ~22s pit delta)
    # PIA at L22 start: HARD, tyre_age = 3 (pitted L18, laps 19-20-21 on HARD)
    # ANT at L22 start: HARD, tyre_age = 0 (just pitted)
    pia_fork_cumul = PIA_CUMUL_L21
    ant_fork_cumul = ANT_CUMUL_L21 + PIT_DELTA

    pia_fork_car = CarModel(
        name="PIA",
        a_accel_low=pia_base.a_accel_low,
        a_accel_high=pia_base.a_accel_high,
        a_brake=pia_base.a_brake,
        v_max=pia_base.v_max,
        deg_pct_per_lap=DEG_HARD,
        tyre_compound="HARD",
        tyre_age=3,
    )
    ant_fork_car = CarModel(
        name="ANT",
        a_accel_low=ant_base.a_accel_low,
        a_accel_high=ant_base.a_accel_high,
        a_brake=ant_base.a_brake,
        v_max=ant_base.v_max,
        deg_pct_per_lap=DEG_HARD,
        tyre_compound="HARD",
        tyre_age=0,
    )

    print(f"\nFork-point (L{FORK_LAP}):")
    print(f"  PIA cumul = {pia_fork_cumul:.1f}s  (HARD, age 3)")
    print(f"  ANT cumul = {ant_fork_cumul:.1f}s  (HARD, age 0, incl. {PIT_DELTA}s pit)")
    print(f"  Starting gap = {ant_fork_cumul - pia_fork_cumul:+.1f}s (PIA ahead)")

    pia_sim = simulate_from_fork(
        segments, pia_fork_car,
        start_lap=FORK_LAP, end_lap=N_LAPS,
        t_race_cumul_init=pia_fork_cumul,
    )
    ant_sim = simulate_from_fork(
        segments, ant_fork_car,
        start_lap=FORK_LAP, end_lap=N_LAPS,
        t_race_cumul_init=ant_fork_cumul,
    )

    pia_sim_cumul = pia_sim.groupby("lap")["t_race_cumul"].max()
    ant_sim_cumul = ant_sim.groupby("lap")["t_race_cumul"].max()

    sim_gap: dict[int, float] = {}
    for lap in range(FORK_LAP, N_LAPS + 1):
        sim_gap[lap] = ant_sim_cumul.loc[lap] - pia_sim_cumul.loc[lap]

    print(f"\nSimulated What-If gap (L{FORK_LAP}–L{N_LAPS}):")
    print(f"  {'Lap':>4}  {'Gap':>9}  (+ = PIA ahead)")
    print(f"  {'-' * 28}")
    for lap in [FORK_LAP, 30, 40, 50, N_LAPS]:
        print(f"  {lap:4d}  {sim_gap[lap]:+9.3f}s")

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Phase 1: real data (solid gray)
    real_laps = sorted(real_gap.keys())
    real_vals = [real_gap[l] for l in real_laps]
    ax.plot(real_laps, real_vals, color="0.35", linewidth=2.5,
            label=f"Real data (L1–{FORK_LAP - 1})", zorder=3)

    # Phase 2: simulated (solid blue)
    sim_laps = sorted(sim_gap.keys())
    sim_vals = [sim_gap[l] for l in sim_laps]
    ax.plot(sim_laps, sim_vals, "b-", linewidth=2.5,
            label=f"Simulated What-If (L{FORK_LAP}–{N_LAPS})", zorder=3)

    # Replace dashed bridge with an arrow annotation showing the pit-stop jump
    if real_laps and sim_laps:
        real_end_y = real_gap[real_laps[-1]]
        sim_start_y = sim_gap[sim_laps[0]]
        ax.annotate(
            "",
            xy=(sim_laps[0], sim_start_y),
            xytext=(real_laps[-1], real_end_y),
            arrowprops=dict(
                arrowstyle="-|>", color="0.5", linewidth=1.5,
                linestyle="--", mutation_scale=15,
            ),
            zorder=2,
        )
        mid_x = (real_laps[-1] + sim_laps[0]) / 2
        mid_y = (real_end_y + sim_start_y) / 2
        ax.text(
            mid_x - 3.5, mid_y,
            f"ANT pits\nnormally\n(+{PIT_DELTA:.0f}s)",
            fontsize=8, color="0.4", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9),
        )

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")

    # Fork-point annotation
    ax.axvline(x=FORK_LAP, color="red", linewidth=1.5, linestyle="--", alpha=0.8)
    fork_y = sim_gap[FORK_LAP]
    ax.annotate(
        f"Fork point L{FORK_LAP}\nANT normal pit\nGap: {fork_y:+.1f}s",
        xy=(FORK_LAP, fork_y),
        xytext=(FORK_LAP + 4, fork_y + 10),
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", linewidth=1.5),
        color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", edgecolor="red", alpha=0.9),
    )

    # PIA pit annotation
    if PIA_PIT_LAP in real_gap:
        ax.axvline(x=PIA_PIT_LAP, color="orange", linewidth=1, linestyle=":", alpha=0.7)
        ax.annotate(
            "PIA pit (L18)",
            xy=(PIA_PIT_LAP, real_gap[PIA_PIT_LAP]),
            xytext=(PIA_PIT_LAP - 8, real_gap[PIA_PIT_LAP] - 8),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="orange"),
            color="orange",
        )

    # SC window shading (what actually happened in reality)
    ax.axvspan(SC_START_LAP, SC_END_LAP, alpha=0.12, color="yellow",
               label=f"Real SC window (L{SC_START_LAP}–{SC_END_LAP})")

    # Final gap annotation
    final_gap = sim_gap[N_LAPS]
    winner = "PIA wins" if final_gap > 0 else "ANT wins"
    ax.annotate(
        f"Final gap: {final_gap:+.1f}s\n{winner}",
        xy=(N_LAPS, final_gap),
        xytext=(N_LAPS - 12, final_gap + 8),
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5),
        color="blue",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="blue"),
    )

    # Explanation text
    ax.text(
        0.02, 0.02,
        f"L1–{FORK_LAP - 1}: Real race data  |  L{FORK_LAP}–{N_LAPS}: Simulated (no SC)\n"
        f"Fork point: ANT pits normally L{ANT_PIT_LAP} (+{PIT_DELTA:.0f}s)\n"
        f"Real race: SC L{SC_START_LAP}–{SC_END_LAP}, ANT gets free pit → wins",
        transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Lap", fontsize=12)
    ax.set_ylabel("Gap (seconds) — positive = PIA ahead", fontsize=12)
    ax.set_title(
        "What-If: 2026 Japanese GP Without Safety Car (Fork-Point Simulation)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.3)

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(_OUTPUT), dpi=150)
    print(f"\nChart saved to {_OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
