#!/usr/bin/env python3
"""Cross-validate simulated lap times against real post-SC clean laps (L28-L53).

This validates both the calibrated car parameters AND the tyre degradation rate.
If deg_pct_per_lap is wrong, error will drift systematically over 25+ laps.

Run:  uv run python scripts/cross_validate.py

Outputs:
  - Console table of per-lap errors
  - outputs/cross_validation.png
"""

from __future__ import annotations

import csv
import math
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
_OUTPUT = _REPO / "outputs" / "cross_validation.png"

REAL_FASTEST_LAP_S = 92.996

DEG_HARD = 0.0001

# Post-SC clean racing starts at L28
VALIDATE_START = 28
VALIDATE_END = 53


def _parse_lap_time(t: str) -> float:
    hms = t.strip().split()[-1]
    h, m, s = hms.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def _load_real_clean_laps(driver: str) -> dict[int, float]:
    """Load clean (TrackStatus=1) lap times for a driver on HARD compound."""
    with open(_RACE_LAPS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    result: dict[int, float] = {}
    for r in rows:
        if r["Driver"] != driver:
            continue
        if r["TrackStatus"] != "1":
            continue
        raw = r["LapTime"].strip()
        if not raw:
            continue
        lap = int(float(r["LapNumber"]))
        if lap < VALIDATE_START or lap > VALIDATE_END:
            continue
        t = _parse_lap_time(raw)
        if t < 110:
            result[lap] = t
    return result


def main() -> None:
    print("Building track and car models...")
    segments = build_track_segments(str(_STRUCTURED_CSV))

    pia_initial = estimate_car_params(str(_STRUCTURED_CSV))
    pia_cal = calibrate_car_params(
        pia_initial, segments, target_lap_s=REAL_FASTEST_LAP_S,
    )
    ant_cal = calibrate_car_from_race_pace(
        pia_cal, segments, 93.003, name="ANT"
    )

    # PIA at L28: HARD, tyre_age = 9 (pitted L18, L19..L27 = 9 laps on HARD)
    # ANT at L28: HARD, tyre_age = 6 (pitted L22 under SC, L23..L27 = 5 + out-lap)
    # We use t_race_cumul_init = 0 since we only compare per-lap times, not cumul
    drivers_config = [
        ("PIA", pia_cal, 9),
        ("ANT", ant_cal, 6),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (name, car_base, tyre_age_at_28) in enumerate(drivers_config):
        real_laps = _load_real_clean_laps(name)
        if not real_laps:
            print(f"\n{name}: No clean laps found for L{VALIDATE_START}–{VALIDATE_END}")
            continue

        fork_car = CarModel(
            name=name,
            a_accel_low=car_base.a_accel_low,
            a_accel_high=car_base.a_accel_high,
            a_brake=car_base.a_brake,
            v_max=car_base.v_max,
            deg_pct_per_lap=DEG_HARD,
            tyre_compound="HARD",
            tyre_age=tyre_age_at_28,
        )

        sim_df = simulate_from_fork(
            segments, fork_car,
            start_lap=VALIDATE_START, end_lap=VALIDATE_END,
            t_race_cumul_init=0.0,
        )
        sim_lap_times = sim_df.groupby("lap")["t_segment"].sum()

        print(f"\n{'='*50}")
        print(f"Cross-Validation: {name} (L{VALIDATE_START}–{VALIDATE_END})")
        print(f"{'='*50}")
        print(f"{'Lap':>4}  {'Sim':>8}  {'Real':>8}  {'Delta':>7}")
        print("-" * 35)

        deltas: list[float] = []
        plot_laps: list[int] = []
        plot_deltas: list[float] = []

        for lap in range(VALIDATE_START, VALIDATE_END + 1):
            sim_t = sim_lap_times.loc[lap]
            if lap in real_laps:
                real_t = real_laps[lap]
                delta = sim_t - real_t
                deltas.append(delta)
                plot_laps.append(lap)
                plot_deltas.append(delta)
                print(f"{lap:4d}  {sim_t:8.3f}  {real_t:8.3f}  {delta:+7.3f}")
            else:
                print(f"{lap:4d}  {sim_t:8.3f}  {'N/A':>8}")

        if deltas:
            mean_err = sum(deltas) / len(deltas)
            abs_errs = [abs(d) for d in deltas]
            max_err = max(abs_errs)
            rmse = math.sqrt(sum(d ** 2 for d in deltas) / len(deltas))

            # Trend: linear regression slope
            n = len(plot_laps)
            x_mean = sum(plot_laps) / n
            y_mean = sum(plot_deltas) / n
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(plot_laps, plot_deltas))
            den = sum((x - x_mean) ** 2 for x in plot_laps)
            slope = num / den if den > 0 else 0

            print(f"\nMean error:  {mean_err:+.3f}s")
            print(f"Max |error|: {max_err:.3f}s")
            print(f"RMSE:        {rmse:.3f}s")
            print(f"Trend slope: {slope:+.4f} s/lap", end="")
            if abs(slope) < 0.01:
                print("  (flat → degradation rate OK)")
            elif slope > 0:
                print("  (growing → sim degrading too slowly)")
            else:
                print("  (shrinking → sim degrading too fast)")

            # Plot
            ax = axes[idx]
            ax.bar(plot_laps, plot_deltas, color="steelblue" if idx == 0 else "coral",
                   alpha=0.7, zorder=3)
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.axhline(y=mean_err, color="red", linewidth=1, linestyle="--",
                       label=f"Mean: {mean_err:+.2f}s")

            # Trend line
            trend_y = [slope * (x - x_mean) + y_mean for x in plot_laps]
            ax.plot(plot_laps, trend_y, "k--", linewidth=1,
                    label=f"Trend: {slope:+.4f} s/lap")

            ax.set_xlabel("Lap", fontsize=11)
            ax.set_title(f"{name}: Sim − Real (L{VALIDATE_START}–{VALIDATE_END})",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Error (seconds) — sim minus real", fontsize=11)

    fig.suptitle("Cross-Validation: Calibrated Simulation vs Real Post-SC Laps",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_OUTPUT), dpi=150)
    print(f"\nChart saved to {_OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
