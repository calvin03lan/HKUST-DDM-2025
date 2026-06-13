#!/usr/bin/env python3
"""Integrated smoke test for the F1 lap simulator PoC.

Run:  uv run python sim/smoke_test.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Resolve paths relative to the repo root
_REPO = Path(__file__).resolve().parents[1]
_STRUCTURED_CSV = _REPO / "data" / "piastri_2026_japan_structured.csv"
_RACE_LAPS_CSV = _REPO / "data" / "race_laps_japan_2026_pia_vs_ant.csv"

sys.path.insert(0, str(_REPO))

from sim.track_model import build_track_segments, print_segments
from sim.car_model import estimate_car_params, calibrate_car_params, print_car_params
from sim.lap_solver import solve_lap
from sim.race_sim import simulate_race

REAL_FASTEST_LAP_S = 92.996
ONE_LAP_ERROR_THRESHOLD = 0.01  # 1% (tighter threshold with calibration)


def _parse_lap_time(td: str) -> float:
    hms = td.strip().split()[-1]
    h, m, s = hms.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def _load_real_pia_laps() -> dict[int, float]:
    """Return {lap_number: lap_time_seconds} for PIA clean laps on HARD."""
    with open(_RACE_LAPS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    result: dict[int, float] = {}
    for r in rows:
        if r["Driver"] == "PIA" and r["TrackStatus"] == "1" and r["Compound"] == "HARD":
            lap = int(float(r["LapNumber"]))
            t = _parse_lap_time(r["LapTime"])
            if t < 100:  # skip out-laps
                result[lap] = t
    return result


def main() -> None:
    failures: list[str] = []

    # --- P0: Track Model ---
    print("=" * 60)
    print("P0: Track Model")
    print("=" * 60)
    segments = build_track_segments(str(_STRUCTURED_CSV))
    print_segments(segments)
    total_dist = sum(s["length"] for s in segments)
    n_segs = len(segments)
    micro = [s for s in segments if s["length"] < 10]
    print(f"\nCheck: {n_segs} segments, total {total_dist:.1f}m, {len(micro)} micro-segments")
    assert 15 <= n_segs <= 50, f"Segment count {n_segs} out of range"
    assert len(micro) == 0, f"Found {len(micro)} micro-segments"
    v_refs = [s["v_ref"] * 3.6 for s in segments]
    assert min(v_refs) > 30, f"v_ref too low: {min(v_refs):.0f} km/h"
    assert max(v_refs) < 400, f"v_ref too high: {max(v_refs):.0f} km/h"
    print("P0: OK\n")

    # --- P1: Car Model (initial estimate + calibration) ---
    print("=" * 60)
    print("P1: Car Model (initial estimate + whole-lap calibration)")
    print("=" * 60)
    car_initial = estimate_car_params(str(_STRUCTURED_CSV))
    print("Initial estimate (telemetry dv/dt):")
    print_car_params(car_initial)

    car = calibrate_car_params(
        car_initial, segments, target_lap_s=REAL_FASTEST_LAP_S,
    )
    print()
    print_car_params(car)

    assert 5 <= car.a_accel_low <= 25, f"a_accel_low out of range: {car.a_accel_low}"
    assert 2 <= car.a_accel_high <= 12, f"a_accel_high out of range: {car.a_accel_high}"
    assert 8 <= car.a_brake <= 60, f"a_brake out of range: {car.a_brake}"
    assert 75 <= car.v_max <= 100, f"v_max out of range: {car.v_max}"
    print("P1: OK\n")

    # --- P2: One-Lap Solver ---
    print("=" * 60)
    print("P2: One-Lap Solver")
    print("=" * 60)
    lap_results = solve_lap(segments, car)
    sim_lap = sum(r.t_segment for r in lap_results)
    error_pct = abs(sim_lap - REAL_FASTEST_LAP_S) / REAL_FASTEST_LAP_S * 100
    print(f"Simulated: {sim_lap:.3f} s")
    print(f"Real:      {REAL_FASTEST_LAP_S:.3f} s")
    print(f"Error:     {error_pct:.2f}%")
    if error_pct > ONE_LAP_ERROR_THRESHOLD * 100:
        failures.append(f"P2: One-lap error {error_pct:.2f}% > {ONE_LAP_ERROR_THRESHOLD*100}%")
        print(f"P2: FAIL (threshold {ONE_LAP_ERROR_THRESHOLD*100}%)\n")
    else:
        print("P2: OK\n")

    # --- P3: 53-Lap Race Simulation ---
    print("=" * 60)
    print("P3: 53-Lap Race Simulation")
    print("=" * 60)
    race_log = simulate_race(segments, car, n_laps=53)
    lap_times = race_log.groupby("lap")["t_segment"].sum()

    real_laps = _load_real_pia_laps()
    comparison_laps = [20, 30, 40, 50, 53]

    print(f"\n{'Lap':>4}  {'Sim':>8}  {'Real':>8}  {'Delta':>7}")
    print("-" * 35)
    for lap_num in comparison_laps:
        sim_t = lap_times.loc[lap_num]
        real_t = real_laps.get(lap_num)
        if real_t:
            delta = sim_t - real_t
            print(f"{lap_num:4d}  {sim_t:8.3f}  {real_t:8.3f}  {delta:+7.3f}")
        else:
            print(f"{lap_num:4d}  {sim_t:8.3f}  {'N/A':>8}")

    # Checks
    all_times = [lap_times.loc[i] for i in range(1, 54)]
    negatives = [t for t in all_times if t <= 0]
    if negatives:
        failures.append(f"P3: Found {len(negatives)} non-positive lap times")

    range_ok = all(85 < t < 120 for t in all_times)
    if not range_ok:
        failures.append("P3: Some lap times outside 85-120s range")

    print(f"\nLap 1:  {all_times[0]:.3f}s")
    print(f"Lap 53: {all_times[-1]:.3f}s")
    print(f"Total increase: {all_times[-1] - all_times[0]:.3f}s")
    print(f"All laps in 85-120s range: {range_ok}")
    print(f"DataFrame shape: {race_log.shape}")
    print(f"Columns: {list(race_log.columns)}")

    if not failures:
        print("\nP3: OK\n")
    else:
        print(f"\nP3: ISSUES — {failures[-1]}\n")

    # --- P4: Cross-validation vs Tier-0 ---
    print("=" * 60)
    print("P4: Cross-validation")
    print("=" * 60)

    # Compute mean clean-lap pace from simulation (laps 20-53)
    sim_clean_mean = lap_times.loc[20:53].mean()

    # From real data
    real_clean = [v for k, v in real_laps.items() if 20 <= k <= 53]
    real_clean_mean = sum(real_clean) / len(real_clean) if real_clean else 0

    print(f"Sim mean pace (laps 20-53):  {sim_clean_mean:.3f}s")
    print(f"Real mean pace (laps 20-53): {real_clean_mean:.3f}s")
    print(f"Delta: {sim_clean_mean - real_clean_mean:+.3f}s")

    # --- Final verdict ---
    print("\n" + "=" * 60)
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        print("RESULT: PASS (P0–P4)")
    print("=" * 60)

    # --- P5: Fork-Point Dual-Driver Simulation ---
    print("\n" + "=" * 60)
    print("P5: Fork-Point Simulation (PIA vs ANT, L22–L53)")
    print("=" * 60)

    from sim.race_sim import simulate_from_fork
    from sim.car_model import calibrate_car_from_race_pace, CarModel

    pia_sim_lap = sum(r.t_segment for r in solve_lap(segments, car))
    ant_real_hard_mean = 93.003
    ant_base = calibrate_car_from_race_pace(car, segments, ant_real_hard_mean, name="ANT")
    ant_sim_lap = sum(r.t_segment for r in solve_lap(segments, ant_base))

    print(f"PIA sim one-lap: {pia_sim_lap:.3f}s")
    print(f"ANT sim one-lap: {ant_sim_lap:.3f}s (calibrated)")
    print(f"Delta: {pia_sim_lap - ant_sim_lap:+.3f}s")

    # Real cumulative times at end of L21
    PIA_CUMUL_L21 = 2017.2
    ANT_CUMUL_L21 = 1998.9
    PIT_DELTA = 22.0
    FORK_LAP = 22
    END_LAP = 53

    # PIA at fork: HARD, tyre age = 3 (pitted L18, drove L19-L21)
    pia_fork = CarModel(
        name="PIA", a_accel_low=car.a_accel_low, a_accel_high=car.a_accel_high,
        a_brake=car.a_brake, v_max=car.v_max,
        deg_pct_per_lap=0.0001, tyre_compound="HARD", tyre_age=3,
    )
    # ANT at fork: HARD, tyre age = 0 (just pitted normally at L22)
    ant_fork = CarModel(
        name="ANT", a_accel_low=ant_base.a_accel_low, a_accel_high=ant_base.a_accel_high,
        a_brake=ant_base.a_brake, v_max=ant_base.v_max,
        deg_pct_per_lap=0.0001, tyre_compound="HARD", tyre_age=0,
    )

    ant_cumul_after_pit = ANT_CUMUL_L21 + PIT_DELTA
    initial_gap = ant_cumul_after_pit - PIA_CUMUL_L21
    print(f"\nFork-point initial conditions:")
    print(f"  PIA cumul L21 = {PIA_CUMUL_L21:.1f}s (HARD, age 3)")
    print(f"  ANT cumul L21 = {ANT_CUMUL_L21:.1f}s + {PIT_DELTA:.0f}s pit = {ant_cumul_after_pit:.1f}s (HARD, age 0)")
    print(f"  Starting gap = {initial_gap:+.1f}s (PIA ahead)")

    pia_sim_fork = simulate_from_fork(
        segments, pia_fork, start_lap=FORK_LAP, end_lap=END_LAP,
        t_race_cumul_init=PIA_CUMUL_L21,
    )
    ant_sim_fork = simulate_from_fork(
        segments, ant_fork, start_lap=FORK_LAP, end_lap=END_LAP,
        t_race_cumul_init=ant_cumul_after_pit,
    )

    pia_cumul = pia_sim_fork.groupby("lap")["t_race_cumul"].max()
    ant_cumul = ant_sim_fork.groupby("lap")["t_race_cumul"].max()

    check_laps = [FORK_LAP, 30, 40, 50, END_LAP]
    print(f"\n{'Lap':>4}  {'Gap':>9}  (+ = PIA ahead)")
    print("-" * 30)
    for lap in check_laps:
        gap = ant_cumul.loc[lap] - pia_cumul.loc[lap]
        print(f"{lap:4d}  {gap:+9.3f}s")

    final_gap = ant_cumul.loc[END_LAP] - pia_cumul.loc[END_LAP]
    print(f"\nFinal gap at L{END_LAP}: {final_gap:+.3f}s")

    # Validation: PIA should be ahead at the end (gap > 0)
    if final_gap <= 0:
        failures.append("P5: PIA not ahead at L53 in fork-point What-If")
    if not (0.5 < final_gap < 15):
        failures.append(f"P5: Final gap {final_gap:.1f}s outside plausible range (0.5–15s)")

    # Verify lap numbers in output match real race lap numbers
    sim_laps = sorted(pia_sim_fork["lap"].unique())
    expected_laps = list(range(FORK_LAP, END_LAP + 1))
    if sim_laps != expected_laps:
        failures.append(f"P5: Lap numbers mismatch: got {sim_laps[0]}–{sim_laps[-1]}, expected {FORK_LAP}–{END_LAP}")

    # Verify lap times are in plausible range
    pia_lt = pia_sim_fork.groupby("lap")["t_segment"].sum()
    ant_lt = ant_sim_fork.groupby("lap")["t_segment"].sum()
    pia_range_ok = all(85 < pia_lt.loc[i] < 120 for i in range(FORK_LAP, END_LAP + 1))
    ant_range_ok = all(85 < ant_lt.loc[i] < 120 for i in range(FORK_LAP, END_LAP + 1))
    if not pia_range_ok:
        failures.append("P5: PIA fork-sim has lap times outside 85-120s range")
    if not ant_range_ok:
        failures.append("P5: ANT fork-sim has lap times outside 85-120s range")

    if not any(f.startswith("P5") for f in failures):
        print("P5: OK\n")
    else:
        for f in failures:
            if f.startswith("P5"):
                print(f"P5: FAIL — {f}\n")

    # --- P6: Cross-Validation (L28–L53 vs real clean laps) ---
    print("=" * 60)
    print("P6: Cross-Validation (sim vs real, L28–L53)")
    print("=" * 60)

    import math

    def _load_real_clean(driver: str) -> dict[int, float]:
        with open(_RACE_LAPS_CSV, newline="") as fh:
            rows = list(csv.DictReader(fh))
        out: dict[int, float] = {}
        for r in rows:
            if r["Driver"] != driver or r["TrackStatus"] != "1":
                continue
            raw = r["LapTime"].strip()
            if not raw:
                continue
            lap = int(float(r["LapNumber"]))
            if 28 <= lap <= 53:
                t = _parse_lap_time(raw)
                if t < 110:
                    out[lap] = t
        return out

    # PIA: HARD, tyre_age=9 at L28 (pitted L18, 9 laps on HARD by L28)
    pia_xval_car = CarModel(
        name="PIA", a_accel_low=car.a_accel_low, a_accel_high=car.a_accel_high,
        a_brake=car.a_brake, v_max=car.v_max,
        deg_pct_per_lap=0.0001, tyre_compound="HARD", tyre_age=9,
    )
    pia_xval = simulate_from_fork(segments, pia_xval_car, start_lap=28, end_lap=53)
    pia_xval_lt = pia_xval.groupby("lap")["t_segment"].sum()

    pia_real = _load_real_clean("PIA")
    pia_deltas = [pia_xval_lt.loc[l] - pia_real[l] for l in pia_real if l in pia_xval_lt.index]
    pia_mae = sum(abs(d) for d in pia_deltas) / len(pia_deltas) if pia_deltas else 999
    pia_rmse = math.sqrt(sum(d**2 for d in pia_deltas) / len(pia_deltas)) if pia_deltas else 999

    print(f"PIA: MAE = {pia_mae:.3f}s, RMSE = {pia_rmse:.3f}s ({len(pia_deltas)} laps)")

    if pia_mae > 1.5:
        failures.append(f"P6: PIA MAE {pia_mae:.2f}s > 1.5s threshold")

    # ANT: HARD, tyre_age=6 at L28
    ant_xval_car = CarModel(
        name="ANT", a_accel_low=ant_base.a_accel_low, a_accel_high=ant_base.a_accel_high,
        a_brake=ant_base.a_brake, v_max=ant_base.v_max,
        deg_pct_per_lap=0.0001, tyre_compound="HARD", tyre_age=6,
    )
    ant_xval = simulate_from_fork(segments, ant_xval_car, start_lap=28, end_lap=53)
    ant_xval_lt = ant_xval.groupby("lap")["t_segment"].sum()

    ant_real = _load_real_clean("ANT")
    ant_deltas = [ant_xval_lt.loc[l] - ant_real[l] for l in ant_real if l in ant_xval_lt.index]
    ant_mae = sum(abs(d) for d in ant_deltas) / len(ant_deltas) if ant_deltas else 999
    ant_rmse = math.sqrt(sum(d**2 for d in ant_deltas) / len(ant_deltas)) if ant_deltas else 999

    print(f"ANT: MAE = {ant_mae:.3f}s, RMSE = {ant_rmse:.3f}s ({len(ant_deltas)} laps)")

    if ant_mae > 1.5:
        failures.append(f"P6: ANT MAE {ant_mae:.2f}s > 1.5s threshold")

    if not any(f.startswith("P6") for f in failures):
        print("P6: OK\n")
    else:
        for f in failures:
            if f.startswith("P6"):
                print(f"P6: FAIL — {f}\n")

    # --- Overall verdict ---
    print("=" * 60)
    if failures:
        print("OVERALL RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        print("OVERALL RESULT: PASS (P0–P6)")
    print("=" * 60)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
