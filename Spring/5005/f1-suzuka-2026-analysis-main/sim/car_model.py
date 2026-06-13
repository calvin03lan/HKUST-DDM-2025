"""Car model: dataclass + parameter estimation from structured telemetry."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

_KMH_TO_MS = 1.0 / 3.6
V_ACCEL_SPLIT_KMH = 200.0
V_ACCEL_SPLIT_MS = V_ACCEL_SPLIT_KMH * _KMH_TO_MS


@dataclass
class CarModel:
    name: str
    a_accel_low: float    # m/s^2, for speeds < 200 km/h
    a_accel_high: float   # m/s^2, for speeds >= 200 km/h
    a_brake: float        # m/s^2, positive (applied as deceleration)
    v_max: float          # m/s
    deg_pct_per_lap: float = 0.0001
    tyre_compound: str = "HARD"
    tyre_age: int = 0

    def a_accel_at(self, v_ms: float) -> float:
        """Return the appropriate acceleration for the given speed."""
        return self.a_accel_low if v_ms < V_ACCEL_SPLIT_MS else self.a_accel_high


def _parse_lap_time(t: str) -> float:
    """Parse FastF1 timedelta string like '0 days 00:01:32.725000' to seconds."""
    hms = t.strip().split()[-1]
    h, m, s = hms.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def estimate_car_params(csv_path: str | Path, name: str = "PIA") -> CarModel:
    """Estimate car performance parameters from a structured telemetry CSV."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    accel_low: list[float] = []
    accel_high: list[float] = []
    brake_samples: list[float] = []

    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        s1 = float(prev["Speed"])
        s2 = float(cur["Speed"])
        t1 = _parse_lap_time(prev["Time"])
        t2 = _parse_lap_time(cur["Time"])
        dt = t2 - t1

        if dt < 0.01:
            continue

        # Acceleration samples
        if (
            cur["Driving_State"] == "Full Acceleration"
            and prev["Driving_State"] == "Full Acceleration"
        ):
            dv = (s2 - s1) * _KMH_TO_MS
            a = dv / dt
            if a > 0.1:
                avg_kmh = (s1 + s2) / 2.0
                if avg_kmh < V_ACCEL_SPLIT_KMH:
                    accel_low.append(a)
                else:
                    accel_high.append(a)

        # Braking samples
        if cur["Driving_State"] == "Heavy Braking" and prev["Driving_State"] in (
            "Heavy Braking",
            "Full Acceleration",
            "Transition",
        ):
            dv = (s1 - s2) * _KMH_TO_MS
            if dv > 0:
                a = dv / dt
                if a > 5.0:
                    brake_samples.append(a)

    v_max_kmh = max(float(r["Speed"]) for r in rows)

    return CarModel(
        name=name,
        a_accel_low=statistics.median(accel_low) if accel_low else 12.0,
        a_accel_high=statistics.median(accel_high) if accel_high else 5.0,
        a_brake=statistics.median(brake_samples) if brake_samples else 30.0,
        v_max=v_max_kmh * _KMH_TO_MS,
    )


def print_car_params(car: CarModel) -> None:
    print(f"Car: {car.name}")
    print(f"  a_accel_low  (< 200 km/h): {car.a_accel_low:6.2f} m/s²")
    print(f"  a_accel_high (≥ 200 km/h): {car.a_accel_high:6.2f} m/s²")
    print(f"  a_brake:                    {car.a_brake:6.2f} m/s²")
    print(f"  v_max:                      {car.v_max:6.2f} m/s  ({car.v_max / _KMH_TO_MS:.1f} km/h)")
    print(f"  deg_pct_per_lap:            {car.deg_pct_per_lap:.6f}")
    print(f"  tyre: {car.tyre_compound}  age: {car.tyre_age}")


def calibrate_car_from_race_pace(
    baseline: CarModel,
    segments: list[dict],
    target_lap_s: float,
    name: str = "ANT",
) -> CarModel:
    """Create a CarModel by scaling v_max until the solver matches target_lap_s.

    Uses binary search on v_max. All other params inherited from *baseline*.
    """
    from .lap_solver import solve_lap

    lo_factor = 0.90
    hi_factor = 1.15

    for _ in range(60):
        mid_factor = (lo_factor + hi_factor) / 2.0
        trial = CarModel(
            name=name,
            a_accel_low=baseline.a_accel_low,
            a_accel_high=baseline.a_accel_high,
            a_brake=baseline.a_brake,
            v_max=baseline.v_max * mid_factor,
            deg_pct_per_lap=baseline.deg_pct_per_lap,
            tyre_compound=baseline.tyre_compound,
            tyre_age=baseline.tyre_age,
        )
        lap_results = solve_lap(segments, trial)
        sim_time = sum(r.t_segment for r in lap_results)
        if sim_time > target_lap_s:
            lo_factor = mid_factor
        else:
            hi_factor = mid_factor

    final_factor = (lo_factor + hi_factor) / 2.0
    return CarModel(
        name=name,
        a_accel_low=baseline.a_accel_low,
        a_accel_high=baseline.a_accel_high,
        a_brake=baseline.a_brake,
        v_max=baseline.v_max * final_factor,
        deg_pct_per_lap=baseline.deg_pct_per_lap,
        tyre_compound=baseline.tyre_compound,
        tyre_age=baseline.tyre_age,
    )


_MAX_PARAM_DRIFT = 0.30  # guard-rail: ±30% from initial estimate


def calibrate_car_params(
    initial: CarModel,
    segments: list[dict],
    target_lap_s: float,
    max_drift: float = _MAX_PARAM_DRIFT,
    rounds: int = 3,
    verbose: bool = True,
) -> CarModel:
    """Calibrate a_accel_low, a_accel_high, a_brake against a target lap time.

    v_max is kept fixed (hard physical constraint from observed top speed).
    Each parameter is searched independently via binary search within
    ±max_drift of the initial estimate, iterated for *rounds* cycles.

    Returns a new CarModel with calibrated effective/lumped parameters.
    """
    from .lap_solver import solve_lap

    def _sim_lap(a_low: float, a_high: float, a_brk: float) -> float:
        trial = CarModel(
            name=initial.name,
            a_accel_low=a_low,
            a_accel_high=a_high,
            a_brake=a_brk,
            v_max=initial.v_max,
        )
        return sum(r.t_segment for r in solve_lap(segments, trial))

    a_low = initial.a_accel_low
    a_high = initial.a_accel_high
    a_brk = initial.a_brake

    bounds = {
        "a_low": (a_low * (1 - max_drift), a_low * (1 + max_drift)),
        "a_high": (a_high * (1 - max_drift), a_high * (1 + max_drift)),
        "a_brake": (a_brk * (1 - max_drift), a_brk * (1 + max_drift)),
    }

    if verbose:
        init_lap = _sim_lap(a_low, a_high, a_brk)
        print(f"Calibration target: {target_lap_s:.3f}s")
        print(f"Initial estimate:   {init_lap:.3f}s  (error {init_lap - target_lap_s:+.3f}s)")
        print(f"Guard-rails: ±{max_drift*100:.0f}% from initial")

    # Phase 1: uniform scaling of all three params to get close to target
    lo_s, hi_s = 1.0 - max_drift, 1.0 + max_drift
    for _ in range(60):
        mid_s = (lo_s + hi_s) / 2.0
        t = _sim_lap(
            initial.a_accel_low * mid_s,
            initial.a_accel_high * mid_s,
            initial.a_brake * mid_s,
        )
        if t > target_lap_s:
            lo_s = mid_s
        else:
            hi_s = mid_s
    scale = (lo_s + hi_s) / 2.0
    a_low = initial.a_accel_low * scale
    a_high = initial.a_accel_high * scale
    a_brk = initial.a_brake * scale
    best_lap = _sim_lap(a_low, a_high, a_brk)
    best = (a_low, a_high, a_brk, best_lap)

    if verbose:
        print(f"  Phase 1 (uniform scale {scale:.4f}): {best_lap:.3f}s  "
              f"(error {best_lap - target_lap_s:+.3f}s)")

    # Phase 2: coordinate descent fine-tuning — only keep if it improves
    for rnd in range(rounds):
        prev_a = (a_low, a_high, a_brk)

        lo, hi = max(bounds["a_low"][0], a_low * 0.95), min(bounds["a_low"][1], a_low * 1.05)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if _sim_lap(mid, a_high, a_brk) > target_lap_s:
                lo = mid
            else:
                hi = mid
        a_low = (lo + hi) / 2.0

        lo, hi = max(bounds["a_high"][0], a_high * 0.95), min(bounds["a_high"][1], a_high * 1.05)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if _sim_lap(a_low, mid, a_brk) > target_lap_s:
                lo = mid
            else:
                hi = mid
        a_high = (lo + hi) / 2.0

        lo, hi = max(bounds["a_brake"][0], a_brk * 0.95), min(bounds["a_brake"][1], a_brk * 1.05)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if _sim_lap(a_low, a_high, mid) > target_lap_s:
                lo = mid
            else:
                hi = mid
        a_brk = (lo + hi) / 2.0

        cur_lap = _sim_lap(a_low, a_high, a_brk)
        if abs(cur_lap - target_lap_s) < abs(best[3] - target_lap_s):
            best = (a_low, a_high, a_brk, cur_lap)

        if verbose:
            print(f"  Phase 2 round {rnd+1}: {cur_lap:.3f}s  "
                  f"(error {cur_lap - target_lap_s:+.3f}s)")

        if abs(cur_lap - _sim_lap(*prev_a)) < 0.001:
            break

    a_low, a_high, a_brk, _ = best

    result = CarModel(
        name=initial.name,
        a_accel_low=a_low,
        a_accel_high=a_high,
        a_brake=a_brk,
        v_max=initial.v_max,
    )

    if verbose:
        final_lap = _sim_lap(a_low, a_high, a_brk)
        print(f"\nCalibration result ({initial.name}):")
        print(f"  {'Param':<14} {'Initial':>8} {'Calibrated':>11} {'Drift':>8}")
        print(f"  {'-'*45}")
        for pname, init_v, cal_v in [
            ("a_accel_low", initial.a_accel_low, a_low),
            ("a_accel_high", initial.a_accel_high, a_high),
            ("a_brake", initial.a_brake, a_brk),
            ("v_max", initial.v_max, initial.v_max),
        ]:
            drift = (cal_v / init_v - 1) * 100 if init_v else 0
            print(f"  {pname:<14} {init_v:8.2f} {cal_v:11.2f} {drift:+7.1f}%")
        print(f"  Sim lap: {final_lap:.3f}s  (target {target_lap_s:.3f}s)")

    return result
