"""Forward-backward point-mass lap solver.

Corner segments have a speed constraint (v_ref) at their midpoint (apex).
The car may enter/exit a corner faster than v_ref and brake/accelerate within
the segment halves.  Straight segments have no constraint besides v_max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .car_model import CarModel, V_ACCEL_SPLIT_MS


@dataclass
class SegmentResult:
    seg_id: int
    d_start: float
    d_end: float
    seg_type: str
    v_entry: float
    v_exit: float
    v_min: float
    t_segment: float


# ---------------------------------------------------------------------------
# Kinematic helpers
# ---------------------------------------------------------------------------

def _accel_exit_speed(v_in: float, dist: float, car: CarModel) -> float:
    """Max speed after accelerating over *dist* from *v_in* (two-band)."""
    if dist <= 0:
        return v_in
    v = v_in
    remaining = dist

    if v < V_ACCEL_SPLIT_MS:
        d_to_split = (V_ACCEL_SPLIT_MS ** 2 - v ** 2) / (2.0 * car.a_accel_low)
        if d_to_split <= remaining:
            v = V_ACCEL_SPLIT_MS
            remaining -= d_to_split
        else:
            return min(math.sqrt(v ** 2 + 2.0 * car.a_accel_low * remaining), car.v_max)

    if remaining > 0:
        v = math.sqrt(v ** 2 + 2.0 * car.a_accel_high * remaining)

    return min(v, car.v_max)


def _brake_entry_speed(v_out: float, dist: float, a_brake: float, v_cap: float) -> float:
    """Max speed at start of a zone, braking to *v_out* over *dist*."""
    if dist <= 0:
        return v_out
    return min(math.sqrt(v_out ** 2 + 2.0 * a_brake * dist), v_cap)


def _time_accel(v_start: float, v_end: float, dist: float, car: CarModel) -> float:
    """Time to cover *dist* while accelerating from v_start toward v_end.

    Handles the two-band split and a possible v_max coast phase.
    """
    if dist <= 0:
        return 0.0
    v_max = car.v_max
    total_t = 0.0
    v = v_start
    remaining = dist

    # Phase A: low-speed acceleration
    if v < V_ACCEL_SPLIT_MS:
        a = car.a_accel_low
        d_to_split = (V_ACCEL_SPLIT_MS ** 2 - v ** 2) / (2.0 * a)
        if d_to_split >= remaining:
            v_final = math.sqrt(v ** 2 + 2.0 * a * remaining)
            v_final = min(v_final, v_max)
            return remaining / ((v + v_final) / 2.0)
        total_t += (V_ACCEL_SPLIT_MS - v) / a
        v = V_ACCEL_SPLIT_MS
        remaining -= d_to_split

    # Phase B: high-speed acceleration (may reach v_max)
    if remaining > 0 and v < v_max:
        a = car.a_accel_high
        d_to_vmax = (v_max ** 2 - v ** 2) / (2.0 * a)
        if d_to_vmax >= remaining:
            v_final = math.sqrt(v ** 2 + 2.0 * a * remaining)
            total_t += (v_final - v) / a
            return total_t
        total_t += (v_max - v) / a
        v = v_max
        remaining -= d_to_vmax

    # Phase C: coast at v_max
    if remaining > 0:
        total_t += remaining / v_max

    return total_t


def _time_brake(v_start: float, v_end: float, dist: float, a_brake: float) -> float:
    """Time to cover *dist* while braking from v_start to v_end."""
    if dist <= 0 or v_start <= v_end:
        return dist / max((v_start + v_end) / 2.0, 0.1)
    return (v_start - v_end) / a_brake


def _time_through_segment(v_entry: float, v_exit: float, length: float,
                          car: CarModel, v_ref: float | None = None,
                          is_corner: bool = False) -> float:
    """Time for any segment.

    For straights: accel -> coast at v_max -> brake.
    For corners: brake to v_ref -> coast at v_ref -> accel from v_ref.
    """
    if length <= 0:
        return 0.0

    if is_corner and v_ref is not None:
        return _time_through_corner(v_entry, v_exit, v_ref, length, car)
    return _time_through_straight(v_entry, v_exit, length, car)


def _time_through_straight(v_entry: float, v_exit: float, length: float,
                           car: CarModel) -> float:
    """Time for a straight segment."""
    v_cap = car.v_max
    a_brk = car.a_brake

    d_accel = _dist_to_accel(v_entry, v_cap, car) if v_entry < v_cap else 0.0
    d_brake = (v_cap ** 2 - v_exit ** 2) / (2.0 * a_brk) if v_cap > v_exit else 0.0

    if d_accel + d_brake <= length:
        t_accel = _time_accel(v_entry, v_cap, d_accel, car) if d_accel > 0 else 0.0
        t_coast = (length - d_accel - d_brake) / v_cap
        t_brake = (v_cap - v_exit) / a_brk if v_cap > v_exit else 0.0
        return t_accel + t_coast + t_brake

    v_peak = _find_peak_speed(v_entry, v_exit, length, car)
    d_acc = _dist_to_accel(v_entry, v_peak, car)
    d_brk = length - d_acc
    t_acc = _time_accel(v_entry, v_peak, d_acc, car) if d_acc > 0 else 0.0
    t_brk = _time_brake(v_peak, v_exit, d_brk, a_brk) if d_brk > 0 else 0.0
    return t_acc + t_brk


def _time_through_corner(v_entry: float, v_exit: float, v_ref: float,
                         length: float, car: CarModel) -> float:
    """Time for a corner: entry -> v_ref (apex) -> exit.

    Phase 1: brake from v_entry to v_ref (or accel if v_entry < v_ref)
    Phase 2: coast at v_ref (if extra distance remains)
    Phase 3: accel from v_ref to v_exit (or brake if v_exit < v_ref)
    """
    a_brk = car.a_brake

    # Phase 1 distance + time
    if v_entry > v_ref:
        d1 = (v_entry ** 2 - v_ref ** 2) / (2.0 * a_brk)
        t1 = (v_entry - v_ref) / a_brk
    elif v_entry < v_ref:
        d1 = _dist_to_accel(v_entry, v_ref, car)
        t1 = _time_accel(v_entry, v_ref, d1, car)
    else:
        d1, t1 = 0.0, 0.0

    # Phase 3 distance + time
    if v_exit > v_ref:
        d3 = _dist_to_accel(v_ref, v_exit, car)
        t3 = _time_accel(v_ref, v_exit, d3, car)
    elif v_exit < v_ref:
        d3 = (v_ref ** 2 - v_exit ** 2) / (2.0 * a_brk)
        t3 = (v_ref - v_exit) / a_brk
    else:
        d3, t3 = 0.0, 0.0

    # Phase 2: coast at v_ref
    d2 = length - d1 - d3
    if d2 >= 0:
        t2 = d2 / v_ref if v_ref > 0.1 else 0.0
        return t1 + t2 + t3

    # Phases overlap: the car never fully reaches/leaves v_ref.
    # Fall back to harmonic-mean approximation.
    v_avg = (v_entry + v_ref + v_exit) / 3.0
    if v_avg < 0.1:
        v_avg = 0.1
    return length / v_avg


def _dist_to_accel(v_from: float, v_to: float, car: CarModel) -> float:
    """Distance needed to accelerate from v_from to v_to (two-band)."""
    if v_to <= v_from:
        return 0.0
    d = 0.0
    v = v_from
    if v < V_ACCEL_SPLIT_MS and v_to > V_ACCEL_SPLIT_MS:
        d += (V_ACCEL_SPLIT_MS ** 2 - v ** 2) / (2.0 * car.a_accel_low)
        v = V_ACCEL_SPLIT_MS
    if v_to <= V_ACCEL_SPLIT_MS:
        d += (v_to ** 2 - v ** 2) / (2.0 * car.a_accel_low)
    elif v >= V_ACCEL_SPLIT_MS:
        d += (v_to ** 2 - v ** 2) / (2.0 * car.a_accel_high)
    return d


def _find_peak_speed(v_entry: float, v_exit: float, length: float, car: CarModel) -> float:
    """Find peak speed when v_max is not reachable (binary search)."""
    lo = max(v_entry, v_exit)
    hi = car.v_max
    a_brk = car.a_brake
    for _ in range(50):
        mid = (lo + hi) / 2.0
        d_acc = _dist_to_accel(v_entry, mid, car)
        d_brk = (mid ** 2 - v_exit ** 2) / (2.0 * a_brk)
        if d_acc + d_brk < length:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_lap(segments: list[dict], car: CarModel) -> list[SegmentResult]:
    """Solve one lap using forward-backward with apex-midpoint constraints."""
    n = len(segments)

    is_corner = [s["seg_type"] != "Straight" for s in segments]
    half_len = [s["length"] / 2.0 for s in segments]

    # --- Forward pass: propagate max achievable speeds at boundaries ---
    # boundary[i] = speed at the boundary between segment i-1 and segment i
    # boundary[0] = entry to first segment = first corner's v_ref
    boundary_fwd = [0.0] * (n + 1)
    boundary_fwd[0] = segments[0]["v_ref"]

    for i in range(n):
        seg = segments[i]
        v_in = boundary_fwd[i]

        if is_corner[i]:
            # First half: entering corner, speed is capped to what braking allows
            # At midpoint (apex): speed = v_ref
            v_apex = seg["v_ref"]
            # Second half: accelerate from apex
            v_out = _accel_exit_speed(v_apex, half_len[i], car)
        else:
            # Straight: accelerate from entry
            v_out = _accel_exit_speed(v_in, seg["length"], car)

        boundary_fwd[i + 1] = v_out

    # --- Backward pass: propagate braking constraints ---
    boundary_bwd = [0.0] * (n + 1)
    boundary_bwd[n] = segments[0]["v_ref"]  # loop back

    for i in range(n - 1, -1, -1):
        seg = segments[i]
        v_out = boundary_bwd[i + 1]

        if is_corner[i]:
            v_apex = seg["v_ref"]
            # Second half exit is after accelerating from apex — already handled.
            # First half: can enter faster than v_ref, brake to apex in half_len
            v_in = _brake_entry_speed(v_apex, half_len[i], car.a_brake, car.v_max)
        else:
            # Straight: can enter at whatever speed allows braking to v_out
            v_in = _brake_entry_speed(v_out, seg["length"], car.a_brake, car.v_max)

        boundary_bwd[i] = v_in

    # --- Combine: actual boundary speeds = min(forward, backward) ---
    boundary = [min(boundary_fwd[i], boundary_bwd[i]) for i in range(n + 1)]

    # --- Compute per-segment results ---
    results: list[SegmentResult] = []
    for i in range(n):
        seg = segments[i]
        v_entry = boundary[i]
        v_exit = boundary[i + 1]

        v_min_seg = min(v_entry, v_exit)
        if is_corner[i]:
            v_min_seg = min(v_min_seg, seg["v_ref"])

        v_seg_limit = seg["v_ref"] if is_corner[i] else car.v_max
        t_seg = _time_through_segment(v_entry, v_exit, seg["length"], car,
                                      v_ref=seg["v_ref"],
                                      is_corner=is_corner[i])

        results.append(SegmentResult(
            seg_id=seg["seg_id"],
            d_start=seg["d_start"],
            d_end=seg["d_end"],
            seg_type=seg["seg_type"],
            v_entry=v_entry,
            v_exit=v_exit,
            v_min=v_min_seg,
            t_segment=t_seg,
        ))

    return results
