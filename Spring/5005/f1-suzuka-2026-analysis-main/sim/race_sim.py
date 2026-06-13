"""Multi-lap race simulator with tyre degradation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from .car_model import CarModel
from .lap_solver import solve_lap


@dataclass
class PitStrategy:
    """Defines when to pit and what compound to switch to.

    Each entry in *stops* is (pit_lap, new_compound, new_deg_pct_per_lap).
    On the pit_lap itself, pit_delta seconds are added to the lap time.
    Tyre age resets to 0 on the lap after the stop.
    """
    stops: list[tuple[int, str, float]] = field(default_factory=list)
    pit_delta_s: float = 22.0


def simulate_race(
    segments: list[dict],
    car: CarModel,
    n_laps: int,
    strategy: PitStrategy | None = None,
) -> pd.DataFrame:
    """Run *n_laps* for a single driver and return a race_log DataFrame.

    Tyre degradation is applied each lap: all corner v_ref values are
    multiplied by (1 - deg_pct_per_lap * tyre_age).
    """
    records: list[dict] = []
    t_race_cumul = 0.0

    working_segments = copy.deepcopy(segments)
    base_v_refs = [s["v_ref"] for s in segments]

    current_compound = car.tyre_compound
    current_deg = car.deg_pct_per_lap
    tyre_age_offset = car.tyre_age

    pit_laps: set[int] = set()
    if strategy:
        for stop_lap, new_compound, new_deg in strategy.stops:
            pit_laps.add(stop_lap)

    for lap in range(1, n_laps + 1):
        # Check for pit stop at end of previous lap
        if strategy and (lap - 1) in pit_laps:
            for stop_lap, new_compound, new_deg in strategy.stops:
                if stop_lap == lap - 1:
                    current_compound = new_compound
                    current_deg = new_deg
                    tyre_age_offset = -(lap - 1)
                    break

        tyre_age = (lap - 1) + tyre_age_offset

        for i, seg in enumerate(working_segments):
            if seg["seg_type"] != "Straight":
                factor = 1.0 - current_deg * tyre_age
                seg["v_ref"] = base_v_refs[i] * max(factor, 0.5)

        lap_results = solve_lap(working_segments, car)
        t_lap_cumul = 0.0

        for res in lap_results:
            t_lap_cumul += res.t_segment
            t_race_cumul += res.t_segment

            records.append({
                "driver": car.name,
                "lap": lap,
                "seg_id": res.seg_id,
                "seg_type": res.seg_type,
                "d_start": res.d_start,
                "d_end": res.d_end,
                "v_entry": res.v_entry,
                "v_exit": res.v_exit,
                "v_min": res.v_min,
                "t_segment": res.t_segment,
                "t_lap_cumul": t_lap_cumul,
                "t_race_cumul": t_race_cumul,
                "tyre_compound": current_compound,
                "tyre_age": tyre_age,
                "v_apex_degraded": (
                    working_segments[res.seg_id]["v_ref"]
                    if working_segments[res.seg_id]["seg_type"] != "Straight"
                    else None
                ),
                "interaction": None,
                "interaction_delta_s": 0.0,
            })

        # Add pit stop time penalty on the pit lap
        if lap in pit_laps:
            t_race_cumul += strategy.pit_delta_s
            records[-1]["t_race_cumul"] = t_race_cumul

    return pd.DataFrame(records)


def simulate_race_multi(
    segments: list[dict],
    cars: list[tuple[CarModel, PitStrategy | None]],
    n_laps: int,
) -> pd.DataFrame:
    """Run independent simulations for multiple drivers and concatenate results."""
    frames: list[pd.DataFrame] = []
    for car, strat in cars:
        df = simulate_race(segments, car, n_laps, strategy=strat)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def simulate_from_fork(
    segments: list[dict],
    car: CarModel,
    start_lap: int,
    end_lap: int,
    t_race_cumul_init: float = 0.0,
) -> pd.DataFrame:
    """Simulate from a fork point with custom initial conditions.

    Lap numbers in the output use real race lap numbers (start_lap..end_lap).
    The car's tyre_compound, tyre_age, and deg_pct_per_lap should be set
    to reflect the state at the fork point.
    """
    n_laps = end_lap - start_lap + 1
    records: list[dict] = []
    t_race_cumul = t_race_cumul_init

    working_segments = copy.deepcopy(segments)
    base_v_refs = [s["v_ref"] for s in segments]

    for i in range(n_laps):
        real_lap = start_lap + i
        tyre_age = car.tyre_age + i

        for j, seg in enumerate(working_segments):
            if seg["seg_type"] != "Straight":
                factor = 1.0 - car.deg_pct_per_lap * tyre_age
                seg["v_ref"] = base_v_refs[j] * max(factor, 0.5)

        lap_results = solve_lap(working_segments, car)
        t_lap_cumul = 0.0

        for res in lap_results:
            t_lap_cumul += res.t_segment
            t_race_cumul += res.t_segment

            records.append({
                "driver": car.name,
                "lap": real_lap,
                "seg_id": res.seg_id,
                "seg_type": res.seg_type,
                "d_start": res.d_start,
                "d_end": res.d_end,
                "v_entry": res.v_entry,
                "v_exit": res.v_exit,
                "v_min": res.v_min,
                "t_segment": res.t_segment,
                "t_lap_cumul": t_lap_cumul,
                "t_race_cumul": t_race_cumul,
                "tyre_compound": car.tyre_compound,
                "tyre_age": tyre_age,
                "v_apex_degraded": (
                    working_segments[res.seg_id]["v_ref"]
                    if working_segments[res.seg_id]["seg_type"] != "Straight"
                    else None
                ),
                "interaction": None,
                "interaction_delta_s": 0.0,
            })

    return pd.DataFrame(records)
