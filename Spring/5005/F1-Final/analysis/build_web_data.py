from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = ROOT / "structured_data"
WEB_PUBLIC = ROOT / "web" / "public"
DATA_DIR = WEB_PUBLIC / "data"
ASSET_DIR = WEB_PUBLIC / "assets"
TRACK_IMAGE = ROOT / "suzuka_straight_mode_zone.png"

TRACK_LENGTH = 5807.0
TOTAL_LAPS = 53
SC_LAPS = set(range(22, 29))
SPEED_BINS = list(range(0, 361, 20))
DRIVER_ORDER = ["ANT", "RUS", "PIA", "NOR", "LEC", "HAM"]

DRIVERS: dict[str, dict[str, str]] = {
    "ANT": {"name": "Kimi Antonelli", "team": "Mercedes", "number": "12", "color": "#00d2be"},
    "RUS": {"name": "George Russell", "team": "Mercedes", "number": "63", "color": "#6ee7ff"},
    "PIA": {"name": "Oscar Piastri", "team": "McLaren", "number": "81", "color": "#ff8700"},
    "NOR": {"name": "Lando Norris", "team": "McLaren", "number": "4", "color": "#ffb000"},
    "LEC": {"name": "Charles Leclerc", "team": "Ferrari", "number": "16", "color": "#dc0000"},
    "HAM": {"name": "Lewis Hamilton", "team": "Ferrari", "number": "44", "color": "#ff4d4d"},
}

PIT_LAPS = {"NOR": 16, "LEC": 17, "PIA": 18, "RUS": 21, "ANT": 22, "HAM": 22}


@dataclass
class ScenarioDriver:
    driver: str
    start_lap: int
    start_gap_s: float
    start_offset_m: float
    compound: str
    tyre_life: float
    must_pit_lap: int | None
    pit_loss_s: float


def td_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def clean_float(value: Any, digits: int = 3) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), digits)


def read_driver_data() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    required = [
        "SessionTime",
        "Time",
        "Speed",
        "Distance",
        "X",
        "Y",
        "LapNumber",
        "Driver",
        "Compound",
        "TyreLife",
        "TrackStatus",
        "Corner_ID",
        "Corner_Phase",
        "Corner_Direction",
        "Corner_SpeedClass",
    ]
    for driver in DRIVER_ORDER:
        path = STRUCTURED_DIR / f"{driver}_structured_data_with_corners.csv"
        df = pd.read_csv(path, usecols=lambda c: c in required)
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").astype("Int64")
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").clip(lower=0, upper=TRACK_LENGTH)
        df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
        df["SessionSeconds"] = td_seconds(df["SessionTime"])
        df["LapSeconds"] = td_seconds(df["Time"])
        df = df.dropna(subset=["LapNumber", "Speed", "Distance", "SessionSeconds"])
        df["LapNumber"] = df["LapNumber"].astype(int)
        df["TrackStatus"] = df["TrackStatus"].astype(str)
        df = df.sort_values(["SessionSeconds", "Distance"]).reset_index(drop=True)
        frames[driver] = add_acceleration(df)
    return frames


def add_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SpeedMS"] = df["Speed"] / 3.6
    dt = df.groupby("LapNumber")["SessionSeconds"].diff()
    dv = df.groupby("LapNumber")["SpeedMS"].diff()
    acc = (dv / dt.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["Acceleration"] = acc.clip(-8.0, 8.0)
    return df


def build_corner_info(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample = pd.concat(frames.values(), ignore_index=True)
    cornered = sample[sample["Corner_ID"].fillna(0).astype(int) > 0].copy()
    for corner_id, group in cornered.groupby("Corner_ID"):
        rows.append(
            {
                "id": int(corner_id),
                "distance": clean_float(group["Distance"].median(), 2),
                "direction": str(group["Corner_Direction"].mode().iat[0]) if not group.empty else "Unknown",
                "speedClass": str(group["Corner_SpeedClass"].mode().iat[0]),
                "minSpeed": clean_float(group["Speed"].quantile(0.05), 1),
            }
        )
    return sorted(rows, key=lambda item: item["id"])


def acceleration_curves(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    valid = df.dropna(subset=["Acceleration", "Speed"])
    valid = valid[(valid["LapNumber"] > 1) & (~valid["LapNumber"].isin(SC_LAPS))]
    valid["SpeedBin"] = pd.cut(valid["Speed"], bins=SPEED_BINS, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for interval, group in valid.groupby("SpeedBin", observed=True):
        center = (float(interval.left) + float(interval.right)) / 2
        accel = group[group["Acceleration"] > 0.15]["Acceleration"]
        decel = group[group["Acceleration"] < -0.15]["Acceleration"]
        rows.append(
            {
                "speed": round(center, 1),
                "accel": clean_float(accel.quantile(0.8), 3),
                "decel": clean_float(decel.quantile(0.2), 3),
            }
        )
    return {"bins": rows}


def corner_minima(df: pd.DataFrame) -> pd.DataFrame:
    work = df[(df["Corner_ID"].fillna(0).astype(int) > 0) & (~df["LapNumber"].isin(SC_LAPS))].copy()
    grouped = (
        work.groupby(["Driver", "LapNumber", "Corner_ID", "Corner_SpeedClass"], dropna=False)["Speed"]
        .min()
        .reset_index(name="minSpeed")
    )
    return grouped


def tire_index(corner_df: pd.DataFrame, telemetry: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    lap_meta = (
        telemetry.groupby(["LapNumber", "Compound"], dropna=False)["TyreLife"]
        .median()
        .reset_index()
        .dropna(subset=["Compound", "TyreLife"])
    )
    enriched = corner_df.merge(lap_meta, on="LapNumber", how="left")
    out: dict[str, list[dict[str, Any]]] = {}
    for compound, group in enriched.groupby("Compound"):
        if compound not in {"MEDIUM", "HARD"}:
            continue
        by_life = group.groupby("TyreLife")["minSpeed"].mean().sort_index()
        if by_life.empty:
            continue
        baseline = by_life.iloc[0] or by_life.mean()
        observed = (by_life / baseline).clip(0.94, 1.05)
        smoothed = observed.rolling(3, center=True, min_periods=1).mean()
        out[str(compound)] = [
            {"tyreLife": int(life), "index": clean_float(value, 4)}
            for life, value in smoothed.items()
            if pd.notna(life)
        ]
    return out


def performance_metrics(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    raw: dict[str, dict[str, float]] = {}
    per_driver: dict[str, Any] = {}
    all_corner_rows = []
    for driver, df in frames.items():
        normal = df[(df["LapNumber"] > 1) & (~df["LapNumber"].isin(SC_LAPS))]
        corner_df = corner_minima(df)
        all_corner_rows.append(corner_df)
        corner_class = corner_df.groupby("Corner_SpeedClass")["minSpeed"].mean()
        metrics = {
            "maxAcceleration": normal["Acceleration"].quantile(0.98),
            "maxDeceleration": abs(normal["Acceleration"].quantile(0.02)),
            "lowCorner": corner_class.get("Low", np.nan),
            "mediumCorner": corner_class.get("Medium", np.nan),
            "highCorner": corner_class.get("High", np.nan),
            "racePace": normal.groupby("LapNumber")["LapSeconds"].max().median(),
        }
        raw[driver] = {k: float(v) for k, v in metrics.items() if pd.notna(v)}
        per_driver[driver] = {
            "driver": driver,
            "curves": acceleration_curves(df),
            "cornerMinima": corner_df.head(220).to_dict(orient="records"),
        }

    ant = raw["ANT"]
    breakdown = []
    for driver in DRIVER_ORDER:
        values = raw[driver]
        rel = {
            key: clean_float((values[key] / ant[key]) if key != "racePace" else (ant[key] / values[key]), 4)
            for key in values
            if key in ant and ant[key]
        }
        breakdown.append({"driver": driver, "absolute": {k: clean_float(v, 3) for k, v in values.items()}, "relativeToANT": rel})

    combined_corner = pd.concat(all_corner_rows, ignore_index=True)
    tires = {
        driver: tire_index(corner_minima(frames[driver]), frames[driver])
        for driver in DRIVER_ORDER
    }
    return {
        "drivers": per_driver,
        "breakdown": breakdown,
        "tireIndex": tires,
        "cornerSummary": combined_corner.groupby(["Driver", "Corner_SpeedClass"])["minSpeed"].mean().reset_index().to_dict(orient="records"),
    }


def build_track_path(frames: dict[str, pd.DataFrame]) -> list[dict[str, float]]:
    pia = frames["PIA"]
    lap = pia[(pia["LapNumber"] == 2) & (pia["Distance"].between(1, TRACK_LENGTH))].copy()
    if lap.empty:
        lap = pia.copy()
    lap = lap.sort_values("Distance").drop_duplicates("Distance")
    lap = lap.iloc[:: max(1, len(lap) // 360)]
    x_min, x_max = lap["X"].min(), lap["X"].max()
    y_min, y_max = lap["Y"].min(), lap["Y"].max()
    return [
        {
            "distance": clean_float(row.Distance, 2),
            "x": clean_float((row.X - x_min) / (x_max - x_min), 5),
            "y": clean_float(1 - ((row.Y - y_min) / (y_max - y_min)), 5),
        }
        for row in lap.itertuples()
    ]


def actual_playback(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    min_session = min(df["SessionSeconds"].min() for df in frames.values())
    drivers_payload = {}
    for driver, df in frames.items():
        work = df[df["LapNumber"].between(1, TOTAL_LAPS)].copy()
        work = work.iloc[::4]
        work["t"] = work["SessionSeconds"] - min_session
        work["totalDistance"] = (work["LapNumber"] - 1) * TRACK_LENGTH + work["Distance"]
        drivers_payload[driver] = [
            {
                "t": clean_float(row.t, 2),
                "lap": int(row.LapNumber),
                "distance": clean_float(row.Distance, 1),
                "totalDistance": clean_float(row.totalDistance, 1),
                "speed": clean_float(row.Speed, 1),
                "accel": clean_float(row.Acceleration, 3),
                "compound": str(row.Compound),
                "tyreLife": clean_float(row.TyreLife, 1),
            }
            for row in work.itertuples()
        ]
    return {"id": "actual", "label": "实际比赛", "startLap": 1, "drivers": drivers_payload}


def median_lap_times(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    lap_times: dict[str, float] = {}
    for driver, df in frames.items():
        normal = df[(df["LapNumber"] > 2) & (~df["LapNumber"].isin(SC_LAPS))]
        times = normal.groupby("LapNumber")["LapSeconds"].max()
        lap_times[driver] = float(times.median())
    return lap_times


def simulate_scenario(
    scenario_id: str,
    label: str,
    start_lap: int,
    configs: list[ScenarioDriver],
    lap_times: dict[str, float],
    tire_adjustment: dict[str, float] | None = None,
) -> dict[str, Any]:
    tire_adjustment = tire_adjustment or {}
    drivers_payload: dict[str, list[dict[str, Any]]] = {}
    finish_times: dict[str, float] = {}
    total_target = TOTAL_LAPS * TRACK_LENGTH

    for cfg in configs:
        base_lap_time = lap_times[cfg.driver]
        t = cfg.start_gap_s
        lap = cfg.start_lap
        distance_total = (start_lap - 1) * TRACK_LENGTH + cfg.start_offset_m
        compound = cfg.compound
        tyre_life = cfg.tyre_life
        points: list[dict[str, Any]] = []
        pit_done = cfg.must_pit_lap is None

        while distance_total < total_target and t < 4000:
            current_lap = max(start_lap, int(distance_total // TRACK_LENGTH) + 1)
            life_factor = tire_life_factor(compound, tyre_life, tire_adjustment.get(cfg.driver, 0.0))
            speed_mps = TRACK_LENGTH / (base_lap_time / max(0.9, life_factor))
            if cfg.must_pit_lap and not pit_done and current_lap > cfg.must_pit_lap:
                t += cfg.pit_loss_s
                compound = "HARD"
                tyre_life = 1.0
                pit_done = True
            points.append(
                {
                    "t": clean_float(t, 2),
                    "lap": current_lap,
                    "distance": clean_float(distance_total % TRACK_LENGTH, 1),
                    "totalDistance": clean_float(distance_total, 1),
                    "speed": clean_float(speed_mps * 3.6, 1),
                    "accel": 0.0,
                    "compound": compound,
                    "tyreLife": clean_float(tyre_life, 1),
                }
            )
            t += 1.0
            distance_total += speed_mps
            if int(distance_total // TRACK_LENGTH) + 1 > lap:
                lap += 1
                tyre_life += 1

        finish_times[cfg.driver] = t
        drivers_payload[cfg.driver] = points

    classification = sorted(finish_times.items(), key=lambda item: item[1])
    return {
        "id": scenario_id,
        "label": label,
        "startLap": start_lap,
        "drivers": drivers_payload,
        "result": [
            {"position": idx + 1, "driver": driver, "finishTime": clean_float(time, 2), "gap": clean_float(time - classification[0][1], 2)}
            for idx, (driver, time) in enumerate(classification)
        ],
    }


def tire_life_factor(compound: str, tyre_life: float, adjustment: float) -> float:
    if compound == "HARD":
        if tyre_life <= 3:
            base = 0.985 + 0.005 * tyre_life
        elif tyre_life <= 20:
            base = 1.025
        else:
            base = 1.025 - min(0.08, (tyre_life - 20) * 0.004)
    else:
        if tyre_life <= 2:
            base = 1.0
        elif tyre_life <= 15:
            base = 1.03
        else:
            base = 1.03 - min(0.09, (tyre_life - 15) * 0.006)
    if base >= 1:
        return max(1.0, base + adjustment)
    return min(1.0, base + adjustment)


def build_scenarios(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    lap_times = median_lap_times(frames)
    scenario_one = simulate_scenario(
        "no_safety_car",
        "假设一：无安全车",
        22,
        [
            ScenarioDriver("ANT", 22, 0.0, 0.0, "MEDIUM", 22, 22, 22.0),
            ScenarioDriver("PIA", 22, 2.6, -160.0, "HARD", 4, None, 0.0),
            ScenarioDriver("RUS", 22, 4.2, -260.0, "HARD", 1, None, 0.0),
            ScenarioDriver("NOR", 22, 8.8, -540.0, "HARD", 6, None, 0.0),
            ScenarioDriver("LEC", 22, 10.2, -630.0, "HARD", 5, None, 0.0),
            ScenarioDriver("HAM", 22, 12.1, -740.0, "MEDIUM", 22, 22, 22.0),
        ],
        lap_times,
    )
    scenario_two = simulate_scenario(
        "late_piastri_stop",
        "假设二：PIA 晚进站",
        28,
        [
            ScenarioDriver("PIA", 28, 0.0, 120.0, "HARD", 5, None, 0.0),
            ScenarioDriver("ANT", 28, 1.5, 0.0, "HARD", 5, None, 0.0),
            ScenarioDriver("RUS", 28, 4.8, -260.0, "HARD", 7, None, 0.0),
            ScenarioDriver("NOR", 28, 8.6, -470.0, "HARD", 12, None, 0.0),
            ScenarioDriver("LEC", 28, 9.8, -540.0, "HARD", 11, None, 0.0),
            ScenarioDriver("HAM", 28, 12.0, -660.0, "HARD", 5, None, 0.0),
        ],
        lap_times,
    )
    return [scenario_one, scenario_two]


def sensitivity_summary(lap_times: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for pia_gap_m in [0, 60, 120, 180]:
        for adjustment in [-0.01, 0.0, 0.01]:
            scenario = simulate_scenario(
                "sensitivity",
                "敏感性分析",
                28,
                [
                    ScenarioDriver("PIA", 28, 0.0, pia_gap_m, "HARD", 5, None, 0.0),
                    ScenarioDriver("ANT", 28, 1.5, 0.0, "HARD", 5, None, 0.0),
                ],
                lap_times,
                {"PIA": adjustment},
            )
            rows.append(
                {
                    "piaInitialGapM": pia_gap_m,
                    "piaTireAdjustment": adjustment,
                    "winner": scenario["result"][0]["driver"],
                    "gap": scenario["result"][1]["gap"],
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if TRACK_IMAGE.exists():
        shutil.copyfile(TRACK_IMAGE, ASSET_DIR / TRACK_IMAGE.name)

    frames = read_driver_data()
    metrics = performance_metrics(frames)
    lap_times = median_lap_times(frames)
    track = {
        "length": TRACK_LENGTH,
        "laps": TOTAL_LAPS,
        "raceDistanceKm": round(TRACK_LENGTH * TOTAL_LAPS / 1000, 2),
        "cornerCount": 18,
        "path": build_track_path(frames),
        "corners": build_corner_info(frames),
    }
    playback = {
        "drivers": DRIVERS,
        "pitLaps": PIT_LAPS,
        "track": track,
        "scLaps": sorted(SC_LAPS),
        "scenarios": [actual_playback(frames), *build_scenarios(frames)],
        "sensitivity": sensitivity_summary(lap_times),
    }
    write_json(DATA_DIR / "telemetry_playback.json", playback)
    write_json(DATA_DIR / "performance_metrics.json", metrics)
    write_json(
        DATA_DIR / "story_content.json",
        {
            "title": "Will Piastri Win?",
            "subtitle": "2026 日本大奖赛皮亚斯特里夺冠虚拟分析",
            "timeline": [
                {"lap": "0-18", "title": "PIA 领跑", "body": "PIA 为防止梅奔双车 undercut，于第 18 圈结束进站。"},
                {"lap": "18-22", "title": "ANT 清洁空气提速", "body": "ANT 在干净空气中释放长距离速度，RUS 第 21 圈进站跟进。"},
                {"lap": "22-28", "title": "安全车改变比赛", "body": "BEA 事故触发安全车，ANT 与 HAM 获得低损失进站窗口。"},
                {"lap": "28-53", "title": "ANT 巡航夺冠", "body": "安全车结束后 ANT 继续领跑，最终领先 PIA 13.7 秒。"},
            ],
        },
    )
    print(f"Wrote web data to {DATA_DIR}")


if __name__ == "__main__":
    main()
