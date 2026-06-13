import json
import os
from typing import Dict, List

import fastf1
import numpy as np
import pandas as pd


YEAR = 2026
RACE = "Japan"
SESSION = "Race"
CACHE_DIR = "fastf1_cache"
RAW_DIR = "raw_data"
STRUCTURED_DIR = "structured_data"

# 用户指定六位车手
DRIVERS: Dict[str, str] = {
    "ANT": "Antonelli",
    "RUS": "Russell",
    "PIA": "Piastri",
    "NOR": "Norris",
    "LEC": "Leclerc",
    "HAM": "Hamilton",
}


def ensure_dirs() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(STRUCTURED_DIR, exist_ok=True)


def format_timedelta(td) -> str:
    if pd.isnull(td):
        return np.nan
    try:
        td = pd.to_timedelta(td)
        total_seconds = td.total_seconds()
        hours = int((total_seconds % 86400) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    except Exception:
        return str(td)


def to_seconds(td_val) -> float:
    try:
        return pd.to_timedelta(td_val).total_seconds()
    except Exception:
        return np.nan


def fetch_raw_for_driver(session, driver_code: str) -> Dict[str, str]:
    laps = session.laps.pick_driver(driver_code)
    if laps.empty:
        raise ValueError(f"{driver_code} has no laps in this session")

    telemetry_frames: List[pd.DataFrame] = []
    summary_records: List[dict] = []

    for _, lap in laps.iterlaps():
        lap_number = int(lap.get("LapNumber", -1))
        if lap_number <= 0:
            continue

        lap_tel = lap.get_telemetry()
        if lap_tel is None or lap_tel.empty:
            continue

        lap_tel = lap_tel.copy()
        lap_tel["LapNumber"] = lap_number
        lap_tel["Driver"] = driver_code
        telemetry_frames.append(lap_tel)

        weather = lap.get_weather_data()
        summary_records.append(
            {
                "LapNumber": lap_number,
                "LapStartTime": str(lap.get("LapStartTime", "")),
                "LapTime": str(lap.get("LapTime", "")),
                "Compound": str(lap.get("Compound", "")),
                "TyreLife": float(lap.get("TyreLife", 0.0))
                if pd.notnull(lap.get("TyreLife", np.nan))
                else np.nan,
                "Sector1Time": str(lap.get("Sector1Time", "")),
                "Sector2Time": str(lap.get("Sector2Time", "")),
                "Sector3Time": str(lap.get("Sector3Time", "")),
                "TrackStatus": str(lap.get("TrackStatus", "")),
                "Weather": {
                    "AirTemp": float(weather.get("AirTemp", np.nan))
                    if pd.notnull(weather.get("AirTemp", np.nan))
                    else np.nan,
                    "TrackTemp": float(weather.get("TrackTemp", np.nan))
                    if pd.notnull(weather.get("TrackTemp", np.nan))
                    else np.nan,
                    "Humidity": float(weather.get("Humidity", np.nan))
                    if pd.notnull(weather.get("Humidity", np.nan))
                    else np.nan,
                    "Pressure": float(weather.get("Pressure", np.nan))
                    if pd.notnull(weather.get("Pressure", np.nan))
                    else np.nan,
                    "WindSpeed": float(weather.get("WindSpeed", np.nan))
                    if pd.notnull(weather.get("WindSpeed", np.nan))
                    else np.nan,
                    "WindDirection": float(weather.get("WindDirection", np.nan))
                    if pd.notnull(weather.get("WindDirection", np.nan))
                    else np.nan,
                    "Rainfall": bool(weather.get("Rainfall", False)),
                },
            }
        )

    if not telemetry_frames:
        raise ValueError(f"{driver_code} has no telemetry laps to export")

    telemetry_df = pd.concat(telemetry_frames, ignore_index=True)
    summary_records = sorted(summary_records, key=lambda x: x["LapNumber"])

    telemetry_path = os.path.join(RAW_DIR, f"{driver_code}_Japan_2026_telemetry.csv")
    summary_path = os.path.join(RAW_DIR, f"{driver_code}_Japan_2026_lap_summary.json")
    telemetry_df.to_csv(telemetry_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    return {"telemetry": telemetry_path, "summary": summary_path}


def structure_driver_data(driver_code: str, telemetry_path: str, summary_path: str) -> Dict[str, str]:
    df = pd.read_csv(telemetry_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_records = json.load(f)

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    cols_to_keep = [
        "SessionTime",
        "Time",
        "RPM",
        "Speed",
        "nGear",
        "Throttle",
        "Brake",
        "DRS",
        "Source",
        "Distance",
        "RelativeDistance",
        "Status",
        "X",
        "Y",
        "Z",
        "LapNumber",
        "Driver",
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    missing_stats = df.isnull().sum().reset_index()
    missing_stats.columns = ["Column", "Missing_Count"]
    missing_stats["Missing_Proportion"] = missing_stats["Missing_Count"] / len(df)
    missing_stats["Type"] = "Missing"
    missing_stats["Recommendation"] = np.where(
        missing_stats["Missing_Count"] > 0,
        "Forward fill or interpolate based on time",
        "No action needed",
    )

    numerical_cols = ["RPM", "Speed", "nGear", "Throttle", "Brake"]
    outlier_records = []
    bounds = {
        "RPM": (0, 15000),
        "Speed": (0, 400),
        "nGear": (0, 8),
        "Throttle": (0, 100),
        "Brake": (0, 1),
    }
    for col in numerical_cols:
        if col not in df.columns:
            continue
        lo, hi = bounds[col]
        outliers = df[(df[col] < lo) | (df[col] > hi)]
        count = len(outliers)
        outlier_records.append(
            {
                "Column": col,
                "Outlier_Count": count,
                "Type": "Outlier",
                "Recommendation": "Cap to logical min/max or interpolate"
                if count > 0
                else "No action needed",
            }
        )

    outlier_df = pd.DataFrame(outlier_records)
    report_df = pd.concat([missing_stats, outlier_df], ignore_index=True)

    report_path = os.path.join(
        STRUCTURED_DIR, f"{driver_code}_missing_and_outliers_report.csv"
    )
    report_df.to_csv(report_path, index=False)

    df = df.ffill().bfill()
    if "RPM" in df.columns:
        df["RPM"] = df["RPM"].clip(0, 15000)
    if "Speed" in df.columns:
        df["Speed"] = df["Speed"].clip(0, 400)
    if "nGear" in df.columns:
        df["nGear"] = df["nGear"].clip(0, 8)
    if "Throttle" in df.columns:
        df["Throttle"] = df["Throttle"].clip(0, 100)
    if "Brake" in df.columns:
        df["Brake"] = df["Brake"].clip(0, 1)

    df["SessionTime_raw"] = pd.to_timedelta(df["SessionTime"], errors="coerce")
    df["Time_raw"] = pd.to_timedelta(df["Time"], errors="coerce")
    df["SessionTime"] = df["SessionTime_raw"].apply(format_timedelta)
    df["Time"] = df["Time_raw"].apply(format_timedelta)

    cleaned_path = os.path.join(STRUCTURED_DIR, f"{driver_code}_cleaned_telemetry.csv")
    df.drop(columns=["SessionTime_raw", "Time_raw"], errors="ignore").to_csv(
        cleaned_path, index=False
    )

    summary_df = pd.DataFrame(summary_records)
    if summary_df.empty:
        raise ValueError(f"{driver_code} summary is empty")

    weather_df = pd.json_normalize(summary_df["Weather"]).add_prefix("Weather_")
    summary_df = pd.concat([summary_df.drop(columns=["Weather"]), weather_df], axis=1)
    summary_df["LapNumber"] = pd.to_numeric(summary_df["LapNumber"], errors="coerce").astype(
        "Int64"
    )
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").astype("Int64")

    summary_cols = [
        "LapNumber",
        "Compound",
        "TyreLife",
        "TrackStatus",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ] + [c for c in summary_df.columns if c.startswith("Weather_")]
    summary_for_merge = summary_df[summary_cols].copy()
    df = df.merge(summary_for_merge, on="LapNumber", how="left")

    # 基于每圈 Time 列与扇区时间累加来做扇区映射
    s1_seconds = (
        summary_df.set_index("LapNumber")["Sector1Time"].apply(to_seconds).to_dict()
    )
    s2_seconds = (
        summary_df.set_index("LapNumber")["Sector2Time"].apply(to_seconds).to_dict()
    )

    time_in_lap_seconds = pd.to_timedelta(df["Time"], errors="coerce").dt.total_seconds()
    sectors = []
    for lap_num, t_sec in zip(df["LapNumber"], time_in_lap_seconds):
        if pd.isna(lap_num) or pd.isna(t_sec):
            sectors.append("Unknown")
            continue
        s1 = s1_seconds.get(int(lap_num), np.nan)
        s2 = s2_seconds.get(int(lap_num), np.nan)
        if pd.isna(s1) or pd.isna(s2):
            sectors.append("Unknown")
        elif t_sec <= s1:
            sectors.append("Sector 1")
        elif t_sec <= (s1 + s2):
            sectors.append("Sector 2")
        else:
            sectors.append("Sector 3")
    df["Sector"] = sectors

    structured_path = os.path.join(STRUCTURED_DIR, f"{driver_code}_structured_data.csv")
    df.to_csv(structured_path, index=False)
    return {
        "report": report_path,
        "cleaned": cleaned_path,
        "structured": structured_path,
    }


def build_corner_info(session) -> pd.DataFrame:
    laps = session.laps
    fast_lap = laps.pick_fastest()
    telemetry = fast_lap.get_telemetry()
    circuit_info = session.get_circuit_info()

    results = []
    for _, corner in circuit_info.corners.iterrows():
        c_dist = corner["Distance"]
        c_num = corner["Number"]

        t_corner = telemetry[
            (telemetry["Distance"] >= c_dist - 150) & (telemetry["Distance"] <= c_dist + 150)
        ]

        if not t_corner.empty:
            apex_data = t_corner[
                (t_corner["Distance"] >= c_dist - 50) & (t_corner["Distance"] <= c_dist + 50)
            ]
            min_speed = apex_data["Speed"].min() if not apex_data.empty else np.nan

            if pd.isna(min_speed):
                speed_class = "Unknown"
            elif min_speed <= 120:
                speed_class = "Low"
            elif min_speed < 200:
                speed_class = "Medium"
            else:
                speed_class = "High"

            entry_dist = c_dist - 100 if speed_class != "High" else c_dist - 50
            exit_dist = c_dist + 50
        else:
            min_speed = np.nan
            speed_class = "Unknown"
            entry_dist = c_dist - 50
            exit_dist = c_dist + 50

        try:
            idx_c = telemetry["Distance"].searchsorted(c_dist)
            idx_b = max(0, idx_c - 10)
            idx_a = min(len(telemetry) - 1, idx_c + 10)
            p_b = (telemetry.iloc[idx_b]["X"], telemetry.iloc[idx_b]["Y"])
            p_c = (telemetry.iloc[idx_c]["X"], telemetry.iloc[idx_c]["Y"])
            p_a = (telemetry.iloc[idx_a]["X"], telemetry.iloc[idx_a]["Y"])
            v1 = (p_c[0] - p_b[0], p_c[1] - p_b[1])
            v2 = (p_a[0] - p_c[0], p_a[1] - p_c[1])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            direction = "Right" if cross < 0 else "Left"
        except Exception:
            direction = "Unknown"

        results.append(
            {
                "Corner_ID": c_num,
                "Apex_Distance": c_dist,
                "Direction": direction,
                "Min_Speed": min_speed,
                "Speed_Class": speed_class,
                "Entry_Distance": entry_dist,
                "Exit_Distance": exit_dist,
            }
        )

    corner_df = pd.DataFrame(results)
    corner_df.to_csv(os.path.join(STRUCTURED_DIR, "corner_info.csv"), index=False)
    return corner_df


def feature_engineering_for_driver(driver_code: str, corner_df: pd.DataFrame) -> str:
    in_path = os.path.join(STRUCTURED_DIR, f"{driver_code}_structured_data.csv")
    out_path = os.path.join(STRUCTURED_DIR, f"{driver_code}_structured_data_with_corners.csv")
    data = pd.read_csv(in_path)

    data["Corner_ID"] = 0
    data["Corner_Phase"] = "Straight"
    data["Corner_Direction"] = "Straight"
    data["Corner_SpeedClass"] = "Straight"

    for _, row in corner_df.iterrows():
        cid = row["Corner_ID"]
        entry_d = row["Entry_Distance"]
        apex_d = row["Apex_Distance"]
        exit_d = row["Exit_Distance"]
        direction = row["Direction"]
        speed_class = row["Speed_Class"]

        mask = (data["Distance"] >= entry_d) & (data["Distance"] <= exit_d)
        data.loc[mask, "Corner_ID"] = cid
        data.loc[mask, "Corner_Direction"] = direction
        data.loc[mask, "Corner_SpeedClass"] = speed_class

        mask_entry = mask & (data["Distance"] <= apex_d)
        mask_exit = mask & (data["Distance"] > apex_d)
        data.loc[mask_entry, "Corner_Phase"] = "Entry"
        data.loc[mask_exit, "Corner_Phase"] = "Exit"

    data.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ensure_dirs()
    fastf1.Cache.enable_cache(CACHE_DIR)

    print(f"Loading session: {YEAR} {RACE} {SESSION}")
    session = fastf1.get_session(YEAR, RACE, SESSION)
    session.load(telemetry=True, weather=True)

    final_files = []
    corner_df = build_corner_info(session)
    print("Saved shared corner info:", os.path.join(STRUCTURED_DIR, "corner_info.csv"))

    for code, name in DRIVERS.items():
        print(f"\n=== Processing {name} ({code}) ===")
        raw_paths = fetch_raw_for_driver(session, code)
        print("Raw telemetry:", raw_paths["telemetry"])
        print("Raw lap summary:", raw_paths["summary"])

        structured_paths = structure_driver_data(
            code, raw_paths["telemetry"], raw_paths["summary"]
        )
        print("Structured file:", structured_paths["structured"])

        final_path = feature_engineering_for_driver(code, corner_df)
        final_files.append(final_path)
        print("Final file:", final_path)

    print("\nCompleted. Six final CSV files:")
    for f in final_files:
        print("-", f)


if __name__ == "__main__":
    main()
