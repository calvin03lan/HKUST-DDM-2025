import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

# ===== 可调参数 =====
START_SEARCH_M = 4000.0
MIN_BRAKE_START_M = 4600.0
BRAKE_ON_THRESHOLD = 0.5

# 2026 赛季近似参数（可按你报告设定调整）
CAR_MASS_KG = 798.0
AIR_DENSITY = 1.18
CDA = 1.35
P_MGUK_MAX_W = 120_000.0


def smooth_accel(accel: np.ndarray) -> np.ndarray:
    n = len(accel)
    if n < 7:
        return accel
    window = min(31, n if n % 2 == 1 else n - 1)
    if window < 7:
        return accel
    return savgol_filter(accel, window_length=window, polyorder=3, mode="interp")


def parse_brake_to_01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    txt = series.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    num = pd.to_numeric(series, errors="coerce")
    return txt.fillna(num).fillna(0.0)


def load_driver_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["TimeSec"] = pd.to_timedelta(df["Time"], errors="coerce").dt.total_seconds()
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Brake01"] = parse_brake_to_01(df["Brake"])

    df = (
        df.dropna(subset=["TimeSec", "Distance", "Speed"])
        .sort_values("Distance")
        .drop_duplicates(subset="Distance", keep="first")
        .reset_index(drop=True)
    )

    t = df["TimeSec"].to_numpy()
    v = df["Speed"].to_numpy() / 3.6  # km/h -> m/s
    a = np.gradient(v, t)             # m/s^2
    a = smooth_accel(a)

    f_drag = 0.5 * AIR_DENSITY * (v ** 2) * CDA
    df["Acceleration"] = a
    df["Fdrag"] = f_drag
    return df


def find_driver_negative_acc_start(df: pd.DataFrame, start_search_m: float) -> float:
    sub = df[(df["Distance"] >= start_search_m) & (df["Brake01"] < BRAKE_ON_THRESHOLD)].copy()
    if len(sub) < 2:
        raise RuntimeError(f"未找到 >= {start_search_m}m 的有效非刹车数据。")

    x = sub["Distance"].to_numpy()
    a = sub["Acceleration"].to_numpy()

    idx = np.where((a[:-1] > 0) & (a[1:] <= 0))[0]
    if len(idx) == 0:
        neg = np.where(a < 0)[0]
        if len(neg) == 0:
            raise RuntimeError(f"未找到 >= {start_search_m}m 的负加速度起点。")
        return float(x[neg[0]])

    i = int(idx[0])
    x1, x2 = x[i], x[i + 1]
    y1, y2 = a[i], a[i + 1]
    if y2 == y1:
        return float(x1)
    return float(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))


def find_brake_start(df: pd.DataFrame, min_distance: float) -> float:
    candidates = df[(df["Distance"] >= min_distance) & (df["Brake01"] >= BRAKE_ON_THRESHOLD)]
    if candidates.empty:
        raise RuntimeError(f"未找到 >= {min_distance}m 的刹车起点。")
    return float(candidates["Distance"].iloc[0])


def calculate_zeta_for_driver(df: pd.DataFrame, driver_code: str, start_m: float, end_m: float) -> dict:
    seg = df[(df["Distance"] >= start_m) & (df["Distance"] <= end_m) & (df["Brake01"] < BRAKE_ON_THRESHOLD)].copy()
    seg = seg[seg["Acceleration"] < 0]

    n_points = len(seg)
    if n_points == 0:
        return {
            "Driver": driver_code,
            "StartM": start_m,
            "EndM": end_m,
            "Zeta": np.nan,
            "AclipMean": np.nan,
            "FdragMeanN": np.nan,
            "Points": 0,
        }

    a_clip_mean = float(seg["Acceleration"].mean())
    f_drag_mean = float(seg["Fdrag"].mean())
    zeta = abs(a_clip_mean * CAR_MASS_KG - f_drag_mean) / P_MGUK_MAX_W

    return {
        "Driver": driver_code,
        "StartM": start_m,
        "EndM": end_m,
        "Zeta": float(zeta),
        "AclipMean": a_clip_mean,
        "FdragMeanN": f_drag_mean,
        "Points": int(n_points),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    drivers = ["HAM", "ANT", "RUS", "LEC"]

    driver_data: dict[str, pd.DataFrame] = {}
    for code in drivers:
        csv_path = project_dir / f"2026_Shanghai_Q_{code}_telemetry.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"缺少文件: {csv_path.name}")
        driver_data[code] = load_driver_data(csv_path)

    results = []
    for code in drivers:
        driver_start = find_driver_negative_acc_start(driver_data[code], START_SEARCH_M)
        driver_end = find_brake_start(driver_data[code], MIN_BRAKE_START_M)
        results.append(calculate_zeta_for_driver(driver_data[code], code, driver_start, driver_end))

    result_df = pd.DataFrame(results).sort_values("Zeta").reset_index(drop=True)
    best = result_df["Zeta"].iloc[0]
    result_df["DeltaVsBest"] = result_df["Zeta"] - best
    result_df["DeltaVsBestPct"] = result_df["DeltaVsBest"] / abs(best) * 100

    print(
        "Forced Recovery Drag Coefficient (zeta) | "
        f"起点=各车手在 >= {START_SEARCH_M:.0f}m 处首次加速度正转负, "
        f"终点=各车手首次刹车点(>= {MIN_BRAKE_START_M:.0f}m)"
    )
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    out_csv = project_dir / "forced_recovery_drag_coefficient_results.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\\n结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
