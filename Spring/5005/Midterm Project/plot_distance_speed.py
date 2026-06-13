import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T13_POSITION_M = 3300
T14_POSITION_M = 4700
BASE_DRIVER_CODE = "LEC"
CROSS_START_M = 4000
CROSS_END_M = 4200
SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 3


def smooth_accel(accel: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = len(accel)
    window = min(window, n if n % 2 == 1 else n - 1)
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return pd.Series(accel).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    return savgol_filter(accel, window_length=window, polyorder=polyorder, mode="interp")


def compute_accel_from_speed_distance(df: pd.DataFrame) -> np.ndarray:
    s = df["Distance"].to_numpy()
    v_ms = df["Speed"].to_numpy() / 3.6
    dv_ds = np.gradient(v_ms, s)
    a_raw = v_ms * dv_ds
    return smooth_accel(a_raw, SAVGOL_WINDOW, SAVGOL_POLYORDER)


def find_positive_to_negative_crossing(
    x: np.ndarray, y: np.ndarray, start_m: float, end_m: float
) -> float | None:
    mask = (x >= start_m) & (x <= end_m)
    xr = x[mask]
    yr = y[mask]
    if len(xr) < 2:
        return None
    idx = np.where((yr[:-1] > 0) & (yr[1:] <= 0))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    x1, x2 = xr[i], xr[i + 1]
    y1, y2 = yr[i], yr[i + 1]
    if y2 == y1:
        return float(x1)
    return float(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    out_path = project_dir / "2026_SH_Q_distance_speed_T13_T14_straight_with_delta.png"

    drivers = [
        ("HAM", "Lewis Hamilton (HAM)", "red"),
        ("ANT", "Kimi Antonelli (ANT)", "#20B2AA"),
        ("RUS", "George Russell (RUS)", "#006400"),
        ("LEC", "Charles Leclerc (LEC)", "#8B0000"),
    ]

    telemetry_data = {}
    for code, label, color in drivers:
        csv_path = project_dir / f"2026_Shanghai_Q_{code}_telemetry.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"缺少文件: {csv_path.name}。请先运行 2026_SH_Q_Download.py 下载四位车手数据。"
            )
        df = pd.read_csv(csv_path)

        # 保证用于绘图的两列是数值，并去除缺失点
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
        df.dropna(subset=["Distance", "Speed"], inplace=True)
        df = df[(df["Distance"] >= T13_POSITION_M) & (df["Distance"] <= T14_POSITION_M)]
        df = df.sort_values("Distance").drop_duplicates(subset="Distance", keep="first")
        if df.empty:
            raise ValueError(
                f"{code} 在 T13-T14 区间内没有可用数据，请检查距离范围设置。"
            )
        telemetry_data[code] = {"df": df, "label": label, "color": color}

    if BASE_DRIVER_CODE not in telemetry_data:
        raise ValueError(f"基准车手 {BASE_DRIVER_CODE} 数据不存在，无法计算 Delta。")

    fig, ax_speed = plt.subplots(figsize=(12, 6), dpi=180)
    for code, info in telemetry_data.items():
        df = info["df"]
        label = info["label"]
        color = info["color"]
        ax_speed.plot(
            df["Distance"],
            df["Speed"],
            color=color,
            linewidth=1.7,
            label=label,
        )

    # 在速度图上标注加速度从正转负的过零点（4000-4200m）
    cross_points = []
    for code, info in telemetry_data.items():
        df = info["df"]
        accel = compute_accel_from_speed_distance(df)
        x_arr = df["Distance"].to_numpy()
        cross_x = find_positive_to_negative_crossing(
            x_arr, accel, CROSS_START_M, CROSS_END_M
        )
        if cross_x is None:
            continue

        speed_at_cross = float(np.interp(cross_x, x_arr, df["Speed"].to_numpy()))
        ax_speed.scatter(cross_x, speed_at_cross, color=info["color"], s=20, zorder=4)
        cross_points.append((code, cross_x, info["color"]))

    # 右轴：以勒克莱尔速度作为基准，计算其他三人的速度差并用无边框面积染色
    lec_df = telemetry_data[BASE_DRIVER_CODE]["df"]
    lec_x = lec_df["Distance"].to_numpy()
    lec_speed = lec_df["Speed"].to_numpy()

    ax_delta = ax_speed.twinx()
    delta_handles = []
    max_abs_delta = 0.0

    for code, info in telemetry_data.items():
        if code == BASE_DRIVER_CODE:
            continue

        x = info["df"]["Distance"].to_numpy()
        speed = info["df"]["Speed"].to_numpy()
        lec_interp = np.interp(x, lec_x, lec_speed)
        delta = speed - lec_interp

        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(delta))))
        ax_delta.fill_between(
            x,
            0,
            delta,
            color=info["color"],
            alpha=0.18,
            linewidth=0,
            edgecolor="none",
            interpolate=True,
        )
        delta_handles.append(
            Patch(
                facecolor=info["color"],
                edgecolor="none",
                alpha=0.18,
                label=f"{code} - {BASE_DRIVER_CODE} Delta",
            )
        )

    ax_speed.set_title("2026 Chinese GP Qualifying: T13-T14 Straight (Speed + Delta vs LEC)")
    ax_speed.set_xlabel("Distance (m)")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_delta.set_ylabel("Delta Speed vs LEC (km/h)")
    ax_speed.set_xlim(T13_POSITION_M, T14_POSITION_M)
    ax_speed.grid(alpha=0.3)
    ax_delta.axhline(0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.7)

    y_min, y_max = ax_speed.get_ylim()
    legend_x = 4200
    legend_y0 = y_min + 0.09 * (y_max - y_min)
    step = 0.045 * (y_max - y_min)
    for i, (code, cross_x, color) in enumerate(sorted(cross_points, key=lambda x: x[0])):
        ax_speed.text(
            legend_x,
            legend_y0 + i * step,
            f"{code}: {cross_x:.1f} m",
            color=color,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
            },
        )

    if max_abs_delta > 0:
        delta_limit = max_abs_delta * 1.1
        ax_delta.set_ylim(-delta_limit, delta_limit)

    ax_speed.legend(loc="upper left", frameon=False)
    ax_delta.legend(handles=delta_handles, loc="upper right", frameon=False, title="Delta Fill")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"T13 位置约: {T13_POSITION_M} m, T14 位置约: {T14_POSITION_M} m")
    print(f"Delta 基准车手: {BASE_DRIVER_CODE} (Leclerc)")
    print(f"图已生成: {out_path}")


if __name__ == "__main__":
    main()
