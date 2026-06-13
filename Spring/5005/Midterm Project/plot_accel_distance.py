import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T13_POSITION_M = 3300
T14_POSITION_M = 4700
CROSS_START_M = 4000
CROSS_END_M = 4200

# 平滑参数（可调）
SAVGOL_WINDOW = 31   # 必须是奇数；越大越平滑
SAVGOL_POLYORDER = 3


def smooth_accel(accel: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = len(accel)
    # 保证 window 合法：奇数、>= polyorder+2、且不超过长度
    window = min(window, n if n % 2 == 1 else n - 1)
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        # 点数太少时退化为滚动平均
        return pd.Series(accel).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    return savgol_filter(accel, window_length=window, polyorder=polyorder, mode="interp")


def compute_accel_from_speed_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    用物理关系 a = v * dv/ds 计算纵向加速度（m/s^2）
    - v: m/s
    - s: m
    """
    s = df["Distance"].to_numpy()
    v_ms = (df["Speed"].to_numpy()) / 3.6
    dv_ds = np.gradient(v_ms, s)       # (m/s)/m = 1/s
    a_raw = v_ms * dv_ds               # m/s^2
    a_smooth = smooth_accel(a_raw, SAVGOL_WINDOW, SAVGOL_POLYORDER)

    out = df.copy()
    out["AccelRaw"] = a_raw
    out["Accel"] = a_smooth
    return out


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
    out_path = project_dir / "2026_SH_Q_accel_distance_T13_T14.png"

    drivers = [
        ("HAM", "Lewis Hamilton (HAM)", "red"),
        ("ANT", "Kimi Antonelli (ANT)", "#20B2AA"),
        ("RUS", "George Russell (RUS)", "#006400"),
        ("LEC", "Charles Leclerc (LEC)", "#8B0000"),
    ]

    telemetry = {}
    for code, label, color in drivers:
        csv_path = project_dir / f"2026_Shanghai_Q_{code}_telemetry.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"缺少文件: {csv_path.name}")

        df = pd.read_csv(csv_path)
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
        df = df.dropna(subset=["Distance", "Speed"])
        df = df[(df["Distance"] >= T13_POSITION_M) & (df["Distance"] <= T14_POSITION_M)]
        df = df.sort_values("Distance").drop_duplicates(subset="Distance", keep="first")

        if len(df) < 7:
            raise ValueError(f"{code} 在 T13-T14 区间点数太少，无法稳定计算加速度")

        df = compute_accel_from_speed_distance(df)
        telemetry[code] = {"df": df, "label": label, "color": color}

    # 用统一距离网格对齐，避免各车手采样点不一致
    common_x = np.linspace(T13_POSITION_M, T14_POSITION_M, 900)
    accel_interp = {}
    for code, info in telemetry.items():
        s = info["df"]["Distance"].to_numpy()
        a = info["df"]["Accel"].to_numpy()
        accel_interp[code] = np.interp(common_x, s, a)

    crossings = {}
    for code, y in accel_interp.items():
        crossings[code] = find_positive_to_negative_crossing(
            common_x, y, CROSS_START_M, CROSS_END_M
        )

    fig, ax_acc = plt.subplots(figsize=(12, 6), dpi=180)

    # 左轴：4位车手加速度线
    cross_points = []
    for code, info in telemetry.items():
        ax_acc.plot(
            common_x,
            accel_interp[code],
            color=info["color"],
            linewidth=1.8,
            label=info["label"],
        )
        cross_x = crossings.get(code)
        if cross_x is not None:
            ax_acc.scatter(cross_x, 0, color=info["color"], s=22, zorder=4)
            cross_points.append((code, cross_x, info["color"]))

    ax_acc.set_title("2026 Chinese GP Q: T13-T14 Straight (Acceleration)")
    ax_acc.set_xlabel("Distance (m)")
    ax_acc.set_ylabel("Acceleration (m/s²)")
    ax_acc.set_xlim(T13_POSITION_M, T14_POSITION_M)
    ax_acc.grid(alpha=0.3)
    ax_acc.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)

    y_min, y_max = ax_acc.get_ylim()
    legend_x = 4200
    legend_y0 = y_min + 0.08 * (y_max - y_min)
    step = 0.06 * (y_max - y_min)
    for i, (code, cross_x, color) in enumerate(sorted(cross_points, key=lambda x: x[0])):
        ax_acc.text(
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

    ax_acc.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"T13~T14: {T13_POSITION_M}m ~ {T14_POSITION_M}m")
    print(f"图已生成: {out_path}")


if __name__ == "__main__":
    main()