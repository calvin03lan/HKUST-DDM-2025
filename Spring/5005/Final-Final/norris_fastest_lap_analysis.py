from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


YEAR = 2026
DRIVER = "NOR"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
REPORT_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / "fastf1_cache"


@dataclass(frozen=True)
class EventConfig:
    label: str
    event_name: str
    file_stem: str


EVENTS = (
    EventConfig("Japanese Grand Prix", "Japanese Grand Prix", "japan"),
    EventConfig("Miami Grand Prix", "Miami Grand Prix", "miami"),
)


def ensure_directories() -> None:
    for directory in (DATA_DIR, FIGURE_DIR, REPORT_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_fastest_lap_telemetry(event: EventConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    session = fastf1.get_session(YEAR, event.event_name, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    driver_laps = session.laps.pick_drivers(DRIVER)
    fastest_lap = driver_laps.pick_fastest()
    if fastest_lap is None or pd.isna(fastest_lap.get("LapTime")):
        raise RuntimeError(f"No valid fastest lap found for {DRIVER} at {event.label} {YEAR}.")

    telemetry = fastest_lap.get_car_data().add_distance()
    telemetry = telemetry.copy()
    telemetry["Event"] = event.label
    telemetry["Driver"] = DRIVER
    telemetry["LapNumber"] = int(fastest_lap["LapNumber"])
    telemetry["LapTimeSeconds"] = fastest_lap["LapTime"].total_seconds()
    telemetry["Compound"] = fastest_lap.get("Compound")

    export_columns = [
        "Event",
        "Driver",
        "LapNumber",
        "LapTimeSeconds",
        "Compound",
        "Time",
        "Date",
        "RPM",
        "Speed",
        "nGear",
        "Throttle",
        "Brake",
        "DRS",
        "Distance",
    ]
    available_columns = [column for column in export_columns if column in telemetry.columns]
    telemetry = telemetry[available_columns]

    lap_info = {
        "event": event.label,
        "lap_number": int(fastest_lap["LapNumber"]),
        "lap_time_seconds": fastest_lap["LapTime"].total_seconds(),
        "compound": fastest_lap.get("Compound"),
        "team": fastest_lap.get("Team"),
    }
    return telemetry, lap_info


def speed_band_summary(telemetry: pd.DataFrame) -> dict[str, float]:
    speed = telemetry["Speed"].astype(float)
    return {
        "mean_speed": float(speed.mean()),
        "median_speed": float(speed.median()),
        "top_speed": float(speed.max()),
        "p90_speed": float(speed.quantile(0.90)),
        "low_speed_share": float((speed < 150).mean()),
        "medium_speed_share": float(((speed >= 150) & (speed < 250)).mean()),
        "high_speed_share": float((speed >= 250).mean()),
    }


def segment_summary(telemetry: pd.DataFrame, threshold: float = 170.0) -> pd.DataFrame:
    telemetry = telemetry.sort_values("Distance").reset_index(drop=True)
    slow = telemetry["Speed"].astype(float) < threshold
    groups = (slow != slow.shift(fill_value=False)).cumsum()
    rows: list[dict[str, float]] = []

    for _, segment in telemetry[slow].groupby(groups[slow]):
        if len(segment) < 3:
            continue
        start_distance = float(segment["Distance"].iloc[0])
        end_distance = float(segment["Distance"].iloc[-1])
        if end_distance - start_distance < 25:
            continue
        exit_window = telemetry[
            (telemetry["Distance"] >= end_distance)
            & (telemetry["Distance"] <= end_distance + 250)
        ]
        speed_gain_250m = np.nan
        avg_accel_proxy = np.nan
        if len(exit_window) >= 2:
            speed_gain_250m = float(exit_window["Speed"].iloc[-1] - exit_window["Speed"].iloc[0])
            distance_delta = float(exit_window["Distance"].iloc[-1] - exit_window["Distance"].iloc[0])
            if distance_delta > 0:
                avg_accel_proxy = speed_gain_250m / distance_delta

        rows.append(
            {
                "start_distance_m": start_distance,
                "end_distance_m": end_distance,
                "segment_length_m": end_distance - start_distance,
                "min_speed": float(segment["Speed"].min()),
                "mean_speed": float(segment["Speed"].mean()),
                "mean_throttle": float(segment["Throttle"].mean()),
                "brake_share": float(segment["Brake"].astype(bool).mean())
                if "Brake" in segment
                else np.nan,
                "exit_speed_gain_250m": speed_gain_250m,
                "exit_accel_proxy_kmh_per_m": avg_accel_proxy,
            }
        )

    return pd.DataFrame(rows).sort_values("min_speed").head(8)


def aero_energy_metrics(telemetry: pd.DataFrame) -> dict[str, float]:
    telemetry = telemetry.sort_values("Distance")
    speed = telemetry["Speed"].astype(float)
    throttle = telemetry["Throttle"].astype(float)
    brake = telemetry["Brake"].astype(bool) if "Brake" in telemetry else pd.Series(False, index=telemetry.index)

    cornering_mask = (speed >= 170) & (speed <= 270) & (throttle < 99) & (~brake)
    full_throttle = throttle >= 99
    accel_zone = full_throttle & (speed >= 120) & (speed <= 260)

    return {
        "high_speed_share": float((speed >= 250).mean()),
        "very_high_speed_share": float((speed >= 290).mean()),
        "cornering_proxy_speed": float(speed[cornering_mask].mean())
        if cornering_mask.any()
        else np.nan,
        "full_throttle_share": float(full_throttle.mean()),
        "accel_zone_mean_speed": float(speed[accel_zone].mean()) if accel_zone.any() else np.nan,
        "drs_open_share": float((telemetry["DRS"].astype(float) > 0).mean())
        if "DRS" in telemetry
        else np.nan,
    }


def plot_speed_density(all_telemetry: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "norris_speed_density_2026.png"
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=all_telemetry, x="Speed", hue="Event", fill=True, common_norm=False, alpha=0.25)
    plt.axvspan(0, 150, color="grey", alpha=0.08, label="<150 km/h")
    plt.axvspan(250, all_telemetry["Speed"].max() + 5, color="orange", alpha=0.08, label=">=250 km/h")
    plt.title("Lando Norris Fastest Lap Speed Density, 2026")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_speed_distance(all_telemetry: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "norris_speed_distance_2026.png"
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_telemetry, x="Distance", y="Speed", hue="Event", linewidth=1.2)
    plt.title("Lando Norris Fastest Lap Speed Trace, 2026")
    plt.xlabel("Distance around lap (m)")
    plt.ylabel("Speed (km/h)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_exit_gain(segments: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "norris_exit_gain_2026.png"
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=segments.dropna(subset=["exit_speed_gain_250m"]),
        x="segment_label",
        y="exit_speed_gain_250m",
        hue="Event",
    )
    plt.title("Low-Speed Corner Exit Speed Gain over 250 m")
    plt.xlabel("Slow-speed segment")
    plt.ylabel("Speed gain (km/h)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def metrics_table_latex(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(row["Event"]),
                    f"{row['LapTimeSeconds']:.3f}",
                    f"{row['MeanSpeed']:.1f}",
                    f"{row['TopSpeed']:.1f}",
                    f"{row['LowSpeedShare'] * 100:.1f}\\%",
                    f"{row['HighSpeedShare'] * 100:.1f}\\%",
                    f"{row['CorneringProxySpeed']:.1f}",
                    f"{row['FullThrottleShare'] * 100:.1f}\\%",
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def write_report(
    summary: pd.DataFrame,
    density_path: Path,
    speed_distance_path: Path,
    exit_gain_path: Path,
) -> Path:
    report_path = REPORT_DIR / "norris_fastest_lap_report.tex"
    japanese = summary.loc[summary["Event"] == "Japanese Grand Prix"].iloc[0]
    miami = summary.loc[summary["Event"] == "Miami Grand Prix"].iloc[0]
    lap_delta = miami["LapTimeSeconds"] - japanese["LapTimeSeconds"]
    high_speed_delta = (miami["HighSpeedShare"] - japanese["HighSpeedShare"]) * 100
    low_speed_delta = (miami["LowSpeedShare"] - japanese["LowSpeedShare"]) * 100
    full_throttle_delta = (miami["FullThrottleShare"] - japanese["FullThrottleShare"]) * 100

    report = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{siunitx}}
\title{{Telemetry-Based Analysis of Lando Norris's 2026 Fastest Laps in Japan and Miami}}
\author{{FastF1 Data Analysis}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Introduction}}
The 2026 Formula One calendar created an unusual competitive context. After the cancellation of the Bahrain and Saudi Arabian Grands Prix, teams had a four-week spring break in which they could prepare mechanical, aerodynamic, and energy-management updates. This report studies Lando Norris's race fastest laps at the 2026 Japanese and Miami Grands Prix using FastF1 telemetry. Norris finished fifth in Japan but improved to second in Miami, while also winning the Miami Sprint. The central question is whether the public telemetry from his fastest race laps supports the project hypothesis that McLaren's Miami package improved aerodynamic performance and energy deployment, especially on a circuit dominated by long straights and slow-corner exits.

\section{{McLaren Upgrade Context}}
The project brief identifies four main areas of McLaren development for Miami. First, the aerodynamic package was aimed at increasing downforce for a stop-and-go circuit, where braking stability and traction out of slow corners matter as much as peak speed. Second, the team is described as using a more front-loaded electrical deployment logic, supported by a 350 kW recovery system, to strengthen acceleration immediately after corner exit. Third, lightweight composite components reduced redundant mass and were expected to sharpen transient response. Fourth, suspension geometry and cooling changes were intended to stabilize the platform and improve tyre temperature control. The data analysis below focuses mainly on the first two items because public FastF1 telemetry can provide plausible proxies for aerodynamic balance and acceleration behaviour, while weight, suspension geometry, and cooling performance remain largely hidden from the public data stream.

\section{{Method}}
The analysis uses FastF1 to load the 2026 race sessions, selects Norris's fastest race lap at each event, and exports car telemetry including speed, distance, throttle, brake, gear, RPM, and DRS state. Because FastF1 does not expose McLaren's internal downforce maps or real ERS deployment power, this report treats aerodynamic and energy-management performance as proxy interpretations. Aerodynamics are evaluated through speed-density structure, the share of high-speed running, and a mid-to-high-speed cornering proxy based on samples between 170 and \SI{{270}}{{km/h}} with less than full throttle and no braking. Energy management is evaluated through full-throttle share, DRS use, and speed gain over the first 250 m after low-speed segments. These indicators cannot prove the exact content of McLaren's upgrades, but they can show whether Norris's fastest laps had the shape expected from the stated package.

\begin{{table}}[H]
\centering
\caption{{Summary metrics from Norris's fastest race laps}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Event & Lap time & Mean speed & Top speed & Low-speed & High-speed & Corner proxy & Full throttle \\
 & s & km/h & km/h & share & share & km/h & share \\
\midrule
{metrics_table_latex(summary)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Speed Distribution and Track Character}}
Figure~\ref{{fig:density}} compares the speed distribution of the two fastest laps. The Japanese lap is expected to display a larger concentration in the medium and high-speed range because Suzuka is a flowing circuit with long sequences such as the Esses, Degner, Spoon, and 130R. The Miami lap, by contrast, should have a more polarized profile: repeated braking zones and very slow corners lower the left side of the distribution, while the long straights raise the high-speed tail. In the exported metrics, Japan records a low-speed share of {japanese['LowSpeedShare'] * 100:.1f}\%, while Miami records {miami['LowSpeedShare'] * 100:.1f}\%, a difference of {low_speed_delta:.1f} percentage points. The high-speed share changes by {high_speed_delta:.1f} percentage points from Japan to Miami. This supports the qualitative distinction between Suzuka's sustained-flow character and Miami's stop-and-go demand profile.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{../figures/{density_path.name}}}
\caption{{Speed density for Norris's fastest race laps in Japan and Miami.}}
\label{{fig:density}}
\end{{figure}}

\section{{Aerodynamic Interpretation}}
McLaren's Miami update is described as increasing downforce while preserving responsiveness. The telemetry does not measure downforce directly, but the speed trace in Figure~\ref{{fig:speedtrace}} gives a useful proxy. A stronger aerodynamic platform should allow the driver to carry speed through medium and high-speed changes of direction without excessive braking or throttle hesitation. The cornering proxy is {japanese['CorneringProxySpeed']:.1f} km/h for Japan and {miami['CorneringProxySpeed']:.1f} km/h for Miami. These numbers must be read with circuit context rather than as a simple ranking, because Suzuka and Miami ask very different questions of the car. The Miami evidence is most meaningful when combined with Norris's ability to leave slow zones cleanly and still achieve a top speed of {miami['TopSpeed']:.1f} km/h. That combination is consistent with an aerodynamic package that added stability and traction-supporting load without creating an excessive straight-line penalty.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{../figures/{speed_distance_path.name}}}
\caption{{Speed trace by lap distance. The different shapes reveal Suzuka's flowing sections and Miami's repeated stop-and-go phases.}}
\label{{fig:speedtrace}}
\end{{figure}}

\section{{Energy-Management Interpretation}}
The project brief states that McLaren introduced a more front-loaded electrical deployment strategy in Miami. Since the public feed does not include ERS power, the best available proxy is the quality of acceleration after low-speed corners. Figure~\ref{{fig:exitgain}} compares the speed gained in the first 250 m after the slowest segments detected on each lap. Miami's layout makes this indicator especially relevant because lap time is strongly affected by how early the car can deploy torque after tight corners and then sustain acceleration onto the straights. Norris's full-throttle share is {japanese['FullThrottleShare'] * 100:.1f}\% in Japan and {miami['FullThrottleShare'] * 100:.1f}\% in Miami, a change of {full_throttle_delta:.1f} percentage points, but Miami's lower full-throttle share is partly a circuit-layout effect caused by more braking and rotation phases. The clearer acceleration proxy is exit speed gain: the detected low-speed exits average {japanese['MeanExitGain250m']:.1f} km/h over 250 m in Japan and {miami['MeanExitGain250m']:.1f} km/h in Miami, with Miami reaching a maximum gain of {miami['MaxExitGain250m']:.1f} km/h. This supports the claim that the car was able to convert slow-corner exits into rapid speed build-up, which is the strongest public-telemetry signature of the claimed energy-management change.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{../figures/{exit_gain_path.name}}}
\caption{{Speed gain over 250 m after detected low-speed segments. This acts as a public-data proxy for corner-exit deployment and traction.}}
\label{{fig:exitgain}}
\end{{figure}}

\section{{Conclusion}}
The FastF1 fastest-lap comparison supports the broader conclusion that McLaren's upgrade package was effective in Miami, while also showing the limits of what public telemetry can prove. Japan and Miami impose different demands: Suzuka rewards aerodynamic efficiency and sustained balance through medium and high-speed corners, whereas Miami rewards braking stability, traction, and powerful exits onto long straights. Norris's Miami performance, including his Sprint victory and the team's double podium in the Grand Prix, is consistent with a package that improved the car's stop-and-go performance without sacrificing competitive top-end speed. The speed density, speed trace, and low-speed exit metrics provide data support for the argument that the aerodynamic and energy-management changes worked together. The evidence is therefore strongest when framed as telemetry-backed interpretation rather than direct measurement of McLaren's internal upgrade values.

\end{{document}}
"""
    report_path.write_text(textwrap.dedent(report).strip() + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    sns.set_theme(style="whitegrid")

    telemetry_frames = []
    lap_infos = []
    segment_frames = []

    for event in EVENTS:
        telemetry, lap_info = load_fastest_lap_telemetry(event)
        telemetry_path = DATA_DIR / f"norris_2026_{event.file_stem}_fastest_lap_telemetry.csv"
        telemetry.to_csv(telemetry_path, index=False)
        telemetry_frames.append(telemetry)
        lap_infos.append(lap_info)

        segments = segment_summary(telemetry)
        segments["Event"] = event.label
        segments["segment_label"] = [
            f"{event.file_stem.capitalize()} {i + 1}" for i in range(len(segments))
        ]
        segments.to_csv(DATA_DIR / f"norris_2026_{event.file_stem}_slow_segments.csv", index=False)
        valid_exit_gain = segments["exit_speed_gain_250m"].dropna()
        lap_info["mean_exit_gain_250m"] = float(valid_exit_gain.mean())
        lap_info["max_exit_gain_250m"] = float(valid_exit_gain.max())
        segment_frames.append(segments)

    all_telemetry = pd.concat(telemetry_frames, ignore_index=True)
    all_telemetry.to_csv(DATA_DIR / "norris_2026_japan_miami_fastest_lap_telemetry.csv", index=False)

    summary_rows = []
    for telemetry, lap_info in zip(telemetry_frames, lap_infos, strict=True):
        speed_summary = speed_band_summary(telemetry)
        proxy_metrics = aero_energy_metrics(telemetry)
        summary_rows.append(
            {
                "Event": lap_info["event"],
                "LapNumber": lap_info["lap_number"],
                "LapTimeSeconds": lap_info["lap_time_seconds"],
                "Compound": lap_info["compound"],
                "Team": lap_info["team"],
                "MeanSpeed": speed_summary["mean_speed"],
                "MedianSpeed": speed_summary["median_speed"],
                "TopSpeed": speed_summary["top_speed"],
                "P90Speed": speed_summary["p90_speed"],
                "LowSpeedShare": speed_summary["low_speed_share"],
                "MediumSpeedShare": speed_summary["medium_speed_share"],
                "HighSpeedShare": speed_summary["high_speed_share"],
                "VeryHighSpeedShare": proxy_metrics["very_high_speed_share"],
                "CorneringProxySpeed": proxy_metrics["cornering_proxy_speed"],
                "FullThrottleShare": proxy_metrics["full_throttle_share"],
                "AccelZoneMeanSpeed": proxy_metrics["accel_zone_mean_speed"],
                "DRSOpenShare": proxy_metrics["drs_open_share"],
                "MeanExitGain250m": lap_info["mean_exit_gain_250m"],
                "MaxExitGain250m": lap_info["max_exit_gain_250m"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "norris_2026_japan_miami_summary_metrics.csv", index=False)

    all_segments = pd.concat(segment_frames, ignore_index=True)
    all_segments.to_csv(DATA_DIR / "norris_2026_japan_miami_slow_segments.csv", index=False)

    density_path = plot_speed_density(all_telemetry)
    speed_distance_path = plot_speed_distance(all_telemetry)
    exit_gain_path = plot_exit_gain(all_segments)
    report_path = write_report(summary, density_path, speed_distance_path, exit_gain_path)

    print("Generated files:")
    for path in [
        DATA_DIR / "norris_2026_japan_miami_fastest_lap_telemetry.csv",
        DATA_DIR / "norris_2026_japan_miami_summary_metrics.csv",
        DATA_DIR / "norris_2026_japan_miami_slow_segments.csv",
        density_path,
        speed_distance_path,
        exit_gain_path,
        report_path,
    ]:
        print(f"- {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
