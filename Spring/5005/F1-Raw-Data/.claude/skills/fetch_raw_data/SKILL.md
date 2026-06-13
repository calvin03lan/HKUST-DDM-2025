---
name: fetch-raw-data
description: Utilize fastf1 to acquire raw F1 telemetry data and lap summaries. Trigger this skill when the user requests to "fetch raw data" or scrape F1 telemetry.
---

# Fetch Raw Data
This skill leverages the `fastf1` Python library to retrieve official Formula 1 telemetry and lap summary data. The core principle is to safely obtain raw data **without any feature engineering operations at this stage**.

## Natural Language Command Parsing
Automatically parse user input in natural language, validate and extract required parameters:
- **Year**: Race year (e.g., 2026, 2025)
- **Grand Prix**: Race name (e.g., Japanese Grand Prix, Japan)
- **Driver**: Full name 、First name、Last name、 3-letter code (e.g., Lando Norris, NOR)
- **Session Type**: Race, Sprint, Qualifying, Practice 1/2/3
- **Data Scope**: All laps / fastest lap / specified lap number
- **Data Type**: Telemetry + lap summary

 Input format example (included but not only natural language):
"Get [year] [grand prix] [driver] [session type] fastest lap / all laps / lap X"
"获取2026年日本大奖赛诺里斯的正赛最快单圈" / "Fetch 2026 Japanese Grand Prix Norris race fastest lap"

## Data Fetching Steps
1. **Environment & Cache Configuration**: Ensure `fastf1` and `pandas` are installed. Set a local cache directory (e.g., `./fastf1_cache`) to avoid API rate limits from duplicate requests, and enable caching via `fastf1.Cache.enable_cache(cache_dir)`.
2. **Load Race Session**: Load event data by year, Grand Prix name, and session type (e.g., 'Race', 'Qualifying').
   Example: `sess = fastf1.get_session(2026, 'Japan', 'Race')`, followed by `sess.load()`.
3. **Target Driver & Fastest Lap Extraction**: Use a driver’s 3-letter abbreviation (e.g., `NOR`) to filter lap data:
   `session.laps.pick_driver('NOR').pick_fastest()`.
4. **High-Frequency Telemetry Acquisition**: Call `.get_telemetry()` to retrieve millisecond-level telemetry data, including speed, RPM, throttle, brake, gear, and X/Y/Z coordinates. Export the resulting DataFrame as a standalone CSV file (e.g., `[driver]_[gp]_[year]_telemetry.csv`).
5. **Lap-Level Summary Extraction**: Extract low-frequency lap metrics using native attributes and methods.
   **Note: Do NOT merge or broadcast these lap-level low-frequency metrics with high-frequency telemetry**:
   - Tire information: `Compound`, `TyreLife`
   - Sector timing: `Sector1Time`, `Sector2Time`, `Sector3Time`
   - Track condition: `TrackStatus`
   - Snapshot weather: Extract AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, WindDirection and Rainfall via `.get_weather_data()`.
6. **Summary Data Export**: Save the extracted lap-level summary as an independent JSON file (e.g., `[driver]_[gp]_[year]_lap_summary.json`) for downstream structured workflow access.
7. Save the generated scripts and CSV，JSON files to the "raw_data" folder in the project directory.