#!/usr/bin/env python3
"""
Step 1: Fetch 2026 Japanese GP Race telemetry for ALL drivers (all clean laps).

Run this script locally with the proxy active (port 7897).
Outputs per-driver per-lap telemetry CSVs into ./raw_laps/ directory.

Usage:
    python step1_fetch_all_data.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Proxy Configuration ──────────────────────────────────────────────────────
os.environ['HTTP_PROXY']  = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

import fastf1
import pandas as pd
import numpy as np

# ── Cache & Output Dirs ──────────────────────────────────────────────────────
BASE_DIR   = r'C:\Users\pc\Desktop\skills'
CACHE_DIR  = os.path.join(BASE_DIR, 'fastf1_cache')
OUTPUT_DIR = os.path.join(BASE_DIR, 'raw_laps')
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── "Normal lap" filter ──────────────────────────────────────────────────────
#   Exclude: formation lap, in-lap, out-lap, SC/VSC/red-flag laps
def is_normal_lap(lap):
    if lap.get('LapNumber', 0) <= 1:
        return False
    if not pd.isnull(lap.get('PitInTime',  pd.NaT)):
        return False
    if not pd.isnull(lap.get('PitOutTime', pd.NaT)):
        return False
    ts = str(lap.get('TrackStatus', ''))
    # Exclude any lap that passed through SC (4), VSC (6), or Red Flag (5)
    if any(s in ts for s in ['4', '5', '6']):
        return False
    return True

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" 2026 F1 Japanese GP — Fetching All Driver Telemetry")
    print("=" * 60)

    print("\n[1/2] Loading session (telemetry=True)…")
    session = fastf1.get_session(2026, 'Japan', 'Race')
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    all_drivers = sorted(session.laps['Driver'].unique())
    print(f"      Drivers found: {all_drivers}")

    # Save lap-level metadata (TrackStatus, PitIn/Out, LapTime, etc.)
    lap_meta_cols = [
        'Driver', 'LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time',
        'Sector3Time', 'Compound', 'TyreLife', 'TrackStatus',
        'PitInTime', 'PitOutTime', 'IsAccurate'
    ]
    available_meta = [c for c in lap_meta_cols if c in session.laps.columns]
    session.laps[available_meta].to_csv(
        os.path.join(OUTPUT_DIR, 'all_laps_meta.csv'), index=False
    )
    print(f"      Lap metadata saved → {OUTPUT_DIR}/all_laps_meta.csv")

    print("\n[2/2] Fetching telemetry for each driver's clean laps…")
    summary_rows = []

    for driver in all_drivers:
        driver_laps = session.laps.pick_driver(driver)
        clean_laps  = driver_laps[driver_laps.apply(is_normal_lap, axis=1)]

        if len(clean_laps) == 0:
            print(f"  {driver}: 0 clean laps — skip")
            continue

        driver_tels = []
        for lap_num, (_, lap) in enumerate(clean_laps.iterrows(), 1):
            try:
                tel = lap.get_telemetry()
                if tel is None or len(tel) < 30:
                    continue
                tel = tel.copy()
                tel['Driver']    = driver
                tel['LapNumber'] = int(lap.get('LapNumber', 0))
                driver_tels.append(tel)
            except Exception as e:
                print(f"    !! {driver} lap {lap.get('LapNumber','?')}: {e}")

        if not driver_tels:
            print(f"  {driver}: telemetry unavailable — skip")
            continue

        combined = pd.concat(driver_tels, ignore_index=True)
        out_path  = os.path.join(OUTPUT_DIR, f'{driver}_telemetry.csv')
        combined.to_csv(out_path, index=False)

        summary_rows.append({
            'Driver': driver,
            'N_Clean_Laps': len(clean_laps),
            'N_Tel_Laps':   len(driver_tels),
            'Tel_Rows':     len(combined),
            'File':         out_path,
        })
        print(f"  {driver}: {len(clean_laps)} clean laps, "
              f"{len(driver_tels)} with telemetry → {out_path}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'fetch_summary.csv'), index=False
    )
    print(f"\n  Fetch summary → {OUTPUT_DIR}/fetch_summary.csv")
    print("\n✓ Step 1 complete. Run step2_compute_metrics.py next.\n")


if __name__ == '__main__':
    main()
