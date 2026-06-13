# F1 Lap Simulator — Interface Specification

Point-mass lap simulator for the 2026 Japanese GP (Suzuka).
Produces per-segment, per-lap telemetry for one or more drivers.

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                         INPUTS                                  │
 │                                                                 │
 │  structured CSV ──► track_model.py ──► 28 segments (track map)  │
 │  structured CSV ──► car_model.py  ──► CarModel (initial guess)  │
 │  real fastest lap ──► calibrate_car_params() ──► calibrated Car  │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                    lap_solver.py                                 │
 │  Forward-backward solve: boundary speeds + segment times        │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     race_sim.py                                  │
 │  53-lap loop with tyre degradation ──► race_log DataFrame       │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                        OUTPUT                                    │
 │  race_log DataFrame (1792 rows = 28 segments × 32 laps × 2)    │
 │  Per-segment: v_entry, v_exit, v_min, t_segment, t_race_cumul   │
 └─────────────────────────────────────────────────────────────────┘
```

## Input: Self-Calibrating Model

The model is **self-calibrating** — no manual parameter input from teammates is needed.

1. **Track model**: Auto-extracted from PIA's structured telemetry CSV (28 segments with corner types and apex speeds)
2. **Car model (PIA)**: Initial telemetry estimate → whole-lap calibration against real fastest lap (92.996s)
3. **Car model (other drivers)**: Scaled from PIA baseline using `calibrate_car_from_race_pace()` — only needs the driver's **real fastest lap time** (one number from FastF1)

See [docs/qa.md](../docs/qa.md) Q1–Q4 for methodology details.

## What I produce (Output Spec)

### Pre-exported CSV files

Run `uv run python scripts/export_race_log.py` to generate:

**`outputs/gap_summary.csv`** (~636 rows) — per-lap gap for baseline + 11 sensitivity scenarios:

| Column | Type | Description |
|--------|------|-------------|
| `scenario` | str | Scenario name: `"baseline"`, `"pit_lap_20"`, `"pit_delta_21"`, `"deg_0.00020"`, etc. |
| `ant_pit_lap` | int | ANT pit stop lap in this scenario |
| `pit_delta_s` | float | Pit stop time penalty (seconds) |
| `deg_rate` | float | HARD tyre degradation rate |
| `lap` | int | Lap number (1–53) |
| `pia_cumul_s` | float | PIA cumulative race time (seconds) |
| `ant_cumul_s` | float | ANT cumulative race time (seconds) |
| `gap_s` | float | ANT_cumul - PIA_cumul (positive = PIA ahead) |
| `source` | str | `"real"` (L1–L21) or `"simulated"` (L22–L53) |

To get baseline only: `df[df["scenario"] == "baseline"]`

**`outputs/sensitivity_results.csv`** (12 rows) — one-row-per-scenario summary:

| Column | Type | Description |
|--------|------|-------------|
| `scenario` | str | Scenario name |
| `ant_pit_lap` | int | ANT pit stop lap |
| `pit_delta_s` | float | Pit stop time penalty |
| `deg_rate` | float | HARD tyre degradation rate |
| `final_gap_s` | float | Gap at L53 (positive = PIA ahead) |
| `winner` | str | `"PIA"` or `"ANT"` |
| `is_baseline` | bool | Whether this is the baseline scenario |

**`outputs/race_log.csv`** (~1800 rows) — per-segment per-lap telemetry (baseline only):

| Column | Type | Description |
|--------|------|-------------|
| `driver` | str | Driver code, e.g. `"PIA"`, `"ANT"` |
| `lap` | int | Lap number (22–53, simulated portion) |
| `seg_id` | int | Segment index (0–27) |
| `seg_type` | str | `"Straight"`, `"High-Speed Corner"`, `"Medium-Speed Corner"`, `"Low-Speed Corner"` |
| `d_start` | float | Segment start distance from start/finish (m) |
| `d_end` | float | Segment end distance (m) |
| `v_entry` | float | Speed entering the segment (m/s) |
| `v_exit` | float | Speed exiting the segment (m/s) |
| `v_min` | float | Minimum speed within the segment (m/s) |
| `t_segment` | float | Time through this segment (s) |
| `t_lap_cumul` | float | Cumulative time within this lap (s) |
| `t_race_cumul` | float | Cumulative time from race start (s) |
| `tyre_compound` | str | `"HARD"`, `"MEDIUM"` |
| `tyre_age` | int | Laps on current tyres |
| `v_apex_degraded` | float | Degraded apex speed for corners (m/s), empty for straights |
| `interaction` | str | Reserved for multi-driver interaction (currently empty) |
| `interaction_delta_s` | float | Reserved time adjustment from interaction (currently `0.0`) |

### Derived outputs (for the presentation)

From the CSV files you can compute:

- **Per-lap times**: `groupby('lap')['t_segment'].sum()`
- **Gap between drivers**: read `gap_summary.csv` directly
- **Speed vs Distance curves**: filter `race_log.csv` by lap, plot `v_entry` / `v_exit` against `d_start`
- **What-If conclusion**: "Without Safety Car, Oscar leads by +1.8 seconds at lap 53"

## How to run

```bash
# Smoke test (validates track model, car model, solver, 53-lap race, cross-validation)
uv run python sim/smoke_test.py

# Export CSVs for teammates
uv run python scripts/export_race_log.py
```

## Current accuracy

| Metric | Value |
|--------|-------|
| One-lap simulated time | 93.052 s |
| Real fastest lap (telemetry) | 92.996 s |
| One-lap error | **0.06%** (after whole-lap calibration) |
| Cross-validation MAE (PIA, L28–L53) | 0.407 s/lap |
| Cross-validation MAE (ANT, L28–L53) | 0.284 s/lap |
| Final What-If gap at L53 | **+1.8 s** (PIA ahead) |

## Limitations

- Two drivers only (PIA + ANT); extendable to more via `calibrate_car_from_race_pace()`
- Track model from Oscar's fastest lap only (not averaged across laps)
- No fuel mass, DRS, or aerodynamic wake modelling (absorbed into calibrated effective parameters)
- Driver interaction columns are reserved but not yet implemented (sensitivity analysis shows gap > 1.2s, so interaction would not trigger)
