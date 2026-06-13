# F1 Suzuka 2026 Analysis

Point-mass lap simulator and What-If analysis for the 2026 Japanese GP at Suzuka.

**Research question:** If there had been no Safety Car, would Oscar Piastri have won?

Chinese version for teammates: [README_zh.md](README_zh.md)

## Project Structure

```
├── sim/                  Core simulator (the main deliverable)
│   ├── track_model.py        28-segment track from structured CSV
│   ├── car_model.py          Driver performance params from telemetry
│   ├── lap_solver.py         Forward-backward point-mass solver
│   ├── race_sim.py           53-lap race with tyre degradation
│   ├── smoke_test.py         Integrated validation (run this first)
│   └── README.md             Interface spec for teammates
│
├── data/                 All input data
│   ├── piastri_2026_japan_structured.csv      Structured telemetry (fastest lap)
│   ├── piastri_2026_japan_fastest_lap_telemetry.csv   Raw telemetry
│   └── race_laps_japan_2026_pia_vs_ant.csv    Race lap times (PIA vs ANT)
│
├── scripts/              Utility scripts
│   ├── export_race_log.py  Export gap_summary.csv and race_log.csv for teammates
│   ├── plot_gap.py          Fork-point gap chart generator
│   ├── cross_validate.py    L28-L53 cross-validation vs real laps
│   ├── sensitivity.py       Sensitivity analysis (pit lap, pit delta, deg rate)
│   ├── pull_race_laps.py     Fetch race laps from FastF1
│   ├── run_draft_insight.py  Scope A/B/C analysis + Tier-0 sensitivity
│   ├── what_if_model.py      Gap projection formulas (Tier-0 + degradation)
│   ├── ingest_team_b.py      Ingest teammate B's lap-time CSV
│   └── parity_check.py       CSV row-count sanity check
│
├── outputs/              Generated charts and data exports
│   ├── gap_summary.csv      Per-lap gap (636 rows, baseline + 11 sensitivity scenarios)
│   ├── sensitivity_results.csv  One-row-per-scenario summary (12 rows)
│   ├── race_log.csv         Per-segment telemetry (1792 rows, PIA + ANT baseline)
│   ├── track_xy.csv         Suzuka centreline (707 points, distance/x/y)
│   ├── gap_vs_lap.png       Fork-point gap chart
│   ├── sensitivity.png      Sensitivity analysis chart
│   └── cross_validation.png Cross-validation error chart
│
├── docs/                 Presentation planning & notes
│   ├── what_if.md            Symbols, formulas, assumptions
│   ├── flow.md               20-min presentation script
│   ├── tasks.md              Team task breakdown
│   ├── qual_sources.md       Bibliography scaffold
│   ├── sync_with_team_B.md   Data handoff notes
│   └── qa.md                 Methodology FAQ (bilingual)
│
└── archive/              Individual assignment (reference only)
    ├── main.py               FastF1 telemetry fetch script
    ├── structure_telemetry.py Structuring engine
    ├── analysis_raw.md       Raw data analysis
    ├── analysis_structured.md Structured data analysis
    ├── comparison_report.tex  LaTeX comparison report
    └── comparison_report.pdf  Compiled report
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run smoke test (validates track model, car model, solver, 53-lap race)
uv run python sim/smoke_test.py
```

## Current Accuracy

| Metric | Value |
|--------|-------|
| One-lap simulated time (calibrated) | 93.052 s |
| Real fastest lap | 92.996 s |
| Error | **0.06%** |
| Cross-validation MAE (PIA, L28–53) | 0.407 s/lap |
| Cross-validation MAE (ANT, L28–53) | 0.284 s/lap |
| Fork-point final gap (no SC) | **PIA wins by +1.8s** |

## How the Simulator Works

The track is modelled as a 1-D line split into 28 segments (straights + corners).

1. **Segment splitting**: Merge consecutive telemetry rows of the same corner type into segments. Each corner has a speed limit (v_ref); straights are limited by top speed (v_max).
2. **Forward-backward solver**: For each segment boundary, compute the max speed from accelerating forward AND the max speed from braking backward. Take the minimum — that's the real speed.
3. **Whole-lap calibration**: Initial car params are estimated from telemetry (median dv/dt). These are then calibrated against the real fastest lap (92.996s) using uniform scaling + coordinate descent, reducing error from 2.4% to 0.06%. The calibrated values are effective/lumped parameters (grey-box model).
4. **Segment timing**: Each segment has three phases — accelerate, coast, brake. Sum all 28 segments = one lap time.
5. **Tyre degradation**: Each lap, corner v_ref drops slightly. Over 53 laps on HARD tyres, total slowdown is ~0.3s.
6. **Pit stops**: A ~22s penalty is added on pit lap; tyre compound and age reset.
7. **Fork-point simulation**: Real data is used for L1–L21 (zero simulation error). At L22 — the lap where the SC changed everything — we fork: in our What-If, ANT pits normally (+22s) instead of getting a free pit under SC. The simulation runs L22–L53 with initial conditions derived from real cumulative times.
8. **Cross-validation**: Simulated L28–L53 lap times are compared against real post-SC clean laps. PIA MAE = 0.407s, ANT MAE = 0.284s.
9. **Gap = difference in cumulative race time**. At L53, PIA finishes +1.8s ahead.

For the full math, see [docs/algorithm.md](docs/algorithm.md).

### Sensitivity Analysis

How robust is "PIA wins by +1.8s"?

- **ANT pit lap** (L20–L24): PIA wins in ALL scenarios
- **Pit delta** (20–24s): PIA loses only if pit stop ≤ 20s (real value is ~22s)
- **Degradation rate**: PIA loses only at 5x baseline rate (contradicted by cross-validation)

See `outputs/sensitivity.png` and [docs/algorithm.md](docs/algorithm.md) Section 9.

### Planned: Driver Interaction Model

Currently both drivers simulate independently. Sensitivity analysis confirms the gap stays above 1.2s in all plausible scenarios, so interaction effects would not change the conclusion. Reserved for future extension.

- **Trigger**: When two cars are within 1.5s on the same segment
- **Effect**: Perturbation to segment time (`interaction_delta_s`)
- **Combinations**: `(+,+)` mutual acceleration, `(-,-)` mutual blocking, `(+,-)` and `(-,+)` asymmetric
- **Data columns**: `interaction` and `interaction_delta_s` are already reserved in `race_log`

### FAQ

See [docs/qa.md](docs/qa.md) for bilingual answers to methodology questions (calibration, validation, grey-box model, etc.).

### Model Assumptions and Limitations

This is a **1D point-mass simulator**. Key simplifications:

| Assumption | Impact |
|-----------|--------|
| Point mass (no car width) | Cannot model overtaking line choices |
| No DRS modelling | Absorbed into calibrated effective parameters |
| No fuel mass change (~0.05s/lap) | Partially offset by degradation rate |
| No weather / track evolution | Reasonable for dry-condition Suzuka race |
| No driver interaction | Sensitivity analysis confirms gap > 1.2s, outside interaction range |
| Effective parameters | Calibrated values are not real physics — they make the simplified model produce correct lap times |
| Single-source track model | Based on PIA's fastest lap only, not multi-lap average |
| Non-unique calibration | 1 target, 3 free params; uniform scaling preserves ratios as mitigation |

Full details: [docs/algorithm.md](docs/algorithm.md) Section 1 and [docs/qa.md](docs/qa.md) Q1–Q8.

### Teammate Data Handoff

The model is **self-calibrating** — no manual parameter input needed. To add a new driver, only their real fastest lap time is required (available from FastF1).

Exported files for visualization:

| File | Rows | Purpose |
|------|------|---------|
| `outputs/gap_summary.csv` | ~636 | Per-lap gap, baseline + 11 sensitivity scenarios (filter by `scenario` column) |
| `outputs/sensitivity_results.csv` | 12 | One-row-per-scenario summary (final gap, winner) for sensitivity bar charts |
| `outputs/race_log.csv` | 1792 | Per-segment telemetry (baseline), for position computation |
| `outputs/track_xy.csv` | 707 | Suzuka centreline (distance, x, y), for 2D track animation |

To place a car on the 2D track: look up `d_start` from `race_log.csv` in `track_xy.csv` to get the (x, y) position.

## Data Sources

All data is sourced programmatically via [FastF1](https://github.com/theOehrly/Fast-F1) from the official F1 timing API.

## Tech Stack

- Python 3.13 + [uv](https://docs.astral.sh/uv/)
- FastF1 for data acquisition
- pandas for race log DataFrames
- Pure-Python physics solver (no ML dependencies)
