---
name: feature-engineering
description: Performs feature engineering on the "structured_data.csv" file. Use when a user initiates a "Feature Engineering" command.
---

In the current project, we have structured data stored in `structured_data.csv`. To enhance the dataset for analysis and modeling, we need to perform feature engineering. The specific steps are as follows:

# High/Medium/Low speed corner recognition

- High-speed corners: Entry speed ≥ 200 km/h (mostly full throttle / 7-8 gear)
- Medium-speed corners: 120-200 km/h (half throttle / 4-6 gear)
- Low-speed corners: ≤ 120 km/h (heavy braking / 1-3 gear)

# Execution

To run feature engineering for corner mapping:
Run the python script `feature_engineering.py` in the `.claude/skills/feature_engineering/` folder. It will:
1. Fetch Fast F1 circuit information (corners and apex distance).
2. Calculate Min Speed inside the corner and Left/Right direction from telemetry coordinates.
3. Compute rough entry points (e.g. apex - 100m for slow/med corners, apex - 50m for high speed) and exit points (apex + 50m).
4. Export the resulting lookup table into `structured_data/corner_info.csv`.
5. Map every telemetry row from `structured_data/structured_data.csv` to the corresponding `Corner_ID` based on its `Distance`.
6. Output a final file `structured_data/structured_data_with_corners.csv` adding `Corner_ID`, `Corner_Phase` (`Entry`/`Exit`/`Straight`), `Corner_Direction` (`Left`/`Right`/`Straight`), and `Corner_SpeedClass` (`Low`/`Medium`/`High`/`Straight`).