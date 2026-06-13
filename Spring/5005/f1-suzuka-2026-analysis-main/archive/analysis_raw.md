# Piastri — Japan 2026 — Raw Lap Telemetry Analysis

**Source:** `piastri_2026_japan_fastest_lap_telemetry.csv` only.  
**Note:** This file has **no** engineered columns (`Corner_Type`, `Driving_State`, `Mini_Sector`). Interpretation is by **distance**, **`Time`**, and **row order** only. Sector or corner naming **cannot** be inferred from the CSV alone.

---

## 1. Data inventory

- **Rows:** 707 samples (~7×10²; one lap of merged telemetry plus header row).
- **Present:** `Speed`, `Throttle`, `Brake` (boolean), `Distance`, `RPM`, `nGear`, `DRS`, `Time`, `RelativeDistance`, `Source` (`interpolation` / `pos` / `car`), `Status` (all samples `OnTrack`), `X`, `Y`, `Z`, `SessionTime`, `Date`, `DriverAhead`, `DistanceToDriverAhead`.
- **Absent in this file:** `Corner_Type`, `Driving_State`, `Mini_Sector`; official lap time; FIA sector times; weather; tyre compound/age; rivals’ laps; lateral acceleration / steering.
- **`RelativeDistance`:** ~0.00017 → **0.9973** on the last row (does not quite reach 1.0).
- **Naming:** Filename implies Piastri / Japan / 2026 and “fastest lap”; the CSV has **no** explicit lap-time or classification column — “fastest lap” is **not provable** from the file alone.

---

## 2. Speed profile

- **Overall `Speed` range:** **73.0–317.0** (stored units; typical FastF1 km/h).
- **Elapsed span from `Time`:** first row `0 days 00:00:00`, last row `0 days 00:01:32.996000` → **~92.996 s** between first and last samples (row-time span, not an official sector split).
- **Slowest region (by `Distance`):** lowest speeds cluster near **~2850–2875 m** (multiple rows **~73–85 km/h**), consistent with a single pronounced slow corner in that segment. **Cannot** name the corner without a track map in the file.
- **Fastest region:** **~310–317 km/h** appears early in row order and again in later segments; see quarter table below.

**Heuristic quarters (by row index):**

| Quarter (rows) | Speed min / max / mean (km/h) | `Brake` true % |
|----------------|-------------------------------|----------------|
| Q1 (0–176)     | 148 / 312 / 227               | 25.0%          |
| Q2 (176–353)   | 89 / 255 / 209                | 31.1%          |
| Q3 (353–530)   | 73 / 288 / 213                | 11.3%          |
| Q4 (530–707)   | 93 / 317 / 238                | 10.2%          |

**Caveat:** Q1 minimum **~148 km/h** — the first ~25% of rows are **not** a standing-start segment; the merged trace may not start exactly at the timing line you care about.

---

## 3. Longitudinal behaviour (Throttle, Brake, Speed)

- **`Throttle`:** min **0**, max **104**, mean **~65.8**.
- **`Brake`:** `True` in **~19.4%** of samples.
- **Full throttle, no brake (`Throttle` ≥ 99, `Brake` false):** **334 samples (~47.2%)** — large fraction of the lap at high longitudinal demand.
- **Brake + throttle > 0 (trail-style overlap, heuristic):** **45 samples (~6.4%)** — present but not dominant; brake pressure / hybrid **cannot** be inferred.
- **Lift / coast–like (`Throttle` < 20, `Brake` false):** includes the **global minimum speed** (~73 km/h) with **partial throttle (~0–20%)** and **no brake** on some samples in the slow cluster — ambiguous vs interpolation/threshold; **cannot** separate without richer brake data.
- **Early rows:** first samples show **`Brake` true** with **high `Throttle` and ~157 km/h** — possible merge/start artefact; interpret the start of the file cautiously.
- **`DRS`:** only value observed is **`0`** for all rows — no evidence of DRS activation in this column.
- **`nGear`:** **2–8** across the lap.

---

## 4. Engineered dimensions

**Not applicable for this file:** `Mini_Sector`, `Corner_Type`, and `Driving_State` are **missing**. Analysis stays on **distance / time / order** only. Any sector delta vs rivals or official splits: **cannot infer from this CSV**.

---

## 5. Limitations

- **Sampling / merging:** `Source` — **`pos` 360**, **`car` 345**, **`interpolation` 2** — mixed position and car streams; values are resampled/interpolated, not raw high-rate logger traces.
- **`Brake` is boolean:** no pressure; no reliable story on lock-ups or small overlaps.
- **No lateral, steering, tyres, weather, session results** in the file — no grip or line-quality conclusion from data alone.
- **`RelativeDistance` < 1** at the last row — slight shortfall vs a full normalized lap; possible edge effects at lap boundaries.

---

## Summary judgment

The trace is **internally consistent** for one on-track lap: **high top speed (~317 km/h)**, a **clear minimum ~73 km/h** around **`Distance` ~2850–2875 m**, **heavy full-throttle share**, and **moderate braking duty (~19% of samples)** with **limited brake+throttle overlap**. **Cannot** confirm official “fastest lap” status, **cannot** compare to other drivers, and **cannot** label corners beyond **distance-based** description without extra data or the structuring pipeline. The **early brake-true + high-throttle** segment warrants caution unless aligned to a known lap-zero reference.
