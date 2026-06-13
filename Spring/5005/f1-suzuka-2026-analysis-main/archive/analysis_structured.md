# Piastri — Japan 2026 — Structured Lap Telemetry Analysis

**Source:** `piastri_2026_japan_structured.csv` only.  
**Note:** `Corner_Type`, `Driving_State`, and `Mini_Sector` are **heuristic / project-defined** labels (structuring pipeline), not official FOM timing splits.

---

## 1. Data inventory

- **Rows:** 707 samples (~one lap of merged telemetry).
- **Present:** `Speed`, `Throttle`, `Brake` (boolean), `Distance`, `RPM`, `nGear`, `DRS`, `Time`, `Source` (`interpolation` / `pos` / `car`), track `X`, `Y`, `Z`, `Status`, `RelativeDistance`, plus `DriverAhead`, `DistanceToDriverAhead`, `SessionTime`, `Date`.
- **Engineered:** `Corner_Type`, `Driving_State`, `Mini_Sector`.
- **Not in file (cannot use):** official lap time, sector times, weather, tyre compound/age, rivals’ laps, lateral acceleration / steering (not in header).

---

## 2. Speed profile

- **Overall `Speed` range:** 73.0–317.0 (typical FastF1 km/h).
- **Slowest sample:** 73.0 at Distance ≈ 2864 m — `Low-Speed Corner`, `Transition`, gear 2, throttle 20, brake False.
- **Fastest sample:** 317.0 at Distance ≈ 4455 m — `Straight`, `Full Acceleration`, gear 8, throttle 99, brake False.
- **Lap length along `Distance`:** ~0 → 5723 m (full lap scale; no sector deltas vs rivals without timing columns).
- **Largest speed swings:** mini-sectors with wide min–max spreads — notably **MS_2**, **MS_5**, **MS_6**, **MS_10** (heavy braking / re-acceleration).

---

## 3. Longitudinal behaviour (Throttle, Brake, Speed)

- **Throttle:** min 0, max 104, mean ≈ 65.8.
- **Brake:** `True` in ~19.4% of rows — braking sparse in time but concentrated at slow phases.
- **`Driving_State` row counts:**

| State            | Rows |
|------------------|------|
| Full Acceleration | 340 |
| Transition       | 188 |
| Heavy Braking    | 93  |
| Trail Braking    | 44  |
| Coasting         | 42  |

- **Coherence:** Early samples show high throttle with `Brake=True` and `Trail Braking`. `Heavy Braking` aligns with large speed drops and zero throttle (e.g. MS_2 region). `Coasting` appears with no throttle and no brake before harder braking in places.
- **`DRS`:** always 0.0 — no samples show DRS open (reason not inferable from CSV).

---

## 4. Engineered dimensions

### `Corner_Type` (counts)

| Label              | Rows |
|--------------------|------|
| Straight           | 252 |
| High-Speed Corner  | 202 |
| Medium-Speed Corner| 193 |
| Low-Speed Corner   | 60  |

Straights associate with `Full Acceleration`; corner labels cluster with transition/braking states per the heuristic — descriptive of labelling, not proof of line quality vs a reference.

### `Mini_Sector` summary

| Mini_Sector | n | Speed min | Speed max | Mean speed | Most common `Driving_State` |
|-------------|---|-----------|-----------|------------|-----------------------------|
| MS_1        | 52 | 157 | 312 | 294 | Full Acceleration |
| MS_2        | 74 | 148 | 299 | 207 | Transition |
| MS_3        | 80 | 163 | 207 | 190 | Transition |
| MS_4        | 62 | 207 | 255 | 243 | Full Acceleration |
| MS_5        | 92 | 73.8 | 246 | 181 | Full Acceleration |
| MS_6        | 74 | 73 | 276 | 211 | Full Acceleration |
| MS_7        | 74 | 153 | 273 | 217 | Transition |
| MS_8        | 55 | 205 | 317 | 284 | Full Acceleration |
| MS_9        | 55 | 275 | 315 | 296 | Full Acceleration |
| MS_10       | 89 | 93 | 290 | 175 | Transition |

**Highlights:**

- **MS_1:** Mostly straight + full throttle; opens with labelled trail braking at high speed — consistent with lap-boundary placement at start/finish style segment.
- **MS_2:** Large speed range, Transition-heavy — high-speed approach and braking sequence.
- **MS_3:** Narrower speed band, Transition-dominated — connected medium-speed sequence in the model.
- **MS_8–MS_9:** Highest average speeds; MS_9 all Full Acceleration — long flat-out in labels.
- **MS_5 / MS_6 / MS_10:** Lower minimum speeds and mixed corner labels — slow corners, braking, and traction phases.

---

## 5. Limitations

- **Sampling:** ~707 points per lap; merge/resample can shift or smooth extrema. `Source` is mostly `pos`/`car` with rare `interpolation`.
- **`Brake` is boolean** — no pedal pressure; `Heavy Braking` / `Trail Braking` are algorithmic classes.
- **No lateral load / steering** — balance, line, under/oversteer not assessable here.
- **`Time` on last row ~** `0 days 00:01:32.996000` from first row’s zero is snippet elapsed time, not necessarily official lap time without confirmed lap start/finish.
- **Session context** (session type, fuel, “clean lap”, traffic): **cannot infer fully**; `DriverAhead` / `DistanceToDriverAhead` exist but are not interpreted into a traffic model in this note.

---

## Bottom line

The lap shows a **coherent longitudinal picture**: long **full-throttle** stretches (especially **MS_8–MS_9**), **focused braking** (~19% brake-on rows; **Heavy Braking** in counts), and **transition / trail-braking** where speed changes fastest. **Global minimum speed (73)** sits in **low-speed corner** context ~2860 m; **global maximum (317)** on **straight** ~4455 m. **DRS reads as never open** in this file.

**Cannot** honestly claim from this CSV alone: pace vs the field, tyre-optimality, or a guaranteed traffic-free lap.
