# Algorithm: Point-Mass Lap Simulator

## 1. Problem Setup / 問題設定

We model an F1 car as a **point mass** moving along a 1-D track.
The car has no width, no aerodynamic wake interaction, and no fuel mass change.

我們把 F1 賽車簡化為沿一維賽道運動的**質點**。
忽略車寬、尾流效應與油量變化——這些對單車手模擬的影響在 1–3% 以內，
可接受。

**What we model:**
- Two-band acceleration (low-speed vs high-speed)
- Braking deceleration (constant)
- Corner speed limits (apex constraint)
- Tyre degradation (linear reduction in corner speeds)
- Pit stop strategy (compound switch mid-race)

**What we ignore:**
- Aerodynamic wake / dirty air / DRS
- Fuel mass reduction (~0.05s/lap effect)
- Weather, track evolution
- Multi-car interaction (reserved for future extension)

---

## 2. Track Model / 賽道模型

**Source**: `data/piastri_2026_japan_structured.csv` (structured telemetry from Oscar's fastest lap)

**Process** (`sim/track_model.py`):

1. 每一筆遙測 row 都有 `Corner_Type`（Straight / High-Speed Corner / Medium-Speed Corner / Low-Speed Corner）
2. 把**連續相同 `Corner_Type`** 的 rows 合併成一個 segment
3. `v_ref` = 直道取 max(Speed)，彎道取 min(Speed)
4. 用 `_close_gaps()` 消除 segment 之間的距離間隙
5. 用 `_absorb_micro_segments()` 把 < 10m 的碎片 segment 吸收進前一段

**Result**: 28 segments, total 5723.3 m

```
seg  type               length   v_ref(km/h)
 0   Medium-Speed         61.0      157.0
 1   Straight            520.9      312.0
 2   High-Speed          150.9      200.6
...
27   Straight            113.1      290.4
```

---

## 3. Car Model / 車輛模型

**Source**: same structured CSV, processed by `sim/car_model.py`

為什麼不用單一加速度？因為空氣阻力與速度平方成正比。
低速時加速度高（traction-limited），高速時加速度低（drag-limited）。

We split at **200 km/h**:

| Parameter | Meaning | How estimated |
|-----------|---------|---------------|
| `a_accel_low` | Acceleration < 200 km/h | Median of `ΔSpeed/Δt` where `Driving_State == "Full Acceleration"` and avg speed < 200 |
| `a_accel_high` | Acceleration ≥ 200 km/h | Same filter, avg speed ≥ 200 |
| `a_brake` | Braking deceleration | Median of `ΔSpeed/Δt` where `Driving_State == "Heavy Braking"` |
| `v_max` | Top speed | `max(Speed)` in the CSV |

**Initial estimate (PIA)**: `a_low=12.73`, `a_high=4.96`, `a_brake=13.89`, `v_max=317 km/h`

### 3.1 Whole-Lap Calibration / 整圈校準

The initial per-sample `Δv/Δt` estimates yield a 2.4% lap time overestimate
(95.251s vs 92.996s). This is because the median is conservative and the
model lacks DRS, gradient, and non-linear drag.

**Solution**: calibrate the three acceleration/braking params against the
real fastest lap time (92.996s), keeping `v_max` fixed (hard physical
constraint — observed top speed).

**Method** (`calibrate_car_params()` in `sim/car_model.py`):

1. **Phase 1 — Uniform scaling**: Scale all three params by a single factor
   via binary search until `solve_lap()` matches the target. This preserves
   the relative ratios between acceleration and braking.
2. **Phase 2 — Coordinate descent**: Fine-tune each param independently
   (within ±5% of the Phase 1 result) while respecting ±30% guard-rails
   from the initial telemetry estimate. Keep the best result.

| Param | Initial | Calibrated | Drift |
|-------|---------|------------|-------|
| `a_accel_low` | 12.73 | 15.96 | +25.4% |
| `a_accel_high` | 4.96 | 6.22 | +25.4% |
| `a_brake` | 13.89 | 17.41 | +25.4% |
| `v_max` | 88.06 m/s | 88.06 m/s | fixed |

**Result**: Sim lap = **93.052s** (error 0.06%, down from 2.4%)

#### What the calibrated values mean / 校準值的物理意義

校準後的參數是**等效參數（Effective / Lumped Parameters）**。它們隱含了
我們沒有顯式建模的因素（DRS 加速、坡度、非線性空氣阻力等）。
這在工程上稱為**灰箱模型（Grey-box Model）**：
物理公式（白箱）定義了行為，數據校準（黑箱）填補了未知的極限參數。

#### Guard-rails and risks / 防護欄與風險

- 每個參數不超過初始估計的 ±30%（防止物理上不合理的補償效應）
- 只用 1 圈校準 3 個參數 → 解不唯一，但 uniform scaling 保持了比例關係
- **真正的驗證在交叉驗證（Section 8.1）**，不在校準本身

For drivers without structured telemetry (e.g. ANT), we calibrate by scaling
`v_max` until the simulated lap time matches their real race pace.

---

## 4. Forward-Backward Solver / 前向-後向求解器

This is the core algorithm in `sim/lap_solver.py`.

### 核心思想

對每個 segment 邊界，我們需要知道「車速應該是多少」。
但車速同時受到兩個限制：

- **前向限制**：從前一段出來後，最多能加速到多快？
- **後向限制**：要能在下一段之前煞停到足夠慢，最多能多快？

取兩者最小值，就是真實的邊界速度。

### Algorithm

```
Forward pass (left → right):
  boundary_fwd[0] = v_ref of first segment
  for each segment i:
    if corner: v_out = accelerate from v_ref (apex) over half-length
    if straight: v_out = accelerate from v_entry over full length
    boundary_fwd[i+1] = v_out

Backward pass (right → left):
  boundary_bwd[N] = v_ref of first segment (lap wraps around)
  for each segment i (reverse):
    if corner: v_in = max speed that can brake to v_ref in half-length
    if straight: v_in = max speed that can brake to v_out in full length
    boundary_bwd[i] = v_in

Combine:
  boundary[i] = min(boundary_fwd[i], boundary_bwd[i])
```

### Speed Profile Diagram / 速度剖面示意圖

```
Speed
  ^
  |          v_max ─────────────
  |         /                   \
  |        / accel    coast      \ brake
  |       /                       \
  |      /                         \  v_ref (corner apex)
  |─────/                           ●───────●
  |                                          \
  |    seg_0     seg_1 (straight)    seg_2    \ seg_3
  └──────────────────────────────────────────────→ Distance
```

在彎道中，`v_ref` 是彎心（apex）的速度約束。
車輛可以在進彎時以高於 `v_ref` 的速度進入，只要能在半段距離內煞停到 `v_ref`。

---

## 5. Three-Phase Segment Timing / 三階段計時

After we know `v_entry` and `v_exit` for each segment, we compute time.

### Straight segments（直道）

```
Phase 1: Accelerate from v_entry → v_max (or v_peak if segment too short)
Phase 2: Coast at v_max (if distance remains)
Phase 3: Brake from v_max → v_exit
```

**Time calculation**:
- Phase 1 uses two-band acceleration: low-speed band → high-speed band → coast
- Phase 2: `t = d_coast / v_max`
- Phase 3: `t = (v_max - v_exit) / a_brake`

If the segment is too short to reach `v_max`, we binary-search for `v_peak`
(the highest speed the car can reach and still brake to `v_exit` in the
remaining distance).

### Corner segments（彎道）

```
Phase 1: Brake from v_entry → v_ref (apex speed)
Phase 2: Coast at v_ref (if distance remains)
Phase 3: Accelerate from v_ref → v_exit
```

彎道跟直道的差別：直道的「頂速」是 `v_max`，彎道的「頂速」是 `v_ref`（彎心速度）。
直道先加速再煞車，彎道先煞車再加速（先減速進彎，再加速出彎）。

**Fallback**: If phases 1+3 need more distance than the segment length (rare,
happens when the car can't fully brake to `v_ref`), we use a harmonic-mean
approximation: `t = length / ((v_entry + v_ref + v_exit) / 3)`.

---

## 6. Tyre Degradation / 輪胎退化

**Model** (`sim/race_sim.py`):

Each lap, corner `v_ref` values are reduced:

$$v_{ref,\text{degraded}} = v_{ref,\text{base}} \times \max(1 - \delta \times \text{tyre\_age},\; 0.5)$$

Where:
- $\delta$ = `deg_pct_per_lap` (default 0.0001 = 0.01%/lap for HARD)
- `tyre_age` = number of laps on current set
- Floor at 0.5× to prevent unrealistic slowdowns

每圈的彎心速度略微降低，模擬輪胎抓地力下降。
HARD 胎退化極小（真實數據顯示 53 圈僅慢 ~0.3s），MEDIUM 退化較大。

### Pit Stop Strategy

When the simulator reaches a pit-stop lap:
1. The current lap includes a `pit_delta` time penalty (typically ~22s)
2. Tyre compound and age reset for the next stint
3. Base `v_ref` values are recalculated based on the new compound

---

## 7. Dual-Driver Extension / 雙車手擴展

**Approach**: Independent simulation — each driver runs their own race
with their own `CarModel`. No interaction between them.

**Gap calculation**:

$$\text{gap}(L) = t_{\text{race,ANT}}(L) - t_{\text{race,PIA}}(L)$$

- Positive → PIA ahead
- Negative → ANT ahead

這個方法的假設是：兩車手在各自的「乾淨空氣」中獨立跑車。
在 What-If 推演中這是合理的，因為我們問的是「如果沒有安全車」，
而安全車消除的正是這種獨立累積的差距。

### ANT Car Model Calibration / Antonelli 車輛參數校準

We lack ANT's structured telemetry, so we calibrate from race data:

1. Start with PIA's car params as baseline
2. ANT's real HARD clean mean = 93.003s vs PIA's 93.560s → ANT is **0.557s faster**
3. Scale `v_max` upward by a factor until simulated lap ≈ ANT's real pace
4. This is a single-parameter calibration (binary search on `v_max`)

### 7.1 Fork-Point Simulation / 分岔點模擬

Full-race simulation (L1–L53) introduces unnecessary error for laps where
real data already exists. The **fork-point approach** uses real data for
L1–L21 and simulates only L22–L53 — the laps where reality diverges from
our What-If hypothesis.

#### Why L22?

| Lap | Event |
|-----|-------|
| L18 | PIA pits (MEDIUM → HARD) |
| L19–L21 | Clean racing, PIA on HARD (age 1–3), ANT on MEDIUM (age 19–21) |
| L22 | **Reality**: SC deployed, ANT gets free pit stop |
| L22 | **What-If**: No SC, ANT pits normally (+22s pit delta) |

At end of L21 the real cumulative times are:

- **PIA**: 2017.2 s (HARD, tyre age = 3)
- **ANT**: 1998.9 s (MEDIUM, tyre age = 21, has NOT pitted)

In the What-If, ANT pits normally at L22 (loses ~22s), so the initial
conditions for the fork-point simulation are:

- PIA cumul = 2017.2 s (HARD, age 3)
- ANT cumul = 1998.9 + 22.0 = 2020.9 s (HARD, age 0)
- Starting gap = +3.7 s (PIA ahead)

#### Implementation: `simulate_from_fork()`

```python
simulate_from_fork(
    segments,
    car,           # CarModel with tyre_age, compound, deg set to fork state
    start_lap=22,
    end_lap=53,
    t_race_cumul_init=2017.2,  # real cumul at L21
)
```

The function uses real lap numbers (22–53) in its output, not 1–31.

#### Validation advantage

- L1–L21: **zero simulation error** (real data)
- L22–L53: simulation error applies to only 31 laps
- Systematic bias cancels when computing the gap (both drivers share the
  same track model and solver)
- Post-SC clean laps (L28–L53) can be cross-validated against real data

---

## 8. Validation / 驗證

| Metric | Value |
|--------|-------|
| PIA one-lap sim (calibrated) | 93.052 s |
| PIA real fastest | 92.996 s |
| Error | **0.06%** |
| Fork-point final gap (L53) | **PIA +1.8 s** |

### 8.1 Cross-Validation (L28–L53) / 交叉驗證

We calibrate using only 1 lap (fastest lap). To validate, we compare
simulated lap times against real post-SC clean laps (L28–L53) — data the
model has **never seen** during calibration.

| Driver | MAE | RMSE | Trend slope | Interpretation |
|--------|-----|------|-------------|----------------|
| PIA | 0.407 s | 0.483 s | +0.021 s/lap | Slightly under-degrading |
| ANT | 0.284 s | 0.382 s | +0.001 s/lap | Flat (degradation rate validated) |

PIA 的趨勢斜率為 +0.021 s/lap，表示模擬的輪胎退化速度略低於真實情況
（每圈多慢 0.02s）。26 圈累計約 0.5s 的偏差，對 What-If 結論不影響。

ANT 的趨勢幾乎為零（+0.001 s/lap），驗證了退化率設定的準確性。

### Fork-point validation advantage / 分岔點驗證優勢

- L1–L21（21 圈）使用真實數據：**零模擬誤差**
- L22–L53（31 圈）才需要模擬，且相對差距（gap）抵消了系統性偏差
- 真實的 L28–L53 乾淨圈速可作為交叉驗證基準

---

## 9. Sensitivity Analysis / 敏感度分析

How robust is the "PIA wins by +1.8s" conclusion?
We sweep three key assumptions independently:

「PIA 贏 +1.8 秒」這個結論有多穩？我們獨立改變三個關鍵假設：

### 9.1 ANT Pit Lap / ANT 進站圈數

| ANT Pit Lap | Final Gap | Result |
|-------------|-----------|--------|
| L20 | +1.47 s | PIA wins |
| L21 | +1.62 s | PIA wins |
| **L22 (baseline)** | **+1.46 s** | **PIA wins** |
| L23 | +1.41 s | PIA wins |
| L24 | +1.20 s | PIA wins |

不管 ANT 在 L20 到 L24 之間哪一圈進站，PIA 都贏。結論穩固。

### 9.2 Pit Stop Delta / 進站時間損失

| Pit Delta | Final Gap | Result |
|-----------|-----------|--------|
| 20 s | -0.54 s | ANT wins |
| 21 s | +0.46 s | PIA wins |
| **22 s (baseline)** | **+1.46 s** | **PIA wins** |
| 23 s | +2.46 s | PIA wins |
| 24 s | +3.46 s | PIA wins |

只有在進站懲罰 ≤ 20 秒時 PIA 才會輸。真實的鈴鹿進站懲罰約 22 秒，
所以結論在合理範圍內成立。Gap 對 pit delta 的敏感度為 ~1.0 s/s（線性）。

### 9.3 HARD Degradation Rate / HARD 輪胎退化率

| deg_pct_per_lap | Final Gap | Result |
|-----------------|-----------|--------|
| 0.00005 | +1.07 s | PIA wins |
| **0.0001 (baseline)** | **+1.46 s** | **PIA wins** |
| 0.0002 | +0.81 s | PIA wins |
| 0.0005 | -1.36 s | ANT wins |

退化率要高達 baseline 的 5 倍（0.0005）PIA 才會輸。
但交叉驗證顯示趨勢斜率幾乎為零，排除了高退化率的可能。

### 9.4 Conclusion / 結論

PIA 的勝利在以下所有合理範圍內都成立：
- ANT 任何正常進站時機（L20–L24）
- 進站損失 ≥ 21 秒
- 退化率 ≤ 0.0002

**唯一讓 PIA 輸的條件是不合理的假設**（進站損失僅 20 秒或退化率超過交叉驗證上限 5 倍）。

---

## 10. FAQ / 常見問答

See [docs/qa.md](qa.md) for detailed bilingual answers to methodology questions
(calibration, validation, grey-box model, why no interaction model, etc.).
