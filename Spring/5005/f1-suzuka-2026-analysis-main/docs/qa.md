# FAQ: Methodology Q&A / 方法論常見問答

This document pre-answers likely questions from teammates and professors.

---

## Q1: How is acceleration calculated? / 加速度怎麼算的？

**Initial estimate**: From the structured telemetry CSV, we compute `a = ΔSpeed / Δt` for consecutive rows where `Driving_State == "Full Acceleration"`. We take the **median** of these samples, split into two speed bands (< 200 km/h and ≥ 200 km/h).

**初始估計**：從結構化遙測 CSV 中，取相鄰兩筆 `Driving_State == "Full Acceleration"` 的資料算 `a = ΔSpeed / Δt`，分成低速（< 200 km/h）和高速（≥ 200 km/h）兩段，各取中位數。

**Problem**: The median is conservative (ignores peak performance) and the model lacks DRS, gradient, and non-linear drag. Result: 95.251s simulated vs 92.996s real = **+2.4% overestimate**.

**問題**：中位數偏保守（忽略峰值表現），且模型缺少 DRS、坡度、非線性空氣阻力。結果：模擬 95.251s vs 真實 92.996s = 高估 2.4%。

**Solution**: Whole-lap calibration (see Q2).

---

## Q2: What is "whole-lap calibration"? / 什麼是「整圈校準」？

We keep the physics engine (`lap_solver.py`) unchanged and **only adjust the input parameters** to match reality.

我們不改物理引擎（`lap_solver.py`），**只調整輸入參數**讓模擬結果對齊真實數據。

**Method**: Multiply all three params (`a_accel_low`, `a_accel_high`, `a_brake`) by a single scale factor. Binary search finds the factor that makes `solve_lap()` output match the real fastest lap (92.996s).

**方法**：把三個參數同時乘以一個倍率。用 binary search 找到讓 `solve_lap()` 輸出等於真實最快圈（92.996s）的倍率。

**Result**: Scale factor = 1.2538 (all params +25.4%). Sim lap = 93.052s (**0.06% error**).

This is a standard **grey-box model** approach: the physics (white-box) defines the behavior; data calibration (black-box) fills in the unknown limit parameters.

這是標準的**灰箱模型**：物理公式（白箱）定義行為，數據校準（黑箱）填補未知的極限參數。

---

## Q3: Why do calibrated values differ from raw telemetry? / 為什麼校準值跟原始遙測不同？

The calibrated values are **effective/lumped parameters**. They absorb all the effects our simplified 1D model does NOT explicitly model:

校準後的值是**等效參數（Lumped Parameters）**，它們吸收了我們簡化的 1D 模型沒有顯式建模的所有效應：

- DRS (adds ~0.3–0.5s/lap on straights)
- Track gradient (Suzuka has elevation changes)
- Non-linear aerodynamic drag (our model uses constant acceleration per band)
- Tyre grip variations within a lap

This is why the calibrated `a_accel_low` (15.96 m/s²) is higher than the raw telemetry median (12.73 m/s²) — it's not the "real" acceleration, it's the value that makes our simplified model produce correct lap times.

這就是為什麼校準後的 `a_accel_low`（15.96）比遙測中位數（12.73）高 — 它不是「真實」加速度，而是讓簡化模型產出正確圈速的等效值。

---

## Q4: Is the calibration solution unique? / 校準的解是唯一的嗎？

**No.** We have 1 target (lap time) and 3 free parameters. Mathematically, infinite (a_low, a_high, a_brake) combinations can produce 92.996s.

**不是。** 1 個目標、3 個自由參數，理論上有無限多組合能跑出 92.996s。

**Mitigation**: Uniform scaling preserves the **ratio** between the three parameters as estimated from telemetry. This is the most physically consistent choice — if all three were underestimated by the same proportion (due to missing DRS/drag effects), scaling them equally is the correct fix.

**緩解措施**：等比例縮放保持了三個參數之間的**比例關係**（來自遙測估計）。如果三者的低估幅度一致（都因為缺少 DRS/阻力效應），等比例放大就是正確的修正。

**Guard-rail**: Each parameter must stay within ±30% of the initial telemetry estimate.

---

## Q5: How do you validate? / 你怎麼驗證？

**Train/test split**:

- **Training set (1 lap)**: Real fastest lap (92.996s) — used for calibration
- **Test set (26 laps)**: Real clean lap times from L28–L53 (post-SC) — **never seen during calibration**

**訓練/測試分離**：

- **訓練集（1 圈）**：真實最快圈（92.996s）— 用於校準
- **測試集（26 圈）**：L28–L53 的真實乾淨圈速（SC 結束後）— **校準過程完全沒有使用**

**Results**:

| Driver | MAE | RMSE | Trend |
|--------|-----|------|-------|
| PIA | 0.407 s/lap | 0.483 s | +0.021 s/lap (slightly under-degrading) |
| ANT | 0.284 s/lap | 0.382 s | +0.001 s/lap (flat, degradation validated) |

---

## Q6: Is the tyre degradation rate correct? / 輪胎退化率正確嗎？

We set `deg_pct_per_lap = 0.0001` for HARD tyres (0.01% per lap reduction in corner speeds).

**Validation**: If the rate is wrong, error will **grow systematically** over 26 laps. The cross-validation trend slope tells us:

**驗證**：如果退化率錯誤，誤差會隨圈數**系統性增長**。交叉驗證的趨勢斜率告訴我們：

- **PIA**: +0.021 s/lap — sim degrading slightly too slowly. Over 26 laps, this accumulates to ~0.5s. Minor.
- **ANT**: +0.001 s/lap — essentially flat. Degradation rate validated.

The slight positive trend for PIA suggests the real HARD compound degrades marginally faster than 0.01%/lap, but not enough to affect the What-If conclusion (+1.8s gap >> 0.5s accumulated error).

---

## Q7: Why fork-point instead of full-race simulation? / 為什麼用分岔點而不是模擬全場？

We have **real data** for L1–L21. Simulating these laps would only introduce unnecessary error.

L1–L21 有**真實數據**。模擬這些圈只會引入不必要的誤差。

The fork point (L22) is where reality diverges from our What-If hypothesis:
- **Reality**: Safety Car deployed, ANT gets free pit stop
- **What-If**: No SC, ANT pits normally (+22s)

By using real data for L1–L21, we get **zero simulation error** for 40% of the race.

用真實數據處理 L1–L21，比賽的 40% 有**零模擬誤差**。

---

## Q8: Why no driver interaction model? / 為什麼沒有車手互動模型？

The gap between PIA and ANT in our What-If simulation ranges from **+1.76s to +3.76s** across L22–L53. This is **always above the 1.5s DRS/interaction threshold**.

模擬中 PIA 和 ANT 的差距在 L22–L53 間為 **+1.76s 到 +3.76s**，**始終超過 1.5s 的 DRS/互動門檻**。

Since the cars are never close enough for slipstream, dirty air, or defensive driving effects to apply, adding an interaction model would:
1. Not change the conclusion
2. Introduce parameters we cannot validate
3. Risk being questioned by the professor ("How did you calibrate the interaction strength?")

The `race_log` DataFrame already reserves `interaction` and `interaction_delta_s` columns for future extension. Sensitivity analysis confirms that even under aggressive assumptions, the gap stays above the interaction threshold.

`race_log` DataFrame 已預留 `interaction` 和 `interaction_delta_s` 欄位供未來擴展。敏感度分析證實即使在較極端的假設下，差距仍高於互動門檻。
