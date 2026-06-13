# MSDM5058 Project II — How `ZHAO_report` Relates to the Assignment

This document maps the questions in **MSDM5058 Project II.pdf** (*Portfolio Management with Prediction Tools*) to what **ZHAO Fuxian** implemented in `ZHAO_report/` and the shared `data/` pipeline. Citations below use paths relative to the project root.

---

## 1. Data processing

**What the assignment asks (PDF §1).**  
Choose two stocks with **more than 4000** daily closes from a credible source; define the riskier stock \(S_\phi(t)\) and safer \(S_\epsilon(t)\); set “today” \(t=0\) so **past : future ≈ 3 : 1**; use **past** data through §6 for learning and **future** from §7 onward for testing; **plot** both price series and daily log returns  
\(X_i(t)=\ln(S_i(t)/S_i(t-1))\).

**step 1 — Data build (`data/build_dataset.py`, `data/manifest.json`, CSVs).**  
- **Logic:** `build_dataset.py` pulls **Yahoo Finance** OHLCV via `yfinance` for `AAL`, `DAL`, and extensions (`SPY`, `QQQ`, `XLK`, `VIX`, `^TNX`, etc.), writes per-ticker CSVs, optional FINRA short volume and a **PCR proxy** from VIX, then calls `write_shared_constants()` to fix the train/test cut.  
- **Analysis:** `manifest.json` records **4590** rows for `AAL`/`DAL` from **2008-01-02** to **2026-03-31**, satisfying the “>4000 days” requirement. The split uses `SPLIT_RATIO = 3/4` on the **intersection** of AAL/DAL calendars, producing `t0_date` **2021-09-02** in `data/shared_constants.json` (≈75% / 25% past/future).  
- **Constants:** `shared_constants.json` stores `epsilon` (half the past-window std of AAL log returns), `p0_est_past` (minimum-variance **two-asset** weight for AAL in the AAL–DAL pair on the past window), and correlation/volatility diagnostics — these support the **group-wide** specification even when the written report emphasises six assets.

**step 2 — Panel load and split (`data/shared.py`, `ZHAO_report/analysis.py`).**  
- **Logic:** `load_panel(extra=...)` aligns calendars and prefixes columns (`AAL_Close`, …). `train_test_split(panel)` splits at `t0_date` from JSON: `past = index <= t0`, `future = index > t0`.  
- **Analysis:** `analysis.py` builds **log returns** for the six-asset universe `["AAL","DAL","SPY","QQQ","XLK","TNX"]`, then locates `split_idx` so the return series matches the same \(t_0\). Past-window moments feed Markowitz/CAPM/BL/Kelly/CVaR/risk-parity; **future** returns feed the §10-style simulator.

**step 3 — Written report (`ZHAO_report/report.tex`, § “Data Preprocessing”).**  
- **Logic:** The report states the **six-asset** subset, log-return definition, \(t_0=\) 2021-09-02, and \(n_{\text{past}}=3439\), \(n_{\text{fut}}=1148\), with a **table** of annualised means and volatilities on the past window.  
- **Analysis:** This **fulfils the spirit** of reproducible preprocessing and the 3:1 split for the extended universe ZHAO chose. It does **not** include dedicated **figures** of \(S_\phi\), \(S_\epsilon\) and their log returns as bullet items in PDF §1; those plots are assignment deliverables that other group reports may carry (see root `README.md`).

---

## 2. Mean–variance analysis (two-asset \(p_J(t,h)\), \(S_J\), Sharpe)

**What the assignment asks (PDF §2).**  
Minimum-risk portfolio \(S_J = p_J S_\phi + (1-p_J)S_\epsilon\); infer \(p_J(t,h)\) using either **all past data** or the last **\(h\)** days (\(h=30,100,300,\infty\)); **plot** \(p_J(t,h)\), \(S_J(t,h)\), and Sharpe \(\gamma_J(t,h)\); **discuss** how performance differs with \(h\).

**step 1 — What ZHAO implements instead (`ZHAO_report/analysis.py`, `report.tex` § “Mean-Variance Analysis and the Markowitz Frontier”).**  
- **Logic:** On the **six-asset** past window, the script computes the **unconstrained** efficient frontier in closed form, **long-only** frontier via SLSQP, **tangency** and **global minimum-variance** weights, and plots `fig01_frontier.png`.  
- **Analysis:** This is **general mean–variance theory** on six assets, not the assignment’s **time-varying two-asset** \(p_J(t,h)\) curves for four values of \(h\). The **two-asset** minimum-risk weight **does** appear indirectly: `p0_est_past` in `shared_constants.json` is used only as a **benchmark** in the out-of-sample simulator (`V_p0` in `analysis.py`), not as a full §2 analysis with multiple \(h\) and Sharpe-of-\(S_J\) plots.

**Coverage verdict:** PDF §2 is **not** completed in the ZHAO write-up as specified; the work is a **different** (and more advanced) Markowitz treatment.

---

## 3. Moving averages and MACD (riskier stock only)

**What the assignment asks (PDF §3).**  
EMA for **\(w=30,100,300\)**; describe effect of \(w\); **compare** to SMA for the same \(w\); **MACD** and signal line; relate crossings to price behaviour.

**step 1 — Code and figure (`ZHAO_report/analysis.py` § “§11 Moving averages”, `report.tex` § “Moving Averages and the MACD Diagnostic”).**  
- **Logic:** Uses `sh.sma`, `sh.ema`, `sh.dema`, `sh.macd` on **AAL** close; produces `fig08_ma_macd.png`.  
- **Analysis:** Windows are **fixed at 20** for SMA/EMA/DEMA and **(12,26,9)** for MACD — **not** 30/100/300 as the PDF requires. The report text notes lag/whipsaw and MACD zero-line behaviour at a high level.  
**Coverage verdict:** **Partial** — methodology matches the topic (MA + MACD on the riskier name) but **does not** follow the required **\(w\in\{30,100,300\}\)** experiment grid.

---

## 4. Probability density (normal vs logistic)

**What the assignment asks (PDF §4).**  
Estimate normal \((\mu,\sigma)\) and logistic \((x^\*,b)\); **plot** integrated normal \(G(x)\) and logistic \(L(x)\) **on top of the empirical CDF** \(F(x)\); comment on reasonableness.

**step 1 — Code vs narrative (`ZHAO_report/analysis.py` § “§9 PDF + Bayes”, `report.tex` § “Probability Density and Bayes Detector”).**  
- **Logic:** `analysis.py` fits **normal** MLE moments and **logistic** via `scipy.stats.logistic.fit` on **past AAL** returns; stores log-likelihoods in `results.json` under `pdf_aal`.  
- **Analysis:** The **report** states parameter values, log-likelihoods, and \(\Delta\)AIC favouring the logistic; it **does not** include the requested **overlay plot** of \(G(x)\) and \(L(x)\) on \(F(x)\) in the LaTeX source reviewed.

**Coverage verdict:** **Parameters reported**; **CDF overlay figure** not present in `report.tex`.

---

## 4.1 Digitisation and conditional distributions

**What the assignment asks (PDF §4.1).**  
Digitise \(X(t)\) into **D/U/H** with threshold \(\varepsilon\); plot **conditional CDFs** \(F(x\mid Y(t+1)=y)\) for \(y\in\{D,U,H\}\); derive/plot **conditional PDFs** \(f_y(x)\).

**step 1 — Code (`data/shared.py`, `ZHAO_report/analysis.py`).**  
- **Logic:** `digitize(X, eps)` implements D/U/H; `epsilon` comes from `shared_constants.json`. `fit_conditional_normals` fits **univariate normal** \((\mu,\sigma)\) to \(X(t)\) given **next-day** label \(Y(t+1)\). The script grids \(x\), evaluates \(\log q_y + \log f_y(x)\), and records **Bayes decision boundaries** in `results["bayes_boundaries"]`.  
- **Analysis:** This is the **computational core** for a Bayes rule (§5) but **no tables/figures** for conditional CDFs/PDFs appear in `report.tex`.

**Coverage verdict:** **Implemented in code**; **not documented** in the ZHAO PDF as §4.1 asks.

---

## 5. Bayes detector

**What the assignment asks (PDF §5).**  
Priors \(P(Y(t+1)=y)\); posterior / decision \(y^\*(x)=\arg\max_y q_y f_y(x)\); **critical** \(x\) where \(y^\*\) changes; **mark** on PDF graph.

**step 1 — Code vs report.**  
- **Logic:** Empirical priors from digitised series; class-conditional normals; argmax over a grid → `bayes_boundaries` in `results.json`.  
- **Analysis:** The **report** mentions Bayes outputs only as **motivation** for the Black–Litterman view (AAL vs DAL), not as a full §5 with priors table, \(y^\*(x)\) plot, or boundary markers on PDF curves.

**Coverage verdict:** **Numerical support exists** in `analysis.py`; **write-up does not** complete PDF §5 as a standalone section.

---

## 6. Association rules (and §6.1 usefulness)

**What the assignment asks (PDF §6–6.1).**  
For **\(k=5\)**, mine rules \(Y\) patterns → next-day \(Y\); top-10 **support** and **confidence** tables; top-10 by **geometric mean** \(\sqrt{sc}\) and by **RPF**; discuss \(\lambda\) in \(u\propto s^{1-\lambda}c^\lambda\) and whether rules look useful.

**step 1 — `ZHAO_report/analysis.py` and `report.tex`.**  
- **Logic / analysis:** **No** association-rule mining appears in `ZHAO_report/analysis.py` or `report.tex`. (`data/shared.py` only lists complexity notes for association rules in `ALGO_COMPLEXITIES` for possible use by **other** reports.)

**Coverage verdict:** **Not addressed** in ZHAO’s deliverable.

---

## 7. Portfolio with one stock and money

**What the assignment asks (PDF §7).**  
State **buy/sell/hold** rules tied to MACD, Bayes, and association rules; simulate \(M,N,S\) with **greed** \(g\), daily compounding at \(r=0.001\%\); plot \(V(t,g)\) for **two** \(g\) values vs minimum-risk benchmark; **Alice / Bob / Charlie** greed-mixing discussion.

**step 1 — What ZHAO does instead (`ZHAO_report/analysis.py` § “§10”, `report.tex` § “Trading Simulator”).**  
- **Logic:** **Six-asset** weights (tangency, risk-parity, BL, half-Kelly) with a **60-day rolling vol-target** scaler \(s_t=\min(2,\sigma^\*/\sigma_{60,t})\), \(\sigma^\*=15\%\) annualised; idle notional earns **daily** `rf_d = 0.001/100`. Benchmarks: buy-and-hold AAL and **two-asset min-risk** \(p_0\) blend.  
- **Analysis:** This is a **risk-managed multi-asset** backtest, **not** the PDF’s single-stock-plus-cash mechanics, explicit **\(g_{\text{aggressive}}\)** / **\(g_{\text{conservative}}\)** trade fractions, or the Alice/Bob/Charlie narrative. MACD is referenced as a **diagnostic** for signals elsewhere, not wired as the §7 rule engine here.

**Coverage verdict:** **Thematically related** (out-of-sample “does the toolkit work?”) but **not** a direct answer to PDF §7.

---

## 8. Portfolio with two stocks (and §8.1 efficient frontier trading)

**What the assignment asks (PDF §8–8.1).**  
Trade **riskier for safer** with the **same** days as §7; initialise from **\(p_J\)** min-risk; plot greedy vs \(V_J\); §8.1: adjust risk along the frontier between \(\sigma_J\) and \(\sigma_\phi\); plot \(p(t)\) vs \(p_J(t)\); replot \(V\) under the new rule.

**step 1 — ZHAO deliverable.**  
- **Logic / analysis:** **No** two-stock barter simulation, **no** \(p(t)\) vs \(p_J(t)\) figure, and **no** §8.1 risk-step trading scheme appear in `ZHAO_report/analysis.py` or `report.tex`.

**Coverage verdict:** **Not addressed.**

---

## 9. Portfolio with two stocks and money

**What the assignment asks (PDF §9).**  
Start with **cash only**; trade **both** stocks with **independent** daily decisions; order of trades matters; combine techniques; plot \(V(t,g_\text{aggressive})\) and \(V(t,g_\text{conservative})\); comment.

**step 1 — ZHAO deliverable.**  
- **Logic / analysis:** The ZHAO simulator starts from **\$100,000** and **implicitly** holds a **vector** of assets via weights, which is closer in spirit to a **long-only or unconstrained weight vector** backtest than to the §9 **sequential cash + two names** story with **per-stock** signals. The report **does not** frame results as PDF §9.

**Coverage verdict:** **Not** structured as §9; closest artefact is the **six-asset** OOS P\&L comparison.

---

## Additional material ZHAO adds (beyond the nine PDF sections)

The report and script substantially develop **CAPM**, **Black–Litterman**, **Kelly / half-Kelly**, **VaR/ES (CVaR)**, **risk parity**, **algorithmic complexity**, and **vol targeting** — valuable extensions but **not** substitutes for missing PDF items (especially §2 as stated, §4.1–§6 plots/tables, and §7–§9 narratives).

---

## Summary table

| PDF section | Topic | In `ZHAO_report`? |
|------------|--------|-------------------|
| §1 | Two stocks, >4000 days, 3:1 split, plots of prices/returns | Data & split **yes**; **price/return figures for the pair** not in `report.tex` |
| §2 | \(p_J(t,h)\), \(S_J\), Sharpe for \(h\in\{30,100,300,\infty\}\) | **No** (six-asset static MV instead; \(p_0\) benchmark only) |
| §3 | EMA/SMA \(w=30,100,300\), MACD | **Partial** (MA/MACD on AAL with **other** windows) |
| §4 | Normal/logistic + CDF overlays | **Params yes**; **CDF overlay plots** not in report |
| §4.1 | Digitisation, conditional CDF/PDF | **In code**; **not** in report |
| §5 | Full Bayes detector write-up | **Boundaries in JSON**; **not** a full §5 in report |
| §6–6.1 | Association rules, usefulness | **Absent** |
| §7 | One stock + cash, \(g\), Alice/Bob/Charlie | **Absent** as specified; **vol-target** multi-asset sim instead |
| §8–8.1 | Two-stock barter + efficient frontier trading | **Absent** |
| §9 | Cash + two stocks, combined scheme | **Absent** as specified |

---

## File reference quick list

| Role | Path |
|------|------|
| Assignment | `MSDM5058 Project II.pdf` |
| Downloader / constants builder | `data/build_dataset.py` |
| Shared loaders, indicators, Bayes helpers | `data/shared.py` |
| Frozen split & \(\varepsilon\), \(p_0\) | `data/shared_constants.json` |
| ZHAO computations & figures | `ZHAO_report/analysis.py` |
| ZHAO PDF (LNCS) | `ZHAO_report/report.tex` |
| Numeric dump | `ZHAO_report/results.json` |
| Group overview | `README.md` |

---

*Generated from the PDF introduction, `data/*`, and all files under `ZHAO_report/` (`report.tex`, `analysis.py`, `results.json`, `refs.bib`).*
