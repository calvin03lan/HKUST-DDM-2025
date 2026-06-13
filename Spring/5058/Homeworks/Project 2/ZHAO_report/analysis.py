"""
ZHAO_report/analysis.py
========================
Modern Portfolio Theory and Risk-Adjusted Optimization.

Companion report: ZHAO Fuxian --- MSDM5058 Project II.

Direction (per group plan):
  - 6-asset universe (AAL, DAL, SPY, QQQ, XLK, ^TNX) for the
    Markowitz analysis.
  - CAPM regression vs SPY.
  - Black-Litterman with Bayes-detector views.
  - Kelly-criterion sizing, CVaR/ES at 5% & 1%, risk-parity allocation.
  - Vol-target portfolio (15% annualised) for the §10 simulator.

The script imports the shared dataset and helpers from data/shared.py
so all four group reports start from identical inputs.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
warnings.filterwarnings("ignore")

from data import shared as sh  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
np.random.seed(sh.SEED)

# ---------------------------------------------------------------------------
# Data load --- six-asset universe
# ---------------------------------------------------------------------------
print("=" * 70)
print("ZHAO_report — Modern Portfolio Theory & Risk")
print("=" * 70)

panel = sh.load_panel(extra=("SPY", "QQQ", "XLK", "TNX", "VIX"))
past, future, t0 = sh.train_test_split(panel)
n_past = len(past)
print(f"  panel: {len(panel)} trading days, t0={t0.date()}, past={n_past}")

UNIVERSE = ["AAL", "DAL", "SPY", "QQQ", "XLK", "TNX"]
prices = pd.DataFrame({tk: panel[f"{tk}_Close"] for tk in UNIVERSE})
prices = prices.dropna()
dates = prices.index
returns = np.log(prices.values[1:] / prices.values[:-1])
n = len(returns)

# split aligned to project's past/future cut
split_idx = int(np.searchsorted(dates[1:], t0))
R_past = returns[:split_idx]
R_fut = returns[split_idx:]
n_pr = len(R_past)
n_fu = len(R_fut)
fut_dates = dates[split_idx + 1:]
print(f"  past returns: {n_pr}, future: {n_fu}")

results: dict = {
    "universe": UNIVERSE,
    "n_total": n, "split_idx": split_idx,
    "split_date": str(t0.date()),
    "date_start": str(dates[0].date()),
    "date_end": str(dates[-1].date()),
}

mu = R_past.mean(axis=0) * 252
Sigma = np.cov(R_past, rowvar=False, ddof=1) * 252
results["mu_ann"] = dict(zip(UNIVERSE, mu.tolist()))
results["sigma_ann"] = dict(zip(UNIVERSE, np.sqrt(np.diag(Sigma)).tolist()))

# ---------------------------------------------------------------------------
# §3 Markowitz frontier (with and without shorting)
# ---------------------------------------------------------------------------
print("[§3] Markowitz frontier (6-asset)")
inv_S = np.linalg.inv(Sigma)
ones = np.ones(len(UNIVERSE))
A = ones @ inv_S @ ones
B = ones @ inv_S @ mu
C = mu @ inv_S @ mu
det = A * C - B * B

mu_grid = np.linspace(mu.min() - 0.05, mu.max() + 0.05, 80)
front_var_unc = []
front_w_unc = []
for mt in mu_grid:
    lam = (C - mt * B) / det
    g = (mt * A - B) / det
    w = lam * (inv_S @ ones) + g * (inv_S @ mu)
    front_var_unc.append(w @ Sigma @ w)
    front_w_unc.append(w)
front_sd_unc = np.sqrt(np.array(front_var_unc))


def constrained_frontier(mu_grid_, mu_, Sigma_):
    out_sd, out_w = [], []
    n_a = len(mu_)
    for mt in mu_grid_:
        cons = ({"type": "eq", "fun": lambda w_: w_.sum() - 1},
                {"type": "eq", "fun": lambda w_: w_ @ mu_ - mt})
        bounds = [(0, 1)] * n_a
        x0 = np.ones(n_a) / n_a
        res = minimize(lambda w_: w_ @ Sigma_ @ w_,
                       x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            out_sd.append(np.sqrt(res.fun))
            out_w.append(res.x)
        else:
            out_sd.append(np.nan); out_w.append(np.full(n_a, np.nan))
    return np.array(out_sd), np.array(out_w)


cons_grid = np.linspace(mu.min() + 1e-4, mu.max() - 1e-4, 60)
front_sd_lo, front_w_lo = constrained_frontier(cons_grid, mu, Sigma)

# Tangency portfolio (max Sharpe with rf=0)
rf_ann = 0.001 / 100 * 252
sharpes = (mu_grid - rf_ann) / front_sd_unc
i_max = int(np.nanargmax(sharpes))
w_tan = front_w_unc[i_max]
mu_tan, sd_tan = mu_grid[i_max], front_sd_unc[i_max]

# Minimum-variance portfolio
i_min = int(np.argmin(front_sd_unc))
w_mv = front_w_unc[i_min]
mu_mv, sd_mv = mu_grid[i_min], front_sd_unc[i_min]

results["mv_portfolio"] = {
    "weights": dict(zip(UNIVERSE, w_mv.tolist())),
    "ann_ret": float(mu_mv), "ann_vol": float(sd_mv),
}
results["tangency_portfolio"] = {
    "weights": dict(zip(UNIVERSE, w_tan.tolist())),
    "ann_ret": float(mu_tan), "ann_vol": float(sd_tan),
    "sharpe": float((mu_tan - rf_ann) / sd_tan),
}

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(front_sd_unc, mu_grid, color="C0", label="Frontier (no constraints)", lw=1.2)
ax.plot(front_sd_lo, cons_grid, color="C1", label="Frontier (long-only)", lw=1.2)
sd_assets = np.sqrt(np.diag(Sigma))
ax.scatter(sd_assets, mu, c="C3", s=50, zorder=5)
for i, lbl in enumerate(UNIVERSE):
    ax.annotate(lbl, (sd_assets[i], mu[i]), fontsize=9)
ax.scatter(sd_tan, mu_tan, c="green", marker="*", s=200, zorder=6,
           label=f"Tangency, Sharpe={results['tangency_portfolio']['sharpe']:.2f}")
ax.scatter(sd_mv, mu_mv, c="purple", marker="D", s=80, zorder=6,
           label="Min-variance")
# CAL
sd_cal = np.linspace(0, sd_assets.max() * 1.5, 50)
ax.plot(sd_cal, rf_ann + (mu_tan - rf_ann) / sd_tan * sd_cal,
        ls="--", color="green", lw=0.8, label="CAL")
ax.set_xlabel("Annualised std-dev")
ax.set_ylabel("Annualised mean")
ax.set_title("Markowitz frontier on 6-asset universe (past window)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "fig01_frontier.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §4 CAPM regression
# ---------------------------------------------------------------------------
print("[§4] CAPM regression vs SPY")
spy_idx = UNIVERSE.index("SPY")
rs_market = R_past[:, spy_idx] - rf_ann / 252
capm = {}
for i, tk in enumerate(UNIVERSE):
    if tk == "SPY":
        continue
    rs_asset = R_past[:, i] - rf_ann / 252
    res = stats.linregress(rs_market, rs_asset)
    capm[tk] = {
        "alpha_daily": float(res.intercept),
        "alpha_ann": float(res.intercept * 252),
        "beta": float(res.slope),
        "r_squared": float(res.rvalue ** 2),
        "p_alpha": float(res.pvalue),
    }
results["capm"] = capm

fig, ax = plt.subplots(figsize=(9, 5))
betas = [capm[t]["beta"] for t in UNIVERSE if t != "SPY"]
alphas_ann = [capm[t]["alpha_ann"] * 100 for t in UNIVERSE if t != "SPY"]
labels_no_spy = [t for t in UNIVERSE if t != "SPY"]
ax.scatter(betas, alphas_ann, c="C0", s=80)
for i, lbl in enumerate(labels_no_spy):
    ax.annotate(lbl, (betas[i], alphas_ann[i]), fontsize=10,
                xytext=(5, 5), textcoords="offset points")
ax.axvline(1.0, color="grey", lw=0.5, ls="--")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_xlabel(r"$\beta$ vs SPY")
ax.set_ylabel(r"Annualised $\alpha$ (\%)")
ax.set_title("CAPM regression: 5-asset alpha-beta plot (past window)")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "fig02_capm.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §5 Black-Litterman with Bayes-detector views
# ---------------------------------------------------------------------------
print("[§5] Black-Litterman")
# Implied equilibrium returns: pi = lambda * Sigma * w_market
# Use SPY-equivalent market weights (uniform proxy) and lambda=2.5
lam = 2.5
w_mkt = np.array([0.15, 0.15, 0.30, 0.20, 0.15, 0.05])
pi_eq = lam * Sigma @ w_mkt

# Views: AAL outperform DAL by 5% (motivated by Bayes detector
# in HU/HUANG reports favouring AAL upticks)
P = np.zeros((1, len(UNIVERSE)))
P[0, UNIVERSE.index("AAL")] = 1.0
P[0, UNIVERSE.index("DAL")] = -1.0
Q = np.array([0.05])  # 5% annualised excess view
Omega = np.diag([0.0001])  # tight confidence
tau = 0.05

A_bl = np.linalg.inv(np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P)
b_bl = np.linalg.inv(tau * Sigma) @ pi_eq + P.T @ np.linalg.inv(Omega) @ Q
mu_bl = A_bl @ b_bl

# BL-implied weights (max Sharpe with mu_bl)
w_bl = np.linalg.solve(lam * Sigma, mu_bl)
w_bl = w_bl / w_bl.sum()

results["black_litterman"] = {
    "pi_eq": dict(zip(UNIVERSE, pi_eq.tolist())),
    "mu_bl": dict(zip(UNIVERSE, mu_bl.tolist())),
    "w_bl": dict(zip(UNIVERSE, w_bl.tolist())),
}

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(UNIVERSE))
width = 0.27
ax.bar(x - width, pi_eq, width, label=r"$\pi_{eq}$", color="grey")
ax.bar(x, mu_bl, width, label=r"$\mu_{BL}$", color="C0")
ax.bar(x + width, mu, width, label=r"$\hat\mu_{hist}$", color="C3")
ax.set_xticks(x); ax.set_xticklabels(UNIVERSE)
ax.set_ylabel("Annualised return")
ax.set_title("Black-Litterman implied returns vs historical")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig03_bl.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §6 Kelly criterion sizing
# ---------------------------------------------------------------------------
print("[§6] Kelly criterion")
# For multiple risky assets: w_kelly = Sigma^-1 * (mu - rf*1)
mu_excess = mu - rf_ann
w_kelly_full = inv_S @ mu_excess
w_kelly = w_kelly_full / max(np.abs(w_kelly_full).sum(), 1.0)  # normalise by L1
w_kelly_half = 0.5 * w_kelly
results["kelly"] = {
    "w_full": dict(zip(UNIVERSE, w_kelly_full.tolist())),
    "w_normalised": dict(zip(UNIVERSE, w_kelly.tolist())),
    "w_half_kelly": dict(zip(UNIVERSE, w_kelly_half.tolist())),
}

# ---------------------------------------------------------------------------
# §7 CVaR / ES at 5% and 1%
# ---------------------------------------------------------------------------
print("[§7] CVaR / ES on past returns")
cvar_table = {}
for i, tk in enumerate(UNIVERSE):
    r = R_past[:, i]
    var5 = float(np.quantile(r, 0.05))
    var1 = float(np.quantile(r, 0.01))
    es5 = float(np.mean(r[r <= var5]))
    es1 = float(np.mean(r[r <= var1]))
    cvar_table[tk] = {
        "VaR_5": var5, "ES_5": es5, "VaR_1": var1, "ES_1": es1,
    }
results["cvar"] = cvar_table

fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(UNIVERSE))
width = 0.2
ax.bar(xs - 1.5*width, [cvar_table[t]["VaR_5"] for t in UNIVERSE], width,
       label="VaR 5%", color="C0")
ax.bar(xs - 0.5*width, [cvar_table[t]["ES_5"] for t in UNIVERSE], width,
       label="ES 5%", color="C0", alpha=0.5)
ax.bar(xs + 0.5*width, [cvar_table[t]["VaR_1"] for t in UNIVERSE], width,
       label="VaR 1%", color="C3")
ax.bar(xs + 1.5*width, [cvar_table[t]["ES_1"] for t in UNIVERSE], width,
       label="ES 1%", color="C3", alpha=0.5)
ax.set_xticks(xs); ax.set_xticklabels(UNIVERSE)
ax.set_ylabel("Daily return")
ax.set_title("VaR / Expected Shortfall (past window)")
ax.axhline(0, color="k", lw=0.5)
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig04_cvar.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §8 Risk-parity allocation
# ---------------------------------------------------------------------------
print("[§8] Risk-parity allocation")


def risk_parity(Sigma_):
    n_a = Sigma_.shape[0]
    target = 1.0 / n_a

    def loss(w):
        w = np.abs(w)
        w /= w.sum()
        port_var = w @ Sigma_ @ w
        marg = Sigma_ @ w
        rc = w * marg / np.sqrt(port_var + 1e-18)
        rc /= rc.sum()
        return float(np.sum((rc - target) ** 2))

    x0 = np.ones(n_a) / n_a
    res = minimize(loss, x0, method="SLSQP",
                   bounds=[(1e-6, 1)] * n_a,
                   constraints={"type": "eq", "fun": lambda w_: w_.sum() - 1})
    w_ = np.abs(res.x); w_ /= w_.sum()
    return w_


w_rp = risk_parity(Sigma)
rc_rp = (w_rp * (Sigma @ w_rp)) / np.sqrt(w_rp @ Sigma @ w_rp + 1e-18)
rc_rp = rc_rp / rc_rp.sum()
results["risk_parity"] = {
    "weights": dict(zip(UNIVERSE, w_rp.tolist())),
    "risk_contrib": dict(zip(UNIVERSE, rc_rp.tolist())),
    "ann_vol": float(np.sqrt(w_rp @ Sigma @ w_rp)),
    "ann_ret": float(w_rp @ mu),
}

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
xs = np.arange(len(UNIVERSE))
ax[0].bar(xs, w_rp * 100, color="C0")
ax[0].set_xticks(xs); ax[0].set_xticklabels(UNIVERSE)
ax[0].set_ylabel("Weight (%)")
ax[0].set_title("Risk-parity weights")
ax[1].bar(xs, rc_rp * 100, color="C2")
ax[1].set_xticks(xs); ax[1].set_xticklabels(UNIVERSE)
ax[1].set_ylabel("Risk contribution (%)")
ax[1].set_title("Marginal risk contribution (≈uniform)")
plt.tight_layout(); plt.savefig(FIG / "fig05_riskparity.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §9 PDF + Bayes detector
# ---------------------------------------------------------------------------
print("[§9] PDF + Bayes detector for AAL/DAL pair")
x1 = R_past[:, UNIVERSE.index("AAL")]
x2 = R_past[:, UNIVERSE.index("DAL")]
mu_n, sigma_n = float(np.mean(x1)), float(np.std(x1, ddof=1))
ll_n = float(np.sum(stats.norm.logpdf(x1, mu_n, sigma_n)))
loc_l, scale_l = stats.logistic.fit(x1)
ll_l = float(np.sum(stats.logistic.logpdf(x1, loc_l, scale_l)))
results["pdf_aal"] = {
    "normal": {"params": [mu_n, sigma_n], "logL": ll_n},
    "logistic": {"params": [float(loc_l), float(scale_l)], "logL": ll_l},
}

eps = sh.load_constants()["epsilon"]
Y_past_aal = sh.digitize(x1, eps)
priors = {y: float(np.mean(Y_past_aal == y)) for y in ("D", "U", "H")}
cond_norm = sh.fit_conditional_normals(x1, Y_past_aal)
xs_ = np.linspace(-0.20, 0.20, 4001)
post = np.column_stack([
    np.log(priors[y]) + stats.norm.logpdf(xs_, *cond_norm[y])
    for y in ("D", "U", "H")
])
pred = np.array(["DUH"[i] for i in np.argmax(post, axis=1)])
boundaries = []
for i in range(1, len(xs_)):
    if pred[i] != pred[i - 1]:
        boundaries.append((float(0.5 * (xs_[i] + xs_[i - 1])),
                           pred[i - 1], pred[i]))
results["bayes_boundaries"] = boundaries
print(f"  Bayes boundaries: {len(boundaries)}")

# ---------------------------------------------------------------------------
# §10 Trading simulators with vol targeting / risk-adjusted
# ---------------------------------------------------------------------------
print("[§10] Vol-target portfolio simulator")
TARGET_VOL = 0.15  # annualised


def sim_vol_target(R_fut_, w, target_vol=TARGET_VOL, V0=1e5):
    """Daily vol-target portfolio: scale gross exposure to keep
    annualised realised vol at target_vol on a 60-day rolling window."""
    n_ = len(R_fut_)
    V = np.empty(n_ + 1); V[0] = V0
    rolling_vol = np.std(R_fut_[:1] @ w, ddof=1)
    history = []
    rf_d = 0.001 / 100
    for t in range(n_):
        history.append(float(R_fut_[t] @ w))
        if len(history) >= 30:
            rv = float(np.std(history[-60:], ddof=1) * np.sqrt(252))
            scale = min(2.0, target_vol / max(rv, 1e-6))
        else:
            scale = 1.0
        ret = float(scale * (R_fut_[t] @ w) + (1 - scale) * rf_d)
        V[t + 1] = V[t] * np.exp(ret)
    return V


# Strategy A: tangency weights with vol targeting
V_tan = sim_vol_target(R_fut, w_tan)
# Strategy B: risk-parity with vol targeting
V_rp = sim_vol_target(R_fut, w_rp)
# Strategy C: BL-weighted with vol targeting
V_bl = sim_vol_target(R_fut, w_bl)
# Strategy D: half-Kelly (no vol target)
V_kelly = np.empty(n_fu + 1); V_kelly[0] = 1e5
for t in range(n_fu):
    rt = float(R_fut[t] @ w_kelly_half)
    V_kelly[t + 1] = V_kelly[t] * np.exp(rt)
# Buy & hold AAL
V_bh = 1e5 * np.exp(np.cumsum(np.concatenate([[0], R_fut[:, UNIVERSE.index("AAL")]])))
# Min-risk: project's classical p_0 between AAL/DAL
p0 = sh.load_constants()["p0_est_past"]
V_p0 = 1e5 * np.exp(np.cumsum(
    np.concatenate([[0], p0 * R_fut[:, 0] + (1 - p0) * R_fut[:, 1]])
))

m_s10 = {
    "Tangency (vol-target 15%)":   sh.compute_metrics(V_tan),
    "Risk-parity (vol-target)":    sh.compute_metrics(V_rp),
    "Black-Litterman (vol-target)": sh.compute_metrics(V_bl),
    "Half-Kelly (no target)":      sh.compute_metrics(V_kelly),
    "Buy & Hold AAL":             sh.compute_metrics(V_bh),
    "Min-risk p0 (AAL/DAL)":     sh.compute_metrics(V_p0),
}
results["s10_metrics"] = m_s10

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot([t0] + list(fut_dates[:n_fu]), V_tan, label="Tangency (vol-target)")
ax.plot([t0] + list(fut_dates[:n_fu]), V_rp, label="Risk-parity (vol-target)")
ax.plot([t0] + list(fut_dates[:n_fu]), V_bl, label="Black-Litterman (vol-target)")
ax.plot([t0] + list(fut_dates[:n_fu]), V_kelly, label="Half-Kelly")
ax.plot([t0] + list(fut_dates[:n_fu]), V_bh, label="Buy & Hold AAL", color="grey")
ax.plot([t0] + list(fut_dates[:n_fu]), V_p0, label="Min-risk $p_0$", color="black", lw=0.8)
ax.set_title("Out-of-sample equity curves (six-asset MPT vs benchmarks)")
ax.legend(fontsize=8)
ax.set_ylabel("Portfolio value")
plt.tight_layout(); plt.savefig(FIG / "fig06_s10.png", dpi=150); plt.close()

# Drawdown panel
def drawdown(V):
    peak = np.maximum.accumulate(V)
    return V / peak - 1


fig, ax = plt.subplots(figsize=(11, 4))
for V_, lbl in [(V_tan, "Tangency"), (V_rp, "Risk-parity"),
                (V_bl, "BL"), (V_bh, "B&H AAL")]:
    ax.plot([t0] + list(fut_dates[:n_fu]), drawdown(V_), label=lbl)
ax.set_title("Drawdown trajectories")
ax.axhline(0, color="k", lw=0.4)
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig07_drawdown.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §11 Moving averages diagnostic on AAL (project §3 requirement)
# ---------------------------------------------------------------------------
print("[§MA] MA + MACD diagnostic")
aal_close = panel["AAL_Close"].values
ma_sma = sh.sma(aal_close, 20)
ma_ema = sh.ema(aal_close, 20)
ma_dema = sh.dema(aal_close, 20)
macd_line, macd_sig, macd_hist = sh.macd(aal_close, 12, 26, 9)

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(panel.index, aal_close, lw=0.5, color="grey", label="Close")
ax[0].plot(panel.index, ma_sma, lw=0.7, label="SMA(20)")
ax[0].plot(panel.index, ma_ema, lw=0.7, label="EMA(20)")
ax[0].plot(panel.index, ma_dema, lw=0.7, label="DEMA(20)")
ax[0].axvline(t0, color="k", ls="--", lw=0.5)
ax[0].set_title("AAL with moving averages and MACD(12,26,9)")
ax[0].legend()
ax[1].plot(panel.index, macd_line, lw=0.6, label="MACD")
ax[1].plot(panel.index, macd_sig, lw=0.6, label="Signal")
ax[1].bar(panel.index, macd_hist, color="grey", alpha=0.5, width=1)
ax[1].axhline(0, color="k", lw=0.4)
ax[1].axvline(t0, color="k", ls="--", lw=0.5)
ax[1].legend()
plt.tight_layout(); plt.savefig(FIG / "fig08_ma_macd.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

n_figs = len(list(FIG.glob("*.png")))
print(f"\nZHAO_report analysis done.\n  figures: {n_figs}\n  results.json: {len(results)} keys")
