"""
HUANG_report/analysis.py
========================
Multi-Indicator Technical Analysis with Signal Engineering.

Companion report: HUANG Wenhao --- MSDM5058 Project II.

Direction (per group plan):
  - Full OHLCV emphasis (intraday range, volume, true range)
  - VIX-regime gating
  - Indicator zoo: SMA / EMA / WMA / DEMA / TEMA, MACD, RSI, Bollinger,
    Stoch K/D, OBV, VWAP, ATR, Kalman trend filter
  - Signal disagreement / consensus matrix
  - Inverse-volatility ensemble + ATR stop-loss
  - VIX-regime greed throttle in §8/9/10 simulators

The script imports the *shared* dataset and helpers from data/shared.py
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
# Data load
# ---------------------------------------------------------------------------
print("=" * 70)
print("HUANG_report — Multi-Indicator Technical Analysis")
print("=" * 70)

panel = sh.load_panel(extra=("VIX", "SPY"))
past, future, t0 = sh.train_test_split(panel)
n_past = len(past)
print(f"  panel: {len(panel)} trading days, t0={t0.date()}, past={n_past}")

aal_close = panel["AAL_Close"].values
aal_high  = panel["AAL_High"].values
aal_low   = panel["AAL_Low"].values
aal_open  = panel["AAL_Open"].values
aal_vol   = panel["AAL_Volume"].values
dal_close = panel["DAL_Close"].values
dal_high  = panel["DAL_High"].values
dal_low   = panel["DAL_Low"].values
dal_vol   = panel["DAL_Volume"].values
vix_close  = panel["VIX_Close"].values
spy_close  = panel["SPY_Close"].values
dates      = panel.index

x1 = sh.log_returns(aal_close)
x2 = sh.log_returns(dal_close)
n = len(x1)
split_idx = n_past - 1
x1_past, x2_past = x1[:split_idx], x2[:split_idx]
x1_fut,  x2_fut  = x1[split_idx:], x2[split_idx:]

results: dict = {
    "n_total": n, "split_idx": split_idx,
    "split_date": str(t0.date()),
    "date_start": str(dates[0].date()),
    "date_end": str(dates[-1].date()),
}

# ---------------------------------------------------------------------------
# §2 Data preprocessing — extra TA features
# ---------------------------------------------------------------------------
print("[§2] Data preprocessing & TA feature engineering")

true_range = np.maximum.reduce([
    aal_high - aal_low,
    np.abs(aal_high - np.concatenate([[aal_close[0]], aal_close[:-1]])),
    np.abs(aal_low  - np.concatenate([[aal_close[0]], aal_close[:-1]])),
])
parkinson = np.sqrt((1 / (4 * np.log(2))) *
                    pd.Series(np.log(aal_high / aal_low) ** 2)
                    .rolling(20, min_periods=5).mean().values)
atr14 = sh.atr(aal_high, aal_low, aal_close, 14)
obv_  = sh.obv(aal_close, aal_vol)
vwap_ = sh.vwap(aal_high, aal_low, aal_close, aal_vol)
vol_z = pd.Series(aal_vol).rolling(20, min_periods=5)
vol_z = ((aal_vol - vol_z.mean().values) / (vol_z.std().values + 1e-12))

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(dates, aal_close, color="C0", lw=0.8, label="AAL Close")
ax[0].plot(dates, vwap_, color="C3", lw=0.6, label="VWAP")
ax[0].set_ylabel("Price (\\$)")
ax[0].legend(loc="upper left", fontsize=8)
ax[0].set_title("AAL price + VWAP, ATR(14), Parkinson 20-day vol")
ax[1].plot(dates, atr14, color="C2", lw=0.8, label="ATR(14)")
ax[1].plot(dates, parkinson * np.nanmean(aal_close), color="C4", lw=0.6,
           label=r"Parkinson$\times \bar{P}$")
ax[1].set_ylabel("Vol")
ax[1].legend(loc="upper left", fontsize=8)
ax[2].bar(dates, vol_z, color="C5", width=1)
ax[2].set_ylabel("Vol z-score (20d)")
ax[2].axhline(2, color="red", lw=0.5)
ax[2].axhline(-2, color="red", lw=0.5)
ax[2].axvline(t0, color="k", ls="--", lw=0.6)
plt.tight_layout(); plt.savefig(FIG / "fig01_ta_features.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §3 Mean-variance — full Markowitz frontier on AAL/DAL/SPY  ===============
# ---------------------------------------------------------------------------
print("[§3] Mean-variance & full Markowitz frontier (AAL/DAL/SPY)")
spy_x = sh.log_returns(spy_close)
m = min(len(x1_past), len(x2_past), len(spy_x[:split_idx]))
R = np.column_stack([x1_past[-m:], x2_past[-m:], spy_x[:split_idx][-m:]])
mu = R.mean(axis=0) * 252
Sigma = np.cov(R, rowvar=False, ddof=1) * 252
labels = ["AAL", "DAL", "SPY"]

# Frontier via parametric mu-target
mu_grid = np.linspace(mu.min(), mu.max(), 60)
front_var = []
front_w = []
ones = np.ones(3)
inv_S = np.linalg.inv(Sigma)
A = ones @ inv_S @ ones
B = ones @ inv_S @ mu
C = mu @ inv_S @ mu
det = A * C - B * B
for mt in mu_grid:
    lam = (C - mt * B) / det
    g = (mt * A - B) / det
    w = lam * (inv_S @ ones) + g * (inv_S @ mu)
    front_var.append(w @ Sigma @ w)
    front_w.append(w)
front_var = np.array(front_var)
front_sd = np.sqrt(front_var)

# Tangency / minimum-variance portfolio
w_mv = (inv_S @ ones) / (ones @ inv_S @ ones)
sd_mv = np.sqrt(w_mv @ Sigma @ w_mv)
mu_mv = w_mv @ mu

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(front_sd, mu_grid, color="C0", lw=1.4, label="Efficient frontier")
ax.scatter(np.sqrt(np.diag(Sigma)), mu, c="C3", marker="o", s=30, zorder=5)
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (np.sqrt(Sigma[i, i]), mu[i]), fontsize=9)
ax.scatter(sd_mv, mu_mv, c="green", marker="*", s=200, zorder=5,
           label=f"MV ({sd_mv:.3f}, {mu_mv:.3f})")
ax.set_xlabel("Annualised std-dev")
ax.set_ylabel("Annualised mean")
ax.set_title("Markowitz frontier on AAL/DAL/SPY (past window)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "fig02_frontier.png", dpi=150); plt.close()

results["mv_portfolio"] = {
    "weights": dict(zip(labels, w_mv.tolist())),
    "ann_ret": float(mu_mv), "ann_vol": float(sd_mv),
}

# ---------------------------------------------------------------------------
# §4 Moving-average family + lag/whipsaw analysis  ============================
# ---------------------------------------------------------------------------
print("[§4] MA family — lag and whipsaw analysis")
W = 20
ma_sma  = sh.sma(aal_close, W)
ma_ema  = sh.ema(aal_close, W)
ma_wma  = sh.wma(aal_close, W)
ma_dema = sh.dema(aal_close, W)
ma_tema = sh.tema(aal_close, W)


def crossings(price, ma):
    sign = np.sign(price - ma)
    return int(np.sum(np.abs(np.diff(sign)) > 0))


def cc_lag(a, b, max_lag=30):
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    a = np.nan_to_num(a); b = np.nan_to_num(b)
    lags = range(-max_lag, max_lag + 1)
    cc = []
    for k in lags:
        if k < 0:
            cc.append(float(np.corrcoef(a[:k], b[-k:])[0, 1]))
        elif k > 0:
            cc.append(float(np.corrcoef(a[k:], b[:-k])[0, 1]))
        else:
            cc.append(float(np.corrcoef(a, b)[0, 1]))
    return list(lags), cc


fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(dates, aal_close, lw=0.5, color="grey", alpha=0.5, label="Close")
for arr, lbl in [(ma_sma, "SMA"), (ma_ema, "EMA"), (ma_wma, "WMA"),
                 (ma_dema, "DEMA"), (ma_tema, "TEMA")]:
    ax.plot(dates, arr, lw=0.7, label=lbl)
ax.legend()
ax.set_title(f"AAL: 5-MA family at $W={W}$")
ax.axvline(t0, color="k", ls="--", lw=0.5)
plt.tight_layout(); plt.savefig(FIG / "fig03_ma_family.png", dpi=150); plt.close()

# whipsaws & lag (cross-corr peak with EMA)
ma_metrics = {}
for arr, lbl in [(ma_sma, "SMA"), (ma_ema, "EMA"), (ma_wma, "WMA"),
                 (ma_dema, "DEMA"), (ma_tema, "TEMA")]:
    lags, cc = cc_lag(aal_close, arr, 30)
    peak_lag = lags[int(np.argmax(cc))]
    ma_metrics[lbl] = {
        "n_crossings": crossings(aal_close, arr),
        "peak_corr_lag": peak_lag,
        "peak_corr": float(np.max(cc)),
    }
results["ma_metrics"] = ma_metrics

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(list(ma_metrics.keys()),
          [m["n_crossings"] for m in ma_metrics.values()], color="C0")
ax[0].set_title("Whipsaws (price/MA crossings, past+future)")
ax[0].set_ylabel("# crossings")
ax[1].bar(list(ma_metrics.keys()),
          [m["peak_corr_lag"] for m in ma_metrics.values()], color="C1")
ax[1].set_title("Cross-corr peak lag (days)")
ax[1].set_ylabel("Lag (smaller = faster)")
plt.tight_layout(); plt.savefig(FIG / "fig04_ma_whipsaw.png", dpi=150); plt.close()

# MACD
macd_line, macd_sig, macd_hist = sh.macd(aal_close, 12, 26, 9)
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(dates, aal_close, color="C0", lw=0.7)
ax[0].set_title("AAL Close + MACD(12,26,9)")
ax[1].plot(dates, macd_line, color="C0", lw=0.6, label="MACD")
ax[1].plot(dates, macd_sig, color="C1", lw=0.6, label="Signal")
ax[1].bar(dates, macd_hist, color="grey", alpha=0.5, width=1)
ax[1].axhline(0, color="k", lw=0.4)
ax[1].axvline(t0, color="k", ls="--", lw=0.5)
ax[1].legend(); ax[1].set_xlabel("Date")
plt.tight_layout(); plt.savefig(FIG / "fig05_macd.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §Extension: Indicator zoo & signal-disagreement matrix  ====================
# ---------------------------------------------------------------------------
print("[§Ext] Indicator zoo + disagreement matrix")
rsi14 = sh.rsi(aal_close, 14)
bb_mid, bb_up, bb_lo = sh.bollinger(aal_close, 20, 2.0)


def stochastic_kd(high, low, close, k=14, d=3):
    high = np.asarray(high); low = np.asarray(low); close = np.asarray(close)
    low_min = pd.Series(low).rolling(k, min_periods=1).min().values
    high_max = pd.Series(high).rolling(k, min_periods=1).max().values
    K = 100 * (close - low_min) / (high_max - low_min + 1e-12)
    D = pd.Series(K).rolling(d, min_periods=1).mean().values
    return K, D


def kalman_trend(z, q=1e-5, r=1e-2):
    """1-D Kalman filter with random-walk state."""
    n = len(z)
    x = np.zeros(n); P = np.zeros(n)
    x[0] = z[0]; P[0] = 1.0
    for k in range(1, n):
        x_pred = x[k - 1]
        P_pred = P[k - 1] + q
        K = P_pred / (P_pred + r)
        x[k] = x_pred + K * (z[k] - x_pred)
        P[k] = (1 - K) * P_pred
    return x


stoch_K, stoch_D = stochastic_kd(aal_high, aal_low, aal_close)
kalman = kalman_trend(aal_close, q=1e-3, r=1.0)


def signal_macd():
    sig = np.where(macd_line > macd_sig, 1, -1)
    return sig


def signal_rsi():
    sig = np.zeros_like(rsi14)
    sig[rsi14 < 30] = 1
    sig[rsi14 > 70] = -1
    return sig


def signal_bb():
    sig = np.zeros_like(aal_close)
    sig[aal_close < bb_lo] = 1
    sig[aal_close > bb_up] = -1
    return sig


def signal_stoch():
    sig = np.zeros_like(stoch_K)
    sig[stoch_K < 20] = 1
    sig[stoch_K > 80] = -1
    return sig


def signal_obv():
    obv_sma = sh.sma(obv_, 20)
    return np.where(obv_ > obv_sma, 1, -1)


def signal_vwap():
    return np.where(aal_close > vwap_, 1, -1)


def signal_kalman():
    return np.where(np.diff(kalman, prepend=kalman[0]) > 0, 1, -1)


sigs = {
    "MACD": signal_macd(), "RSI": signal_rsi(), "BB": signal_bb(),
    "STOCH": signal_stoch(), "OBV": signal_obv(),
    "VWAP": signal_vwap(), "KALMAN": signal_kalman(),
}
sig_names = list(sigs.keys())
S = np.column_stack([sigs[k] for k in sig_names])
S_past = S[:n_past]
disagree = np.zeros((len(sig_names), len(sig_names)))
for i, _ in enumerate(sig_names):
    for j, _ in enumerate(sig_names):
        # disagreement = fraction of opposite-sign days
        a, b = S_past[:, i], S_past[:, j]
        mask = (a != 0) & (b != 0)
        if mask.sum() == 0:
            disagree[i, j] = 0
        else:
            disagree[i, j] = float(np.mean(a[mask] != b[mask]))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(disagree, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(sig_names))); ax.set_xticklabels(sig_names, rotation=30)
ax.set_yticks(range(len(sig_names))); ax.set_yticklabels(sig_names)
for i in range(len(sig_names)):
    for j in range(len(sig_names)):
        ax.text(j, i, f"{disagree[i,j]:.2f}", ha="center", va="center",
                color="black" if disagree[i, j] < 0.5 else "white", fontsize=8)
plt.colorbar(im, ax=ax)
ax.set_title("Signal disagreement matrix (past window)")
plt.tight_layout(); plt.savefig(FIG / "fig06_disagreement.png", dpi=150); plt.close()

results["signals"] = {
    "names": sig_names,
    "disagreement_matrix": disagree.tolist(),
}

# Composite consensus signal
consensus = S.sum(axis=1)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(dates, consensus, color="C0", lw=0.6)
ax.axhline(0, color="k", lw=0.4)
ax.fill_between(dates, 0, consensus, where=consensus > 0,
                color="C2", alpha=0.4, label="Bullish")
ax.fill_between(dates, 0, consensus, where=consensus < 0,
                color="C3", alpha=0.4, label="Bearish")
ax.set_title("Consensus signal (sum of 7 indicators)")
ax.axvline(t0, color="k", ls="--", lw=0.5)
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig07_consensus.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §5 PDF — Normal vs Logistic   ==============================================
# ---------------------------------------------------------------------------
print("[§5] PDF: Normal vs Logistic on past")
mu_n, sigma_n = float(np.mean(x1_past)), float(np.std(x1_past, ddof=1))
loc_l, scale_l = stats.logistic.fit(x1_past)
ll_n = float(np.sum(stats.norm.logpdf(x1_past, mu_n, sigma_n)))
ll_l = float(np.sum(stats.logistic.logpdf(x1_past, loc_l, scale_l)))
aic_n = -2 * ll_n + 2 * 2
aic_l = -2 * ll_l + 2 * 2
ks_n = stats.kstest(x1_past, "norm", args=(mu_n, sigma_n))
ks_l = stats.kstest(x1_past, "logistic", args=(loc_l, scale_l))

results["pdf"] = {
    "normal": {"params": [mu_n, sigma_n], "logL": ll_n, "AIC": aic_n,
               "KS_p": float(ks_n.pvalue)},
    "logistic": {"params": [loc_l, scale_l], "logL": ll_l, "AIC": aic_l,
                 "KS_p": float(ks_l.pvalue)},
}

xs = np.linspace(x1_past.min(), x1_past.max(), 600)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(x1_past, bins=80, density=True, alpha=0.4, color="grey")
ax.plot(xs, stats.norm.pdf(xs, mu_n, sigma_n), label="Normal", color="C0")
ax.plot(xs, stats.logistic.pdf(xs, loc_l, scale_l), label="Logistic",
        color="C3")
ax.set_title("AAL log returns — Normal vs Logistic ML fit")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig08_pdf.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §6 Bayes detector under both likelihoods   =================================
# ---------------------------------------------------------------------------
print("[§6] Bayes detector (Normal & Logistic likelihoods)")
eps = sh.load_constants()["epsilon"]
Y_past = sh.digitize(x1_past, eps)
priors = {y: float(np.mean(Y_past == y)) for y in ("D", "U", "H")}
cond_norm = sh.fit_conditional_normals(x1_past, Y_past)


def cond_logistic(X, Y):
    out = {}
    for y in ("D", "U", "H"):
        sub = X[Y == y]
        if len(sub) < 5:
            out[y] = (0.0, 1.0)
        else:
            loc, scale = stats.logistic.fit(sub)
            out[y] = (float(loc), float(scale))
    return out


cond_log = cond_logistic(x1_past, Y_past)


def find_boundaries(cond, priors_, dist="normal"):
    xs_ = np.linspace(-0.20, 0.20, 4001)
    if dist == "normal":
        post = np.column_stack([
            np.log(priors_[y]) + stats.norm.logpdf(xs_, *cond[y])
            for y in ("D", "U", "H")
        ])
    else:
        post = np.column_stack([
            np.log(priors_[y]) + stats.logistic.logpdf(xs_, *cond[y])
            for y in ("D", "U", "H")
        ])
    pred = np.array(["DUH"[i] for i in np.argmax(post, axis=1)])
    boundaries = []
    for i in range(1, len(xs_)):
        if pred[i] != pred[i - 1]:
            boundaries.append((float(0.5 * (xs_[i] + xs_[i - 1])),
                               pred[i - 1], pred[i]))
    return boundaries


bnd_n = find_boundaries(cond_norm, priors, "normal")
bnd_l = find_boundaries(cond_log, priors, "logistic")
results["bayes"] = {"normal": bnd_n, "logistic": bnd_l}
print(f"  Normal boundaries: {len(bnd_n)}; Logistic boundaries: {len(bnd_l)}")

# Misclassification on past window
def predict(X, cond, priors_, dist):
    if dist == "normal":
        ll = np.column_stack([
            np.log(priors_[y]) + stats.norm.logpdf(X, *cond[y])
            for y in ("D", "U", "H")
        ])
    else:
        ll = np.column_stack([
            np.log(priors_[y]) + stats.logistic.logpdf(X, *cond[y])
            for y in ("D", "U", "H")
        ])
    return np.array(["DUH"[i] for i in np.argmax(ll, axis=1)])


pred_n = predict(x1_past, cond_norm, priors, "normal")
pred_l = predict(x1_past, cond_log, priors, "logistic")
acc_n = float(np.mean(pred_n == Y_past))
acc_l = float(np.mean(pred_l == Y_past))
results["bayes_accuracy"] = {"normal": acc_n, "logistic": acc_l}

fig, ax = plt.subplots(figsize=(9, 5))
xs_ = np.linspace(-0.15, 0.15, 1001)
for y, c in [("D", "C3"), ("U", "C2"), ("H", "C0")]:
    ax.plot(xs_, priors[y] * stats.norm.pdf(xs_, *cond_norm[y]),
            color=c, lw=1.4, label=f"$\\pi_{y}f^N(x|{y})$")
    ax.plot(xs_, priors[y] * stats.logistic.pdf(xs_, *cond_log[y]),
            color=c, lw=1.0, ls="--", label=f"$\\pi_{y}f^L(x|{y})$")
for x_, _, _ in bnd_n:
    ax.axvline(x_, color="k", lw=0.5)
ax.set_title("Bayes detector — Normal (solid) vs Logistic (dashed)")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(FIG / "fig09_bayes.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §7 Association rules — k=5  ================================================
# ---------------------------------------------------------------------------
print("[§7] Association rules (k=5)")


def kgram_rules(Y, k):
    pat_count = {}
    pat_outcome = {}
    for i in range(len(Y) - k):
        p = "".join(Y[i:i + k]); o = Y[i + k]
        pat_count[p] = pat_count.get(p, 0) + 1
        pat_outcome.setdefault(p, {"D": 0, "U": 0, "H": 0})[o] += 1
    rules = []
    N = len(Y) - k
    for p, c in pat_count.items():
        for y_, cy in pat_outcome[p].items():
            if cy == 0: continue
            rules.append({
                "pattern": p, "outcome": y_,
                "support": cy / N,
                "confidence": cy / c,
            })
    return rules


rules5 = kgram_rules(Y_past, 5)
rules5.sort(key=lambda r: r["support"], reverse=True)
results["top_rules"] = rules5[:10]

# ---------------------------------------------------------------------------
# §8 Trading simulator — VIX-regime greed throttle  ===========================
# ---------------------------------------------------------------------------
print("[§8] Simulator I — VIX-regime greed throttle")
vix_past_med = float(np.nanmedian(vix_close[:n_past]))
results["vix_past_median"] = vix_past_med

future_dates = dates[split_idx + 1:]
fut_vix = vix_close[split_idx + 1: split_idx + 1 + len(x1_fut)]


def sim_one(x_ret, vix, signal, greed_low, greed_high, V0=1e5, rf=0.001/100):
    n_ = len(x_ret)
    V = np.empty(n_ + 1); V[0] = V0
    for t in range(n_):
        v = vix[t] if t < len(vix) else vix[-1]
        g = greed_low if v >= vix_past_med else greed_high
        s = signal[t]
        # +s: long with weight g, -s: cash, 0: cash
        if s > 0:
            V[t + 1] = V[t] * ((1 - g) * (1 + rf) + g * np.exp(x_ret[t]))
        elif s < 0:
            V[t + 1] = V[t] * (1 + rf)
        else:
            V[t + 1] = V[t] * (1 + rf)
    return V


# Inverse-volatility ensemble of indicator signals
sig_fut = S[n_past:]
inv_vol = 1.0 / (np.std(S_past, axis=0, ddof=1) + 1e-9)
inv_vol = inv_vol / inv_vol.sum()
sig_ens = sig_fut @ inv_vol
sig_ens_disc = np.sign(sig_ens)

# Pure MACD baseline
sig_macd_fut = signal_macd()[n_past:][:len(x1_fut)]
sig_rsi_fut  = signal_rsi()[n_past:][:len(x1_fut)]
sig_ens_disc = sig_ens_disc[:len(x1_fut)]

V_macd = sim_one(x1_fut, fut_vix, sig_macd_fut, 0.2, 0.7)
V_rsi  = sim_one(x1_fut, fut_vix, sig_rsi_fut, 0.2, 0.7)
V_ens  = sim_one(x1_fut, fut_vix, sig_ens_disc, 0.2, 0.7)
V_bh   = 1e5 * np.exp(np.cumsum(np.concatenate([[0], x1_fut])))

m_s8 = {
    "MACD (VIX-throttle)":  sh.compute_metrics(V_macd),
    "RSI (VIX-throttle)":   sh.compute_metrics(V_rsi),
    "Ensemble (inv-vol)":   sh.compute_metrics(V_ens),
    "Buy & Hold AAL":      sh.compute_metrics(V_bh),
}
results["s8_metrics"] = m_s8

fig, ax = plt.subplots(figsize=(11, 5))
fut_idx = future_dates[:len(V_macd) - 1]
ax.plot([t0] + list(fut_idx), V_macd, label="MACD")
ax.plot([t0] + list(fut_idx), V_rsi, label="RSI")
ax.plot([t0] + list(fut_idx), V_ens, label="Ensemble (inv-vol)")
ax.plot([t0] + list(fut_idx), V_bh, label="Buy & Hold", color="grey")
ax.set_title("§8 Out-of-sample equity curves (one stock + cash)")
ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("V")
plt.tight_layout(); plt.savefig(FIG / "fig10_s8.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §9 Two-stocks portfolio with EF + ATR stop-loss   ===========================
# ---------------------------------------------------------------------------
print("[§9] Simulator II — two stocks, EF + ATR stop")
atr_pct = atr14[n_past:][:len(x1_fut)] / aal_close[n_past:][:len(x1_fut)]
atr_pct = np.where(np.isnan(atr_pct), 0.05, atr_pct)


def sim_two(x_a, x_b, p, vix, atr_pct_, greed_low=0.5, greed_high=1.0):
    n_ = len(x_a)
    V = np.empty(n_ + 1); V[0] = 1e5
    in_position = True
    cooldown = 0
    for t in range(n_):
        v = vix[t] if t < len(vix) else vix[-1]
        g = greed_high if v < vix_past_med else greed_low
        # ATR stop-loss: if loss > 2*ATR%, cut to cash for 5 days
        ret_p = p * x_a[t] + (1 - p) * x_b[t]
        if cooldown > 0:
            cooldown -= 1
            V[t + 1] = V[t] * (1 + 0.001 / 100)
            continue
        V[t + 1] = V[t] * np.exp(g * ret_p)
        if ret_p < -2.0 * atr_pct_[t]:
            cooldown = 5
    return V


p0 = sh.load_constants()["p0_est_past"]
V_p0   = sim_two(x1_fut, x2_fut, p0,  fut_vix, atr_pct, 1.0, 1.0)
V_agg  = sim_two(x1_fut, x2_fut, 0.7, fut_vix, atr_pct, 0.5, 1.0)
V_cons = sim_two(x1_fut, x2_fut, 0.3, fut_vix, atr_pct, 0.3, 0.6)

m_s9 = {
    "Min-risk (p=p0, ATR-stop)":  sh.compute_metrics(V_p0),
    "Aggressive (p=0.7, gated)":  sh.compute_metrics(V_agg),
    "Conservative (p=0.3, gated)": sh.compute_metrics(V_cons),
}
results["s9_metrics"] = m_s9

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot([t0] + list(fut_idx), V_p0,   label=f"$p=p_0={p0:.2f}$ + ATR stop")
ax.plot([t0] + list(fut_idx), V_agg,  label="$p=0.7$, VIX-gated")
ax.plot([t0] + list(fut_idx), V_cons, label="$p=0.3$, VIX-gated")
ax.set_title("§9 Two-stocks portfolio with VIX-throttle and ATR stop-loss")
ax.legend(); ax.set_ylabel("V")
plt.tight_layout(); plt.savefig(FIG / "fig11_s9.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# §10 Two stocks + money — full TA ensemble  =================================
# ---------------------------------------------------------------------------
print("[§10] Simulator III — two stocks + cash, ensemble greed")


def sim_three(x_a, x_b, p, vix, sig_arr, atr_pct_):
    """Two-stocks-and-money simulator with consensus-weighted greed,
    VIX-regime gating and ATR-based stop-loss.  Position fraction g lies
    in [0, 1] and represents the share invested in the (p, 1-p) sleeve."""
    n_ = len(x_a)
    V = np.empty(n_ + 1); V[0] = 1e5
    cooldown = 0
    for t in range(n_):
        v = vix[t] if t < len(vix) else vix[-1]
        cons = sig_arr[t]  # signed consensus (sum of ±1 indicators)
        # Map consensus to long-only greed in [0, 1]; throttle by VIX regime.
        base = 0.7 * max(np.tanh(0.4 * cons), 0.0)
        throttle = max(0.2, 1.0 - 0.4 * (v / vix_past_med - 1.0))
        g = float(np.clip(base * throttle, 0.0, 1.0))
        if cooldown > 0:
            cooldown -= 1
            V[t + 1] = V[t] * (1 + 0.001 / 100)
            continue
        ret_p = p * x_a[t] + (1 - p) * x_b[t]
        V[t + 1] = V[t] * ((1 - g) * (1 + 0.001 / 100) + g * np.exp(ret_p))
        if ret_p < -2.0 * atr_pct_[t]:
            cooldown = 5
    return V


cons_fut = consensus[n_past:][:len(x1_fut)]
V_full = sim_three(x1_fut, x2_fut, p0, fut_vix, cons_fut, atr_pct)

m_s10 = {
    "Ensemble greed (p=p0)":  sh.compute_metrics(V_full),
    "Aggressive ensemble":    sh.compute_metrics(
        sim_three(x1_fut, x2_fut, 0.7, fut_vix, cons_fut, atr_pct)
    ),
    "Min-risk benchmark":     sh.compute_metrics(V_p0),
}
results["s10_metrics"] = m_s10

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot([t0] + list(fut_idx), V_full, label="Ensemble greed (p=p0)")
ax.plot([t0] + list(fut_idx), V_p0, label="Min-risk benchmark", ls="--")
ax.set_title("§10 Two-stocks-and-money with consensus-greed + VIX gate + ATR stop")
ax.legend(); plt.tight_layout()
plt.savefig(FIG / "fig12_s10.png", dpi=150); plt.close()

# Drawdown panel
def drawdown(V):
    peak = np.maximum.accumulate(V)
    return V / peak - 1


fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between([t0] + list(fut_idx), drawdown(V_full), 0,
                color="C3", alpha=0.5, label="Ensemble")
ax.fill_between([t0] + list(fut_idx), drawdown(V_p0), 0,
                color="grey", alpha=0.3, label="Min-risk")
ax.set_title("§10 Drawdown trajectories")
ax.legend(); plt.tight_layout()
plt.savefig(FIG / "fig13_drawdown.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

n_figs = len(list(FIG.glob("*.png")))
print(f"\nHUANG_report analysis done.\n  figures: {n_figs}\n  results.json: {len(results)} keys")
