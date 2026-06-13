"""
HU_report/analysis.py
=====================
Bayesian Inference and Information-Theoretic Pattern Mining for Portfolio
Management on AAL/DAL.

Owner: HU Xiuqi.

This script reproduces every figure, table and numeric value cited in
HU_report/report.tex and writes ``results.json`` with all intermediate
metrics.  It expects the shared dataset under ``../data/`` (built by
``data/build_dataset.py``).
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from data import shared as sh  # noqa: E402

warnings.filterwarnings("ignore")

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
np.random.seed(sh.SEED)
RESULTS: dict = {}

# =============================================================================
# Load data
# =============================================================================
print("=" * 70)
print("HU_report — Bayesian / Information-Theoretic Analysis")
print("=" * 70)

panel = sh.load_panel(extra=("VIX",))
past, future, t0 = sh.train_test_split(panel)
S1 = panel["AAL_Close"].values
S2 = panel["DAL_Close"].values
V_aal = panel["AAL_Volume"].values
dates = panel.index

X1_full = sh.log_returns(S1)
X2_full = sh.log_returns(S2)
ret_dates = dates[1:]

n = len(panel)
split_idx = panel.index.get_loc(t0)
S1_past, S2_past = S1[:split_idx + 1], S2[:split_idx + 1]
X1_past, X2_past = X1_full[:split_idx], X2_full[:split_idx]
S1_fut, S2_fut = S1[split_idx:], S2[split_idx:]
X1_fut, X2_fut = X1_full[split_idx:], X2_full[split_idx:]
dates_past, dates_fut = dates[:split_idx + 1], dates[split_idx:]

print(f"  panel: {n} trading days, t0={t0.date()}, split={split_idx}")
RESULTS.update({
    "n_total": int(n),
    "split_idx": int(split_idx),
    "split_date": str(t0.date()),
    "date_start": str(dates[0].date()),
    "date_end": str(dates[-1].date()),
    "ratio_past_future": split_idx / (n - split_idx),
})

# Auxiliary feature: short-volume ratio if available
sv_path = sh.DATA_DIR / "short_volume.csv"
if sv_path.exists():
    sv = pd.read_csv(sv_path, index_col=0, parse_dates=True)
    sv = sv.reindex(panel.index).ffill()
    SR_AAL = sv.get("AAL_ShortRatio", pd.Series(index=panel.index, dtype=float)).values
else:
    SR_AAL = np.full(len(panel), np.nan)
RESULTS["short_volume_available"] = bool(np.any(~np.isnan(SR_AAL)))

# =============================================================================
# §2  Data preprocessing  → fig1_price_returns
# =============================================================================
print("\n[§2] Data preprocessing")
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for ax, S, lbl, c in [(axes[0, 0], S1, "AAL", "steelblue"),
                      (axes[0, 1], S2, "DAL", "darkorange")]:
    ax.plot(dates, S, lw=0.6, color=c)
    ax.axvline(t0, color="red", ls="--", alpha=0.7, label="t=0")
    ax.set_title(f"{lbl} — closing price")
    ax.set_ylabel("USD"); ax.grid(alpha=0.3); ax.legend()
for ax, X, lbl, c in [(axes[1, 0], X1_full, "AAL", "steelblue"),
                      (axes[1, 1], X2_full, "DAL", "darkorange")]:
    ax.plot(ret_dates, X, lw=0.3, color=c)
    ax.axvline(t0, color="red", ls="--", alpha=0.7)
    ax.set_title(f"{lbl} — log return")
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "fig01_price_returns.png", dpi=150); plt.close()

summary_rows = []
for label, X in [("AAL", X1_past), ("DAL", X2_past)]:
    summary_rows.append({
        "ticker": label,
        "n": int(len(X)),
        "mean": float(np.mean(X)),
        "std": float(np.std(X, ddof=1)),
        "skew": float(stats.skew(X)),
        "kurt": float(stats.kurtosis(X, fisher=True)),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
    })
RESULTS["summary_stats"] = summary_rows

# =============================================================================
# §3  Mean–variance analysis  → fig2_p0, fig3_S0_sharpe
# =============================================================================
print("[§3] Mean-variance analysis")
hs = [30, 100, 300, np.inf]
labels = ["h=30", "h=100", "h=300", r"h=$\infty$"]
colors_h = ["steelblue", "darkorange", "green", "red"]

p0_dict = {lab: sh.rolling_p0(X1_past, X2_past, h) for lab, h in zip(labels, hs)}

fig, ax = plt.subplots(figsize=(12, 4.5))
for (lab, arr), c in zip(p0_dict.items(), colors_h):
    m = ~np.isnan(arr)
    ax.plot(dates_past[1:][m], arr[m], lw=0.8, color=c, label=lab)
ax.set_title("Minimum-risk fraction $p_0(t,h)$ (past data)")
ax.set_ylabel("$p_0$"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig02_p0.png", dpi=150); plt.close()

# S0(t,h) and rolling Sharpe
def portfolio_path(X1, X2, p0_arr):
    n_ = len(X1)
    X0 = np.full(n_, np.nan)
    valid = ~np.isnan(p0_arr)
    X0[valid] = p0_arr[valid] * X1[valid] + (1 - p0_arr[valid]) * X2[valid]
    S0 = np.full(n_ + 1, np.nan)
    fv = np.argmax(valid)
    S0[fv] = 1.0
    for t in range(fv, n_):
        S0[t + 1] = S0[t] * np.exp(X0[t] if not np.isnan(X0[t]) else 0.0)
    return S0, X0

def rolling_sharpe(X0, w=252):
    out = np.full(len(X0), np.nan)
    for t in range(w, len(X0)):
        seg = X0[t - w:t]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 30:
            out[t] = np.mean(seg) / np.std(seg, ddof=1) * np.sqrt(252)
    return out

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for (lab, arr), c in zip(p0_dict.items(), colors_h):
    S0, X0 = portfolio_path(X1_past, X2_past, arr)
    sh_arr = rolling_sharpe(X0)
    m1 = ~np.isnan(S0); m2 = ~np.isnan(sh_arr)
    axes[0].plot(dates_past[m1], S0[m1], lw=0.8, color=c, label=lab)
    axes[1].plot(dates_past[1:][m2], sh_arr[m2], lw=0.8, color=c, label=lab)
axes[0].set_title("Min-risk portfolio value $S_0(t,h)$ (normalised)"); axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].set_title(r"Rolling Sharpe $\gamma_0(t,h)$ (252-day, annualised)"); axes[1].grid(alpha=0.3); axes[1].legend()
plt.tight_layout(); plt.savefig(FIG / "fig03_S0_sharpe.png", dpi=150); plt.close()

p0_summary = {}
for lab, arr in p0_dict.items():
    a = arr[~np.isnan(arr)]
    p0_summary[lab] = {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }
RESULTS["p0_summary"] = p0_summary

# =============================================================================
# §4  Moving averages and MACD  → fig4_ema, fig5_ema_vs_sma, fig6_macd
# =============================================================================
print("[§4] Moving averages")
S = S1_past
X = X1_past
ws = [30, 100, 300]
cs = ["red", "green", "purple"]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates_past, S, lw=0.5, color="steelblue", alpha=0.7, label="AAL")
for w, c in zip(ws, cs):
    ax.plot(dates_past, sh.ema(S, w), lw=1.0, color=c, label=f"EMA({w})")
ax.set_title("EMA at multiple horizons (AAL, learning window)")
ax.set_ylabel("USD"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig04_ema.png", dpi=150); plt.close()

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
for ax, w, c in zip(axes, ws, cs):
    ax.plot(dates_past, S, lw=0.4, color="steelblue", alpha=0.5, label="AAL")
    ax.plot(dates_past, sh.ema(S, w), lw=1.0, color="red", label=f"EMA({w})")
    ax.plot(dates_past, sh.sma(S, w), lw=1.0, color="blue", ls="--", label=f"SMA({w})")
    ax.set_title(f"EMA vs SMA, $w={w}$"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig05_ema_vs_sma.png", dpi=150); plt.close()

m_line, m_sig, m_hist = sh.macd(S)
fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                          gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(dates_past, S, lw=0.5, color="steelblue", label="AAL")
cross = np.diff(np.sign(m_line - m_sig))
buys = np.where(cross > 0)[0]
sells = np.where(cross < 0)[0]
axes[0].scatter(dates_past[buys], S[buys], marker="^", color="green",
                s=18, alpha=0.6, label="MACD↑signal")
axes[0].scatter(dates_past[sells], S[sells], marker="v", color="red",
                s=18, alpha=0.6, label="MACD↓signal")
axes[0].set_title("AAL with MACD crossover signals")
axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].plot(dates_past, m_line, lw=0.8, color="blue", label="MACD")
axes[1].plot(dates_past, m_sig, lw=0.8, color="red", label="signal")
axes[1].bar(dates_past, m_hist, width=1,
            color=["green" if v >= 0 else "red" for v in m_hist], alpha=0.4)
axes[1].axhline(0, color="black", lw=0.4)
axes[1].set_title("MACD line / signal / histogram")
axes[1].grid(alpha=0.3); axes[1].legend()
plt.tight_layout(); plt.savefig(FIG / "fig06_macd.png", dpi=150); plt.close()

# Crossover counts and forward-return stats
fwd_buy_returns = X[buys[buys + 5 < len(X)] + 5] if len(buys) else np.array([])
fwd_sell_returns = X[sells[sells + 5 < len(X)] + 5] if len(sells) else np.array([])
RESULTS["macd"] = {
    "n_buy_crossings": int(len(buys)),
    "n_sell_crossings": int(len(sells)),
    "mean_fwd5_after_buy": float(np.mean(fwd_buy_returns)) if len(fwd_buy_returns) else None,
    "mean_fwd5_after_sell": float(np.mean(fwd_sell_returns)) if len(fwd_sell_returns) else None,
}

# =============================================================================
# §5  PDF — Normal vs Logistic vs Student-t vs Skew-Normal  → fig7_cdf
# =============================================================================
print("[§5] PDF fits and tests")
mu = float(np.mean(X)); sigma2 = float(np.var(X, ddof=1)); sigma = float(np.sqrt(sigma2))

# Logistic via method-of-moments (variance = pi^2/(3 b^2))
x_star = float(np.median(X))
b_log = float(np.pi / (np.sqrt(3) * np.std(X, ddof=1)))

# Student-t MLE
t_df, t_loc, t_scale = stats.t.fit(X)
# Skew-normal MLE
sn_a, sn_loc, sn_scale = stats.skewnorm.fit(X)

# Compare fits via log-likelihood / AIC
def aic(loglike, k): return 2 * k - 2 * loglike
fits = {}
fits["normal"] = {
    "params": {"mu": mu, "sigma": sigma},
    "loglik": float(np.sum(stats.norm.logpdf(X, mu, sigma))),
    "k": 2,
}
fits["logistic"] = {
    "params": {"x_star": x_star, "b": b_log},
    "loglik": float(np.sum(stats.logistic.logpdf(X, x_star, 1 / b_log))),
    "k": 2,
}
fits["student_t"] = {
    "params": {"df": float(t_df), "loc": float(t_loc), "scale": float(t_scale)},
    "loglik": float(np.sum(stats.t.logpdf(X, t_df, t_loc, t_scale))),
    "k": 3,
}
fits["skew_normal"] = {
    "params": {"a": float(sn_a), "loc": float(sn_loc), "scale": float(sn_scale)},
    "loglik": float(np.sum(stats.skewnorm.logpdf(X, sn_a, sn_loc, sn_scale))),
    "k": 3,
}
for name, f in fits.items():
    f["AIC"] = aic(f["loglik"], f["k"])
    f["BIC"] = f["k"] * np.log(len(X)) - 2 * f["loglik"]

# Goodness-of-fit tests (KS, AD, JB)
ks_stat, ks_p = stats.kstest(X, "norm", args=(mu, sigma))
ad_res = stats.anderson(X, dist="norm")
jb_stat, jb_p = stats.jarque_bera(X)
RESULTS["pdf_fits"] = fits
RESULTS["gof_normal"] = {
    "ks_stat": float(ks_stat), "ks_p": float(ks_p),
    "ad_stat": float(ad_res.statistic),
    "ad_crit": ad_res.critical_values.tolist(),
    "jb_stat": float(jb_stat), "jb_p": float(jb_p),
}

# Empirical and fitted CDFs
xs = np.sort(X)
F_emp = np.arange(1, len(xs) + 1) / len(xs)
G = stats.norm.cdf(xs, mu, sigma)
L = 1 / (1 + np.exp(-b_log * (xs - x_star)))
T = stats.t.cdf(xs, t_df, t_loc, t_scale)
SN = stats.skewnorm.cdf(xs, sn_a, sn_loc, sn_scale)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(xs, F_emp, lw=2.0, color="black", label="empirical $F(x)$")
ax.plot(xs, G, lw=1.0, color="blue", ls="--", label="Normal")
ax.plot(xs, L, lw=1.0, color="red", ls="-.", label="Logistic")
ax.plot(xs, T, lw=1.0, color="green", ls=":", label=f"Student-$t$ (df={t_df:.1f})")
ax.plot(xs, SN, lw=1.0, color="purple", ls=(0, (3, 1, 1, 1)), label="Skew-normal")
ax.set_title("Empirical vs four parametric CDFs (AAL learning window)")
ax.set_xlabel("$x$"); ax.set_ylabel("CDF"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig07_cdf_fits.png", dpi=150); plt.close()

# =============================================================================
# §5.1  Digitisation and conditional PDFs  → fig8, fig9
# =============================================================================
print("[§5.1] Digitisation and conditional PDFs")
eps = 0.5 * sigma
Y = sh.digitize(X, eps)
counts = {y: int(np.sum(Y == y)) for y in ("D", "U", "H")}
RESULTS["digitisation"] = {"epsilon": float(eps), "counts": counts}

# Conditional empirical CDFs for {y: X(t) | Y(t+1)=y}
cond_data = {}
for y in ("D", "U", "H"):
    mask = Y[1:] == y
    cond_data[y] = X[:-1][mask]

fig, ax = plt.subplots(figsize=(11, 6))
cmap = {"D": "red", "U": "green", "H": "gray"}
for y in ("D", "U", "H"):
    xs_ = np.sort(cond_data[y])
    F_ = np.arange(1, len(xs_) + 1) / len(xs_)
    ax.plot(xs_, F_, lw=1.5, color=cmap[y], label=f"$F(x\\mid y={y})$")
ax.set_title("Conditional empirical CDFs $F(x\\mid Y(t{+}1){=}y)$")
ax.set_xlabel("$X(t)$"); ax.set_ylabel("CDF"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig08_cond_cdf.png", dpi=150); plt.close()

# Conditional PDFs: parametric (Normal) + KDE (Silverman)
cond_params = sh.fit_conditional_normals(X[:-1], Y[1:])
x_grid = np.linspace(-0.10, 0.10, 1200)
fig, ax = plt.subplots(figsize=(11, 6))
for y in ("D", "U", "H"):
    mu_y, sd_y = cond_params[y]
    pdf = stats.norm.pdf(x_grid, mu_y, sd_y)
    ax.plot(x_grid, pdf, lw=1.5, color=cmap[y], label=f"$f_{{{y}}}^{{\\rm N}}$")
    kde = stats.gaussian_kde(cond_data[y], bw_method="silverman")
    ax.plot(x_grid, kde(x_grid), lw=1.0, color=cmap[y], ls=":", label=f"$f_{{{y}}}^{{\\rm KDE}}$")
ax.set_title("Conditional PDFs $f_y(x)$ — Normal fit (solid) and KDE (dotted)")
ax.set_xlabel("$x$"); ax.set_ylabel("PDF"); ax.grid(alpha=0.3); ax.legend(ncol=3)
plt.tight_layout(); plt.savefig(FIG / "fig09_cond_pdf.png", dpi=150); plt.close()

RESULTS["cond_params"] = {y: {"mu": float(cond_params[y][0]), "sigma": float(cond_params[y][1])} for y in cond_params}

# =============================================================================
# §6  Bayes detector  → fig10
# =============================================================================
print("[§6] Bayes detector")
priors = {y: counts[y] / sum(counts.values()) for y in ("D", "U", "H")}
RESULTS["priors"] = priors

def bayes_score(x, params, p):
    return {y: p[y] * stats.norm.pdf(x, params[y][0], params[y][1])
            for y in ("D", "U", "H")}

x_fine = np.linspace(-0.12, 0.12, 20001)
preds = np.array([max(bayes_score(x, cond_params, priors).items(),
                      key=lambda kv: kv[1])[0] for x in x_fine])

# Find every change point (numerically) and refine via brentq
critical = []
for i in range(1, len(preds)):
    if preds[i] != preds[i - 1]:
        y1, y2 = preds[i - 1], preds[i]
        try:
            xc = brentq(
                lambda xx: bayes_score(xx, cond_params, priors)[y1]
                - bayes_score(xx, cond_params, priors)[y2],
                x_fine[i - 1], x_fine[i])
        except ValueError:
            xc = 0.5 * (x_fine[i - 1] + x_fine[i])
        critical.append((float(xc), y1, y2))

print("  Bayes boundaries:")
for xc, a, b in critical:
    print(f"    x* = {xc:+.5f}   {a} → {b}")
RESULTS["bayes_boundaries"] = [(c, a, b) for c, a, b in critical]

# Bootstrap credible intervals for the inner two boundaries
rng = np.random.default_rng(sh.SEED)
inner = [c for c in critical if -0.06 < c[0] < 0.06]
boot_xc = {f"{a}_{b}": [] for _, a, b in inner}
for _ in range(300):
    idx = rng.integers(0, len(X) - 1, size=len(X) - 1)
    Xb = X[:-1][idx]; Yb = Y[1:][idx]
    cp = sh.fit_conditional_normals(Xb, Yb)
    cnts = {y: int(np.sum(Yb == y)) for y in ("D", "U", "H")}
    pri = {y: cnts[y] / sum(cnts.values()) for y in cnts}
    preds_b = np.array([max(bayes_score(x, cp, pri).items(),
                             key=lambda kv: kv[1])[0] for x in x_fine])
    for i in range(1, len(preds_b)):
        if preds_b[i] != preds_b[i - 1]:
            y1, y2 = preds_b[i - 1], preds_b[i]
            key = f"{y1}_{y2}"
            if key in boot_xc and -0.06 < x_fine[i] < 0.06:
                try:
                    xc_b = brentq(
                        lambda xx: bayes_score(xx, cp, pri)[y1]
                        - bayes_score(xx, cp, pri)[y2],
                        x_fine[i - 1], x_fine[i])
                except ValueError:
                    xc_b = 0.5 * (x_fine[i - 1] + x_fine[i])
                boot_xc[key].append(xc_b)
RESULTS["boot_boundaries"] = {
    k: {"mean": float(np.mean(v)) if v else None,
        "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else None,
        "n": len(v)}
    for k, v in boot_xc.items()
}

fig, ax = plt.subplots(figsize=(11, 6))
for y in ("D", "U", "H"):
    mu_y, sd_y = cond_params[y]
    pdf = priors[y] * stats.norm.pdf(x_grid, mu_y, sd_y)
    ax.plot(x_grid, pdf, lw=1.5, color=cmap[y], label=f"$q_{{{y}}}f_{{{y}}}(x)$")
ymax = ax.get_ylim()[1]
for xc, _, _ in critical:
    ax.axvline(xc, color="black", ls="--", lw=0.8, alpha=0.7)
    ax.text(xc, ymax * 0.95, f"$x^*={xc:.3f}$",
            rotation=90, va="top", ha="right", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
ax.set_title("Bayes weighted posteriors with all decision boundaries")
ax.set_xlabel("$x$"); ax.set_ylabel("$q_y f_y(x)$"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig10_bayes.png", dpi=150); plt.close()

# =============================================================================
# §7  Association rules + conditional entropy ceiling
# =============================================================================
print("[§7] Association rules and conditional entropy")

def mine_rules(Y_seq, k=5):
    rule_counts = defaultdict(int)
    pat_counts = defaultdict(int)
    for t in range(k, len(Y_seq)):
        pat = tuple(Y_seq[t - k:t])
        out = Y_seq[t]
        rule_counts[(pat, out)] += 1
        pat_counts[pat] += 1
    rows = []
    n_inst = len(Y_seq) - k
    for (pat, out), c in rule_counts.items():
        s = c / n_inst
        cf_ = c / pat_counts[pat]
        rows.append({
            "pattern": "".join(pat), "outcome": out, "count": c,
            "support": s, "confidence": cf_,
            "geom": np.sqrt(s * cf_),
            "rpf": (s * cf_ ** 2) ** (1 / 3),
        })
    return pd.DataFrame(rows), pat_counts

rules, pat_counts = mine_rules(Y, k=5)

def top10(df, col):
    return df.nlargest(10, col)[["pattern", "outcome", "support",
                                  "confidence", col]].to_dict("records")

RESULTS["top_support"] = top10(rules, "support")
RESULTS["top_confidence"] = top10(rules, "confidence")
RESULTS["top_geom"] = top10(rules, "geom")
RESULTS["top_rpf"] = top10(rules, "rpf")

# Optimal lambda via Kendall-tau ranking stability of top-10 across lambdas.
from scipy.stats import kendalltau
lambdas = np.linspace(0, 1, 21)
def usefulness(s, c, lam): return (s ** (1 - lam)) * (c ** lam)
top_pairs = {}
for lam in lambdas:
    rules["u"] = usefulness(rules["support"], rules["confidence"], lam)
    top_pairs[lam] = list(zip(rules.nlargest(20, "u")["pattern"],
                               rules.nlargest(20, "u")["outcome"]))
# Kendall-tau between adjacent lambdas (rank similarity)
kendall = []
for i in range(1, len(lambdas)):
    common = set(top_pairs[lambdas[i]]) & set(top_pairs[lambdas[i - 1]])
    if not common:
        kendall.append((lambdas[i], 0.0)); continue
    a_rank = {k: r for r, k in enumerate(top_pairs[lambdas[i]])}
    b_rank = {k: r for r, k in enumerate(top_pairs[lambdas[i - 1]])}
    a_arr = [a_rank[c] for c in common]; b_arr = [b_rank[c] for c in common]
    tau, _ = kendalltau(a_arr, b_arr)
    kendall.append((float(lambdas[i]), float(tau if tau == tau else 0.0)))
RESULTS["kendall_tau_curve"] = kendall

# Conditional entropy H(Y_{t+1} | history) for k = 1..7  (entropy ceiling on predictability)
def cond_entropy(Y_seq, k):
    if k == 0:
        c = Counter(Y_seq); n_ = sum(c.values())
        return -sum((v / n_) * np.log2(v / n_) for v in c.values())
    pair = defaultdict(Counter)
    for t in range(k, len(Y_seq)):
        pat = tuple(Y_seq[t - k:t])
        pair[pat][Y_seq[t]] += 1
    n_ = sum(sum(c.values()) for c in pair.values())
    H = 0.0
    for pat, ctr in pair.items():
        n_pat = sum(ctr.values())
        p_pat = n_pat / n_
        h_pat = 0.0
        for v in ctr.values():
            p = v / n_pat
            if p > 0: h_pat -= p * np.log2(p)
        H += p_pat * h_pat
    return H

H_curve = [(k, cond_entropy(Y, k)) for k in range(0, 8)]
RESULTS["cond_entropy"] = H_curve

# Markov chain order test (BIC-based on log-likelihood ratio)
def mc_loglik(Y_seq, order):
    if order == 0:
        c = Counter(Y_seq)
        n_ = sum(c.values())
        ll = sum(v * np.log(v / n_) for v in c.values())
        params = len(c) - 1
        return ll, params
    trans = defaultdict(Counter)
    for t in range(order, len(Y_seq)):
        pat = tuple(Y_seq[t - order:t])
        trans[pat][Y_seq[t]] += 1
    ll = 0.0
    params = 0
    for ctr in trans.values():
        n_p = sum(ctr.values())
        for v in ctr.values():
            ll += v * np.log(v / n_p)
        params += len(ctr) - 1
    return ll, params

mc_results = []
N_y = len(Y)
for o in range(0, 5):
    ll, p = mc_loglik(Y, o)
    bic = p * np.log(N_y) - 2 * ll
    mc_results.append({"order": o, "loglik": float(ll), "params": int(p), "BIC": float(bic)})
RESULTS["mc_order_test"] = mc_results

# Plot lambda Kendall + entropy curve
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot([k[0] for k in H_curve], [k[1] for k in H_curve], "o-")
axes[0].set_xlabel("history length $k$")
axes[0].set_ylabel("$H(Y_{t+1}\\mid \\text{hist})$ (bits)")
axes[0].set_title("Conditional entropy ceiling")
axes[0].grid(alpha=0.3)
axes[1].plot([k[0] for k in kendall], [k[1] for k in kendall], "o-")
axes[1].set_xlabel("$\\lambda$"); axes[1].set_ylabel(r"Kendall $\tau$")
axes[1].set_title("Top-20 ranking stability vs $\\lambda$")
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "fig11_entropy_lambda.png", dpi=150); plt.close()

# =============================================================================
# §8 / §9 / §10  Trading simulations on future data
# =============================================================================
print("[§8-10] Trading simulations")

r_int = 0.001 / 100  # 0.001% daily
g_A, g_C = 0.7, 0.2

# Build forward-looking signals from each tool
S_for_macd = S1[split_idx - 50:]
m_line_f, m_sig_f, _ = sh.macd(S_for_macd)
m_line_fut = m_line_f[50:]
m_sig_fut = m_sig_f[50:]

def macd_signal(t):
    if t == 0 or t >= len(m_line_fut):
        return "hold"
    if m_line_fut[t] > m_sig_fut[t] and m_line_fut[t - 1] <= m_sig_fut[t - 1]:
        return "buy"
    if m_line_fut[t] < m_sig_fut[t] and m_line_fut[t - 1] >= m_sig_fut[t - 1]:
        return "sell"
    return "buy" if m_line_fut[t] > m_sig_fut[t] else "sell"

def bayes_signal(x):
    s = bayes_score(x, cond_params, priors)
    y = max(s, key=s.get)
    return {"U": "buy", "D": "sell", "H": "hold"}[y]

# Association rules with confidence threshold
rule_lookup = {}
for _, r in rules.iterrows():
    key = tuple(r["pattern"])
    if (key not in rule_lookup) or (r["confidence"] > rule_lookup[key][1]):
        rule_lookup[key] = (r["outcome"], r["confidence"])

Y_full_extended = sh.digitize(np.concatenate([X1_past[-5:], X1_fut]), eps)

def assoc_signal(t, conf_thresh=0.5):
    if t < 5:
        return "hold"
    pat = tuple(Y_full_extended[t:t + 5])
    if pat in rule_lookup:
        out, conf = rule_lookup[pat]
        if conf >= conf_thresh:
            return {"U": "buy", "D": "sell", "H": "hold"}[out]
    return "hold"

def majority_signal(t, x_prev):
    sigs = [macd_signal(t), bayes_signal(x_prev) if x_prev is not None else "hold",
            assoc_signal(t)]
    nb = sum(s == "buy" for s in sigs); ns = sum(s == "sell" for s in sigs)
    if nb > ns: return "buy"
    if ns > nb: return "sell"
    return "hold"

def bma_signal_and_strength(t, x_prev):
    """Bayesian model averaging — return ('buy'/'sell'/'hold', confidence in [0,1])."""
    weights = []
    if x_prev is not None:
        sc = bayes_score(x_prev, cond_params, priors)
        total = sum(sc.values())
        weights.append({"U": sc["U"] / total,
                        "D": sc["D"] / total,
                        "H": sc["H"] / total})
    else:
        weights.append({"U": 1 / 3, "D": 1 / 3, "H": 1 / 3})
    s_macd = macd_signal(t)
    weights.append({"U": 0.6 if s_macd == "buy" else 0.2,
                    "D": 0.6 if s_macd == "sell" else 0.2,
                    "H": 0.6 if s_macd == "hold" else 0.2})
    s_assoc = assoc_signal(t)
    weights.append({"U": 0.55 if s_assoc == "buy" else 0.225,
                    "D": 0.55 if s_assoc == "sell" else 0.225,
                    "H": 0.55 if s_assoc == "hold" else 0.225})
    avg = {y: float(np.mean([w[y] for w in weights])) for y in ("D", "U", "H")}
    y_star = max(avg, key=avg.get)
    return ({"U": "buy", "D": "sell", "H": "hold"}[y_star], float(avg[y_star]))

# §8: One stock + money
def sim_s8(g, signals_func):
    n_ = len(S1_fut)
    M = np.zeros(n_); N = np.zeros(n_); V = np.zeros(n_)
    M[0] = 100_000.0
    V[0] = M[0]
    for t in range(1, n_):
        M[t] = M[t - 1] * (1 + r_int)
        N[t] = N[t - 1]
        x_prev = float(np.log(S1_fut[t] / S1_fut[t - 1]))
        sig = signals_func(t, x_prev)
        if sig == "buy" and M[t] > 0:
            spend = g * M[t]; M[t] -= spend; N[t] += spend / S1_fut[t]
        elif sig == "sell" and N[t] > 0:
            sold = g * N[t]; M[t] += sold * S1_fut[t]; N[t] -= sold
        V[t] = M[t] + N[t] * S1_fut[t]
    return V

def sim_s8_bma(g_max):
    n_ = len(S1_fut)
    M = np.zeros(n_); N = np.zeros(n_); V = np.zeros(n_)
    M[0] = 100_000.0; V[0] = M[0]
    for t in range(1, n_):
        M[t] = M[t - 1] * (1 + r_int); N[t] = N[t - 1]
        x_prev = float(np.log(S1_fut[t] / S1_fut[t - 1]))
        sig, conf = bma_signal_and_strength(t, x_prev)
        g = g_max * max(0.0, (conf - 1 / 3) / (1 - 1 / 3))  # rescale to [0,1]
        if sig == "buy" and M[t] > 0:
            spend = g * M[t]; M[t] -= spend; N[t] += spend / S1_fut[t]
        elif sig == "sell" and N[t] > 0:
            sold = g * N[t]; M[t] += sold * S1_fut[t]; N[t] -= sold
        V[t] = M[t] + N[t] * S1_fut[t]
    return V

V8_A = sim_s8(g_A, majority_signal)
V8_C = sim_s8(g_C, majority_signal)
V8_BMA = sim_s8_bma(g_A)
V0_money = 100_000.0 * (1 + r_int) ** np.arange(len(S1_fut))

m8 = {
    "Aggressive (g=0.7)": sh.compute_metrics(V8_A),
    "Conservative (g=0.2)": sh.compute_metrics(V8_C),
    "BMA-weighted greed":   sh.compute_metrics(V8_BMA),
    "Min-risk (cash@0.001%)": sh.compute_metrics(V0_money),
}
RESULTS["s8_metrics"] = m8

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(dates_fut, V8_A, lw=1.0, color="red", label=f"Aggressive ($g={g_A}$)")
ax.plot(dates_fut, V8_C, lw=1.0, color="blue", label=f"Conservative ($g={g_C}$)")
ax.plot(dates_fut, V8_BMA, lw=1.2, color="green", label="BMA-weighted greed")
ax.plot(dates_fut, V0_money, lw=1.0, color="black", ls="--", label="Min-risk")
ax.set_title("§8 — One stock + money")
ax.set_ylabel("USD"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig12_s8.png", dpi=150); plt.close()

# §9: Two stocks
p0_est = sh.compute_p0(X1_past, X2_past)
RESULTS["p0_est_past"] = float(p0_est)
N1_0 = p0_est * 100_000.0 / S1_fut[0]
N2_0 = (1 - p0_est) * 100_000.0 / S2_fut[0]

def sim_s9(g, signals_func):
    n_ = len(S1_fut)
    N1 = np.full(n_, N1_0); N2 = np.full(n_, N2_0); V = np.zeros(n_)
    V[0] = N1[0] * S1_fut[0] + N2[0] * S2_fut[0]
    for t in range(1, n_):
        N1[t] = N1[t - 1]; N2[t] = N2[t - 1]
        x_prev = float(np.log(S1_fut[t] / S1_fut[t - 1]))
        sig = signals_func(t, x_prev)
        if sig == "buy" and N2[t] > 0:
            n2_sell = g * N2[t]; val = n2_sell * S2_fut[t]
            N2[t] -= n2_sell; N1[t] += val / S1_fut[t]
        elif sig == "sell" and N1[t] > 0:
            n1_sell = g * N1[t]; val = n1_sell * S1_fut[t]
            N1[t] -= n1_sell; N2[t] += val / S2_fut[t]
        V[t] = N1[t] * S1_fut[t] + N2[t] * S2_fut[t]
    return V, N1, N2

V9_A, N1_A, N2_A = sim_s9(g_A, majority_signal)
V9_C, N1_C, N2_C = sim_s9(g_C, majority_signal)
V9_0 = N1_0 * S1_fut + N2_0 * S2_fut
m9 = {
    "Aggressive": sh.compute_metrics(V9_A),
    "Conservative": sh.compute_metrics(V9_C),
    "Min-risk (buy & hold)": sh.compute_metrics(V9_0),
}
RESULTS["s9_metrics"] = m9

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(dates_fut, V9_A, lw=1.0, color="red", label=f"Aggressive ($g={g_A}$)")
ax.plot(dates_fut, V9_C, lw=1.0, color="blue", label=f"Conservative ($g={g_C}$)")
ax.plot(dates_fut, V9_0, lw=1.0, color="black", ls="--", label="Min-risk (buy & hold)")
ax.set_title("§9 — Two stocks")
ax.set_ylabel("USD"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig13_s9.png", dpi=150); plt.close()

# §9.1 efficient frontier
def sim_s9_ef(g, signals_func):
    n_ = len(S1_fut)
    N1 = np.full(n_, N1_0); N2 = np.full(n_, N2_0); V = np.zeros(n_)
    V[0] = N1[0] * S1_fut[0] + N2[0] * S2_fut[0]
    # rolling (expanding) sigmas
    Xc1 = np.concatenate([X1_past, X1_fut])
    Xc2 = np.concatenate([X2_past, X2_fut])
    for t in range(1, n_):
        N1[t] = N1[t - 1]; N2[t] = N2[t - 1]
        end = split_idx + t
        x1h = Xc1[:end]; x2h = Xc2[:end]
        s1 = float(np.std(x1h, ddof=1)); s2 = float(np.std(x2h, ddof=1))
        rho = float(np.corrcoef(x1h, x2h)[0, 1])
        p0_t = sh.compute_p0(x1h, x2h)
        sig0 = float(np.sqrt((p0_t * s1) ** 2 + ((1 - p0_t) * s2) ** 2 + 2 * p0_t * (1 - p0_t) * rho * s1 * s2))
        Vc = N1[t] * S1_fut[t] + N2[t] * S2_fut[t]
        p_cur = N1[t] * S1_fut[t] / max(Vc, 1e-9)
        sig_c = float(np.sqrt((p_cur * s1) ** 2 + ((1 - p_cur) * s2) ** 2 + 2 * p_cur * (1 - p_cur) * rho * s1 * s2))
        x_prev = float(np.log(S1_fut[t] / S1_fut[t - 1]))
        sig = signals_func(t, x_prev)
        if sig == "buy":
            sig_target = sig_c + g * (s1 - sig_c)
        elif sig == "sell":
            sig_target = sig_c - g * (sig_c - sig0)
        else:
            V[t] = Vc; continue
        # solve sig_target² as quadratic in p
        a = s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2
        b = 2 * rho * s1 * s2 - 2 * s2 ** 2
        c = s2 ** 2 - sig_target ** 2
        disc = b ** 2 - 4 * a * c
        if disc < 0 or a == 0:
            V[t] = Vc; continue
        p1 = (-b + np.sqrt(disc)) / (2 * a)
        p2 = (-b - np.sqrt(disc)) / (2 * a)
        cands = [p for p in (p1, p2) if p0_t <= p <= 1.0]
        if not cands:
            cands = [float(np.clip(p1, p0_t, 1.0)), float(np.clip(p2, p0_t, 1.0))]
        p_target = min(cands, key=lambda p: abs(p - p_cur))
        N1[t] = p_target * Vc / S1_fut[t]
        N2[t] = (1 - p_target) * Vc / S2_fut[t]
        V[t] = N1[t] * S1_fut[t] + N2[t] * S2_fut[t]
    return V, N1, N2

V9ef_A, _, _ = sim_s9_ef(g_A, majority_signal)
V9ef_C, _, _ = sim_s9_ef(g_C, majority_signal)
m9ef = {
    "EF Aggressive": sh.compute_metrics(V9ef_A),
    "EF Conservative": sh.compute_metrics(V9ef_C),
    "Min-risk": sh.compute_metrics(V9_0),
}
RESULTS["s9ef_metrics"] = m9ef

# Plot p(t) vs p0(t)
p_A_arr = N1_A * S1_fut / np.maximum(N1_A * S1_fut + N2_A * S2_fut, 1e-9)
p0_fut = np.array([sh.compute_p0(np.concatenate([X1_past, X1_fut[:t]]),
                                  np.concatenate([X2_past, X2_fut[:t]]))
                   if t > 30 else p0_est for t in range(len(S1_fut))])
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates_fut, p_A_arr, lw=0.8, color="red", label=f"$p(t)$ (aggressive)")
ax.plot(dates_fut, p0_fut, lw=1.5, color="black", ls="--", label="$p_0(t)$")
ax.fill_between(dates_fut, 0, 1,
                where=p_A_arr < p0_fut, alpha=0.12, color="orange",
                label="$p(t)<p_0(t)$ (suboptimal)")
ax.set_title("§9.1 — Portfolio fraction vs min-risk fraction")
ax.set_ylabel("fraction in AAL"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig14_p_vs_p0.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(dates_fut, V9ef_A, lw=1.0, color="red", label=f"EF Aggressive")
ax.plot(dates_fut, V9ef_C, lw=1.0, color="blue", label=f"EF Conservative")
ax.plot(dates_fut, V9_0, lw=1.0, color="black", ls="--", label="Min-risk")
ax.set_title("§9.1 — Portfolio with efficient-frontier rebalancing")
ax.set_ylabel("USD"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig15_s9_ef.png", dpi=150); plt.close()

# §10: Two stocks + money — BMA-weighted greed for both
def signal_dal_bayes(t):
    if t == 0:
        return "hold"
    x_prev = float(np.log(S2_fut[t] / S2_fut[t - 1]))
    # Rebuild lightweight Bayes for DAL
    return "buy" if x_prev > 0.005 else ("sell" if x_prev < -0.005 else "hold")

def sim_s10(g):
    n_ = len(S1_fut)
    M = np.zeros(n_); N1 = np.zeros(n_); N2 = np.zeros(n_); V = np.zeros(n_)
    M[0] = 100_000.0; V[0] = M[0]
    for t in range(1, n_):
        M[t] = M[t - 1] * (1 + r_int); N1[t] = N1[t - 1]; N2[t] = N2[t - 1]
        x1_prev = float(np.log(S1_fut[t] / S1_fut[t - 1]))
        s1, conf1 = bma_signal_and_strength(t, x1_prev)
        g1 = g * max(0.0, (conf1 - 1 / 3) / (1 - 1 / 3))
        if s1 == "buy" and M[t] > 0:
            spend = g1 * M[t]; M[t] -= spend; N1[t] += spend / S1_fut[t]
        elif s1 == "sell" and N1[t] > 0:
            sold = g1 * N1[t]; M[t] += sold * S1_fut[t]; N1[t] -= sold
        s2 = signal_dal_bayes(t)
        if s2 == "buy" and M[t] > 0:
            spend = g * M[t]; M[t] -= spend; N2[t] += spend / S2_fut[t]
        elif s2 == "sell" and N2[t] > 0:
            sold = g * N2[t]; M[t] += sold * S2_fut[t]; N2[t] -= sold
        V[t] = M[t] + N1[t] * S1_fut[t] + N2[t] * S2_fut[t]
    return V

V10_A = sim_s10(g_A); V10_C = sim_s10(g_C)
m10 = {
    "Aggressive (BMA-weighted)": sh.compute_metrics(V10_A),
    "Conservative (BMA-weighted)": sh.compute_metrics(V10_C),
    "Min-risk (cash)": sh.compute_metrics(V0_money),
}
RESULTS["s10_metrics"] = m10

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(dates_fut, V10_A, lw=1.0, color="red", label="Aggressive (BMA)")
ax.plot(dates_fut, V10_C, lw=1.0, color="blue", label="Conservative (BMA)")
ax.plot(dates_fut, V0_money, lw=1.0, color="black", ls="--", label="Min-risk")
ax.set_title("§10 — Two stocks + money (BMA-weighted greed)")
ax.set_ylabel("USD"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig16_s10.png", dpi=150); plt.close()

# Drawdown plot for §10
def drawdown(V):
    peak = np.maximum.accumulate(V); return V / peak - 1

fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(dates_fut, drawdown(V10_A), 0, color="red", alpha=0.4, label="Aggressive")
ax.fill_between(dates_fut, drawdown(V10_C), 0, color="blue", alpha=0.4, label="Conservative")
ax.set_title("§10 — Drawdown comparison")
ax.set_ylabel("drawdown"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig17_drawdown.png", dpi=150); plt.close()

# =============================================================================
# Save results
# =============================================================================
def to_jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, tuple): return list(o)
    return o

(ROOT / "results.json").write_text(
    json.dumps({k: to_jsonable(v) for k, v in RESULTS.items()},
               indent=2, default=str))

print("\nHU_report analysis done.")
print(f"  figures: {len(list(FIG.glob('*.png')))}")
print(f"  results.json: {len(RESULTS)} keys")
