"""
LAN_report/analysis.py
======================
Machine-Learning and Time-Series Forecasting for Trading.

Companion report: LAN Tianwei --- MSDM5058 Project II.

Direction (per group plan):
  - ARIMA(p,d,q) and GARCH(1,1) baselines.
  - Random Forest, XGBoost classifiers; lightweight LSTM (PyTorch) classifier
    when the toolchain allows; otherwise an MLP fallback.
  - Walk-forward cross-validation with rolling 252-day window.
  - SHAP feature importance for the tree models.
  - PrefixSpan-style sequential mining as alternative to brute-force k-grams.
  - ML-probability-weighted greed (replaces classical Bayes detector signal).
  - Meta-labelling à la López de Prado for trade selection.

The script imports the shared dataset and helpers from data/shared.py so all
four group reports start from identical inputs.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
warnings.filterwarnings("ignore")

from data import shared as sh  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
np.random.seed(sh.SEED)

print("=" * 70)
print("LAN_report — ML & Time-Series Forecasting")
print("=" * 70)

# ---------------------------------------------------------------------------
# Imports for ML stack
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, brier_score_loss, roc_auc_score,
)
import xgboost as xgb
import statsmodels.api as smapi
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# NOTE: torch on macOS Python 3.13 has shown CPU-thread deadlocks for the
# small LSTM used here.  We fall back to a sklearn MLP with the same
# rolling-window inputs --- it is comparable for the 1-day classification
# problem and trains in seconds.
HAS_TORCH = False
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Data: enriched panel with macro + volume features
# ---------------------------------------------------------------------------
panel = sh.load_panel(extra=("VIX", "SPY", "TNX", "XLK"))
past, future, t0 = sh.train_test_split(panel)
print(f"Panel: {len(panel):d} rows  ({panel.index[0].date()} → {panel.index[-1].date()})")
print(f"Train: {len(past):d} rows  Test (post t0={t0.date()}): {len(future):d} rows")

# Returns
P_a = panel["AAL_Close"].to_numpy(float)
P_m = panel["DAL_Close"].to_numpy(float)
X_a = sh.log_returns(P_a)
X_m = sh.log_returns(P_m)
dates = panel.index[1:]

# Volume features
vol_a = panel["AAL_Volume"].to_numpy(float)
vol_m = panel["DAL_Volume"].to_numpy(float)
log_vol_a = np.log(vol_a + 1.0)
log_vol_m = np.log(vol_m + 1.0)

# Macro features (lagged so they are available at decision time)
vix = panel["VIX_Close"].to_numpy(float)
spy = panel["SPY_Close"].to_numpy(float)
tnx = panel["TNX_Close"].to_numpy(float)
xlk = panel["XLK_Close"].to_numpy(float)
ret_spy = sh.log_returns(spy)
ret_xlk = sh.log_returns(xlk)
ret_tnx = sh.log_returns(np.maximum(tnx, 1e-6))

# Technical indicators on AAL (representative)
macd_line, macd_sig, macd_hist = sh.macd(P_a)
rsi_a = sh.rsi(P_a)
boll_m, boll_u, boll_l = sh.bollinger(P_a)
atr_a = sh.atr(panel["AAL_High"].to_numpy(float),
               panel["AAL_Low"].to_numpy(float),
               P_a)

# ---------------------------------------------------------------------------
# Feature matrix
#
# Each row at index t (t >= 1) describes information available at end of day t
# and predicts sign(X_a(t+1)).
# ---------------------------------------------------------------------------
def shift1(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out


# Align indicator/macro to the X_a index (returns are length N-1, indicators
# are length N — drop their first element).
ind_idx = slice(1, None)


feat_dict = {
    "ret_aal_lag1": shift1(X_a),
    "ret_dal_lag1": shift1(X_m),
    "ret_aal_lag2": np.concatenate([[np.nan, np.nan], X_a[:-2]]),
    "ret_dal_lag2": np.concatenate([[np.nan, np.nan], X_m[:-2]]),
    "ret_spy_lag1":  shift1(ret_spy),
    "ret_xlk_lag1":  shift1(ret_xlk),
    "ret_tnx_lag1":  shift1(ret_tnx),
    "vol_z_aal":    (log_vol_a[ind_idx] - pd.Series(log_vol_a[ind_idx]).rolling(20).mean().to_numpy())
                     / pd.Series(log_vol_a[ind_idx]).rolling(20).std().to_numpy(),
    "vol_z_dal":    (log_vol_m[ind_idx] - pd.Series(log_vol_m[ind_idx]).rolling(20).mean().to_numpy())
                     / pd.Series(log_vol_m[ind_idx]).rolling(20).std().to_numpy(),
    "vix_lvl":       vix[ind_idx],
    "vix_chg":       np.diff(vix, prepend=vix[0])[ind_idx],
    "macd_hist":     macd_hist[ind_idx],
    "rsi":           rsi_a[ind_idx],
    "boll_z":        (P_a[ind_idx] - boll_m[ind_idx]) / np.where(np.isnan(boll_u[ind_idx] - boll_m[ind_idx]),
                                                                  1e-12,
                                                                  (boll_u[ind_idx] - boll_m[ind_idx]) / 2),
    "atr_pct":       atr_a[ind_idx] / P_a[ind_idx],
}
F = pd.DataFrame(feat_dict, index=dates)
F = F.replace([np.inf, -np.inf], np.nan)

# Targets
sign_next = (X_a[1:] > 0).astype(int)        # next-day binary direction
y_target = pd.Series(np.concatenate([sign_next, [np.nan]]), index=dates)

# Drop rows with any NaN
df = F.copy()
df["y"] = y_target
df = df.dropna()
print(f"Features (after dropna): {df.shape[0]:d} rows × {df.shape[1] - 1:d} cols")

# ---------------------------------------------------------------------------
# Train / test split aligned with the project's t0
# ---------------------------------------------------------------------------
df_tr = df.loc[df.index <= t0]
df_te = df.loc[df.index > t0]
X_tr_full, y_tr_full = df_tr.drop(columns=["y"]).to_numpy(), df_tr["y"].to_numpy().astype(int)
X_te,      y_te      = df_te.drop(columns=["y"]).to_numpy(), df_te["y"].to_numpy().astype(int)
feat_names = list(df_tr.drop(columns=["y"]).columns)

# Standardise (fit on train only)
scaler = StandardScaler().fit(X_tr_full)
X_tr_full_s = scaler.transform(X_tr_full)
X_te_s = scaler.transform(X_te)

# ---------------------------------------------------------------------------
# Walk-forward cross-validation on training data
# ---------------------------------------------------------------------------
def walk_forward_splits(n: int, train_w: int = 252, step: int = 63):
    """Yield (idx_train, idx_val) pairs over a rolling window."""
    out = []
    start = 0
    while start + train_w + step <= n:
        idx_tr = np.arange(start, start + train_w)
        idx_va = np.arange(start + train_w, start + train_w + step)
        out.append((idx_tr, idx_va))
        start += step
    return out


splits = walk_forward_splits(len(y_tr_full), train_w=252, step=126)
print(f"Walk-forward folds: {len(splits)}")


def _eval_clf(name, model, Xtr, ytr, Xva, yva):
    t0c = time.time()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xva)[:, 1]
    preds = (proba > 0.5).astype(int)
    return {
        "model": name,
        "acc": accuracy_score(yva, preds),
        "f1": f1_score(yva, preds, zero_division=0),
        "log_loss": log_loss(yva, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "brier": brier_score_loss(yva, proba),
        "auc": roc_auc_score(yva, proba) if len(np.unique(yva)) > 1 else float("nan"),
        "fit_sec": time.time() - t0c,
    }


# Baselines
def mk_logreg():
    return LogisticRegression(max_iter=200, n_jobs=1)


def mk_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  random_state=sh.SEED, n_jobs=-1)


def mk_xgb():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=sh.SEED, eval_metric="logloss",
        n_jobs=-1, verbosity=0,
    )


cv_records = []
print("Running walk-forward CV…")
for fold_id, (idx_tr, idx_va) in enumerate(splits):
    Xtr_, ytr_ = X_tr_full_s[idx_tr], y_tr_full[idx_tr]
    Xva_, yva_ = X_tr_full_s[idx_va], y_tr_full[idx_va]
    if len(np.unique(ytr_)) < 2 or len(np.unique(yva_)) < 2:
        continue
    for name, mk in (
        ("LogReg", mk_logreg),
        ("RandomForest", mk_rf),
        ("XGBoost", mk_xgb),
    ):
        rec = _eval_clf(name, mk(), Xtr_, ytr_, Xva_, yva_)
        rec["fold"] = fold_id
        cv_records.append(rec)

cv_df = pd.DataFrame(cv_records)
cv_summary = cv_df.groupby("model")[["acc", "f1", "log_loss", "brier", "auc"]].mean()
cv_summary["folds"] = cv_df.groupby("model").size()
print(cv_summary.round(4).to_string())

# ---------------------------------------------------------------------------
# Final fit on the full training window, evaluate on the future window
# ---------------------------------------------------------------------------
final_models = {
    "LogReg": mk_logreg().fit(X_tr_full_s, y_tr_full),
    "RandomForest": mk_rf().fit(X_tr_full_s, y_tr_full),
    "XGBoost": mk_xgb().fit(X_tr_full_s, y_tr_full),
}

oos_records = {}
for name, model in final_models.items():
    proba = model.predict_proba(X_te_s)[:, 1]
    preds = (proba > 0.5).astype(int)
    oos_records[name] = {
        "acc": accuracy_score(y_te, preds),
        "f1": f1_score(y_te, preds, zero_division=0),
        "log_loss": log_loss(y_te, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "brier": brier_score_loss(y_te, proba),
        "auc": roc_auc_score(y_te, proba),
    }
print("Out-of-sample performance:")
print(pd.DataFrame(oos_records).T.round(4).to_string())

# ---------------------------------------------------------------------------
# Optional: lightweight LSTM (PyTorch). Falls back to MLP if torch missing.
# ---------------------------------------------------------------------------
def make_seq_tensors(X, y, win=10):
    """Roll up a window of `win` rows into one sample."""
    n = len(X) - win + 1
    out_x = np.zeros((n, win, X.shape[1]), dtype=np.float32)
    out_y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out_x[i] = X[i: i + win]
        out_y[i] = y[i + win - 1]
    return out_x, out_y


print("Training MLP on rolling-window sequence features…")
win = 10
Xs_tr, ys_tr = make_seq_tensors(X_tr_full_s, y_tr_full, win)
Xs_te, ys_te = make_seq_tensors(X_te_s, y_te, win)
# Flatten the (n, win, d) tensor into (n, win*d) for MLPClassifier.
Xs_tr_flat = Xs_tr.reshape(len(Xs_tr), -1)
Xs_te_flat = Xs_te.reshape(len(Xs_te), -1)
mlp_t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                    solver="adam", learning_rate_init=1e-3,
                    max_iter=40, random_state=sh.SEED, batch_size=64,
                    early_stopping=False)
mlp.fit(Xs_tr_flat, ys_tr.astype(int))
p_mlp = np.clip(mlp.predict_proba(Xs_te_flat)[:, 1], 1e-6, 1 - 1e-6)
pred_mlp = (p_mlp > 0.5).astype(int)
y_mlp_int = ys_te.astype(int)
mlp_record = {
    "acc": accuracy_score(y_mlp_int, pred_mlp),
    "f1": f1_score(y_mlp_int, pred_mlp, zero_division=0),
    "log_loss": log_loss(y_mlp_int, p_mlp, labels=[0, 1]),
    "brier": brier_score_loss(y_mlp_int, p_mlp),
    "auc": roc_auc_score(y_mlp_int, p_mlp) if len(np.unique(y_mlp_int)) > 1 else float("nan"),
    "fit_sec": time.time() - mlp_t0,
}
oos_records["MLP"] = {k: mlp_record[k] for k in ("acc", "f1", "log_loss", "brier", "auc")}
print(f"MLP OOS: {oos_records['MLP']}")

# ---------------------------------------------------------------------------
# Feature importance via XGBoost gain & permutation
# ---------------------------------------------------------------------------
xgb_model = final_models["XGBoost"]
gains = xgb_model.feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "xgb_gain": gains}).sort_values(
    "xgb_gain", ascending=False
)
print("Feature importance (XGBoost):")
print(imp_df.to_string(index=False))

# Permutation importance on the OOS test
def permutation_importance(model, X, y, baseline_proba_fn, n_rep=5, rng=None):
    rng = rng or np.random.default_rng(sh.SEED)
    base = baseline_proba_fn(model, X)
    base_loss = log_loss(y, np.clip(base, 1e-6, 1 - 1e-6), labels=[0, 1])
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        diffs = []
        for _ in range(n_rep):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            p = baseline_proba_fn(model, Xp)
            diffs.append(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]) - base_loss)
        out[j] = float(np.mean(diffs))
    return out


def _proba(m, X):
    return m.predict_proba(X)[:, 1]


perm = permutation_importance(xgb_model, X_te_s, y_te, _proba, n_rep=2)
imp_df["perm_oos"] = perm
imp_df = imp_df.sort_values("perm_oos", ascending=False)
print("Permutation importance on OOS:")
print(imp_df.to_string(index=False))

# Plot importances
fig, ax = plt.subplots(figsize=(7, 5))
imp_plot = imp_df.sort_values("perm_oos")
ax.barh(imp_plot["feature"], imp_plot["perm_oos"], color="#3a76d8")
ax.set_xlabel("Increase in log-loss when permuted")
ax.set_title("Permutation feature importance (XGBoost, OOS)")
plt.tight_layout()
plt.savefig(FIG / "feature_importance.png", dpi=140)
plt.close()

# ---------------------------------------------------------------------------
# ARIMA / GARCH baselines on AAL log-returns
# ---------------------------------------------------------------------------
print("Fitting ARIMA(1,0,1) baseline…")
ret_tr = X_a[: len(panel.loc[panel.index <= t0]) - 1]
ret_te = X_a[len(panel.loc[panel.index <= t0]) - 1:]
arima_res = ARIMA(ret_tr * 100, order=(1, 0, 1)).fit()
print(arima_res.summary().tables[1])

# 1-step ARIMA forecasts on OOS.  Refit every K days; in between, reuse the
# current parameters and just feed new observations via .apply (cheap).
print("Walk-forward ARIMA forecasts on OOS (refit every 126d)…")
arima_signs_correct = 0
arima_n = 0
arima_forecast = []
hist = list(ret_tr * 100)
refit_every = 126
m = ARIMA(np.array(hist[-500:]), order=(1, 0, 1)).fit(method_kwargs={"warn_convergence": False})
for k, r in enumerate(ret_te * 100):
    if k > 0 and k % refit_every == 0:
        m = ARIMA(np.array(hist[-500:]), order=(1, 0, 1)).fit(method_kwargs={"warn_convergence": False})
    try:
        f = float(m.forecast(steps=1)[0])
    except Exception:
        f = 0.0
    arima_forecast.append(f)
    if r != 0:
        arima_signs_correct += int(np.sign(f) == np.sign(r))
        arima_n += 1
    hist.append(r)
    # Cheaply incorporate the new observation without refitting parameters.
    if k % refit_every != refit_every - 1:
        try:
            m = m.append([r], refit=False)
        except Exception:
            pass
arima_acc = arima_signs_correct / max(arima_n, 1)
print(f"ARIMA(1,0,1) sign-accuracy on OOS: {arima_acc:.4f} over {arima_n} obs")

print("Fitting GARCH(1,1) baseline…")
garch = arch_model(ret_tr * 100, vol="Garch", p=1, q=1, dist="t")
garch_res = garch.fit(disp="off")
print(garch_res.summary().tables[1])

# Forecast next-day vol on OOS via rolling fit (every 63 trading days).
print("Walk-forward GARCH(1,1) forecasts on OOS (refit every 63d)…")
garch_var = []
hist = list(ret_tr * 100)
m = arch_model(np.array(hist[-500:]), vol="Garch", p=1, q=1, dist="t").fit(disp="off")
for k, r in enumerate(ret_te * 100):
    if k > 0 and k % 63 == 0:
        m = arch_model(np.array(hist[-500:]), vol="Garch", p=1, q=1, dist="t").fit(disp="off")
    f = m.forecast(horizon=1, reindex=False)
    garch_var.append(float(f.variance.iloc[-1, 0]))
    hist.append(r)
garch_vol = np.sqrt(np.array(garch_var)) / 100.0  # back to daily-decimal scale
print(f"GARCH OOS mean forecast vol (daily): {np.mean(garch_vol):.4%}")

# ---------------------------------------------------------------------------
# PrefixSpan-style sequential mining (alternative to brute-force k-grams)
# ---------------------------------------------------------------------------
def prefix_span(seq, min_sup=80, max_len=4):
    """Tiny PrefixSpan implementation for a single long string sequence.

    Returns dict pattern → support (count of occurrences in sliding windows).
    """
    n = len(seq)
    counts = Counter(seq)
    freq = {tuple([s]): c for s, c in counts.items() if c >= min_sup}
    queue = list(freq.items())
    while queue:
        pat, _ = queue.pop()
        if len(pat) >= max_len:
            continue
        local = Counter()
        L = len(pat)
        for i in range(n - L):
            if tuple(seq[i: i + L]) == pat:
                local[seq[i + L]] += 1
        for nxt, c in local.items():
            if c >= min_sup:
                new = pat + (nxt,)
                freq[new] = c
                queue.append((new, c))
    return freq


eps = sh.load_constants()["epsilon"]
sym_a = sh.digitize(X_a, eps)
patterns = prefix_span(list(sym_a), min_sup=120, max_len=4)
top_patterns = sorted(patterns.items(), key=lambda kv: -kv[1])[:12]
print("Top frequent patterns (PrefixSpan):")
for p, c in top_patterns:
    print(f"  {''.join(p):<6}  {c}")

# Conditional probability of next-up vs each pattern
seq = list(sym_a)
N = len(seq)
patt_stats = []
for pat, sup in patterns.items():
    L = len(pat)
    if N - L - 1 <= 0:
        continue
    nxt = []
    for i in range(N - L):
        if tuple(seq[i: i + L]) == pat:
            if i + L < N:
                nxt.append(seq[i + L])
    if len(nxt) < 50:
        continue
    cnt = Counter(nxt)
    total = sum(cnt.values())
    p_up = cnt["U"] / total
    patt_stats.append((pat, total, p_up))
patt_stats.sort(key=lambda r: abs(r[2] - 0.5), reverse=True)
print("Most informative patterns (|P(U|pat)-0.5|):")
for pat, n_, p in patt_stats[:8]:
    print(f"  {''.join(pat):<6}  n={n_:5d}  P(U|pat)={p:.3f}")

# ---------------------------------------------------------------------------
# Trading simulators on the test window
# ---------------------------------------------------------------------------
print("Building trading simulators on the OOS window…")
ret_a_test = X_a[len(past) - 1:]
ret_m_test = X_m[len(past) - 1:]

# Buy-and-hold baselines
def simulate_bh(rets):
    V = np.empty(len(rets) + 1)
    V[0] = 1.0
    for t, r in enumerate(rets):
        V[t + 1] = V[t] * np.exp(r)
    return V


V_bh_a = simulate_bh(ret_a_test)
V_bh_m = simulate_bh(ret_m_test)

# Align ML probability series with test returns
proba_xgb_full = pd.Series(final_models["XGBoost"].predict_proba(X_te_s)[:, 1],
                           index=df_te.index)
proba_rf_full = pd.Series(final_models["RandomForest"].predict_proba(X_te_s)[:, 1],
                          index=df_te.index)


def simulate_prob_weighted(prob_series, ret_a_idx, half_kelly=True):
    """g(t) = clip(2 * (p - 0.5), 0, 1) (* 0.5 if half_kelly).
    Cash earns rf_daily."""
    rf = 0.001 / 100
    idx = panel.index[len(past):]  # future dates aligned with ret_a_test
    V = np.empty(len(ret_a_test) + 1)
    V[0] = 1.0
    p = prob_series.reindex(idx).ffill().fillna(0.5).to_numpy()
    for t, r in enumerate(ret_a_test):
        g = float(np.clip(2.0 * (p[t] - 0.5), 0.0, 1.0))
        if half_kelly:
            g *= 0.5
        V[t + 1] = V[t] * ((1 - g) * (1 + rf) + g * np.exp(r))
    return V


V_xgb = simulate_prob_weighted(proba_xgb_full, ret_a_test, half_kelly=False)
V_rf = simulate_prob_weighted(proba_rf_full, ret_a_test, half_kelly=False)
V_xgb_hk = simulate_prob_weighted(proba_xgb_full, ret_a_test, half_kelly=True)


def simulate_meta_label(prob_series, ret_a, threshold=0.55):
    """Meta-labelling à la López de Prado. A primary 'always long' model is
    refined by a secondary classifier: only take the trade when secondary
    probability of up exceeds threshold."""
    rf = 0.001 / 100
    idx = panel.index[len(past):]
    V = np.empty(len(ret_a) + 1)
    V[0] = 1.0
    p = prob_series.reindex(idx).ffill().fillna(0.5).to_numpy()
    for t, r in enumerate(ret_a):
        take = p[t] >= threshold
        V[t + 1] = V[t] * (np.exp(r) if take else (1 + rf))
    return V


V_meta = simulate_meta_label(proba_xgb_full, ret_a_test, threshold=0.55)

# GARCH vol-target overlay on the XGB-weighted strategy
def simulate_voltarget(prob_series, ret_a, garch_vol_, sigma_star=0.15 / np.sqrt(252)):
    rf = 0.001 / 100
    idx = panel.index[len(past):]
    V = np.empty(len(ret_a) + 1)
    V[0] = 1.0
    p = prob_series.reindex(idx).ffill().fillna(0.5).to_numpy()
    for t, r in enumerate(ret_a):
        g_raw = float(np.clip(2.0 * (p[t] - 0.5), 0.0, 1.0))
        scale = 1.0 if t >= len(garch_vol_) else min(1.5, sigma_star / max(garch_vol_[t], 1e-4))
        g = float(np.clip(g_raw * scale, 0.0, 1.0))
        V[t + 1] = V[t] * ((1 - g) * (1 + rf) + g * np.exp(r))
    return V


V_volt = simulate_voltarget(proba_xgb_full, ret_a_test, garch_vol)

metrics = {
    "Buy & Hold AAL": sh.compute_metrics(V_bh_a),
    "Buy & Hold DAL": sh.compute_metrics(V_bh_m),
    "XGB (full)": sh.compute_metrics(V_xgb),
    "XGB (half-Kelly)": sh.compute_metrics(V_xgb_hk),
    "Random Forest": sh.compute_metrics(V_rf),
    "Meta-label@0.55": sh.compute_metrics(V_meta),
    "XGB + GARCH vol-target": sh.compute_metrics(V_volt),
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("Plotting…")

# Walk-forward CV summary
fig, ax = plt.subplots(figsize=(7, 4))
for name, grp in cv_df.groupby("model"):
    ax.plot(grp["fold"], grp["acc"], marker="o", label=name, alpha=0.7)
ax.set_xlabel("Walk-forward fold")
ax.set_ylabel("Accuracy")
ax.set_title("Walk-forward CV accuracy by model")
ax.axhline(0.5, color="grey", lw=0.6, ls="--")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "wfcv_accuracy.png", dpi=140)
plt.close()

# OOS probability vs realised return
fig, ax = plt.subplots(figsize=(8, 4))
xgb_p = proba_xgb_full.reindex(panel.index[len(past):]).ffill().fillna(0.5)
ax.plot(xgb_p.index, xgb_p.values, color="#1f77b4", lw=0.8, label="P(up | xgb)")
ax2 = ax.twinx()
ax2.plot(panel.index[len(past):], np.cumsum(ret_a_test) * 100, color="#d62728",
         lw=0.8, label="cum log-ret AAL")
ax.set_ylabel("P(up)")
ax2.set_ylabel("cum log-ret (%)")
ax.set_title("XGBoost up-probability and AAL cumulative log-return on OOS")
plt.tight_layout()
plt.savefig(FIG / "xgb_proba_vs_return.png", dpi=140)
plt.close()

# GARCH forecast vol
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(panel.index[len(past):][: len(garch_vol)],
        np.array(garch_vol) * np.sqrt(252) * 100, color="#ff7f0e", lw=0.9)
ax.set_ylabel("Annualised vol (\\%)")
ax.set_title("GARCH(1,1) one-step-ahead conditional volatility on OOS")
plt.tight_layout()
plt.savefig(FIG / "garch_vol.png", dpi=140)
plt.close()

# Equity curves
fig, ax = plt.subplots(figsize=(8, 4.4))
test_dates = panel.index[len(past) - 1:]
for name, V in [
    ("Buy & Hold AAL", V_bh_a),
    ("XGB", V_xgb),
    ("XGB (half-Kelly)", V_xgb_hk),
    ("Random Forest", V_rf),
    ("Meta-label@0.55", V_meta),
    ("XGB + GARCH vol-target", V_volt),
]:
    ax.plot(test_dates[: len(V)], V, lw=1.0, label=name)
ax.set_ylabel("Wealth (initial = 1)")
ax.set_title("Out-of-sample equity curves (post t0)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "equity_curves.png", dpi=140)
plt.close()

# Confusion-matrix-style probability calibration plot
def calibration(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        out.append((float(np.mean(p[mask])), float(np.mean(y[mask])), int(mask.sum())))
    return np.array(out)


fig, ax = plt.subplots(figsize=(5, 5))
for name, m in final_models.items():
    p = m.predict_proba(X_te_s)[:, 1]
    cal = calibration(p, y_te)
    ax.plot(cal[:, 0], cal[:, 1], "-o", lw=1.0, label=name, alpha=0.85)
ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.7)
ax.set_xlabel("Predicted P(up)")
ax.set_ylabel("Empirical P(up)")
ax.set_title("Reliability diagram (OOS)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "calibration.png", dpi=140)
plt.close()

# ARIMA residuals diagnostic
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
res = arima_res.resid
ax[0].plot(res, lw=0.5)
ax[0].set_title("ARIMA(1,0,1) residuals (in-sample)")
smapi.qqplot(res, line="45", ax=ax[1])
ax[1].set_title("Normal Q-Q plot")
plt.tight_layout()
plt.savefig(FIG / "arima_residuals.png", dpi=140)
plt.close()

# PrefixSpan top patterns bar chart
fig, ax = plt.subplots(figsize=(6, 3.4))
labs = ["".join(p) for p, _ in top_patterns]
vals = [c for _, c in top_patterns]
ax.bar(labs, vals, color="#2ca02c", alpha=0.85)
ax.set_ylabel("Support")
ax.set_title("PrefixSpan: top frequent patterns on digitised AAL returns")
plt.tight_layout()
plt.savefig(FIG / "prefixspan.png", dpi=140)
plt.close()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "panel_rows": int(len(panel)),
    "train_rows": int(len(past)),
    "test_rows": int(len(future)),
    "t0_date": str(t0.date()),
    "feat_names": feat_names,
    "cv_summary": cv_summary.round(4).to_dict(),
    "oos": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in oos_records.items()},
    "feat_importance": imp_df.to_dict(orient="list"),
    "arima_acc_oos": float(arima_acc),
    "arima_aic": float(arima_res.aic),
    "garch_loglik": float(garch_res.loglikelihood),
    "garch_vol_mean_ann": float(np.mean(garch_vol) * np.sqrt(252)),
    "top_patterns": [(("".join(p), int(c))) for p, c in top_patterns],
    "metrics": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
}
(ROOT / "results.json").write_text(json.dumps(results, indent=2))

print("=" * 70)
print("LAN_report — analysis complete")
print(f"Figures written to {FIG.relative_to(PROJECT)}")
print("=" * 70)
