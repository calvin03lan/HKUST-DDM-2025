"""
shared.py — common dataset loaders, technical indicators and metric helpers
shared by all four MSDM5058 Project II reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SEED = 20260516


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_constants() -> dict:
    return json.loads((DATA_DIR / "shared_constants.json").read_text())


def _read_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def load_panel(extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return a wide DataFrame with AAL + DAL OHLCV plus optional extras.

    Columns are flat strings ``{TICKER}_{Field}``.  Index is the intersection
    of the tickers' trading days so every row is fully populated for the core
    pair.
    """
    core = ["AAL", "DAL"]
    frames = {tk: _read_csv(tk) for tk in core + list(extra)}
    common = None
    for tk in core:
        idx = frames[tk].index
        common = idx if common is None else common.intersection(idx)
    panel_parts = []
    for tk, df in frames.items():
        df2 = df.copy()
        df2.columns = [f"{tk}_{c}" for c in df2.columns]
        panel_parts.append(df2)
    panel = pd.concat(panel_parts, axis=1).loc[common]
    panel = panel.dropna(subset=[f"{tk}_Close" for tk in core])
    return panel


def train_test_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the panel at the project's t=0 (3:1 past:future)."""
    consts = load_constants()
    t0 = pd.Timestamp(consts["t0_date"])
    past = panel.loc[panel.index <= t0]
    future = panel.loc[panel.index > t0]
    return past, future, t0


# ---------------------------------------------------------------------------
# Returns / digitisation
# ---------------------------------------------------------------------------
def log_returns(prices: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.log(p[1:] / p[:-1])


def digitize(X: np.ndarray, eps: float) -> np.ndarray:
    Y = np.full(len(X), "H", dtype="<U1")
    Y[X < -eps] = "D"
    Y[X > eps] = "U"
    return Y


# ---------------------------------------------------------------------------
# Min-risk fraction
# ---------------------------------------------------------------------------
def compute_p0(X1: np.ndarray, X2: np.ndarray) -> float:
    var1 = float(np.var(X1, ddof=1))
    var2 = float(np.var(X2, ddof=1))
    cov12 = float(np.cov(X1, X2, ddof=1)[0, 1])
    denom = var1 + var2 - 2 * cov12
    if denom <= 0:
        return 0.5
    return float(np.clip((var2 - cov12) / denom, 0.0, 1.0))


def rolling_p0(X1: np.ndarray, X2: np.ndarray, h: int | float) -> np.ndarray:
    """O(N) rolling-window p_0 via running sums.

    For finite ``h``: use a sliding window of ``h`` returns.
    For ``h == np.inf``: use the expanding window.
    """
    n = len(X1)
    out = np.full(n, np.nan)
    expanding = (h == np.inf) or (isinstance(h, (int, np.integer)) and h >= n)
    if expanding:
        s1 = np.cumsum(X1); s2 = np.cumsum(X2)
        s11 = np.cumsum(X1 * X1); s22 = np.cumsum(X2 * X2)
        s12 = np.cumsum(X1 * X2)
        for t in range(2, n):
            m1 = s1[t] / (t + 1); m2 = s2[t] / (t + 1)
            v1 = s11[t] / (t + 1) - m1 * m1
            v2 = s22[t] / (t + 1) - m2 * m2
            c12 = s12[t] / (t + 1) - m1 * m2
            denom = v1 + v2 - 2 * c12
            if denom > 1e-18:
                out[t] = float(np.clip((v2 - c12) / denom, 0.0, 1.0))
        return out

    H = int(h)
    # cumulative sums for fast windowed mean / variance
    c1 = np.concatenate([[0.0], np.cumsum(X1)])
    c2 = np.concatenate([[0.0], np.cumsum(X2)])
    c11 = np.concatenate([[0.0], np.cumsum(X1 * X1)])
    c22 = np.concatenate([[0.0], np.cumsum(X2 * X2)])
    c12 = np.concatenate([[0.0], np.cumsum(X1 * X2)])
    for t in range(H, n):
        a = t - H + 1
        b = t + 1
        m1 = (c1[b] - c1[a]) / H
        m2 = (c2[b] - c2[a]) / H
        v1 = (c11[b] - c11[a]) / H - m1 * m1
        v2 = (c22[b] - c22[a]) / H - m2 * m2
        cc = (c12[b] - c12[a]) / H - m1 * m2
        denom = v1 + v2 - 2 * cc
        if denom > 1e-18:
            out[t] = float(np.clip((v2 - cc) / denom, 0.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# Moving averages and friends (O(N) where possible)
# ---------------------------------------------------------------------------
def sma(x: np.ndarray, w: int) -> np.ndarray:
    """O(N) simple moving average via cumulative sums."""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full_like(x, np.nan)
    if w > len(x):
        return out
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def ema(x: np.ndarray, w: int) -> np.ndarray:
    """O(N) exponential moving average."""
    x = np.asarray(x, dtype=float)
    a = 2.0 / (w + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wma(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    out = np.full_like(x, np.nan)
    if w > len(x):
        return out
    for i in range(w - 1, len(x)):
        out[i] = float(np.dot(x[i - w + 1: i + 1], weights))
    return out


def dema(x: np.ndarray, w: int) -> np.ndarray:
    e1 = ema(x, w)
    e2 = ema(e1, w)
    return 2 * e1 - e2


def tema(x: np.ndarray, w: int) -> np.ndarray:
    e1 = ema(x, w)
    e2 = ema(e1, w)
    e3 = ema(e2, w)
    return 3 * e1 - 3 * e2 + e3


def macd(x: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(x, fast) - ema(x, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def rsi(x: np.ndarray, w: int = 14) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    diff = np.diff(x, prepend=x[0])
    up = np.maximum(diff, 0.0)
    dn = np.maximum(-diff, 0.0)
    # Wilder smoothing (recursive EMA with alpha = 1/w)
    a = 1.0 / w
    avg_up = np.empty_like(x)
    avg_dn = np.empty_like(x)
    avg_up[0] = up[0]
    avg_dn[0] = dn[0]
    for i in range(1, len(x)):
        avg_up[i] = a * up[i] + (1 - a) * avg_up[i - 1]
        avg_dn[i] = a * dn[i] + (1 - a) * avg_dn[i - 1]
    rs = avg_up / np.where(avg_dn == 0, 1e-12, avg_dn)
    return 100 - 100 / (1 + rs)


def bollinger(x: np.ndarray, w: int = 20, k: float = 2.0):
    m = sma(x, w)
    # rolling std via cumulative sums
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    cc = np.concatenate([[0.0], np.cumsum(x * x)])
    s = np.full_like(x, np.nan)
    if w <= len(x):
        mean = (c[w:] - c[:-w]) / w
        var = (cc[w:] - cc[:-w]) / w - mean * mean
        var = np.maximum(var, 0.0)
        s[w - 1:] = np.sqrt(var)
    upper = m + k * s
    lower = m - k * s
    return m, upper, lower


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, w: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    return ema(tr, w)


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    sign = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(sign * volume)


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray) -> np.ndarray:
    typical = (high + low + close) / 3.0
    cum_vol = np.cumsum(volume)
    cum_pv = np.cumsum(typical * volume)
    return cum_pv / np.where(cum_vol == 0, 1e-12, cum_vol)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(V: np.ndarray, periods: int = 252,
                    rf_daily: float = 0.001 / 100) -> dict:
    """Return final value, total return, ann. return, ann. vol, Sharpe,
    Sortino, max drawdown, Calmar, hit ratio."""
    V = np.asarray(V, dtype=float)
    V = V[~np.isnan(V)]
    if len(V) < 2:
        return {k: float("nan") for k in (
            "final", "total_ret", "ann_ret", "ann_vol", "sharpe",
            "sortino", "max_dd", "calmar", "hit_ratio")}

    rets = np.diff(V) / V[:-1]
    final = V[-1]
    total_ret = final / V[0] - 1
    n = len(rets)
    ann_ret = (1 + total_ret) ** (periods / n) - 1
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(periods))
    excess = rets - rf_daily
    sharpe = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-12) * np.sqrt(periods))
    downside = excess[excess < 0]
    sortino = float(np.mean(excess) / (np.std(downside, ddof=1) + 1e-12) * np.sqrt(periods)) \
        if len(downside) > 1 else float("nan")
    peak = np.maximum.accumulate(V)
    dd = (V / peak - 1)
    max_dd = float(dd.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else float("nan")
    hit_ratio = float(np.mean(rets > 0))
    return {
        "final": float(final),
        "total_ret": float(total_ret),
        "ann_ret": float(ann_ret),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "hit_ratio": hit_ratio,
    }


def metrics_table_latex(metrics: dict[str, dict], name_map: dict[str, str] | None = None) -> str:
    """Render a metrics dict-of-dicts as a LaTeX booktabs table body."""
    name_map = name_map or {}
    rows = []
    header = (
        "Strategy & Final V (\\$) & Total Ret. & Ann. Ret. & Ann. Vol & Sharpe "
        "& Sortino & Max DD & Calmar & Hit"
    )
    for key, m in metrics.items():
        label = name_map.get(key, key)
        rows.append(
            f"{label} & {m['final']:,.0f} & {m['total_ret']*100:.2f}\\% & "
            f"{m['ann_ret']*100:.2f}\\% & {m['ann_vol']*100:.2f}\\% & "
            f"{m['sharpe']:.2f} & {m['sortino']:.2f} & "
            f"{m['max_dd']*100:.2f}\\% & {m['calmar']:.2f} & {m['hit_ratio']*100:.1f}\\%"
        )
    body = " \\\\\n".join(rows) + " \\\\"
    return header + " \\\\\n\\midrule\n" + body


# ---------------------------------------------------------------------------
# Conditional / Bayes utilities
# ---------------------------------------------------------------------------
def fit_conditional_normals(X: np.ndarray, Y_next: np.ndarray) -> dict[str, tuple[float, float]]:
    """Return {y: (mu, sigma)} for X(t) given Y(t+1)=y."""
    out = {}
    for y in ("D", "U", "H"):
        mask = Y_next == y
        if mask.sum() < 2:
            out[y] = (0.0, 1.0)
        else:
            d = X[mask]
            out[y] = (float(np.mean(d)), float(np.std(d, ddof=1)))
    return out


# ---------------------------------------------------------------------------
# Algorithmic complexities (one source of truth used by every report's §12)
# ---------------------------------------------------------------------------
ALGO_COMPLEXITIES: list[dict] = [
    {"algo": "Log return $X_i(t)$", "time": r"$O(N)$", "space": r"$O(N)$",
     "notes": "Single vectorised diff."},
    {"algo": "$p_0(t,h)$ rolling (this work)", "time": r"$O(N)$",
     "space": r"$O(N)$",
     "notes": r"Cumulative sums of $X_1,X_2,X_1^2,X_2^2,X_1X_2$. Naive: $O(NH)$."},
    {"algo": "EMA / DEMA / TEMA", "time": r"$O(N)$", "space": r"$O(N)$",
     "notes": r"Single recursion $\alpha=2/(w+1)$."},
    {"algo": "SMA (this work)", "time": r"$O(N)$", "space": r"$O(N)$",
     "notes": r"Cumulative-sum trick. Naive: $O(NW)$."},
    {"algo": "MACD (12,26,9)", "time": r"$O(N)$", "space": r"$O(N)$",
     "notes": "Three EMAs."},
    {"algo": "RSI(14), Bollinger, ATR", "time": r"$O(N)$", "space": r"$O(N)$",
     "notes": "Wilder smoothing / cumulative sums."},
    {"algo": "Empirical CDF", "time": r"$O(N\log N)$", "space": r"$O(N)$",
     "notes": "Sort once."},
    {"algo": "Normal/Logistic ML fits", "time": r"$O(N)$", "space": r"$O(1)$",
     "notes": "Closed form moments."},
    {"algo": "Bayes posterior $q_y f_y(x)$",
     "time": r"$O(K)$ per query", "space": r"$O(1)$",
     "notes": r"$K=3$ hypotheses; boundaries via $O(K^2 G)$ grid + Brent."},
    {"algo": "Association-rule support/confidence",
     "time": r"$O(NK)$", "space": r"$O(3^{k+1})$",
     "notes": r"$K=k+1$; one pass over digitised series."},
    {"algo": "Portfolio simulation (S7--S9)",
     "time": r"$O(T)$", "space": r"$O(T)$",
     "notes": "Single forward pass over future days."},
    {"algo": "Efficient-frontier rebalancing",
     "time": r"$O(T)$", "space": r"$O(T)$",
     "notes": "Closed-form quadratic in $p$ each step."},
]


def algo_complexity_latex_rows() -> str:
    rows = []
    for r in ALGO_COMPLEXITIES:
        rows.append(f"{r['algo']} & {r['time']} & {r['space']} & {r['notes']}")
    return " \\\\\n".join(rows) + " \\\\"
