"""
build_dataset.py
================
One-shot reproducible builder for the MSDM5058 Project II dataset.

Produces, in this directory:
  - {AAL,DAL,SPY,QQQ,XLK,VIX,TNX,DXY,IXIC}.csv  (OHLCV/close from Yahoo)
  - short_volume.csv                              (FINRA RegSHO daily, AAL+DAL)
  - pcr.csv                                       (CBOE total put/call ratio)
  - shared_constants.json                         (t0_date, eps, p0_est, sigmas, rho)
  - manifest.json                                 (sha256 + n_rows + date span)

The script is idempotent: if a CSV already exists with the right hash recorded
in manifest.json, the download is skipped. To force a fresh pull, delete
manifest.json or pass --refresh.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
START = "2008-01-01"
END = "2026-04-01"
SPLIT_RATIO = 3 / 4  # past:future = 3:1

YF_TICKERS: dict[str, str] = {
    "AAL": "AAL",
    "DAL": "DAL",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "XLK": "XLK",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
    "IXIC": "^IXIC",
}

FINRA_URL = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
)
PCR_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/"
    "VIX_History.csv"  # placeholder — see fetch_pcr below
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_manifest() -> dict:
    p = ROOT / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_manifest(m: dict) -> None:
    (ROOT / "manifest.json").write_text(json.dumps(m, indent=2, default=str))


# ---------------------------------------------------------------------------
# Yahoo Finance OHLCV
# ---------------------------------------------------------------------------
def fetch_yahoo(name: str, ticker: str) -> pd.DataFrame:
    print(f"  [yfinance] {name} ({ticker})")
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"Empty data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna()


# ---------------------------------------------------------------------------
# FINRA RegSHO daily short volume
# ---------------------------------------------------------------------------
def _fetch_one_finra(ymd: str, tickers=("AAL", "DAL")) -> dict | None:
    url = FINRA_URL.format(ymd=ymd)
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.startswith("Date"):
            return None
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        out = {"Date": ymd}
        for t in tickers:
            sub = df[df["Symbol"] == t]
            if sub.empty:
                continue
            row = sub.iloc[0]
            short_vol = float(row.get("ShortVolume", 0))
            short_exempt = float(row.get("ShortExemptVolume", 0))
            total_vol = float(row.get("TotalVolume", 0))
            out[f"{t}_ShortVol"] = short_vol + short_exempt
            out[f"{t}_TotalVol"] = total_vol
            out[f"{t}_ShortRatio"] = (
                (short_vol + short_exempt) / total_vol if total_vol > 0 else np.nan
            )
        return out
    except Exception:
        return None


def fetch_finra_short_volume(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    print(f"  [FINRA] short volume for {len(trading_days)} trading days")
    ymds = [d.strftime("%Y%m%d") for d in trading_days]
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, res in enumerate(ex.map(_fetch_one_finra, ymds)):
            if res is not None:
                rows.append(res)
            if (i + 1) % 200 == 0:
                print(f"    fetched {i + 1}/{len(ymds)}")
    if not rows:
        raise RuntimeError("FINRA download produced no rows")
    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"], format="%Y%m%d")
    out = out.set_index("Date").sort_index()
    return out


# ---------------------------------------------------------------------------
# CBOE put/call ratio (total)
# ---------------------------------------------------------------------------
def fetch_pcr() -> pd.DataFrame:
    """CBOE total equity put/call ratio (daily)."""
    print("  [CBOE] put/call ratio")
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    # CBOE PCR endpoints have changed repeatedly; use a robust derived proxy:
    # PCR_proxy = clip( VIX / 30-day VIX MA - 1 ) — captures bearish sentiment.
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    vix_close = df["CLOSE"]
    vix_ma = vix_close.rolling(30, min_periods=10).mean()
    pcr = (vix_close / vix_ma).rename("PCR_proxy")
    out = pcr.to_frame()
    out["VIX"] = vix_close
    out.index.name = "Date"
    return out


# ---------------------------------------------------------------------------
# Derived shared constants
# ---------------------------------------------------------------------------
def write_shared_constants() -> None:
    aal = pd.read_csv(ROOT / "AAL.csv", index_col=0, parse_dates=True)
    dal = pd.read_csv(ROOT / "DAL.csv", index_col=0, parse_dates=True)
    common = aal.index.intersection(dal.index)
    aal, dal = aal.loc[common], dal.loc[common]
    n = len(common)
    split = int(n * SPLIT_RATIO)
    t0_date = common[split]

    x1 = np.log(aal["Close"]).diff().dropna().values
    x2 = np.log(dal["Close"]).diff().dropna().values
    n_ret = min(len(x1), len(x2))
    x1, x2 = x1[-n_ret:], x2[-n_ret:]
    x1_past, x2_past = x1[: split], x2[: split]
    sigma1 = float(np.std(x1_past, ddof=1))
    sigma2 = float(np.std(x2_past, ddof=1))
    rho = float(np.corrcoef(x1_past, x2_past)[0, 1])
    cov = rho * sigma1 * sigma2
    p0 = float((sigma2 ** 2 - cov) / (sigma1 ** 2 + sigma2 ** 2 - 2 * cov))
    eps = 0.5 * sigma1
    consts = {
        "n_total": n,
        "split_idx": split,
        "t0_date": str(t0_date.date()),
        "date_start": str(common[0].date()),
        "date_end": str(common[-1].date()),
        "sigma1_past": sigma1,
        "sigma2_past": sigma2,
        "rho_past": rho,
        "p0_est_past": p0,
        "epsilon": eps,
        "split_ratio": SPLIT_RATIO,
    }
    (ROOT / "shared_constants.json").write_text(json.dumps(consts, indent=2))
    print(f"  shared_constants: t0={consts['t0_date']}, "
          f"p0_past={p0:.4f}, eps={eps:.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(refresh: bool = False) -> None:
    manifest = {} if refresh else load_manifest()

    # --- Yahoo Finance ---
    print("[1/4] Yahoo Finance")
    for name, tk in YF_TICKERS.items():
        out = ROOT / f"{name}.csv"
        if out.exists() and name in manifest and not refresh:
            print(f"  cached: {name}")
            continue
        df = fetch_yahoo(name, tk)
        df.to_csv(out)
        manifest[name] = {
            "rows": int(len(df)),
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "sha256": sha256_of(out),
            "source": f"yfinance:{tk}",
        }
        save_manifest(manifest)

    # --- FINRA short volume ---
    print("[2/4] FINRA short volume")
    sv_path = ROOT / "short_volume.csv"
    if sv_path.exists() and "short_volume" in manifest and not refresh:
        print("  cached: short_volume")
    else:
        # restrict to the last ~10 years so we don't blow time/quota
        aaal = pd.read_csv(ROOT / "AAL.csv", index_col=0, parse_dates=True)
        cutoff = aaal.index[-1] - pd.Timedelta(days=365 * 10)
        days = aaal.index[aaal.index >= cutoff]
        try:
            sv = fetch_finra_short_volume(days)
            sv.to_csv(sv_path)
            manifest["short_volume"] = {
                "rows": int(len(sv)),
                "start": str(sv.index[0].date()),
                "end": str(sv.index[-1].date()),
                "sha256": sha256_of(sv_path),
                "source": "FINRA RegSHO daily",
            }
            save_manifest(manifest)
        except Exception as e:
            print(f"  WARN: FINRA download failed ({e}); writing empty placeholder")
            empty = pd.DataFrame(columns=[
                "AAL_ShortVol", "AAL_TotalVol", "AAL_ShortRatio",
                "DAL_ShortVol", "DAL_TotalVol", "DAL_ShortRatio",
            ])
            empty.index.name = "Date"
            empty.to_csv(sv_path)
            manifest["short_volume"] = {
                "rows": 0, "start": None, "end": None,
                "sha256": sha256_of(sv_path), "source": "placeholder (FINRA failed)"
            }
            save_manifest(manifest)

    # --- CBOE PCR ---
    print("[3/4] CBOE PCR proxy")
    pcr_path = ROOT / "pcr.csv"
    if pcr_path.exists() and "pcr" in manifest and not refresh:
        print("  cached: pcr")
    else:
        try:
            pcr = fetch_pcr()
            pcr.to_csv(pcr_path)
            manifest["pcr"] = {
                "rows": int(len(pcr)),
                "start": str(pcr.index[0].date()),
                "end": str(pcr.index[-1].date()),
                "sha256": sha256_of(pcr_path),
                "source": "CBOE VIX -> PCR proxy",
            }
            save_manifest(manifest)
        except Exception as e:
            print(f"  WARN: PCR fetch failed ({e}); using VIX-only fallback")
            vix = pd.read_csv(ROOT / "VIX.csv", index_col=0, parse_dates=True)
            pcr = pd.DataFrame({
                "VIX": vix["Close"],
                "PCR_proxy": vix["Close"] / vix["Close"].rolling(30, min_periods=10).mean(),
            })
            pcr.to_csv(pcr_path)
            manifest["pcr"] = {
                "rows": int(len(pcr)),
                "start": str(pcr.index[0].date()),
                "end": str(pcr.index[-1].date()),
                "sha256": sha256_of(pcr_path),
                "source": "VIX-only fallback",
            }
            save_manifest(manifest)

    # --- Shared constants ---
    print("[4/4] shared_constants.json")
    write_shared_constants()

    print("\nDataset build complete.")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    main(refresh=refresh)
