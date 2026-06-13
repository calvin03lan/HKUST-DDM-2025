# MSDM5058 Project II — Group Submission

Four independent reports on the same shared dataset.

## Authors and directions

| Folder | Author | Direction |
|---|---|---|
| `HU_report/`    | HU Xiuqi      | Bayesian inference and information-theoretic pattern mining |
| `HUANG_report/` | HUANG Wenhao  | Multi-indicator technical analysis with signal engineering |
| `ZHAO_report/`  | ZHAO Fuxian   | Modern portfolio theory and risk-adjusted optimisation |
| `LAN_report/`   | LAN Tianwei   | Machine-learning and time-series forecasting |

Each report covers the nine sections required by the project specification (data preprocessing, mean-variance, MA/MACD, PDF, Bayes detector, association rules, three portfolio simulators) plus a person-specific extension chapter and an algorithmic-complexity table.

## Folder layout

```
Project 2/
├── MSDM5058 Project II.pdf          original assignment PDF
├── README.md                        this file
├── data/                            shared dataset and helpers
│   ├── build_dataset.py             one-shot downloader (yfinance + FINRA + CBOE)
│   ├── shared.py                    common API: load_panel, indicators, metrics
│   ├── shared_constants.json        t0_date, ε, p0_est, σ₁, σ₂, ρ
│   ├── manifest.json                SHA-256 cache for reproducibility
│   ├── AAL.csv  DAL.csv  SPY.csv  QQQ.csv  XLK.csv  VIX.csv
│   ├── TNX.csv   IXIC.csv   DXY.csv
│   ├── short_volume.csv             FINRA RegSHO daily short-volume
│   └── pcr.csv                      CBOE put/call ratio proxy
├── HU_report/      analysis.py  refs.bib  report.tex  figures/  results.json
├── HUANG_report/   analysis.py  refs.bib  report.tex  figures/  results.json
├── ZHAO_report/    analysis.py  refs.bib  report.tex  figures/  results.json
├── LAN_report/     analysis.py  refs.bib  report.tex  figures/  results.json
└── _legacy/                         original draft retained for reference
```

## Build instructions

### 1. Build the shared dataset (one-shot, idempotent)

```bash
cd "MSDM 5058/Project 2"
python3 data/build_dataset.py
```

Downloads OHLCV for the nine tickers, the FINRA short-volume archive, and the CBOE put/call ratio proxy, then writes a `manifest.json` with SHA-256 hashes. Subsequent runs skip downloads if the on-disk hashes still match.

### 2. Run each report's analysis

```bash
python3 HU_report/analysis.py
python3 HUANG_report/analysis.py
python3 ZHAO_report/analysis.py
python3 LAN_report/analysis.py
```

Each script imports from `data/shared.py`, writes its own figures into `<report>/figures/`, and dumps numerical results to `<report>/results.json`. All scripts pin `np.random.seed(20260516)` (`sh.SEED`) so re-runs are deterministic.

### 3. Compile each report

```bash
cd HU_report   # or HUANG_report / ZHAO_report / LAN_report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

LNCS class (`llncs`) and `splncs04.bst` are taken from the system TeX Live distribution (TeX Live 2024 or later).

## Dependencies

Python 3.10+:

```
numpy pandas scipy matplotlib scikit-learn statsmodels arch xgboost yfinance requests
```

LaTeX: TeX Live 2023+ with `llncs` and `splncs04.bst`, plus standard packages
(`graphicx`, `amsmath`, `booktabs`, `siunitx`, `hyperref`, `cleveref`,
`algorithm`, `algpseudocode`).

## Reproducibility

* Every script uses `np.random.seed(sh.SEED)` with `SEED=20260516`.
* The data layer caches every downloaded file with its SHA-256 in `data/manifest.json`; if a hash matches, the file is reused.
* The train/test split point `t0 = 2021-09-02` is recorded once in `data/shared_constants.json` and read by every report.
