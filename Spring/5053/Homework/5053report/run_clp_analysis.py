from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "0002_HK_clp_adjusted_close_2011_2024.csv"
OUTPUT_DIR = ROOT / "output" / "clp"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
TEXT_DIR = OUTPUT_DIR / "text"

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"


@dataclass
class ArmaChoice:
    order: tuple[int, int]
    result: object
    table: pd.DataFrame


@dataclass
class VolatilityChoice:
    name: str
    result: object
    table: pd.DataFrame


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, FIG_DIR, TABLE_DIR, TEXT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["log_return"]).copy()
    df["squared_return"] = df["log_return"] ** 2
    return df


def descriptive_stats(returns: pd.Series) -> pd.DataFrame:
    jb_stat, jb_pvalue = jarque_bera(returns)
    stats = pd.DataFrame(
        [
            {
                "count": returns.count(),
                "mean": returns.mean(),
                "sd": returns.std(ddof=1),
                "min": returns.min(),
                "max": returns.max(),
                "skewness": returns.skew(),
                "kurtosis": returns.kurt(),
                "jarque_bera": jb_stat,
                "jb_pvalue": jb_pvalue,
            }
        ]
    )
    return stats


def stationarity_table(df: pd.DataFrame) -> pd.DataFrame:
    price_adf = adfuller(df["log_price"], autolag="AIC")
    return_adf = adfuller(df["log_return"], autolag="AIC")
    lb_returns = acorr_ljungbox(df["log_return"], lags=[10, 20], return_df=True)
    lb_squared = acorr_ljungbox(df["squared_return"], lags=[10, 20], return_df=True)

    rows = [
        {
            "series": "log_price",
            "test": "ADF",
            "lag": price_adf[2],
            "stat": price_adf[0],
            "pvalue": price_adf[1],
        },
        {
            "series": "log_return",
            "test": "ADF",
            "lag": return_adf[2],
            "stat": return_adf[0],
            "pvalue": return_adf[1],
        },
    ]
    for lag in (10, 20):
        rows.append(
            {
                "series": "log_return",
                "test": "Ljung-Box",
                "lag": lag,
                "stat": lb_returns.loc[lag, "lb_stat"],
                "pvalue": lb_returns.loc[lag, "lb_pvalue"],
            }
        )
        rows.append(
            {
                "series": "squared_return",
                "test": "Ljung-Box",
                "lag": lag,
                "stat": lb_squared.loc[lag, "lb_stat"],
                "pvalue": lb_squared.loc[lag, "lb_pvalue"],
            }
        )
    return pd.DataFrame(rows)


def fit_arma_candidates(train_returns: pd.Series) -> ArmaChoice:
    rows: list[dict[str, float | int | str]] = []
    best_result = None
    best_order = None
    best_bic = np.inf

    for p in range(4):
        for q in range(4):
            try:
                model = ARIMA(train_returns, order=(p, 0, q), trend="c")
                result = model.fit()
                resid = pd.Series(result.resid, index=train_returns.index).dropna()
                lb = acorr_ljungbox(resid, lags=[10], return_df=True)
                rows.append(
                    {
                        "model": f"ARMA({p},{q})",
                        "p": p,
                        "q": q,
                        "aic": result.aic,
                        "bic": result.bic,
                        "lb_stat_lag10": lb["lb_stat"].iloc[0],
                        "lb_pvalue_lag10": lb["lb_pvalue"].iloc[0],
                    }
                )
                if result.bic < best_bic:
                    best_bic = result.bic
                    best_result = result
                    best_order = (p, q)
            except Exception as exc:
                rows.append(
                    {
                        "model": f"ARMA({p},{q})",
                        "p": p,
                        "q": q,
                        "aic": np.nan,
                        "bic": np.nan,
                        "lb_stat_lag10": np.nan,
                        "lb_pvalue_lag10": np.nan,
                        "error": str(exc),
                    }
                )

    table = pd.DataFrame(rows).sort_values(["bic", "aic"], na_position="last").reset_index(drop=True)
    if best_result is None or best_order is None:
        raise RuntimeError("No ARMA candidate converged.")
    return ArmaChoice(order=best_order, result=best_result, table=table)


def residual_diagnostics(residuals: pd.Series) -> pd.DataFrame:
    lb_returns = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    lb_squared = acorr_ljungbox(residuals**2, lags=[10, 20], return_df=True)
    arch_lm = het_arch(residuals, nlags=10)

    rows = []
    for lag in (10, 20):
        rows.append(
            {
                "series": "arma_residual",
                "test": "Ljung-Box",
                "lag": lag,
                "stat": lb_returns.loc[lag, "lb_stat"],
                "pvalue": lb_returns.loc[lag, "lb_pvalue"],
            }
        )
        rows.append(
            {
                "series": "arma_residual_squared",
                "test": "Ljung-Box",
                "lag": lag,
                "stat": lb_squared.loc[lag, "lb_stat"],
                "pvalue": lb_squared.loc[lag, "lb_pvalue"],
            }
        )
    rows.append(
        {
            "series": "arma_residual_squared",
            "test": "ARCH-LM",
            "lag": 10,
            "stat": arch_lm[0],
            "pvalue": arch_lm[1],
        }
    )
    return pd.DataFrame(rows)


def _vol_model_specs() -> list[dict[str, object]]:
    return [
        {"name": "GARCH(1,1)", "vol": "GARCH", "p": 1, "o": 0, "q": 1},
        {"name": "EGARCH(1,1)", "vol": "EGARCH", "p": 1, "o": 1, "q": 1},
        {"name": "GJR-GARCH(1,1)", "vol": "GARCH", "p": 1, "o": 1, "q": 1},
    ]


def fit_volatility_models(train_resid: pd.Series) -> VolatilityChoice:
    rows: list[dict[str, float | str]] = []
    best_name = None
    best_result = None
    best_bic = np.inf

    scaled_resid = train_resid.astype(float) * 10.0
    for spec in _vol_model_specs():
        model = arch_model(
            scaled_resid,
            mean="Constant",
            vol=spec["vol"],
            p=spec["p"],
            o=spec["o"],
            q=spec["q"],
            dist="normal",
            rescale=False,
        )
        result = model.fit(disp="off")
        params = result.params
        alpha = params.get("alpha[1]", np.nan)
        beta = params.get("beta[1]", np.nan)
        gamma = params.get("gamma[1]", np.nan)
        if spec["name"].startswith("GJR"):
            persistence = alpha + beta + 0.5 * gamma
        elif spec["name"].startswith("GARCH"):
            persistence = alpha + beta
        else:
            persistence = beta
        rows.append(
            {
                "model": spec["name"],
                "aic": result.aic,
                "bic": result.bic,
                "omega": params.get("omega", np.nan),
                "alpha1": alpha,
                "beta1": beta,
                "gamma1": gamma,
                "persistence_proxy": persistence,
                "loglikelihood": result.loglikelihood,
            }
        )
        if result.bic < best_bic:
            best_bic = result.bic
            best_name = spec["name"]
            best_result = result

    table = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    if best_name is None or best_result is None:
        raise RuntimeError("No volatility model converged.")
    return VolatilityChoice(name=best_name, result=best_result, table=table)


def fit_final_volatility_model(full_resid: pd.Series, name: str):
    scaled_resid = full_resid.astype(float) * 10.0
    spec = next(spec for spec in _vol_model_specs() if spec["name"] == name)
    model = arch_model(
        scaled_resid,
        mean="Constant",
        vol=spec["vol"],
        p=spec["p"],
        o=spec["o"],
        q=spec["q"],
        dist="normal",
        rescale=False,
    )
    return model.fit(disp="off")


def rolling_arma_forecast(result, test_returns: pd.Series, forecast_dates: pd.Series) -> pd.Series:
    history = result
    forecasts = []
    for step, actual in enumerate(test_returns.tolist(), start=1):
        date = pd.Timestamp(forecast_dates.iloc[step - 1])
        value = float(history.forecast(steps=1).iloc[0])
        forecasts.append({"date": date, "forecast_arma": value})
        next_index = history.nobs
        orig_endog = history.model.data.orig_endog
        if isinstance(orig_endog, pd.DataFrame):
            new_obs = pd.DataFrame([[actual]], index=[next_index], columns=orig_endog.columns)
        else:
            new_obs = pd.Series([actual], index=[next_index], name=getattr(orig_endog, "name", None))
        history = history.append(new_obs, refit=False)
    return pd.DataFrame(forecasts).set_index("date")["forecast_arma"]


def garch_test_forecast(
    train_returns: pd.Series, test_returns: pd.Series, forecast_dates: pd.Series, ar_lags: int
) -> pd.Series:
    combined = pd.concat([train_returns, test_returns])
    preds = []

    for idx in range(len(train_returns), len(combined)):
        history = combined.iloc[:idx]
        model = arch_model(
            history,
            mean="ARX" if ar_lags > 0 else "Constant",
            lags=ar_lags if ar_lags > 0 else None,
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="normal",
            rescale=False,
        )
        result = model.fit(disp="off")
        forecast = result.forecast(horizon=1, reindex=False)
        mean_value = float(forecast.mean.iloc[-1, 0])
        preds.append({"date": pd.Timestamp(forecast_dates.iloc[idx - len(train_returns)]), "forecast_garch": mean_value})
    return pd.DataFrame(preds).set_index("date")["forecast_garch"]


def forecasting_table(actual: pd.Series, arma_forecast: pd.Series, garch_forecast: pd.Series) -> pd.DataFrame:
    rows = []
    for name, series in [("ARMA", arma_forecast), ("AR-GARCH", garch_forecast)]:
        aligned = pd.concat([actual.rename("actual"), series.rename("forecast")], axis=1).dropna()
        rows.append(
            {
                "model": name,
                "rmse": np.sqrt(np.mean((aligned["actual"] - aligned["forecast"]) ** 2)),
                "mae": np.mean(np.abs(aligned["actual"] - aligned["forecast"])),
                "directional_accuracy": np.mean(
                    np.sign(aligned["actual"]) == np.sign(aligned["forecast"])
                ),
            }
        )
    return pd.DataFrame(rows)


def save_tables(
    desc: pd.DataFrame,
    stationarity: pd.DataFrame,
    arma_table: pd.DataFrame,
    resid_diag: pd.DataFrame,
    vol_table: pd.DataFrame,
    forecast_table_df: pd.DataFrame,
    forecast_series: pd.DataFrame,
) -> None:
    desc.to_csv(TABLE_DIR / "table1_descriptive_statistics.csv", index=False)
    stationarity.to_csv(TABLE_DIR / "table2_stationarity_tests.csv", index=False)
    arma_table.to_csv(TABLE_DIR / "table3_arma_candidates.csv", index=False)
    resid_diag.to_csv(TABLE_DIR / "table2b_residual_diagnostics.csv", index=False)
    vol_table.to_csv(TABLE_DIR / "table4_garch_models.csv", index=False)
    forecast_table_df.to_csv(TABLE_DIR / "table5_forecasting_performance.csv", index=False)
    forecast_series.to_csv(TABLE_DIR / "forecast_series.csv", index=False)


def plot_price_and_log_price(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["date"], df["adj_close"], color="#1f77b4", linewidth=1.1)
    axes[0].set_title("CLP Adjusted Close Price")
    axes[0].set_ylabel("Adjusted Close")
    axes[1].plot(df["date"], df["log_price"], color="#ff7f0e", linewidth=1.1)
    axes[1].set_title("CLP Log Price")
    axes[1].set_ylabel("Log Price")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_price_and_log_price.png", dpi=200)
    plt.close(fig)


def plot_returns(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["date"], df["log_return"], color="#2ca02c", linewidth=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("CLP Log Return")
    ax.set_ylabel("Log Return (%)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_log_return_plot.png", dpi=200)
    plt.close(fig)


def plot_acf_pacf_returns(returns: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_acf(returns, lags=30, ax=axes[0])
    axes[0].set_title("ACF of CLP Returns")
    plot_pacf(returns, lags=30, ax=axes[1], method="ywm")
    axes[1].set_title("PACF of CLP Returns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_acf_pacf_returns.png", dpi=200)
    plt.close(fig)


def plot_acf_squared_returns(returns: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_acf(returns**2, lags=30, ax=ax)
    ax.set_title("ACF of Squared Returns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_acf_squared_returns.png", dpi=200)
    plt.close(fig)


def plot_conditional_volatility(dates: pd.Series, volatility: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, volatility, color="#d62728", linewidth=0.9)
    ax.set_title("Fitted Conditional Volatility")
    ax.set_ylabel("Conditional Volatility")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_conditional_volatility.png", dpi=200)
    plt.close(fig)


def plot_forecast_vs_actual(forecast_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(forecast_df["date"], forecast_df["actual"], color="#111111", label="Actual", linewidth=0.9)
    axes[0].plot(
        forecast_df["date"],
        forecast_df["forecast_arma"],
        color="#1f77b4",
        label="ARMA Forecast",
        linewidth=0.9,
    )
    axes[0].set_title("ARMA Forecast vs Actual")
    axes[0].legend()

    axes[1].plot(forecast_df["date"], forecast_df["actual"], color="#111111", label="Actual", linewidth=0.9)
    axes[1].plot(
        forecast_df["date"],
        forecast_df["forecast_garch"],
        color="#d62728",
        label="AR-GARCH Forecast",
        linewidth=0.9,
    )
    axes[1].set_title("AR-GARCH Forecast vs Actual")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure6_forecast_vs_actual.png", dpi=200)
    plt.close(fig)


def save_text_summary(
    arma_choice: ArmaChoice,
    vol_choice: VolatilityChoice,
    forecast_table_df: pd.DataFrame,
) -> None:
    lines = [
        "CLP analysis summary",
        f"Best ARMA order by BIC: ARMA{arma_choice.order}",
        f"Best volatility model by BIC: {vol_choice.name}",
        "",
        "Forecasting performance:",
        forecast_table_df.to_string(index=False),
    ]
    (TEXT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = load_data()

    train_mask = df["date"] <= TRAIN_END
    test_mask = df["date"] >= TEST_START
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    desc = descriptive_stats(df["log_return"])
    stationarity = stationarity_table(df)

    train_dates = train["date"].reset_index(drop=True)
    test_dates = test["date"].reset_index(drop=True)
    train_returns = train["log_return"].reset_index(drop=True)
    test_returns = test["log_return"].reset_index(drop=True)
    full_returns = df["log_return"].reset_index(drop=True)

    arma_choice = fit_arma_candidates(train_returns)
    arma_resid_train = pd.Series(np.asarray(arma_choice.result.resid), index=train_dates).dropna()
    resid_diag = residual_diagnostics(arma_resid_train)

    vol_choice = fit_volatility_models(arma_resid_train)

    full_arma = ARIMA(full_returns, order=(arma_choice.order[0], 0, arma_choice.order[1]), trend="c").fit()
    full_arma_resid = pd.Series(np.asarray(full_arma.resid), index=df["date"]).dropna()
    final_vol = fit_final_volatility_model(full_arma_resid, vol_choice.name)
    conditional_vol = pd.Series(final_vol.conditional_volatility / 10.0, index=full_arma_resid.index)

    arma_forecast = rolling_arma_forecast(arma_choice.result, test_returns, test_dates)
    garch_forecast = garch_test_forecast(train_returns, test_returns, test_dates, ar_lags=arma_choice.order[0])
    actual_test = pd.Series(test_returns.to_numpy(), index=pd.Index(test_dates, name="date"))
    forecast_table_df = forecasting_table(actual_test, arma_forecast, garch_forecast)
    forecast_df = (
        pd.concat(
            [
                actual_test.rename("actual"),
                arma_forecast,
                garch_forecast,
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    save_tables(
        desc,
        stationarity,
        arma_choice.table,
        resid_diag,
        vol_choice.table,
        forecast_table_df,
        forecast_df,
    )
    save_text_summary(arma_choice, vol_choice, forecast_table_df)

    plot_price_and_log_price(df)
    plot_returns(df)
    plot_acf_pacf_returns(df["log_return"])
    plot_acf_squared_returns(df["log_return"])
    plot_conditional_volatility(full_arma_resid.index, conditional_vol)
    plot_forecast_vs_actual(forecast_df)

    print("Analysis complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Selected mean model: ARMA{arma_choice.order}")
    print(f"Selected volatility model: {vol_choice.name}")
    print(forecast_table_df.to_string(index=False))


if __name__ == "__main__":
    main()
