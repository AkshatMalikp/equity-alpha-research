"""Step 7: turn forecasts into a long-short portfolio, trade it with realistic costs,
and measure what an investor would actually have earned.

Conventions
  - rebalance at the close of each month-end; the forecast is the one made that day
  - $1 of capital, $1 long and $1 short (gross exposure 2, dollar-neutral)
  - positions drift with prices between rebalances; turnover is measured against the
    drifted positions, which is what you would really have to trade
  - a missing return (no trade that day, or after a delisting) counts as 0 for that day,
    so a delisted stock's position just sits as cash until the next rebalance.
    CRSP returns already include the final delisting return.
  - lag=1 trades one day AFTER the forecast date (implementation-lag robustness check)
"""
import io
import zipfile
import numpy as np
import pandas as pd

from alpha.evaluate import newey_west_tstat

DAYS = 252


# ====================================================================== weights
def quintile_weights(pred: pd.Series, q: int = 5, min_stocks: int = 50) -> pd.Series:
    """Long the top fifth equally ($1 in total), short the bottom fifth equally (-$1)."""
    s = pred.dropna()
    if len(s) < min_stocks:
        return pd.Series(dtype=float)
    bucket = pd.qcut(s.rank(method="first"), q, labels=False)
    top, bot = s.index[bucket == q - 1], s.index[bucket == 0]
    w = pd.Series(0.0, index=s.index)
    w[top] = 1 / len(top)
    w[bot] = -1 / len(bot)
    return w[w != 0]


def rank_weights(pred: pd.Series, min_stocks: int = 50) -> pd.Series:
    """Weights proportional to the demeaned rank, scaled to $1 long and $1 short."""
    s = pred.dropna()
    if len(s) < min_stocks:
        return pd.Series(dtype=float)
    x = s.rank() - s.rank().mean()
    return x / x.abs().sum() * 2


# ====================================================================== costs
def half_spread_cost(spread: pd.DataFrame, window: int = 21, cap: float = 0.05) -> pd.DataFrame:
    """One-way cost per dollar traded = half the quoted bid-ask spread, averaged over the
    trailing 21 days (only past data). Missing values get that day's median."""
    hs = 0.5 * spread.rolling(window, min_periods=5).mean()
    hs = hs.apply(lambda row: row.fillna(row.median()), axis=1)
    return hs.clip(upper=cap)


def impact_cost(returns, close, volume, trade_dollars: pd.Series, date, k: float = 1.0, window: int = 21):
    """Square-root market impact: k * daily volatility * sqrt(trade size / average daily
    dollar volume). Empirical rule of thumb; k is uncertain (roughly 0.5 to 1)."""
    past = returns.loc[:date].tail(window)
    sigma = past.std()
    adv = (close * volume).loc[:date].tail(window).mean()
    adv = adv.where(adv > 0)                                  # zero volume -> unknown, not infinite
    names = trade_dollars.index
    frac = (trade_dollars / adv.reindex(names)).clip(lower=0)
    cost = (k * sigma.reindex(names) * np.sqrt(frac)).replace([np.inf, -np.inf], np.nan)
    return cost.fillna(cost.median()).fillna(0).clip(upper=0.10)


# ====================================================================== simulation
def simulate(weights: dict, returns: pd.DataFrame, cost_rate=0.0, lag: int = 0,
             borrow_bps: float = 25.0, impact=None) -> pd.DataFrame:
    """Daily simulation.
    weights   : {rebalance date: Series of target weights (fractions of equity)}
    cost_rate : flat one-way cost per $ traded (e.g. 0.001 = 10 bps) OR a date x stock grid
    impact    : optional function(trade_dollars_per_$equity: Series, date) -> extra cost rate
    Returns a daily frame with gross, cost, borrow, net returns and turnover on trade days."""
    cols = returns.columns
    col_ix = {c: i for i, c in enumerate(cols)}
    R = returns.fillna(0.0).to_numpy()
    dates = returns.index
    trade_at = {}
    for d, w in weights.items():
        k = dates.searchsorted(d) + lag
        if k < len(dates) and len(w):
            trade_at[k] = (d, w)

    pos = np.zeros(len(cols))
    out = np.zeros((len(dates), 4))                           # gross, cost, borrow, turnover
    for t in range(len(dates)):
        if t > 0 and pos.any():
            r_p = pos @ R[t]
            out[t, 0] = r_p
            out[t, 2] = borrow_bps / 1e4 / DAYS * (-pos[pos < 0]).sum()
            pos = pos * (1 + R[t]) / (1 + r_p)                # drift, as a fraction of equity
        if t in trade_at:
            d, w = trade_at[t]
            target = np.zeros(len(cols))
            idx = [col_ix[c] for c in w.index if c in col_ix]
            target[idx] = w[[c for c in w.index if c in col_ix]].to_numpy()
            trade = np.abs(target - pos)
            if isinstance(cost_rate, pd.DataFrame):
                row = cost_rate.loc[:dates[t]].iloc[-1].reindex(cols).fillna(cost_rate.loc[:dates[t]].iloc[-1].median())
                rate = row.to_numpy()
            else:
                rate = np.full(len(cols), float(cost_rate))
            if impact is not None:
                traded = pd.Series(trade, index=cols)
                traded = traded[traded > 0]
                rate = rate + impact(traded, dates[t]).reindex(cols).fillna(0).to_numpy()
            out[t, 1] = trade @ rate
            out[t, 3] = trade.sum()
            pos = target
    df = pd.DataFrame(out, index=dates, columns=["gross", "cost", "borrow", "turnover"])
    df["net"] = df["gross"] - df["cost"] - df["borrow"]
    first = min(trade_at) if trade_at else len(dates)
    return df.iloc[first:]


# ====================================================================== metrics
def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()


def performance(daily: pd.DataFrame) -> dict:
    """Annualised figures from the daily simulation. Long-short is self-financing, so the
    Sharpe ratio needs no risk-free subtraction."""
    years = len(daily) / DAYS
    g, n = daily["gross"], daily["net"]
    monthly = (1 + n).groupby(n.index.to_period("M")).prod() - 1
    _, t_nw, _ = newey_west_tstat(monthly, lags=3)
    sr = n.mean() / n.std() * np.sqrt(DAYS) if n.std() > 0 else np.nan
    traded_per_year = daily["turnover"].sum() / years            # $ traded per $1 equity per year
    gross_ann = g.mean() * DAYS
    return {
        "gross_return": gross_ann,
        "net_return": n.mean() * DAYS,
        "volatility": n.std() * np.sqrt(DAYS),
        "sharpe_gross": g.mean() / g.std() * np.sqrt(DAYS) if g.std() > 0 else np.nan,
        "sharpe_net": sr,
        "sharpe_se": np.sqrt((1 + sr ** 2 / 2) / years) if years > 0 else np.nan,
        "t_net_monthly_nw": t_nw,
        "max_drawdown": max_drawdown(n),
        "hit_rate_monthly": (monthly > 0).mean(),
        "traded_per_year": traded_per_year,
        "cost_per_year": daily["cost"].sum() / years,
        "breakeven_bps": gross_ann / traded_per_year * 1e4 if traded_per_year > 0 else np.nan,
        "years": years,
    }


# ====================================================================== Fama-French
FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}


def parse_french_monthly(text: str) -> pd.DataFrame:
    """Parse the MONTHLY block of a Ken French CSV (rows keyed yyyymm, values in percent)."""
    rows, header = [], None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            if len(parts) > 1 and parts[0] == "" and any(c.isalpha() for c in "".join(parts[1:])):
                header = parts[1:]
            continue
        if parts[0].isdigit() and len(parts[0]) == 6:
            rows.append([int(parts[0])] + [float(x) for x in parts[1:len(header) + 1]])
        elif rows:
            break                                               # end of the monthly block
    df = pd.DataFrame(rows, columns=["yyyymm"] + header).set_index("yyyymm") / 100
    df.index = pd.PeriodIndex([f"{i // 100}-{i % 100:02d}" for i in df.index], freq="M")
    return df


def load_french_factors(cache_dir) -> pd.DataFrame:
    """Download (once) and parse the Fama-French 5 factors + momentum, monthly."""
    from pathlib import Path
    import urllib.request
    cache = Path(cache_dir) / "ff_factors_monthly.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df
    frames = []
    for url in FF_URLS.values():
        raw = urllib.request.urlopen(url, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        frames.append(parse_french_monthly(text))
    df = frames[0].join(frames[1], how="inner")
    df.columns = [c.replace(" ", "") for c in df.columns]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def factor_regression(monthly: pd.Series, factors: pd.DataFrame, lags: int = 3) -> dict:
    """OLS of monthly strategy returns on Mkt-RF, SMB, HML, RMW, CMA, Mom with Newey-West
    standard errors. The intercept is the alpha: return NOT explained by known factors."""
    cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in factors.columns]
    data = pd.concat([monthly.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    X = np.column_stack([np.ones(len(data)), data[cols].to_numpy()])
    y = data["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        G = (X[j:] * e[j:, None]).T @ (X[:-j] * e[:-j, None]) / n
        S += w * (G + G.T)
    se = np.sqrt(np.diag(n * XtX_inv @ S @ XtX_inv))
    out = {"alpha_annual": beta[0] * 12, "alpha_t": beta[0] / se[0], "months": n,
           "r2": 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))}
    for c, b, s in zip(cols, beta[1:], se[1:]):
        out[f"beta_{c}"] = b
        out[f"t_{c}"] = b / s
    return out