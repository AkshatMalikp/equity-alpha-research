"""Step 5: does each signal predict returns?

Three tests, from simplest to strictest:
  1. Rank IC      - each day, rank correlation between signal and forward return
  2. Quantiles    - each month, sort stocks into 5 buckets by signal; do top buckets earn more?
  3. Fama-MacBeth - each month, regress returns on signals; does a signal still matter
                    after controlling for the others?
Then a multiple-testing correction, because we test 8 signals at once.

Every function here only compares a signal at date t with returns AFTER t
(forward returns / next-period returns built in step 3 or by period_returns below).
"""
from math import e, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

N01 = NormalDist()
EULER_GAMMA = 0.5772156649015329


# ====================================================================== statistics helpers
def newey_west_tstat(x, lags: int) -> tuple:
    """Mean of a series and its t-statistic with Newey-West (HAC) standard errors.
    Needed because overlapping forward returns make consecutive values correlated,
    which makes the ordinary t-stat far too large. lags=0 gives the ordinary t-stat.
    Returns (mean, t_stat, n_obs)."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, n
    mu = x.mean()
    d = x - mu
    var = d @ d / n
    for j in range(1, min(lags, n - 1) + 1):
        w = 1 - j / (lags + 1)                 # Bartlett weight
        var += 2 * w * (d[j:] @ d[:-j]) / n
    se = sqrt(max(var, 0) / n)
    return mu, (mu / se if se > 0 else np.nan), n


def p_value(t) -> float:
    """Two-sided p-value of a t-stat (normal approximation; fine for hundreds of observations)."""
    return np.nan if pd.isna(t) else 2 * (1 - N01.cdf(abs(t)))


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-adjusted p-values: controls the chance of ANY false positive among the tests."""
    p = pvals.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([min(1.0, (m - i) * v) for i, v in enumerate(p.to_numpy())])
    return pd.Series(adj, index=p.index).reindex(pvals.index)


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """BH-adjusted p-values: controls the expected SHARE of false discoveries."""
    p = pvals.dropna().sort_values()
    m = len(p)
    raw = [v * m / (i + 1) for i, v in enumerate(p.to_numpy())]
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    return pd.Series(np.minimum(adj, 1.0), index=p.index).reindex(pvals.index)


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr_trials) -> float:
    """Bailey & Lopez de Prado (2014): probability that the true Sharpe ratio is above zero,
    given that the best of N trials was picked.
    sr, sr_trials: PER-PERIOD (e.g. monthly) Sharpe ratios, not annualised.
    kurt: ordinary kurtosis (normal = 3), not excess."""
    trials = np.asarray(pd.Series(sr_trials).dropna(), dtype=float)
    N = len(trials)
    if N < 2 or n_obs < 3:
        return np.nan
    v = trials.var(ddof=1)
    sr0 = sqrt(v) * ((1 - EULER_GAMMA) * N01.inv_cdf(1 - 1 / N) + EULER_GAMMA * N01.inv_cdf(1 - 1 / (N * e)))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return np.nan
    return N01.cdf((sr - sr0) * sqrt(n_obs - 1) / sqrt(denom))


# ====================================================================== 1. rank IC
def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_stocks: int = 30) -> pd.Series:
    """Per date: Spearman rank correlation, across stocks, between the signal at t and the
    forward return from t. Only stocks with both values count (the signal grid is already
    members-only from cs_rank). Dates with fewer than min_stocks stocks give NaN."""
    s, f = signal.align(fwd, join="inner")
    valid = s.notna() & f.notna()
    rs = s.where(valid).rank(axis=1)
    rf = f.where(valid).rank(axis=1)
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rf = rf.sub(rf.mean(axis=1), axis=0)
    num = (rs * rf).sum(axis=1)
    den = np.sqrt((rs ** 2).sum(axis=1) * (rf ** 2).sum(axis=1))
    ic = num / den.where(den > 0)
    ic[valid.sum(axis=1) < min_stocks] = np.nan
    return ic.rename("ic")


def ic_summary(ic: pd.Series, lags: int) -> dict:
    """Mean IC, its Newey-West t-stat, ICIR (mean / std) and share of days with IC > 0."""
    ic = ic.dropna()
    mean, t, n = newey_west_tstat(ic, lags)
    sd = ic.std()
    return {"mean_ic": mean, "ic_t_nw": t, "icir": mean / sd if sd > 0 else np.nan,
            "pct_positive": (ic > 0).mean(), "n_days": n}


# ====================================================================== monthly holding periods
def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().to_numpy())


def period_returns(returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Row d = compounded return of each stock from the day AFTER d up to and including the
    next rebalance date. The last date has no next period and is dropped.
    A stock with no data in the period gives NaN; a stock whose data stops mid-period
    (e.g. delisting) compounds only the days it has (optimistic; see the delisting note)."""
    out = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        window = returns.loc[(returns.index > d0) & (returns.index <= d1)]
        out[d0] = (1 + window).prod(min_count=1) - 1
    return pd.DataFrame(out).T


# ====================================================================== 2. quantile portfolios
def quantile_returns(signal: pd.DataFrame, per_rets: pd.DataFrame, q: int = 5, min_stocks: int = 50):
    """Each rebalance date: split stocks with a signal into q equal buckets (1 = lowest,
    q = highest), hold each bucket equally weighted for the next period.
    Returns (bucket_returns: date x bucket, turnover: date x ['top', 'bottom'])."""
    rows, turn = {}, {}
    prev_top = prev_bot = None
    for d in per_rets.index:
        if d not in signal.index:
            continue
        s = signal.loc[d].dropna()
        r = per_rets.loc[d].reindex(s.index)
        ok = r.notna()
        s, r = s[ok], r[ok]
        if len(s) < min_stocks:
            continue
        bucket = pd.qcut(s.rank(method="first"), q, labels=range(1, q + 1))
        rows[d] = r.groupby(bucket, observed=False).mean()
        top, bot = set(s[bucket == q].index), set(s[bucket == 1].index)
        if prev_top is not None:
            turn[d] = {"top": 1 - len(top & prev_top) / len(top), "bottom": 1 - len(bot & prev_bot) / len(bot)}
        prev_top, prev_bot = top, bot
    return pd.DataFrame(rows).T, pd.DataFrame(turn).T


def quantile_summary(bucket_rets: pd.DataFrame, turnover: pd.DataFrame,
                     periods_per_year: int = 12, lags: int = 3) -> dict:
    """Average return per bucket, the top-minus-bottom spread and how it behaves."""
    q = bucket_rets.columns.max()
    means = bucket_rets.mean()
    spread = (bucket_rets[q] - bucket_rets[1]).dropna()
    m, t, n = newey_west_tstat(spread, lags)
    sd = spread.std()
    out = {f"Q{k}_monthly": means[k] for k in bucket_rets.columns}
    out.update({
        "spread_monthly": m,
        "spread_annual": m * periods_per_year,
        "spread_t_nw": t,
        "spread_sharpe_annual": m / sd * sqrt(periods_per_year) if sd > 0 else np.nan,
        "monotonicity": pd.Series(range(1, q + 1)).corr(means.reset_index(drop=True).rank()),
        "turnover_top": turnover["top"].mean() if len(turnover) else np.nan,
        "turnover_bottom": turnover["bottom"].mean() if len(turnover) else np.nan,
        "n_months": n,
    })
    return out


# ====================================================================== 3. Fama-MacBeth
def fama_macbeth(signals: dict, per_rets: pd.DataFrame, min_stocks: int = 50) -> pd.DataFrame:
    """Each rebalance date: OLS across stocks of next-period return on the given signals
    (plus a constant). Returns one row of slopes per date.
    With cs_rank signals (range -0.5..+0.5), a slope is roughly the return difference
    between the highest- and lowest-ranked stock, holding the other signals fixed."""
    names = list(signals)
    rows = {}
    for d in per_rets.index:
        X = pd.concat([signals[k].loc[d].rename(k) if d in signals[k].index else pd.Series(dtype=float, name=k)
                       for k in names], axis=1)
        y = per_rets.loc[d].rename("y")
        data = pd.concat([X, y], axis=1).dropna()
        if len(data) < max(min_stocks, len(names) + 2):
            continue
        A = np.column_stack([np.ones(len(data)), data[names].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, data["y"].to_numpy(), rcond=None)
        rows[d] = pd.Series(coef[1:], index=names)
    return pd.DataFrame(rows).T


def fm_summary(slopes: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Average slope per signal and its Newey-West t-stat."""
    res = {}
    for k in slopes.columns:
        m, t, n = newey_west_tstat(slopes[k], lags)
        res[k] = {"fm_slope_monthly": m, "fm_t_nw": t, "n_months": n}
    return pd.DataFrame(res).T