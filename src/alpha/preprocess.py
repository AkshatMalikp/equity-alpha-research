"""Step 4: cross-sectional preprocessing.

Every operation here works ONE DATE AT A TIME, across the stocks that were index
members on that date. Nothing uses statistics from other dates, so nothing here
can leak future information.
"""
import numpy as np
import pandas as pd


def _align(mask: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    return mask.reindex(index=like.index, columns=like.columns, fill_value=False).astype(bool)


def members_only(signal: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Blank out every stock that was not an index member that day."""
    return signal.where(_align(mask, signal))


def winsorize(signal: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Per date, clip values below the 1st / above the 99th percentile of that date's cross-section."""
    lo = signal.quantile(lower, axis=1)
    hi = signal.quantile(upper, axis=1)
    return signal.clip(lower=lo, upper=hi, axis=0)


def cs_rank(signal: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Per date, rank among that day's members, scaled to a symmetric range around 0
    (lowest about -0.5, highest about +0.5, median 0). Non-members and NaN stay NaN."""
    s = members_only(signal, mask)
    r = s.rank(axis=1, method="average")
    n = s.notna().sum(axis=1)
    return (r.sub(0.5)).div(n, axis=0) - 0.5


def cs_zscore(signal: pd.DataFrame, mask: pd.DataFrame, winsor: bool = True) -> pd.DataFrame:
    """Per date: members only -> optional winsorise -> subtract mean, divide by std."""
    s = members_only(signal, mask)
    if winsor:
        s = winsorize(s)
    mu = s.mean(axis=1)
    sd = s.std(axis=1)
    return s.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)


def neutralize(score: pd.DataFrame, exposures: dict, min_stocks: int = 30) -> pd.DataFrame:
    """Per date, regress the score on the exposures (plus a constant) across stocks and
    keep the residual, so the result has no linear exposure to them.
    exposures: {name: date x ticker DataFrame}, e.g. {"beta": beta_score}.
    Optional step; compare results with and without it."""
    names = list(exposures)
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    X_all = {k: v.reindex_like(score) for k, v in exposures.items()}
    for d in score.index:
        y = score.loc[d]
        X = pd.concat([X_all[k].loc[d].rename(k) for k in names], axis=1)
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < min_stocks:
            continue
        A = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        out.loc[d, ok[ok].index] = y[ok].to_numpy() - A @ coef
    return out