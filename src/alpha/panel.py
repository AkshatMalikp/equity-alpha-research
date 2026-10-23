import numpy as np
import pandas as pd


def pivot_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Convert long prices to a date-by-ticker grid for one field."""
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()


def daily_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns, keeping missing prices as NaN."""
    return adj_close.pct_change(fill_method=None)


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compound returns from t+1 through t+horizon for each date t."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    compounded = (1 + returns).rolling(
        window=horizon,
        min_periods=horizon,
    ).apply(np.prod, raw=True)
    return compounded.shift(-horizon) - 1


def membership_mask(membership, dates, tickers) -> pd.DataFrame:
    """Boolean grid: True if ticker was an index member on that date.
    Each date uses the membership row in force on it (same as-of rule as members_on)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    tickers = list(tickers)
    pos = membership.index.searchsorted(dates, side="right") - 1
    if (pos < 0).any():
        raise ValueError("some dates are before the first membership row")

    col = {t: j for j, t in enumerate(tickers)}
    rows = {}
    for p in np.unique(pos):                      # build each membership row once
        arr = np.zeros(len(tickers), dtype=bool)
        for t in membership.iloc[p]["tickers"]:
            j = col.get(t)
            if j is not None:
                arr[j] = True
        rows[p] = arr

    data = np.vstack([rows[p] for p in pos])
    return pd.DataFrame(data, index=dates, columns=tickers)


def coverage_by_year(
    mask: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize member-days with prices and coverage percentage by year."""
    member_mask = mask.fillna(False).astype(bool)
    prices_available = adj_close.reindex(
        index=member_mask.index,
        columns=member_mask.columns,
    ).notna()

    member_days = member_mask.sum(axis=1).groupby(member_mask.index.year).sum()
    member_days_with_price = (
        (member_mask & prices_available)
        .sum(axis=1)
        .groupby(member_mask.index.year)
        .sum()
    )
    coverage_pct = (
        member_days_with_price
        .div(member_days.where(member_days != 0))
        .mul(100)
    )

    return pd.DataFrame(
        {
            "member_days": member_days,
            "member_days_with_price": member_days_with_price,
            "coverage_pct": coverage_pct,
        }
    ).rename_axis("year")

def find_reused_tickers(adj_close, mask, grace_days=30) -> pd.DataFrame:
    """Flag tickers whose Yahoo history probably belongs to a different company.
    starts_after_membership: prices begin after the ticker left the index (certain reuse)
    starts_mid_membership: prices begin > grace_days after it joined (suspicious, check by hand)"""
    rows = []
    for tk in adj_close.columns:
        if tk not in mask.columns or not mask[tk].any():
            continue
        first_price = adj_close[tk].first_valid_index()
        if first_price is None:
            continue
        member_dates = mask.index[mask[tk]]
        first_member, last_member = member_dates.min(), member_dates.max()
        if first_price > last_member:
            reason = "starts_after_membership"
        elif first_price > first_member + pd.Timedelta(days=grace_days):
            reason = "starts_mid_membership"
        else:
            continue
        rows.append({"ticker": tk, "reason": reason, "first_price": first_price.date(),
                     "first_member": first_member.date(), "last_member": last_member.date()})
    return pd.DataFrame(rows, columns=["ticker", "reason", "first_price", "first_member", "last_member"])    
