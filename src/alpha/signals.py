"""Step 4: the eight raw signals.

Every function takes date x ticker grids and returns a grid of the same shape.
Convention: a HIGHER value means a HIGHER expected future return, so signals
where "low is good" (volatility, beta, IVOL, MAX, reversal) carry a minus sign.

Lookahead rules followed everywhere:
  - rolling windows end at row t (trailing, never centred)
  - only positive shifts (looking back), never shift(-k)
  - a value is NaN unless enough of its window has data (min_frac)
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _min_periods(window: int, min_frac: float) -> int:
    return max(2, int(np.ceil(min_frac * window)))


# ---------------------------------------------------------------- market proxy
def market_return(returns: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """Equal-weighted average return of the stocks that were index members that day.
    Used as the 'market' for beta and idiosyncratic volatility."""
    member = mask.reindex(index=returns.index, columns=returns.columns, fill_value=False)
    return returns.where(member).mean(axis=1)


# ---------------------------------------------------------------- 1. momentum
def momentum_12_1(adj_close: pd.DataFrame, min_frac: float = 0.8) -> pd.DataFrame:
    """P(t-21) / P(t-252) - 1: the 12-month return, skipping the most recent month.
    Jegadeesh & Titman (1993). Sign: +."""
    enough = adj_close.notna().rolling(TRADING_DAYS, min_periods=1).sum() >= min_frac * TRADING_DAYS
    mom = adj_close.shift(21) / adj_close.shift(TRADING_DAYS) - 1
    return mom.where(enough)


# ---------------------------------------------------------------- 2. reversal
def short_term_reversal(adj_close: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """-(P(t) / P(t-5) - 1): minus last week's return, so last week's losers score high.
    Jegadeesh (1990), Lehmann (1990). Sign: -."""
    return -(adj_close / adj_close.shift(window) - 1)


# ---------------------------------------------------------------- 3. 52-week high
def high_52w(adj_close: pd.DataFrame, min_frac: float = 0.8) -> pd.DataFrame:
    """P(t) / max(P over the last 252 days, including today). Always <= 1.
    George & Hwang (2004). Sign: +."""
    peak = adj_close.rolling(TRADING_DAYS, min_periods=_min_periods(TRADING_DAYS, min_frac)).max()
    return (adj_close / peak).where(adj_close.notna())


# ---------------------------------------------------------------- 4. volatility
def volatility(returns: pd.DataFrame, window: int = 63, min_frac: float = 0.8) -> pd.DataFrame:
    """-(annualised std of daily returns over the last 63 days).
    Ang et al. (2006), Blitz & van Vliet (2007). Sign: -."""
    vol = returns.rolling(window, min_periods=_min_periods(window, min_frac)).std() * np.sqrt(TRADING_DAYS)
    return -vol


# ---------------------------------------------------------------- helpers for 5 and 6
def _rolling_market_moments(returns, market, window, min_frac):
    """Rolling cov(r_i, m), var(m) and var(r_i), each computed over the days where
    BOTH the stock and the market have a return (pairwise), so beta is consistent."""
    mp = _min_periods(window, min_frac)
    m = pd.DataFrame(np.repeat(market.to_numpy()[:, None], returns.shape[1], axis=1),
                     index=returns.index, columns=returns.columns)
    both = returns.notna() & m.notna()
    r = returns.where(both)
    m = m.where(both)
    roll = lambda x: x.rolling(window, min_periods=mp)
    mean_r, mean_m = roll(r).mean(), roll(m).mean()
    cov = roll(r * m).mean() - mean_r * mean_m
    var_m = roll(m * m).mean() - mean_m ** 2
    var_r = roll(r * r).mean() - mean_r ** 2
    return cov, var_m, var_r


# ---------------------------------------------------------------- 5. beta
def beta(returns: pd.DataFrame, market: pd.Series, window: int = TRADING_DAYS,
         min_frac: float = 0.8) -> pd.DataFrame:
    """-(slope of daily stock returns on market returns over the last 252 days).
    Black (1972), Frazzini & Pedersen (2014). Sign: -."""
    cov, var_m, _ = _rolling_market_moments(returns, market, window, min_frac)
    b = cov / var_m.where(var_m > 0)
    return -b


# ---------------------------------------------------------------- 6. idiosyncratic vol
def idio_vol(returns: pd.DataFrame, market: pd.Series, window: int = 21,
             min_frac: float = 0.8) -> pd.DataFrame:
    """-(annualised std of residuals from a market-model regression over the last 21 days).
    residual variance = var(r) - cov(r,m)^2 / var(m).
    Ang, Hodrick, Xing & Zhang (2006) used Fama-French 3-factor residuals; this is the
    market-only simplification. Sign: -."""
    cov, var_m, var_r = _rolling_market_moments(returns, market, window, min_frac)
    resid_var = (var_r - cov ** 2 / var_m.where(var_m > 0)).clip(lower=0)
    return -np.sqrt(resid_var) * np.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------- 7. MAX
def max_return(returns: pd.DataFrame, window: int = 21, min_frac: float = 0.8) -> pd.DataFrame:
    """-(largest single daily return over the last 21 days).
    Bali, Cakici & Whitelaw (2011). Sign: -."""
    return -returns.rolling(window, min_periods=_min_periods(window, min_frac)).max()


# ---------------------------------------------------------------- 8. Amihud
def amihud(returns: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
           window: int = 21, min_frac: float = 0.8) -> pd.DataFrame:
    """log of mean(|r| / dollar volume) over the last 21 days, dollar volume = RAW close x volume.
    Higher = less liquid. Amihud (2002). Sign: + (illiquid stocks should pay a premium).
    Log because the raw ratio is extremely skewed."""
    dollar_vol = (close * volume).reindex_like(returns)
    dollar_vol = dollar_vol.where(dollar_vol > 0)          # zero volume -> unknown, not infinite
    ratio = returns.abs() / dollar_vol
    illiq = ratio.rolling(window, min_periods=_min_periods(window, min_frac)).mean()
    return np.log(illiq.where(illiq > 0) * 1e9)           # scale only so the numbers are readable


# ---------------------------------------------------------------- all together
SIGNAL_NAMES = ["momentum", "reversal", "high_52w", "volatility",
                "beta", "idio_vol", "max_return", "amihud"]


def compute_all(adj_close, returns, close, volume, mask) -> dict:
    """Compute all eight raw signals. Returns {name: date x ticker DataFrame}."""
    mkt = market_return(returns, mask)
    return {
        "momentum": momentum_12_1(adj_close),
        "reversal": short_term_reversal(adj_close),
        "high_52w": high_52w(adj_close),
        "volatility": volatility(returns),
        "beta": beta(returns, mkt),
        "idio_vol": idio_vol(returns, mkt),
        "max_return": max_return(returns),
        "amihud": amihud(returns, close, volume),
    }