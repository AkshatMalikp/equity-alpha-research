"""Tests for step 5. Run from the repo root:  PYTHONPATH=src pytest -q"""
import numpy as np
import pandas as pd
import pytest

from alpha import evaluate as ev


def planted(n_days=600, n_stocks=200, true_ic=0.10, seed=0):
    """Synthetic world: tomorrow's return = true_ic * today's signal + noise.
    Also returns a pure-noise signal that predicts nothing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_stocks)]
    sig = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)
    noise_sig = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)
    eps = rng.normal(size=(n_days, n_stocks))
    fwd = 0.01 * (true_ic * sig + np.sqrt(1 - true_ic ** 2) * eps)    # return realised after t
    return sig, noise_sig, fwd


# ---------------------------------------------------------------- statistics helpers
def test_newey_west_lag0_is_plain_tstat():
    x = np.random.default_rng(1).normal(0.1, 1, 500)
    mean, t, n = ev.newey_west_tstat(x, lags=0)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert t == pytest.approx(plain)


def test_newey_west_shrinks_t_for_overlapping_data():
    rng = np.random.default_rng(2)
    daily = pd.Series(rng.normal(0.0005, 0.01, 3000))
    overlapping = daily.rolling(21).sum().dropna()          # like a 21-day forward return series
    _, t_naive, _ = ev.newey_west_tstat(overlapping, 0)
    _, t_nw, _ = ev.newey_west_tstat(overlapping, 20)
    assert abs(t_nw) < abs(t_naive) / 2


def test_holm_and_bh_known_values():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    h = ev.holm(p)
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
    bh = ev.benjamini_hochberg(p)
    assert bh["a"] == pytest.approx(0.03) and bh["c"] == pytest.approx(0.04) and bh["b"] == pytest.approx(0.04)


def test_deflated_sharpe_falls_as_trials_grow():
    rng = np.random.default_rng(3)
    few = rng.normal(0, 0.05, 3)
    many = rng.normal(0, 0.05, 100)
    d_few = ev.deflated_sharpe(0.15, 240, 0.0, 3.0, few)
    d_many = ev.deflated_sharpe(0.15, 240, 0.0, 3.0, many)
    assert 0 <= d_many < d_few <= 1


# ---------------------------------------------------------------- 1. rank IC
def test_rank_ic_matches_manual_spearman():
    rng = np.random.default_rng(4)
    s = pd.DataFrame([rng.normal(size=40)], index=pd.bdate_range("2020-01-01", periods=1))
    f = pd.DataFrame([rng.normal(size=40)], index=s.index)
    ic = ev.rank_ic(s, f, min_stocks=10).iloc[0]
    manual = np.corrcoef(s.iloc[0].rank(), f.iloc[0].rank())[0, 1]
    assert ic == pytest.approx(manual)


def test_rank_ic_recovers_planted_signal_and_ignores_noise():
    sig, noise_sig, fwd = planted()
    good = ev.ic_summary(ev.rank_ic(sig, fwd), lags=0)
    bad = ev.ic_summary(ev.rank_ic(noise_sig, fwd), lags=0)
    assert good["mean_ic"] == pytest.approx(0.10, abs=0.015)
    assert good["ic_t_nw"] > 10
    assert abs(bad["mean_ic"]) < 0.01 and abs(bad["ic_t_nw"]) < 3


def test_rank_ic_skips_thin_dates():
    sig, _, fwd = planted(n_days=3, n_stocks=10)
    assert ev.rank_ic(sig, fwd, min_stocks=30).isna().all()


# ---------------------------------------------------------------- holding periods
def test_period_returns_compounds_between_rebalance_dates():
    idx = pd.bdate_range("2020-01-27", "2020-02-28")
    r = pd.DataFrame({"A": 0.01}, index=idx)
    dates = ev.rebalance_dates(idx)
    pr = ev.period_returns(r, dates)
    n_feb = ((idx > pd.Timestamp("2020-01-31")) & (idx <= pd.Timestamp("2020-02-28"))).sum()
    assert list(pr.index) == [pd.Timestamp("2020-01-31")]
    assert pr.iloc[0, 0] == pytest.approx(1.01 ** n_feb - 1)


# ---------------------------------------------------------------- 2. quantiles
def test_quantiles_monotone_when_signal_is_perfect():
    rng = np.random.default_rng(5)
    dates = pd.DatetimeIndex(pd.date_range("2015-01-31", periods=24, freq="ME"))
    cols = [f"S{i}" for i in range(100)]
    per = pd.DataFrame(rng.normal(0, 0.05, (24, 100)), index=dates, columns=cols)
    sig = per.copy()                                   # signal = the future return itself
    b, turn = ev.quantile_returns(sig, per)
    summ = ev.quantile_summary(b, turn)
    assert summ["monotonicity"] == pytest.approx(1.0)
    assert summ["spread_monthly"] > 0


# ---------------------------------------------------------------- 3. Fama-MacBeth
def test_fama_macbeth_recovers_known_slope_and_drops_copies():
    rng = np.random.default_rng(6)
    dates = pd.DatetimeIndex(pd.date_range("2015-01-31", periods=60, freq="ME"))
    cols = [f"S{i}" for i in range(300)]
    a = pd.DataFrame(rng.uniform(-0.5, 0.5, (60, 300)), index=dates, columns=cols)
    noise = pd.DataFrame(rng.uniform(-0.5, 0.5, (60, 300)), index=dates, columns=cols)
    per = 0.02 * a + pd.DataFrame(rng.normal(0, 0.03, (60, 300)), index=dates, columns=cols)
    s = ev.fm_summary(ev.fama_macbeth({"a": a, "noise": noise}, per))
    assert s.loc["a", "fm_slope_monthly"] == pytest.approx(0.02, abs=0.003)
    assert s.loc["a", "fm_t_nw"] > 10
    assert abs(s.loc["noise", "fm_t_nw"]) < 3