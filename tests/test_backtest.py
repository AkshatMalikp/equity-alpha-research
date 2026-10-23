"""Tests for step 7. Run from the repo root:  PYTHONPATH=src pytest -q"""
import numpy as np
import pandas as pd
import pytest

from alpha import backtest as bt


def two_stock_world(n=30):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({"A": 0.01, "B": -0.01}, index=idx)
    return rets


def test_quintile_weights_are_dollar_neutral():
    pred = pd.Series(np.arange(100.0), index=[f"S{i}" for i in range(100)])
    w = bt.quintile_weights(pred)
    assert w.sum() == pytest.approx(0)
    assert w[w > 0].sum() == pytest.approx(1) and w[w < 0].sum() == pytest.approx(-1)
    assert set(w[w > 0].index) == {f"S{i}" for i in range(80, 100)}


def test_rank_weights_gross_two():
    pred = pd.Series(np.random.default_rng(0).normal(size=200))
    w = bt.rank_weights(pred)
    assert w.sum() == pytest.approx(0, abs=1e-12) and w.abs().sum() == pytest.approx(2)


def test_long_winner_short_loser_earns_both_legs():
    rets = two_stock_world()
    w = {rets.index[0]: pd.Series({"A": 1.0, "B": -1.0})}
    d = bt.simulate(w, rets, borrow_bps=0)
    assert d["gross"].iloc[1] == pytest.approx(0.02)          # +1% long, +1% from the short
    assert d["gross"].iloc[0] == 0                            # nothing earned on the trade day itself


def test_returns_before_first_trade_are_excluded_and_lag_shifts_trading():
    rets = two_stock_world()
    w = {rets.index[3]: pd.Series({"A": 1.0, "B": -1.0})}
    d0 = bt.simulate(w, rets, borrow_bps=0)
    d1 = bt.simulate(w, rets, borrow_bps=0, lag=1)
    assert d0.index[0] == rets.index[3] and d1.index[0] == rets.index[4]


def test_flat_cost_charged_on_turnover():
    rets = two_stock_world()
    w = {rets.index[0]: pd.Series({"A": 1.0, "B": -1.0})}
    d = bt.simulate(w, rets, cost_rate=0.001, borrow_bps=0)
    assert d["turnover"].iloc[0] == pytest.approx(2.0)
    assert d["cost"].iloc[0] == pytest.approx(0.002)


def test_turnover_measured_against_drifted_positions():
    rets = two_stock_world()
    w = {rets.index[0]: pd.Series({"A": 0.5, "B": -0.5}), rets.index[5]: pd.Series({"A": 0.5, "B": -0.5})}
    d = bt.simulate(w, rets, borrow_bps=0)
    assert 0 < d["turnover"].iloc[5] < 0.2                   # small rebalancing trade, not a full re-buy


def test_breakeven_cost_formula():
    rets = two_stock_world(260)
    w = {rets.index[i]: pd.Series({"A": 1.0, "B": -1.0}) * (1 if i % 2 == 0 else -1) for i in range(0, 260, 21)}
    perf = bt.performance(bt.simulate(w, rets, borrow_bps=0))
    assert perf["breakeven_bps"] == pytest.approx(perf["gross_return"] / perf["traded_per_year"] * 1e4)


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert bt.max_drawdown(r) == pytest.approx(-0.5)


def test_parse_french_monthly():
    text = ("This file was created by CMPT_ME_BEME_RETS\n\n"
            ",Mkt-RF,SMB,HML,RF\n"
            "196307,   -0.39,   -0.48,   -0.81,    0.27\n"
            "196308,    5.07,   -0.80,    1.70,    0.25\n"
            "\n Annual Factors: January-December\n,Mkt-RF,SMB,HML,RF\n1964,  10,  1, 2, 3\n")
    df = bt.parse_french_monthly(text)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 2 and df.loc[pd.Period("1963-08", "M"), "Mkt-RF"] == pytest.approx(0.0507)


def test_factor_regression_recovers_alpha_and_beta():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2000-01", periods=240, freq="M")
    f = pd.DataFrame({"Mkt-RF": rng.normal(0.005, 0.04, 240)}, index=idx)
    y = pd.Series(0.002 + 0.5 * f["Mkt-RF"] + rng.normal(0, 0.005, 240), index=idx)
    out = bt.factor_regression(y, f)
    assert out["beta_Mkt-RF"] == pytest.approx(0.5, abs=0.03)
    assert out["alpha_annual"] == pytest.approx(0.024, abs=0.01)