"""Tests for step 4. Run from the repo root:  PYTHONPATH=src pytest -q"""
import numpy as np
import pandas as pd
import pytest

from alpha import signals as sg
from alpha.preprocess import cs_rank, cs_zscore, winsorize, members_only, neutralize


# ---------------------------------------------------------------- fake data
def fake_data(n_days=400, n_stocks=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_stocks)]
    r = rng.normal(0, 0.02, (n_days, n_stocks))
    adj = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
    rets = adj.pct_change(fill_method=None)
    close = adj * 1.3
    volume = pd.DataFrame(rng.integers(1e5, 1e7, (n_days, n_stocks)).astype(float), index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    return adj, rets, close, volume, mask


def run_all(adj, rets, close, volume, mask):
    return sg.compute_all(adj, rets, close, volume, mask)


# ---------------------------------------------------------------- 1. no lookahead (the key test)
@pytest.mark.parametrize("name", sg.SIGNAL_NAMES)
def test_no_lookahead(name):
    """Scramble every input AFTER day t. Every signal value on or before t must not change."""
    adj, rets, close, volume, mask = fake_data()
    t = 300
    before = run_all(adj, rets, close, volume, mask)[name]

    rng = np.random.default_rng(99)
    adj2, close2, vol2 = adj.copy(), close.copy(), volume.copy()
    adj2.iloc[t + 1:] = adj2.iloc[t + 1:].to_numpy() * rng.uniform(0.2, 5, adj2.iloc[t + 1:].shape)
    close2.iloc[t + 1:] = close2.iloc[t + 1:].to_numpy() * 7
    vol2.iloc[t + 1:] = vol2.iloc[t + 1:].to_numpy() * 0.01
    rets2 = adj2.pct_change(fill_method=None)
    after = run_all(adj2, rets2, close2, vol2, mask)[name]

    pd.testing.assert_frame_equal(before.iloc[: t + 1], after.iloc[: t + 1])


# ---------------------------------------------------------------- 2. known values
def test_momentum_by_hand():
    adj, *_ = fake_data()
    mom = sg.momentum_12_1(adj)
    d = adj.index[300]
    expected = adj["S0"].iloc[300 - 21] / adj["S0"].iloc[300 - 252] - 1
    assert mom.loc[d, "S0"] == pytest.approx(expected)
    assert mom.iloc[:252].isna().all().all()          # not enough history yet


def test_reversal_sign():
    idx = pd.bdate_range("2020-01-01", periods=10)
    adj = pd.DataFrame({"UP": np.linspace(100, 120, 10), "DOWN": np.linspace(100, 80, 10)}, index=idx)
    rev = sg.short_term_reversal(adj)
    assert rev.iloc[-1]["DOWN"] > 0 > rev.iloc[-1]["UP"]   # last week's loser scores high


def test_high_52w_at_most_one():
    adj, *_ = fake_data()
    h = sg.high_52w(adj).stack().dropna()
    assert (h <= 1 + 1e-12).all() and (h > 0).all()


def test_beta_recovers_known_value():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=300)
    m = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    rets = pd.DataFrame({"B2": 2 * m + rng.normal(0, 0.001, 300), "B05": 0.5 * m}, index=idx)
    b = -sg.beta(rets, m)                                    # undo the sign convention
    assert b["B2"].iloc[-1] == pytest.approx(2.0, abs=0.05)
    assert b["B05"].iloc[-1] == pytest.approx(0.5, abs=1e-6)


def test_idio_vol_zero_for_pure_market_stock():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2010-01-01", periods=100)
    m = pd.Series(rng.normal(0, 0.01, 100), index=idx)
    rets = pd.DataFrame({"PURE": 1.5 * m, "NOISY": m + rng.normal(0, 0.02, 100)}, index=idx)
    iv = -sg.idio_vol(rets, m)
    assert iv["PURE"].iloc[-1] == pytest.approx(0, abs=1e-6)
    assert iv["NOISY"].iloc[-1] > 0.1


def test_max_return_picks_the_spike():
    idx = pd.bdate_range("2020-01-01", periods=30)
    r = pd.DataFrame({"A": 0.001}, index=idx)
    r.iloc[25, 0] = 0.30
    assert -sg.max_return(r).iloc[-1, 0] == pytest.approx(0.30)


def test_amihud_illiquid_scores_higher():
    idx = pd.bdate_range("2020-01-01", periods=30)
    r = pd.DataFrame({"LIQ": 0.01, "ILLIQ": 0.01}, index=idx)
    close = pd.DataFrame(50.0, index=idx, columns=r.columns)
    vol = pd.DataFrame({"LIQ": 1e7, "ILLIQ": 1e5}, index=idx)
    a = sg.amihud(r, close, vol).iloc[-1]
    assert a["ILLIQ"] > a["LIQ"]


def test_missing_price_never_becomes_a_value():
    adj, *_ = fake_data()
    adj.iloc[350, 0] = np.nan
    assert np.isnan(sg.high_52w(adj).iloc[350, 0])


# ---------------------------------------------------------------- 3. preprocessing
def test_cs_rank_range_and_members_only():
    idx = pd.bdate_range("2020-01-01", periods=2)
    sig = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 2.0], "C": [3.0, 1.0], "D": [9.0, 9.0]}, index=idx)
    mask = pd.DataFrame({"A": True, "B": True, "C": True, "D": False}, index=idx)
    r = cs_rank(sig, mask)
    assert r["D"].isna().all()                               # non-member stays NaN
    assert r.loc[idx[0]].dropna().tolist() == pytest.approx([-1 / 3, 0, 1 / 3])
    assert r.loc[idx[0], "A"] < r.loc[idx[0], "C"]           # higher signal -> higher rank


def test_cs_zscore_mean_zero_std_one():
    adj, *_ = fake_data(n_stocks=50)
    mask = pd.DataFrame(True, index=adj.index, columns=adj.columns)
    z = cs_zscore(adj, mask, winsor=False)
    assert z.mean(axis=1).abs().max() < 1e-9
    assert (z.std(axis=1) - 1).abs().max() < 1e-9


def test_winsorize_clips_outlier():
    idx = pd.bdate_range("2020-01-01", periods=1)
    vals = list(range(99)) + [10_000]
    sig = pd.DataFrame([vals], index=idx, columns=[f"S{i}" for i in range(100)])
    w = winsorize(sig)
    assert w.iloc[0].max() < 200


def test_neutralize_removes_exposure():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = [f"S{i}" for i in range(200)]
    expo = pd.DataFrame(rng.normal(size=(3, 200)), index=idx, columns=cols)
    score = 2 * expo + pd.DataFrame(rng.normal(size=(3, 200)), index=idx, columns=cols)
    res = neutralize(score, {"x": expo})
    for d in idx:
        assert abs(np.corrcoef(res.loc[d], expo.loc[d])[0, 1]) < 1e-8