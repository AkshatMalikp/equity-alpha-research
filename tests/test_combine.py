"""Tests for step 6. Run from the repo root:  PYTHONPATH=src pytest -q"""
import numpy as np
import pandas as pd
import pytest

from alpha import combine as cb

FEATS = ["a", "b", "c"]


def fake_long(n_months=96, n_stocks=150, seed=0, strength=0.3):
    """Monthly panel where feature 'a' predicts next-month returns, 'b' and 'c' are noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    cols = [f"P{i}" for i in range(n_stocks)]
    ranks = {f: pd.DataFrame(rng.uniform(-0.5, 0.5, (n_months, n_stocks)), index=dates, columns=cols) for f in FEATS}
    per = 0.05 * (strength * ranks["a"] + rng.normal(0, 0.3, (n_months, n_stocks)))
    per = per.iloc[:-1]                                       # last date has no next period
    return cb.to_long(ranks, per), ranks, per


def test_to_long_ranks_target_within_each_date():
    df, _, _ = fake_long(n_months=5, n_stocks=50)
    g = df[df["date"] == df["date"].min()]
    assert g["y"].min() > -0.5 and g["y"].max() < 0.5
    assert g["y"].rank().corr(g["ret"].rank()) == pytest.approx(1.0)


def test_label_end_is_next_rebalance_date():
    df, _, per = fake_long(n_months=5, n_stocks=50)
    d0, d1 = per.index[0], per.index[1]
    assert (df.loc[df["date"] == d0, "label_end"] == d1).all()


def test_walk_forward_never_trains_on_unfinished_labels():
    df, _, _ = fake_long()
    seen = {}

    def spy(train, feats):
        seen[train["date"].max()] = train["label_end"].max()
        return lambda X: X[feats].mean(axis=1)

    cb.walk_forward(df, FEATS, spy, first_test_year=2003, verbose=False)
    for Y in range(2003, 2008):
        test_start = df.loc[df["date"].dt.year == Y, "date"].min()
        max_label_end = max(v for k, v in seen.items() if k < test_start)
        assert max_label_end <= test_start


@pytest.mark.parametrize("name", ["equal", "ic_weighted", "ridge", "lgbm"])
def test_predictions_ignore_the_future(name):
    """Scramble every label from year 2004 on: predictions for 2004 must not change,
    because the 2004 model is trained only on data before 2004."""
    pytest.importorskip("lightgbm") if name == "lgbm" else None
    df, _, _ = fake_long()
    p1, _ = cb.walk_forward(df, FEATS, cb.COMBINERS[name], first_test_year=2004, verbose=False)
    df2 = df.copy()
    future = df2["date"].dt.year >= 2004
    df2.loc[future, ["y", "ret"]] = np.random.default_rng(9).permutation(df2.loc[future, ["y", "ret"]].to_numpy())
    p2, _ = cb.walk_forward(df2, FEATS, cb.COMBINERS[name], first_test_year=2004, verbose=False)
    in_2004 = df.loc[p1.index, "date"].dt.year == 2004
    pd.testing.assert_series_equal(p1[in_2004.values], p2[in_2004.values])


def test_ridge_puts_weight_on_the_predictive_feature():
    df, _, _ = fake_long(n_months=120, n_stocks=300)
    cb.fit_ridge(df[df["y"].notna()], FEATS)
    w = cb.fit_ridge.last
    assert w["a"] > 5 * max(abs(w["b"]), abs(w["c"]))


def test_ic_weighted_ignores_noise_features():
    df, _, _ = fake_long(n_months=120, n_stocks=300, strength=1.0)
    pred = cb.fit_ic_weighted(df[df["y"].notna()], FEATS)
    X = pd.DataFrame({"a": [0.5, -0.5], "b": [0.0, 0.0], "c": [0.0, 0.0]})
    p = pred(X)
    assert p.iloc[0] > p.iloc[1]


def test_lgbm_beats_noise_out_of_sample():
    pytest.importorskip("lightgbm")
    df, _, _ = fake_long(n_months=120, n_stocks=300, strength=1.0)
    p, _ = cb.walk_forward(df, FEATS, cb.fit_lgbm, first_test_year=2005, verbose=False)
    test = df.loc[p.index].assign(p=p.values)
    ics = [g["p"].rank().corr(g["y"].rank()) for _, g in test.dropna(subset=["y"]).groupby("date")]
    assert np.mean(ics) > 0.1