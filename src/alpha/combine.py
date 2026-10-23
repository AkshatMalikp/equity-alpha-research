"""Step 6: combine the eight signals into one forecast, strictly out of sample.

Setup
  - one row per (rebalance date, stock): X = the 8 cross-sectional ranks at the month-end,
    y = the stock's NEXT-month return, ranked across stocks that date (-0.5 .. +0.5).
    Ranking y removes the market's overall move, which a long-short portfolio cancels anyway.
  - expanding walk-forward by calendar year: to predict year Y, train only on rows whose
    label period ENDS on or before the first test date (that is the purge).
  - four combiners: equal weight, IC-weighted, ridge, LightGBM.
  - every tuning choice (ridge alpha, LightGBM trees) is made on the LAST part of the
    training window only, never on the test year.
"""
import numpy as np
import pandas as pd

from alpha.evaluate import rank_ic


# ====================================================================== data layout
def to_long(ranks: dict, per_rets: pd.DataFrame) -> pd.DataFrame:
    """Rank grids + next-period returns -> long table with columns
    date, permno, <signal names>, ret (next-period return), y (ranked ret), label_end."""
    dates = per_rets.index
    cols = {k: g.reindex(index=dates).stack(future_stack=True) for k, g in ranks.items()}
    X = pd.DataFrame(cols)
    X = X[X.notna().any(axis=1)]                              # members only (non-members are all-NaN)
    ret = per_rets.stack(future_stack=True).rename("ret")
    df = X.join(ret, how="left")
    df.index = df.index.set_names(["date", "permno"])
    df = df.reset_index()
    r = df.groupby("date")["ret"].rank(method="average")
    n = df.groupby("date")["ret"].transform("count")
    df["y"] = (r - 0.5) / n - 0.5
    next_date = pd.Series(dates[1:].tolist() + [pd.NaT], index=dates)
    df["label_end"] = df["date"].map(next_date)
    return df


def to_grid(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="permno", values=col).sort_index()


# ====================================================================== combiners
# Each "fit" takes a training table and returns a predict(X) function.
def fit_equal(train, feats):
    return lambda X: X[feats].mean(axis=1)


def _per_date_ic(df, feats, target="y"):
    """Monthly rank IC of each feature with the target, one row per date."""
    out = {}
    for d, g in df.groupby("date"):
        g = g.dropna(subset=[target])
        if len(g) < 30:
            continue
        rt = g[target].rank()
        out[d] = {f: g[f].rank().corr(rt) for f in feats}
    return pd.DataFrame(out).T


def fit_ic_weighted(train, feats, lookback_months: int = 36):
    """Weight each signal by its average IC over the last 36 training months; negative -> 0."""
    last = sorted(train["date"].unique())[-lookback_months:]
    ic = _per_date_ic(train[train["date"].isin(last)], feats).mean().clip(lower=0)
    w = ic / ic.sum() if ic.sum() > 0 else pd.Series(1 / len(feats), index=feats)
    return lambda X: (X[feats].fillna(0) * w).sum(axis=1)


def _split_train_val(train, val_frac: float = 0.2, min_val_months: int = 12):
    dates = sorted(train["date"].unique())
    n_val = max(min_val_months, int(len(dates) * val_frac))
    cut = dates[-n_val]
    fit = train[train["label_end"] <= cut]                    # purge between inner train and val
    val = train[train["date"] >= cut]
    return fit, val


def _mean_ic(pred, df):
    tmp = df[["date"]].assign(p=pred.values, y=df["y"].values)
    ics = [g["p"].rank().corr(g["y"].rank()) for _, g in tmp.groupby("date") if len(g) >= 30]
    return np.nanmean(ics) if ics else np.nan


def fit_ridge(train, feats, alphas=(0.1, 1, 10, 100, 1000, 10000)):
    """Ridge regression of ranked next-month return on the 8 ranks. Alpha is chosen by the
    mean rank IC on the last 20% of the training window, then the model is refit on all of it.
    Missing ranks are filled with 0, the cross-sectional middle."""
    from sklearn.linear_model import Ridge
    fit, val = _split_train_val(train)
    Xf, yf = fit[feats].fillna(0), fit["y"]
    best = max(alphas, key=lambda a: _mean_ic(pd.Series(Ridge(alpha=a).fit(Xf, yf).predict(val[feats].fillna(0))), val))
    model = Ridge(alpha=best).fit(train[feats].fillna(0), train["y"])
    fit_ridge.last = {"alpha": best, **dict(zip(feats, model.coef_))}
    return lambda X: pd.Series(model.predict(X[feats].fillna(0)), index=X.index)


LGBM_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, max_depth=4,
                   min_child_samples=1000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=10.0, n_estimators=1000, verbose=-1, random_state=0)


def fit_lgbm(train, feats, params=None):
    """Gradient-boosted trees, deliberately small and heavily regularised for noisy return data.
    The number of trees is chosen by early stopping on the last 20% of the training window,
    then the model is refit on all of it with that many trees."""
    import warnings
    import lightgbm as lgb
    params = {**LGBM_PARAMS, **(params or {})}
    fit, val = _split_train_val(train)
    m = lgb.LGBMRegressor(**params)
    with warnings.catch_warnings():                           # eval_set is renamed in newer LightGBM
        warnings.simplefilter("ignore")
        m.fit(fit[feats], fit["y"], eval_set=[(val[feats], val["y"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    n_trees = max(20, m.best_iteration_ or 20)
    final = lgb.LGBMRegressor(**{**params, "n_estimators": n_trees}).fit(train[feats], train["y"])
    fit_lgbm.last = {"n_trees": n_trees, **dict(zip(feats, final.booster_.feature_importance("gain")))}
    return lambda X: pd.Series(final.predict(X[feats]), index=X.index)


COMBINERS = {"equal": fit_equal, "ic_weighted": fit_ic_weighted, "ridge": fit_ridge, "lgbm": fit_lgbm}


# ====================================================================== walk-forward
def walk_forward(df: pd.DataFrame, feats: list, fit_fn, first_test_year: int, verbose: bool = True):
    """Expanding window, one fit per calendar year. Returns (predictions aligned to df rows
    in the test years, per-year info such as chosen alpha / feature importance)."""
    preds, info = [], {}
    labelled = df[df["y"].notna()]
    for Y in range(first_test_year, df["date"].dt.year.max() + 1):
        test = df[df["date"].dt.year == Y]
        if test.empty:
            continue
        test_start = test["date"].min()
        train = labelled[labelled["label_end"] <= test_start]    # purge: labels must be finished
        if train["date"].nunique() < 24:
            continue
        predict = fit_fn(train, feats)
        preds.append(predict(test))
        info[Y] = getattr(fit_fn, "last", {}).copy() if hasattr(fit_fn, "last") else {}
        if verbose:
            print(f"    {Y}: trained on {train['date'].nunique()} months", flush=True)
    return pd.concat(preds).rename("pred"), pd.DataFrame(info).T