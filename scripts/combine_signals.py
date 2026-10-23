"""Step 6: combine the eight signals out of sample and compare four combiners.

Run from the repo root (after evaluate_signals.py):
    DATA=data/processed_crsp RESULTS=results/step6_crsp EVAL_START=1997-01-01 \
        PYTHONPATH=src python scripts/combine_signals.py

Pre-registered choices (fixed before looking at results):
  - monthly rebalance on the last trading day; target = next month's return, ranked per date
  - expanding walk-forward, one refit per calendar year, first test year 2002 (5 years of training)
  - tuning only on the last 20% of each training window
  - the comparison that decides "does ML help": LightGBM vs ridge, paired monthly IC difference
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from alpha.signals import SIGNAL_NAMES
from alpha import evaluate as ev
from alpha import combine as cb

PROCESSED = Path(os.environ.get("DATA", "data/processed"))
SIGNALS = PROCESSED / "signals"
OUT = Path(os.environ.get("RESULTS", "results/step6"))
EVAL_START = os.environ.get("EVAL_START", "2006-01-01")
FIRST_TEST_YEAR = int(os.environ.get("FIRST_TEST_YEAR", str(pd.Timestamp(EVAL_START).year + 5)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rets = pd.read_parquet(PROCESSED / "returns.parquet")
    ranks = {k: pd.read_parquet(SIGNALS / f"rank_{k}.parquet").loc[EVAL_START:] for k in SIGNAL_NAMES}
    dates = ev.rebalance_dates(rets.loc[EVAL_START:].index)
    per = ev.period_returns(rets, dates)
    df = cb.to_long(ranks, per)
    print(f"panel: {len(df):,} stock-months, {df['date'].nunique()} months, "
          f"first test year {FIRST_TEST_YEAR}")

    models = dict(cb.COMBINERS)
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("LightGBM not installed: skipping it (pip install lightgbm)")
        models.pop("lgbm")

    monthly_ic, rows, extras = {}, {}, {}
    test_mask = df["date"].dt.year >= FIRST_TEST_YEAR
    per_test = per.loc[per.index.year >= FIRST_TEST_YEAR]

    # reference: the best single signal, same test months
    baseline = df.loc[test_mask, ["date", "permno", "reversal"]].rename(columns={"reversal": "pred"})
    candidates = {"reversal_only": baseline}

    for name, fn in models.items():
        print(f"\n{name} ...")
        p, info = cb.walk_forward(df, SIGNAL_NAMES, fn, FIRST_TEST_YEAR)
        candidates[name] = df.loc[p.index, ["date", "permno"]].assign(pred=p.values)
        if len(info.columns):
            info.to_csv(OUT / f"fold_info_{name}.csv")
            extras[name] = info

    for name, c in candidates.items():
        grid = cb.to_grid(c, "pred")
        grid.to_parquet(OUT / f"pred_{name}.parquet")
        ic = ev.rank_ic(grid, per_test)
        monthly_ic[name] = ic
        b, turn = ev.quantile_returns(grid, per_test)
        q = ev.quantile_summary(b, turn)
        m, t, n = ev.newey_west_tstat(ic, lags=3)
        rows[name] = {"mean_ic": m, "ic_t_nw": t, "icir": m / ic.std(), "pct_months_ic_pos": (ic > 0).mean(),
                      "Q5_Q1_annual": q["spread_annual"], "Q5_Q1_sharpe": q["spread_sharpe_annual"],
                      "monotonicity": q["monotonicity"], "turnover_top": q["turnover_top"], "months": n}

    summary = pd.DataFrame(rows).T
    ic_df = pd.DataFrame(monthly_ic)
    by_year = ic_df.groupby(ic_df.index.year).mean()
    summary.to_csv(OUT / "summary.csv")
    by_year.to_csv(OUT / "ic_by_year.csv")

    pd.set_option("display.width", 200)
    show = summary.copy()
    for c in ["pct_months_ic_pos", "Q5_Q1_annual", "turnover_top"]:
        show[c] = (show[c] * 100).round(1)
    print(f"\n=== Out-of-sample results, {FIRST_TEST_YEAR}-{per_test.index[-1].year} (monthly) ===")
    print(show.round(3).to_string())

    print("\n=== Paired tests: is the difference in monthly IC real? ===")
    pairs = [("ridge", "equal"), ("ic_weighted", "equal"), ("lgbm", "ridge"), ("lgbm", "equal"), ("ridge", "reversal_only")]
    for a, b in pairs:
        if a in ic_df and b in ic_df:
            d = (ic_df[a] - ic_df[b]).dropna()
            m, t, n = ev.newey_west_tstat(d, lags=3)
            print(f"  {a:12} - {b:13}: mean IC difference {m:+.4f}, t = {t:+.2f}")

    print("\n=== Mean IC by year (does any model win consistently?) ===")
    print(by_year.round(3).to_string())

    if "ridge" in extras:
        print("\n=== Ridge: chosen alpha and coefficients per year (stability check) ===")
        print(extras["ridge"].round(3).to_string())
    if "lgbm" in extras:
        imp = extras["lgbm"].drop(columns="n_trees")
        print("\n=== LightGBM: share of total gain per signal, averaged over years ===")
        print((imp.div(imp.sum(axis=1), axis=0).mean() * 100).round(1).sort_values(ascending=False).to_string())
        print(f"trees per year: {extras['lgbm']['n_trees'].astype(int).tolist()}")
    print(f"\nPredictions saved to {OUT}/pred_<model>.parquet for step 7.")


if __name__ == "__main__":
    main()