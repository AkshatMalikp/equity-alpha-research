"""Step 7: backtest the step 6 forecasts with realistic costs.

Run from the repo root (after combine_signals.py):
    DATA=data/processed_crsp PRED=results/step6_crsp RESULTS=results/step7_crsp \
        PYTHONPATH=src python scripts/backtest_strategies.py

Pre-registered main setting (fixed before looking at results):
  - monthly rebalance, long top fifth / short bottom fifth, $1 long and $1 short
  - cost = half the quoted bid-ask spread (CRSP, trailing 21 days), borrow fee 25 bps a year
Everything else (flat costs, lag, fund size, weights, subperiods) is a robustness check.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from alpha import backtest as bt

PROCESSED = Path(os.environ.get("DATA", "data/processed_crsp"))
PRED = Path(os.environ.get("PRED", "results/step6_crsp"))
OUT = Path(os.environ.get("RESULTS", "results/step7_crsp"))
STRATEGIES = ["reversal_only", "equal", "ic_weighted", "ridge", "lgbm"]


def run(pred, rets, cost, weights_fn=bt.quintile_weights, **kw):
    w = {d: weights_fn(pred.loc[d]) for d in pred.index}
    return bt.simulate(w, rets, cost_rate=cost, **kw)


def pct(df, cols):
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = (df[c] * 100).round(2)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rets = pd.read_parquet(PROCESSED / "returns.parquet")
    spread = pd.read_parquet(PROCESSED / "spread.parquet")
    close = pd.read_parquet(PROCESSED / "close.parquet")
    volume = pd.read_parquet(PROCESSED / "volume.parquet")
    preds = {s: pd.read_parquet(PRED / f"pred_{s}.parquet") for s in STRATEGIES if (PRED / f"pred_{s}.parquet").exists()}
    first = min(p.index.min() for p in preds.values())
    rets = rets.loc[first:]
    hs = bt.half_spread_cost(spread).loc[first:]
    print(f"strategies: {list(preds)}; {first.date()} to {rets.index[-1].date()}")
    print(f"median half-spread paid: {hs.stack().median() * 1e4:.1f} bps")

    # ------------------------------------------------------------ 1. main result
    main_rows, daily = {}, {}
    for s, p in preds.items():
        d = run(p, rets, hs)
        d.to_csv(OUT / f"daily_{s}.csv")
        daily[s] = d
        main_rows[s] = bt.performance(d)
    main_tab = pd.DataFrame(main_rows).T
    main_tab.to_csv(OUT / "main.csv")

    # ------------------------------------------------------------ 2. flat cost sensitivity
    sens = {}
    for s, p in preds.items():
        sens[s] = {f"{c} bps": bt.performance(run(p, rets, c / 1e4))["sharpe_net"] for c in [0, 5, 10, 20, 30]}
    sens = pd.DataFrame(sens).T
    sens.to_csv(OUT / "cost_sensitivity.csv")

    # ------------------------------------------------------------ 3. one-day implementation lag
    lag = {}
    for s, p in preds.items():
        lag[s] = {"gross Sharpe, lag 0": main_rows[s]["sharpe_gross"],
                  "gross Sharpe, lag 1": bt.performance(run(p, rets, 0.0, lag=1))["sharpe_gross"],
                  "net Sharpe, lag 1": bt.performance(run(p, rets, hs, lag=1))["sharpe_net"]}
    lag = pd.DataFrame(lag).T
    lag.to_csv(OUT / "lag_test.csv")

    # ------------------------------------------------------------ 4. fund size (market impact)
    cap_rows = {}
    for s in [x for x in ["reversal_only", "lgbm", "equal"] if x in preds]:
        row = {"spread only": main_rows[s]["sharpe_net"]}
        for aum in [100e6, 1e9]:
            imp = lambda traded, date, aum=aum: bt.impact_cost(rets, close, volume, traded * aum, date)
            row[f"${aum / 1e6:,.0f}M"] = bt.performance(run(preds[s], rets, hs, impact=imp))["sharpe_net"]
        cap_rows[s] = row
    cap = pd.DataFrame(cap_rows).T
    cap.to_csv(OUT / "capacity.csv")

    # ------------------------------------------------------------ 5. weighting scheme
    wts = {}
    for s, p in preds.items():
        wts[s] = {"quintile": main_rows[s]["sharpe_net"],
                  "rank-weighted": bt.performance(run(p, rets, hs, weights_fn=bt.rank_weights))["sharpe_net"]}
    wts = pd.DataFrame(wts).T

    # ------------------------------------------------------------ 6. subperiods
    sub = {}
    for s, d in daily.items():
        sub[s] = {f"{a}-{b}": bt.performance(d.loc[str(a):str(b)])["sharpe_net"]
                  for a, b in [(2002, 2009), (2010, 2017), (2018, 2025)]}
    sub = pd.DataFrame(sub).T
    sub.to_csv(OUT / "subperiods.csv")

    # ------------------------------------------------------------ 7. Fama-French
    ff_tab = None
    try:
        ff = bt.load_french_factors("data/raw")
        rows = {}
        for s, d in daily.items():
            monthly = (1 + d["net"]).groupby(d.index.to_period("M")).prod() - 1
            rows[s] = bt.factor_regression(monthly, ff)
        ff_tab = pd.DataFrame(rows).T
        ff_tab.to_csv(OUT / "factor_regression.csv")
    except Exception as e:                                       # no internet, site down, ...
        print(f"(skipping Fama-French regression: {e})")

    # ------------------------------------------------------------ print
    pd.set_option("display.width", 220)
    print("\n=== 1. Main result: quintile long-short, half-spread costs, 25 bps borrow ===")
    show = pct(main_tab, ["gross_return", "net_return", "volatility", "max_drawdown", "hit_rate_monthly", "cost_per_year"])
    show["traded_per_year"] = show["traded_per_year"].round(1)
    show["breakeven_bps"] = show["breakeven_bps"].round(1)
    print(show.drop(columns=["years"]).round(3).to_string())
    print("(returns, volatility, drawdown, hit rate, cost in %; traded_per_year = $ traded per $1 of capital)")

    print("\n=== 2. Net Sharpe at flat one-way costs ===")
    print(sens.round(2).to_string())
    print("\n=== 3. One-day implementation lag (bid-ask bounce check) ===")
    print(lag.round(2).to_string())
    print("\n=== 4. Net Sharpe by fund size (half-spread + square-root impact) ===")
    print(cap.round(2).to_string())
    print("\n=== 5. Net Sharpe: quintile vs rank weights ===")
    print(wts.round(2).to_string())
    print("\n=== 6. Net Sharpe by subperiod ===")
    print(sub.round(2).to_string())
    if ff_tab is not None:
        print("\n=== 7. Fama-French 5 factors + momentum, monthly net returns (Newey-West t) ===")
        cols = ["alpha_annual", "alpha_t"] + [c for c in ff_tab.columns if c.startswith("beta_")] + ["r2"]
        t = ff_tab[cols].copy()
        t["alpha_annual"] = (t["alpha_annual"] * 100).round(2)
        print(t.round(2).to_string())
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()