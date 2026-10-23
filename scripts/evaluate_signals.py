"""Step 5: test whether each signal predicts returns.

Run from the repo root (after build_signals.py):
    PYTHONPATH=src python scripts/evaluate_signals.py                                  # Yahoo
    DATA=data/processed_crsp RESULTS=results/step5_crsp EVAL_START=1997-01-01 \
        PYTHONPATH=src python scripts/evaluate_signals.py                              # CRSP

Pre-registered choices (decided BEFORE looking at results, write them in the README):
  - primary test: 21-day rank IC, Newey-West t-stat with 20 lags
  - significance bar: |t| > 3 (Harvey-Liu-Zhu), plus Holm and BH corrections over the 8 signals
  - portfolios: monthly rebalance on the last trading day, 5 equal-weighted buckets
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from alpha.signals import SIGNAL_NAMES
from alpha import evaluate as ev

PROCESSED = Path(os.environ.get("DATA", "data/processed"))
SIGNALS = PROCESSED / "signals"
OUT = Path(os.environ.get("RESULTS", "results/step5"))
EVAL_START = os.environ.get("EVAL_START", "2006-01-01")   # one year after prices start
HORIZONS = [1, 5, 10, 21, 63]
PRIMARY_H = 21


def fwd_from_returns(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Same definition as panel.forward_returns (days t+1..t+h), via log sums because it is faster."""
    return np.expm1(np.log1p(returns).rolling(h).sum().shift(-h))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rets = pd.read_parquet(PROCESSED / "returns.parquet")
    ranks = {k: pd.read_parquet(SIGNALS / f"rank_{k}.parquet").loc[EVAL_START:] for k in SIGNAL_NAMES}

    fwd = {}
    for h in HORIZONS:
        path = PROCESSED / f"fwd_{h}.parquet"
        fwd[h] = pd.read_parquet(path) if path.exists() else fwd_from_returns(rets, h)
        fwd[h] = fwd[h].loc[EVAL_START:]

    # ------------------------------------------------------------ 1. rank IC and its decay
    print("1/4  rank IC ...")
    ic_rows, decay = {}, {}
    for k in SIGNAL_NAMES:
        decay[k] = {}
        for h in HORIZONS:
            ic = ev.rank_ic(ranks[k], fwd[h])
            decay[k][f"{h}d"] = ic.mean()
            if h == PRIMARY_H:
                ic.to_frame().to_parquet(OUT / f"ic_daily_{k}.parquet")
                ic_rows[k] = ev.ic_summary(ic, lags=h - 1)
    ic_table = pd.DataFrame(ic_rows).T
    decay_table = pd.DataFrame(decay).T

    # ------------------------------------------------------------ 2. quantile portfolios
    print("2/4  quintile portfolios ...")
    dates = ev.rebalance_dates(rets.loc[EVAL_START:].index)
    per = ev.period_returns(rets, dates)
    q_rows, spreads = {}, {}
    for k in SIGNAL_NAMES:
        b, turn = ev.quantile_returns(ranks[k], per)
        b.to_csv(OUT / f"quintiles_{k}.csv")
        q_rows[k] = ev.quantile_summary(b, turn)
        spreads[k] = (b[5] - b[1]).dropna()
    q_table = pd.DataFrame(q_rows).T

    # ------------------------------------------------------------ 3. Fama-MacBeth
    print("3/4  Fama-MacBeth ...")
    uni = pd.concat({k: ev.fm_summary(ev.fama_macbeth({k: ranks[k]}, per)).loc[k] for k in SIGNAL_NAMES}, axis=1).T
    multi_slopes = ev.fama_macbeth(ranks, per)
    multi_slopes.to_csv(OUT / "fm_multivariate_slopes.csv")
    multi = ev.fm_summary(multi_slopes)

    # ------------------------------------------------------------ 4. multiple testing
    print("4/4  multiple-testing correction ...")
    p = ic_table["ic_t_nw"].map(ev.p_value)
    summary = pd.DataFrame({
        "mean_ic_21d": ic_table["mean_ic"],
        "ic_t_nw": ic_table["ic_t_nw"],
        "icir": ic_table["icir"],
        "pct_days_ic_pos": ic_table["pct_positive"],
        "Q5_Q1_annual": q_table["spread_annual"],
        "Q5_Q1_t_nw": q_table["spread_t_nw"],
        "Q5_Q1_sharpe": q_table["spread_sharpe_annual"],
        "monotonicity": q_table["monotonicity"],
        "turnover_top": q_table["turnover_top"],
        "fm_t_uni": uni["fm_t_nw"],
        "fm_t_multi": multi["fm_t_nw"],
        "p_value": p,
        "p_holm": ev.holm(p),
        "p_bh": ev.benjamini_hochberg(p),
    })
    summary["passes_t3"] = summary["ic_t_nw"].abs() > 3

    # Deflated Sharpe of the best long-short quintile spread, with the 8 signals as the trials
    monthly_sr = {k: s.mean() / s.std() for k, s in spreads.items() if s.std() > 0}
    best = max(monthly_sr, key=monthly_sr.get)
    s = spreads[best]
    dsr = ev.deflated_sharpe(monthly_sr[best], len(s), s.skew(), s.kurt() + 3, list(monthly_sr.values()))

    # ------------------------------------------------------------ save and print
    summary.to_csv(OUT / "summary.csv")
    ic_table.to_csv(OUT / "ic_summary.csv")
    decay_table.to_csv(OUT / "ic_decay.csv")
    q_table.to_csv(OUT / "quintile_summary.csv")
    uni.to_csv(OUT / "fm_univariate.csv")
    multi.to_csv(OUT / "fm_multivariate.csv")

    pd.set_option("display.width", 200)
    print("\n=== Summary (21-day horizon, monthly portfolios) ===")
    show = summary.copy()
    for c in ["Q5_Q1_annual", "pct_days_ic_pos", "turnover_top"]:
        show[c] = (show[c] * 100).round(1)
    print(show.drop(columns=["p_value"]).round(3).to_string())
    print("\n=== Mean rank IC by horizon (IC decay) ===")
    print(decay_table.round(4).to_string())
    print(f"\nBest long-short spread: {best}, annual Sharpe {monthly_sr[best] * np.sqrt(12):.2f}, "
          f"deflated Sharpe probability {dsr:.2f} (8 trials)")
    print(f"\nEvaluated {len(per)} months, {EVAL_START} to {per.index[-1].date()}. Results saved to {OUT}/")


if __name__ == "__main__":
    main()