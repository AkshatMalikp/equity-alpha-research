"""Step 4: compute the eight signals and their cross-sectional ranks.

Run from the repo root (after build_panel.py):
    PYTHONPATH=src python scripts/build_signals.py                         # Yahoo data
    DATA=data/processed_crsp PYTHONPATH=src python scripts/build_signals.py  # CRSP data
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from alpha.prices import load_prices
from alpha.panel import pivot_field
from alpha.signals import compute_all, SIGNAL_NAMES
from alpha.preprocess import cs_rank

START = "2005-01-01"
PROCESSED = Path(os.environ.get("DATA", "data/processed"))
OUT = PROCESSED / "signals"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Inputs from step 3
    adj = pd.read_parquet(PROCESSED / "adj_close.parquet")
    rets = pd.read_parquet(PROCESSED / "returns.parquet")
    mask = pd.read_parquet(PROCESSED / "mask.parquet")
    # Raw close and volume (for Amihud), same stocks and dates as adj
    if (PROCESSED / "close.parquet").exists():                 # CRSP build saves them directly
        close = pd.read_parquet(PROCESSED / "close.parquet").reindex_like(adj)
        volume = pd.read_parquet(PROCESSED / "volume.parquet").reindex_like(adj)
    else:                                                      # Yahoo: rebuild from raw files
        excluded = set(pd.read_csv(PROCESSED / "excluded_tickers.csv")["ticker"])
        prices = load_prices("data/raw/prices")
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices[(prices["date"] >= START) & (~prices["ticker"].isin(excluded))]
        close = pivot_field(prices, "close").reindex_like(adj)
        volume = pivot_field(prices, "volume").reindex_like(adj)

    member = mask.reindex(index=adj.index, columns=adj.columns, fill_value=False).astype(bool)

    # Compute, rank, save
    raw = compute_all(adj, rets, close, volume, member)
    ranks = {}
    for name in SIGNAL_NAMES:
        raw[name].to_parquet(OUT / f"raw_{name}.parquet")
        ranks[name] = cs_rank(raw[name], member)
        ranks[name].to_parquet(OUT / f"rank_{name}.parquet")
        print(f"saved {name}")

    # Check 1: coverage, share of members with a valid signal value
    print("\nCoverage (% of members with a value):")
    rows = []
    for d in ["1996-03-01", "2005-03-01", "2006-01-03", "2020-01-02"]:
        k = adj.index.searchsorted(pd.Timestamp(d))
        if k >= len(adj.index):
            continue
        d = adj.index[k]
        n = member.loc[d].sum()
        if n == 0:
            continue
        rows.append({"date": d.date(), **{k: round(100 * ranks[k].loc[d].notna().sum() / n, 1)
                                          for k in SIGNAL_NAMES}})
    print(pd.DataFrame(rows).set_index("date").T.to_string())

    # Check 2: momentum by hand
    t = "AAPL" if "AAPL" in adj.columns else "14593"            # 14593 = Apple's PERMNO
    d = pd.Timestamp("2015-06-01")
    i = adj.index.get_loc(d)
    by_hand = adj[t].iloc[i - 21] / adj[t].iloc[i - 252] - 1
    print(f"\nAAPL momentum on {d.date()}: by hand {by_hand:.6f}  function {raw['momentum'].loc[d, t]:.6f}")

    # Check 3: average cross-sectional rank correlation, on month-end dates
    month_ends = adj.groupby(adj.index.to_period("M")).tail(1).index
    corrs = []
    for d in month_ends:
        frame = pd.DataFrame({k: ranks[k].loc[d] for k in SIGNAL_NAMES}).dropna()
        if len(frame) > 50:
            corrs.append(frame.corr().to_numpy())
    avg = pd.DataFrame(np.mean(corrs, axis=0), index=SIGNAL_NAMES, columns=SIGNAL_NAMES)
    print("\nAverage cross-sectional rank correlation (month-ends):")
    print(avg.round(2).to_string())


if __name__ == "__main__":
    main()