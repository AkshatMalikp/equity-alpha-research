from pathlib import Path
import pandas as pd
from alpha.universe import load_membership
from alpha.prices import tickers_since, load_prices
from alpha.panel import (pivot_field, daily_returns, forward_returns,
                         membership_mask, coverage_by_year,find_reused_tickers)

START = "2005-01-01"
OUT = Path("data/processed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_membership("data/raw/sp500_membership.csv")

    prices = load_prices("data/raw/prices")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["date"] >= START]

    adj = pivot_field(prices, "adj_close")
    universe = tickers_since(m, START)
    mask = membership_mask(m, adj.index, universe)

    # Fix 1 + 2: detect reuse, exclude certain cases, print suspicious ones for review
    flags = find_reused_tickers(adj, mask)
    print(flags["reason"].value_counts())
    print(flags[flags["reason"] == "starts_mid_membership"].head(10).to_string(index=False))
    certain = flags[flags["reason"] == "starts_after_membership"][["ticker", "reason"]]
    mid = flags[flags["reason"] == "starts_mid_membership"]
    mid_bad = [tk for tk, fp in zip(mid["ticker"], pd.to_datetime(mid["first_price"]))
               if mask.loc[mask.index >= fp, tk].iloc[0]]
    print("mid-membership excluded:", mid_bad)
    mid_bad_df = pd.DataFrame({"ticker": mid_bad, "reason": "starts_mid_membership_overlap"})
    manual = pd.DataFrame({"ticker": ["CPWR", "EP","PARA"], "reason": ["manual_bad_data"] * 3})
    excluded = pd.concat([certain, manual,mid_bad_df], ignore_index=True)
    excluded.to_csv(OUT / "excluded_tickers.csv", index=False)
    adj = adj.drop(columns=excluded["ticker"], errors="ignore")

    rets = daily_returns(adj)
    fwd = {h: forward_returns(rets, h) for h in (1, 5, 21)}

    adj.to_parquet(OUT / "adj_close.parquet")
    rets.to_parquet(OUT / "returns.parquet")
    for h, df in fwd.items():
        df.to_parquet(OUT / f"fwd_{h}.parquet")
    mask.to_parquet(OUT / "mask.parquet")

    cov = coverage_by_year(mask, adj)
    cov.to_csv(OUT / "coverage_by_year.csv")
    print(cov.round(1))
    print("overall coverage %:", round(100 * cov.member_days_with_price.sum() / cov.member_days.sum(), 1))

    # Fix 3: extreme returns on member-days only
    member = mask.reindex(index=rets.index, columns=rets.columns, fill_value=False)
    print(rets.where(member).stack().abs().nlargest(10))

    # fwd_5 spot check (unchanged)
    t, d = "AAPL", pd.Timestamp("2015-06-01")
    nxt = rets.loc[rets.index > d, t].iloc[:5]
    print("by hand:", round((1 + nxt).prod() - 1, 6), " fwd_5:", round(fwd[5].loc[d, t], 6))


if __name__ == "__main__":
    main()