"""CRSP step B: download daily data for every mapped S&P 500 PERMNO and build the panel.

Run from the repo root (after crsp_map_membership.py):
    PYTHONPATH=src python scripts/crsp_build_panel.py YOUR_WRDS_USERNAME

Writes the same files the Yahoo pipeline wrote, keyed by PERMNO instead of ticker, to
data/processed_crsp/ so steps 4 and 5 can run on it unchanged:
    adj_close, returns, close, volume, mask, fwd_1, fwd_5, fwd_21  (+ spread, cap for step 7)
CRSP data is licensed: everything here stays under data/ (git-ignored). Never commit it.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from alpha import crsp

RAW = Path("data/raw/crsp")
DAILY = RAW / "daily"
OUT = Path("data/processed_crsp")
PRICE_START = "1995-01-01"      # one year before membership starts, for the 12-month signals
PRICE_END = "2025-12-31"        # last date in your CRSP daily file


def fwd(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Compounded return over days t+1..t+h (same definition as panel.forward_returns)."""
    return np.expm1(np.log1p(returns).rolling(h).sum().shift(-h))


def main(username: str):
    OUT.mkdir(parents=True, exist_ok=True)
    iv = pd.read_csv(RAW / "sp500_permno_intervals.csv", parse_dates=["start", "end", "from", "to"])
    permnos = sorted(iv["permno"].astype(int).unique())
    print(f"{len(permnos)} PERMNOs to download")

    # ------------------------------------------------------------ download (skips finished batches)
    n_batches = -(-len(permnos) // 100)
    if len(list(DAILY.glob("daily_*.parquet"))) < n_batches:
        db = crsp.connect(username)
        crsp.download_daily(db, permnos, PRICE_START, PRICE_END, DAILY)
        db.close()

    # ------------------------------------------------------------ build grids
    daily = crsp.load_daily(DAILY)
    print(f"daily rows: {len(daily):,}")
    rets = crsp.grid(daily, "dlyret")
    close = crsp.grid(daily, "dlyprc").abs()                   # raw price that day
    volume = crsp.grid(daily, "dlyvol")
    bid, ask = crsp.grid(daily, "dlybid"), crsp.grid(daily, "dlyask")
    mid = (bid + ask) / 2
    spread = ((ask - bid) / mid).where((bid > 0) & (ask > bid))
    cap = crsp.grid(daily, "dlycap")
    adj = crsp.total_return_index(rets)
    mask = crsp.intervals_mask(iv, rets.index, rets.columns)

    files = {"adj_close": adj, "returns": rets, "close": close, "volume": volume,
             "mask": mask, "spread": spread, "cap": cap}
    for h in (1, 5, 21):
        files[f"fwd_{h}"] = fwd(rets, h)
    for name, df in files.items():
        df.to_parquet(OUT / f"{name}.parquet")
    print(f"saved {len(files)} grids to {OUT}/  shape {rets.shape}")

    # ------------------------------------------------------------ checks
    member = mask.reindex_like(rets).fillna(False).astype(bool)
    yr = member.index.year
    cov = pd.DataFrame({
        "member_days": member.sum(axis=1).groupby(yr).sum(),
        "with_return": (member & rets.notna()).sum(axis=1).groupby(yr).sum(),
    })
    cov["coverage_pct"] = 100 * cov["with_return"] / cov["member_days"]
    cov = cov[cov["member_days"] > 0]
    print("\nCoverage of index member-days by year (compare: Yahoo was 53% in 2005):")
    print(cov.round(1).to_string())
    print(f"overall: {100 * cov['with_return'].sum() / cov['member_days'].sum():.1f}%")
    print(f"members per day: min {member.sum(axis=1)[member.sum(axis=1) > 0].min()}, "
          f"median {member.sum(axis=1)[member.sum(axis=1) > 0].median():.0f}")

    big = rets.where(member).stack()
    print("\n10 largest absolute member-day returns (should be real events):")
    names = iv.drop_duplicates("permno").set_index(iv.drop_duplicates("permno")["permno"].astype(str))["issuer"]
    top = big.abs().nlargest(10)
    for (d, p), v in top.items():
        print(f"  {d.date()}  {p:>6}  {big[(d, p)]:+.1%}  {names.get(p, '')}")

    delist = daily[daily["dlydelflg"] == "Y"]
    in_index = [member.at[d, p] if (d in member.index and p in member.columns) else False
                for d, p in zip(delist["dlycaldt"], delist["permno"])]
    print(f"\nDelisting days while in the index: {sum(in_index)}  (their final returns are now included)")

    a = "14593"     # Apple
    if a in rets.columns:
        d = pd.Timestamp("2015-06-01")
        nxt = rets.loc[rets.index > d, a].iloc[:5]
        print(f"\nAAPL fwd_5 check on {d.date()}: by hand {(1 + nxt).prod() - 1:.6f}  "
              f"file {files['fwd_5'].loc[d, a]:.6f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")