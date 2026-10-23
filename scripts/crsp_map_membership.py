"""CRSP step A: map S&P 500 membership (tickers by date) to CRSP PERMNOs.

Run from the repo root:
    PYTHONPATH=src python scripts/crsp_map_membership.py YOUR_WRDS_USERNAME

Downloads the CRSP ticker history once (cached in data/raw/crsp/), then reports how well
the membership file maps. Nothing downstream changes yet.
"""
import sys
from pathlib import Path
import pandas as pd

from alpha.universe import load_membership
from alpha import crsp

RAW = Path("data/raw/crsp")
SEC_CACHE = RAW / "security_history.parquet"


def main(username: str):
    RAW.mkdir(parents=True, exist_ok=True)
    if SEC_CACHE.exists():
        sec = pd.read_parquet(SEC_CACHE)
    else:
        db = crsp.connect(username)
        sec = crsp.load_security_history(db)
        sec.to_parquet(SEC_CACHE)
        db.close()
    print(f"CRSP security history: {len(sec):,} rows, {sec['permno'].nunique():,} PERMNOs")

    m = load_membership("data/raw/sp500_membership.csv")
    spells = crsp.membership_spells(m)
    iv = crsp.map_all(spells, sec)
    cov = crsp.spell_coverage(spells, iv)

    iv.to_csv(RAW / "sp500_permno_intervals.csv", index=False)
    cov.to_csv(RAW / "sp500_spell_coverage.csv", index=False)

    # ---------------------------------------------------------------- report
    total_days = cov["days"].sum()
    print(f"\nIndex stays (ticker spells) within CRSP dates: {len(cov):,}  ({cov['ticker'].nunique():,} tickers)")
    print(f"Fully matched (>=99% of days):  {(cov['coverage'] >= 0.99).mean():.1%} of stays")
    print(f"Not matched at all:             {(cov['coverage'] == 0).mean():.1%} of stays")
    print(f"Member-days matched:            {cov['matched_days'].clip(upper=cov['days']).sum() / total_days:.1%}")
    print(f"Stays with 2+ candidate PERMNOs (best one picked): {(iv['n_candidates'] > 1).sum()}")

    print("\nKnown cases (ticker, stay start -> PERMNO, issuer):")
    lehman = sorted(t for t in cov["ticker"].unique() if t.startswith("LEH"))
    print(f"  Lehman appears in the membership file as: {lehman or 'nothing starting with LEH'}")
    for t in ["AAPL", "C", "CMCSA", "FOXA", "BRK.B", "GOOGL", "FB", "META", "CPWR", "FNMA", "MTLQQ"] + lehman:
        rows = cov[cov["ticker"] == t]
        for _, r in rows.iterrows():
            permnos = "-" if pd.isna(r["permnos"]) else r["permnos"]
            issuer = "NOT MATCHED" if pd.isna(r["issuer"]) else r["issuer"]
            print(f"  {t:6} {r['start'].date()}  -> {permnos:>12}  {issuer}")
        if rows.empty:
            print(f"  {t:6} not in membership file")

    worst = cov[cov["coverage"] < 0.99].sort_values("days", ascending=False).head(40)
    print("\n40 longest stays that are not fully matched:")
    print(worst[["ticker", "start", "end", "days", "coverage", "permnos", "issuer"]].to_string(index=False))

    via = iv[iv["lookup"] != iv["ticker"]].drop_duplicates(["ticker", "start"])
    print(f"\nStays matched through an alias ({len(via)}). Check each issuer is the right company:")
    print(via[["ticker", "lookup", "start", "end", "permno", "issuer"]].to_string(index=False))

    multi = iv[iv["n_candidates"] > 1].head(25)
    if len(multi):
        print("\nStays where several PERMNOs used the ticker; the one shown was picked (first 20):")
        print(f"(total: {(iv['n_candidates'] > 1).sum()}; full list in data/raw/crsp/sp500_permno_intervals.csv)")
        print(multi[["ticker", "start", "end", "permno", "issuer", "n_candidates"]].to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2 and not SEC_CACHE.exists():
        sys.exit("usage: PYTHONPATH=src python scripts/crsp_map_membership.py YOUR_WRDS_USERNAME")
    main(sys.argv[1] if len(sys.argv) > 1 else "")