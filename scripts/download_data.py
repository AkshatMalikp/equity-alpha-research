import pandas as pd
from pathlib import Path
from alpha.universe import load_membership
from alpha.prices import tickers_since, download_prices

START = "2005-01-01"
MEMBERSHIP_PATH = "data/raw/sp500_membership.csv"
PRICES_DIR = "data/raw/prices"


def main():
    m = load_membership(MEMBERSHIP_PATH)
    tickers = tickers_since(m, START)
    print("total tickers:", len(tickers))

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    failed = download_prices(tickers, START, end, PRICES_DIR)

    saved = len(list(Path(PRICES_DIR).glob("*.parquet")))
    print("saved:", saved, "failed:", len(failed))
    print("sample failures:", failed[:10])


if __name__ == "__main__":
    main()