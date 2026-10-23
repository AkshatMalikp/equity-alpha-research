import pandas as pd

def load_membership(path: str) -> pd.DataFrame:
    """
    Read the membership CSV.
    Returns a DataFrame indexed by date (sorted ascending), one column
    'tickers' holding a set of ticker strings per row.
    """
    
    membership = pd.read_csv(path, parse_dates=["date"])
    assert membership["date"].is_unique
    membership["tickers"] = membership["tickers"].map(
        lambda tickers: (
            set()
            if pd.isna(tickers)
            else {
                ticker.strip()
                for ticker in tickers.split(",")
                if ticker.strip()
            }
        )
    )
    return membership.set_index("date").sort_index()

def members_on(membership: pd.DataFrame, date: str) -> set[str]:

    target_date=pd.Timestamp(date)
    available_dates = membership.index[membership.index <= target_date]

    return set(membership.loc[available_dates[-1], "tickers"])
