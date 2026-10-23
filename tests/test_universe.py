import pandas as pd

from alpha.universe import load_membership


def test_load_membership_parses_tickers_and_sorts_dates(tmp_path):
    csv_path = tmp_path / "membership.csv"
    csv_path.write_text(
        'date,tickers\n'
        '2024-01-03,"MSFT, AAPL,MSFT"\n'
        '2024-01-02,\n'
    )

    membership = load_membership(str(csv_path))

    assert membership.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert membership.columns.tolist() == ["tickers"]
    assert membership.iloc[0]["tickers"] == set()
    assert membership.iloc[1]["tickers"] == {"AAPL", "MSFT"}
