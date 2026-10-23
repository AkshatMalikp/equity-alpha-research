import pandas as pd
import pytest

from alpha.panel import (
    coverage_by_year,
    daily_returns,
    forward_returns,
    membership_mask,
    pivot_field,
)


def test_pivot_field_creates_date_by_ticker_grid():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            "ticker": ["BBB", "AAA"],
            "close": [20.0, 10.0],
        }
    )

    result = pivot_field(prices, "close")

    assert result.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert result.loc["2024-01-02", "AAA"] == 10.0


def test_daily_returns_preserves_missing_prices_as_nan():
    prices = pd.DataFrame({"AAA": [10.0, None, 12.0]})

    result = daily_returns(prices)

    assert pd.isna(result.iloc[1, 0])
    assert pd.isna(result.iloc[2, 0])


def test_forward_returns_compounds_only_future_returns():
    returns = pd.DataFrame({"AAA": [0.10, 0.20, 0.30, 0.40]})

    result = forward_returns(returns, 2)

    assert result.iloc[0, 0] == pytest.approx((1.20 * 1.30) - 1)
    assert result.iloc[1, 0] == pytest.approx((1.30 * 1.40) - 1)
    assert result.iloc[-2:, 0].isna().all()


def test_membership_mask_uses_membership_in_force_on_each_date():
    membership = pd.DataFrame(
        {"tickers": [{"AAA"}, {"BBB"}]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    result = membership_mask(
        membership,
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ["AAA", "BBB"],
    )
    assert result.loc["2024-01-02"].tolist() == [True, False]
    assert result.loc["2024-01-03"].tolist() == [True, False]   # still the 01-02 row
    assert result.loc["2024-01-04"].tolist() == [False, True]   # change counts on its own day


def test_membership_mask_rejects_dates_before_data():
    membership = pd.DataFrame(
        {"tickers": [{"AAA"}]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    with pytest.raises(ValueError):
        membership_mask(membership, pd.to_datetime(["2024-01-01"]), ["AAA"])

def test_coverage_by_year_counts_only_member_days_with_prices():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    mask = pd.DataFrame({"AAA": [True, True], "BBB": [False, True]}, index=dates)
    prices = pd.DataFrame({"AAA": [10.0, None], "BBB": [None, 20.0]}, index=dates)

    result = coverage_by_year(mask, prices)

    assert result.loc[2024, "member_days"] == 3
    assert result.loc[2024, "member_days_with_price"] == 2
    assert result.loc[2024, "coverage_pct"] == pytest.approx(200 / 3)
