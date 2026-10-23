import pandas as pd
from pathlib import Path
import yfinance as yf
import warnings

from alpha.universe import members_on


def tickers_since(membership, start) -> list[str]:

    start_date = pd.Timestamp(start)
    tickers = set(members_on(membership, start_date))
    later_rows = membership.loc[membership.index > start_date, "tickers"]
    for row in later_rows:
        tickers.update(row)

    return sorted(tickers)    
    """Sorted list of every ticker that was a member at any point on or after `start`.
    Careful: also include tickers in the row *in force* on `start`, not just rows after it."""

def to_yahoo(ticker: str) -> str:
    return ticker.replace(".", "-")
    """Convert to Yahoo's format. Share classes use '-' not '.': BRK.B -> BRK-B."""

def download_prices(tickers, start, end, out_dir) -> list[str]:
    """Download daily OHLCV in batches of 50 and save one Parquet file per ticker.

    Existing files are skipped. Return tickers with empty results or errors.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        ticker
        for ticker in tickers
        if not (output_dir / f"{ticker}.parquet").exists()
    ]
    failed = []

    for offset in range(0, len(pending), 50):
        batch = pending[offset:offset + 50]
        yahoo_to_original = {to_yahoo(ticker): ticker for ticker in batch}
        try:
            batch_prices = yf.download(
                tickers=list(yahoo_to_original),
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
            )
        except Exception as error:
            failed.extend(batch)
            warnings.warn(f"Could not download batch {batch}: {error}")
            continue

        for yahoo_ticker, ticker in yahoo_to_original.items():
            try:
                if isinstance(batch_prices.columns, pd.MultiIndex):
                    ticker_level = next(
                        (
                            level
                            for level in range(batch_prices.columns.nlevels)
                            if yahoo_ticker
                            in batch_prices.columns.get_level_values(level)
                        ),
                        None,
                    )
                    if ticker_level is None:
                        raise ValueError("Ticker missing from batch results")
                    prices = batch_prices.xs(
                        yahoo_ticker, axis=1, level=ticker_level
                    ).copy()
                elif len(batch) == 1:
                    prices = batch_prices.copy()
                else:
                    raise ValueError("Expected ticker-grouped columns for batch")

                prices = prices.dropna(how="all")
                if prices.empty:
                    raise ValueError("No price data returned")

                prices = prices.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                    }
                )
                price_columns = [
                    "open", "high", "low", "close", "adj_close", "volume"
                ]
                missing_columns = set(price_columns) - set(prices.columns)
                if missing_columns:
                    raise ValueError(
                        f"Missing price columns: {sorted(missing_columns)}"
                    )

                prices.index.name = "date"
                prices = prices[price_columns]
                prices["ticker"] = ticker
                prices.reset_index().to_parquet(
                    output_dir / f"{ticker}.parquet",
                    index=False,
                )
            except Exception as error:
                failed.append(ticker)
                warnings.warn(f"Could not save prices for {ticker}: {error}")

    return failed

def load_prices(out_dir) -> pd.DataFrame:
    """Load all saved ticker Parquet files into one DataFrame."""
    columns = [
        "date", "ticker", "open", "high", "low",
        "close", "adj_close", "volume",
    ]
    files = sorted(Path(out_dir).glob("*.parquet"))

    if not files:
        return pd.DataFrame(columns=columns)

    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)[columns]