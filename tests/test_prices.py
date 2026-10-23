import pandas as pd

from alpha import prices


def test_download_prices_requests_batches_of_50(tmp_path, monkeypatch):
    requested_batches = []

    def fake_download(*, tickers, **kwargs):
        requested_batches.append(tickers)
        columns = pd.MultiIndex.from_product(
            [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        return pd.DataFrame(
            [[1.0, 2.0, 0.5, 1.5, 1.4, 100] * len(tickers)],
            index=pd.to_datetime(["2024-01-02"]),
            columns=columns,
        )

    monkeypatch.setattr(prices.yf, "download", fake_download)
    tickers = [f"TICKER{i}" for i in range(51)]

    failed = prices.download_prices(tickers, "2024-01-01", "2024-02-01", tmp_path)

    assert failed == []
    assert [len(batch) for batch in requested_batches] == [50, 1]
    assert len(list(tmp_path.glob("*.parquet"))) == 51
