import pandas as pd

# Attempt to import yfinance. Provide a lightweight stub if import fails due to missing
# dependencies or platform incompatibilities (e.g., `websockets.sync`).
try:
    import yfinance as yf  # noqa: F401  # imported for its Ticker/download utilities
except Exception:  # pragma: no cover – broad but acceptable for graceful degradation
    from types import SimpleNamespace

    import pandas as pd  # already imported but re-import for stub context

    class _DummyTicker:
        """Minimal stub replicating the subset of yfinance.Ticker interface needed here."""

        def __init__(self, symbol):
            self.ticker = symbol

        def history(self, *args, **kwargs):  # noqa: D401
            """Return an empty DataFrame for price history."""
            return pd.DataFrame()

        @staticmethod
        def download(*args, **kwargs):  # noqa: D401
            """Return an empty DataFrame as placeholder market data."""
            return pd.DataFrame()

    # Expose stub under same attribute names as yfinance
    yf = SimpleNamespace(Ticker=_DummyTicker, download=_DummyTicker.download)
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored.",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
        ] = False,
    ):
        df = None
        data = None

        if not online:
            try:
                data = pd.read_csv(
                    os.path.join(
                        data_dir,
                        f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                    )
                )
                df = wrap(data)
            except FileNotFoundError:
                raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
        else:
            # Get today's date as YYYY-mm-dd to add to cache
            today_date = pd.Timestamp.today()
            curr_date = pd.to_datetime(curr_date)

            end_date = today_date
            start_date = today_date - pd.DateOffset(years=15)
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")

            # Get config and ensure cache directory exists
            config = get_config()
            os.makedirs(config["data_cache_dir"], exist_ok=True)

            data_file = os.path.join(
                config["data_cache_dir"],
                f"{symbol}-YFin-data-{start_date}-{end_date}.csv",
            )

            if os.path.exists(data_file):
                data = pd.read_csv(data_file)
                data["Date"] = pd.to_datetime(data["Date"])
            else:
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                )
                data = data.reset_index()
                data.to_csv(data_file, index=False)

            df = wrap(data)

            # -----------------------------------------------------------------
            # Ensure a canonical "Date" column exists regardless of stockstats
            # internal name-mangling (it converts all columns to lowercase).
            # -----------------------------------------------------------------

            if "Date" not in df.columns:
                # stockstats typically lowercases column names; fall back to
                # that variant or derive from the index.
                if "date" in df.columns:
                    df["Date"] = df["date"]
                else:
                    # Fallback: use the original DataFrame's index if it looks
                    # like a datetime index.
                    try:
                        df["Date"] = pd.to_datetime(data.index)
                    except Exception:
                        # As a last resort, create a sequential placeholder.
                        df["Date"] = pd.date_range(start=start_date, periods=len(df))

            # Normalise date to string YYYY-MM-DD for downstream comparisons.
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            curr_date = curr_date.strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
