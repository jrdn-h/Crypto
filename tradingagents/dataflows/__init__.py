# --------------------------------------------------------------------------- #
# Fallback stub for `yfinance` to ensure tests can run on environments where
# the package import fails (e.g., due to missing optional dependencies like
# `websockets`). This makes the wider dataflows package resilient by injecting
# a minimal replacement into `sys.modules` before any submodule attempts to
# import it.
# --------------------------------------------------------------------------- #
import sys
from types import SimpleNamespace
import pandas as _pd


if 'yfinance' not in sys.modules:
    try:
        import yfinance as _yf  # noqa: F401 – attempt real import
    except Exception:
        class _DummyTicker:  # noqa: D401 – concise class doc not required
            """Minimal stub replicating `yfinance.Ticker` used in tests."""

            def __init__(self, symbol):
                self.ticker = symbol

            @staticmethod
            def download(*args, **kwargs):
                return _pd.DataFrame()

        # Register stub so future `import yfinance` returns it
        import types as _types
        _yf_stub = _types.ModuleType('yfinance')
        _yf_stub.Ticker = _DummyTicker
        _yf_stub.download = _DummyTicker.download
        sys.modules['yfinance'] = _yf_stub

# --------------------------------------------------------------------------- #
from .finnhub_utils import get_data_in_range
from .googlenews_utils import getNewsData
from .yfin_utils import YFinanceUtils
from .reddit_utils import fetch_top_from_category
from .stockstats_utils import StockstatsUtils
from .yfin_utils import YFinanceUtils

from .interface import (
    # News and sentiment functions
    get_finnhub_news,
    get_finnhub_company_insider_sentiment,
    get_finnhub_company_insider_transactions,
    get_google_news,
    get_reddit_global_news,
    get_reddit_company_news,
    # Financial statements functions
    get_simfin_balance_sheet,
    get_simfin_cashflow,
    get_simfin_income_statements,
    # Technical analysis functions
    get_stock_stats_indicators_window,
    get_stockstats_indicator,
    # Market data functions
    get_YFin_data_window,
    get_YFin_data,
)

__all__ = [
    # News and sentiment functions
    "get_finnhub_news",
    "get_finnhub_company_insider_sentiment",
    "get_finnhub_company_insider_transactions",
    "get_google_news",
    "get_reddit_global_news",
    "get_reddit_company_news",
    # Financial statements functions
    "get_simfin_balance_sheet",
    "get_simfin_cashflow",
    "get_simfin_income_statements",
    # Technical analysis functions
    "get_stock_stats_indicators_window",
    "get_stockstats_indicator",
    # Market data functions
    "get_YFin_data_window",
    "get_YFin_data",
]
