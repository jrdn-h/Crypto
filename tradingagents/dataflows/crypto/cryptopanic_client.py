"""
CryptoPanic news client for crypto market news.

This client implements the NewsClient interface to fetch crypto news from CryptoPanic
using both RSS feeds (free) and REST API (free tier with optional auth token).
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import quote_plus

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..base_interfaces import AssetClass, DataQuality, NewsClient, NewsItem
from .caching import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


class CryptoPanicClient(NewsClient):
    """CryptoPanic news client for crypto market news."""
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize CryptoPanic client.
        
        Args:
            api_token: Optional CryptoPanic API token for higher rate limits
            cache_manager: Cache manager for response caching
            rate_limiter: Rate limiter for API calls
        """
        self.api_token = api_token
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # CryptoPanic rate limits: 100 req/day without token, 500 req/day with free token
        default_config = RateLimitConfig(requests_per_minute=15)
        self.rate_limiter = rate_limiter or RateLimiter(default_config)
        
        self.base_url = "https://cryptopanic.com/api/v1"
        self.rss_url = "https://cryptopanic.com/news/rss"
        
        # Common crypto symbols for relevance scoring
        self.crypto_symbols = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'bnb', 'cardano', 'ada',
            'solana', 'sol', 'xrp', 'ripple', 'dogecoin', 'doge', 'avalanche', 'avax',
            'polygon', 'matic', 'chainlink', 'link', 'polkadot', 'dot', 'tron', 'trx',
            'litecoin', 'ltc', 'shiba', 'shib', 'uniswap', 'uni', 'cosmos', 'atom',
            'filecoin', 'fil', 'aptos', 'apt', 'near', 'arbitrum', 'arb', 'optimism',
            'op', 'toncoin', 'ton', 'hedera', 'hbar', 'cronos', 'cro', 'qtum'
        }
    
    @property
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        return AssetClass.CRYPTO
    
    async def get_news(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        """
        Get crypto news from CryptoPanic.
        
        Args:
            symbol: Crypto symbol to filter news (e.g., 'BTC', 'ETH')
            start_date: Start date for news filtering
            end_date: End date for news filtering
            limit: Maximum number of news items to return
            
        Returns:
            List of NewsItem objects
        """
        cache_key = f"cryptopanic_news_{symbol}_{start_date}_{end_date}_{limit}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached CryptoPanic news for {symbol or 'global'}")
            return [NewsItem(**item) for item in cached_result]
        
        news_items = []
        
        try:
            # Use API if we have a token, otherwise use RSS
            if self.api_token:
                news_items = await self._fetch_api_news(symbol, start_date, end_date, limit)
            else:
                news_items = await self._fetch_rss_news(symbol, start_date, end_date, limit)
                
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
            # Fallback to RSS if API fails
            if self.api_token:
                try:
                    news_items = await self._fetch_rss_news(symbol, start_date, end_date, limit)
                except Exception as fallback_e:
                    logger.error(f"RSS fallback also failed: {fallback_e}")
        
        # Cache the results
        if news_items:
            cache_data = [item.dict() for item in news_items]
            await self.cache_manager.set(cache_key, cache_data, ttl=300)  # 5 minutes
        
        return news_items
    
    async def get_global_news(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        """Get global crypto market news."""
        return await self.get_news(None, start_date, end_date, limit)
    
    async def _fetch_api_news(
        self,
        symbol: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> List[NewsItem]:
        """Fetch news using CryptoPanic API."""
        await self.rate_limiter.acquire()
        
        # Build API parameters
        params = {
            "auth_token": self.api_token,
            "public": "true",
            "kind": "news",
            "limit": min(limit, 50)  # API max is 50 per request
        }
        
        if symbol:
            # Map common symbols to CryptoPanic currency codes
            currency_map = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
                'ADA': 'cardano', 'SOL': 'solana', 'XRP': 'ripple',
                'DOGE': 'dogecoin', 'AVAX': 'avalanche-2', 'MATIC': 'matic-network',
                'LINK': 'chainlink', 'DOT': 'polkadot', 'TRX': 'tron',
                'LTC': 'litecoin', 'SHIB': 'shiba-inu', 'UNI': 'uniswap'
            }
            params["currencies"] = currency_map.get(symbol.upper(), symbol.lower())
        
        news_items = []
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/posts"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        
                        for item in results:
                            news_item = await self._parse_api_news_item(item, symbol)
                            if news_item:
                                # Filter by date if specified
                                if start_date and news_item.published_at < start_date:
                                    continue
                                if end_date and news_item.published_at > end_date:
                                    continue
                                news_items.append(news_item)
                                
                    else:
                        logger.warning(f"CryptoPanic API error: {response.status}")
                        
            except Exception as e:
                logger.error(f"Error calling CryptoPanic API: {e}")
                raise
        
        return news_items[:limit]
    
    async def _fetch_rss_news(
        self,
        symbol: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> List[NewsItem]:
        """Fetch news using CryptoPanic RSS feed."""
        await self.rate_limiter.acquire()
        
        news_items = []
        
        try:
            # Fetch RSS feed
            async with aiohttp.ClientSession() as session:
                async with session.get(self.rss_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries:
                            news_item = await self._parse_rss_news_item(entry, symbol)
                            if news_item:
                                # Filter by date if specified
                                if start_date and news_item.published_at < start_date:
                                    continue
                                if end_date and news_item.published_at > end_date:
                                    continue
                                news_items.append(news_item)
                                
                    else:
                        logger.warning(f"CryptoPanic RSS error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic RSS: {e}")
            raise
        
        return news_items[:limit]
    
    async def _parse_api_news_item(self, item: dict, target_symbol: Optional[str]) -> Optional[NewsItem]:
        """Parse API response item into NewsItem."""
        try:
            published_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            
            # Extract symbols mentioned
            symbols_mentioned = []
            currencies = item.get("currencies", [])
            for currency in currencies:
                symbol = currency.get("code", "").upper()
                if symbol:
                    symbols_mentioned.append(symbol)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                item.get("title", ""), 
                item.get("summary", ""),
                target_symbol,
                symbols_mentioned
            )
            
            # Simple sentiment analysis based on keywords
            sentiment_score = self._analyze_sentiment(
                item.get("title", "") + " " + item.get("summary", "")
            )
            
            return NewsItem(
                title=item.get("title", ""),
                summary=item.get("summary", "")[:500],  # Limit summary length
                url=item.get("url"),
                published_at=published_at,
                source="CryptoPanic",
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                symbols_mentioned=symbols_mentioned,
                asset_class=AssetClass.CRYPTO,
                data_quality=DataQuality.HIGH
            )
            
        except Exception as e:
            logger.warning(f"Error parsing CryptoPanic API item: {e}")
            return None
    
    async def _parse_rss_news_item(self, entry, target_symbol: Optional[str]) -> Optional[NewsItem]:
        """Parse RSS feed entry into NewsItem."""
        try:
            # Parse published date
            published_at = datetime.fromisoformat(entry.published.replace("Z", "+00:00"))
            
            # Extract content
            title = entry.title
            summary = BeautifulSoup(entry.summary, 'html.parser').get_text()[:500]
            
            # Extract symbols from title and summary
            symbols_mentioned = self._extract_symbols_from_text(title + " " + summary)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                title, summary, target_symbol, symbols_mentioned
            )
            
            # Simple sentiment analysis
            sentiment_score = self._analyze_sentiment(title + " " + summary)
            
            return NewsItem(
                title=title,
                summary=summary,
                url=entry.link,
                published_at=published_at,
                source="CryptoPanic",
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                symbols_mentioned=symbols_mentioned,
                asset_class=AssetClass.CRYPTO,
                data_quality=DataQuality.MEDIUM
            )
            
        except Exception as e:
            logger.warning(f"Error parsing CryptoPanic RSS item: {e}")
            return None
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract crypto symbols from text."""
        symbols = []
        text_lower = text.lower()
        
        for symbol in self.crypto_symbols:
            if symbol in text_lower:
                # Convert common names to symbols
                symbol_map = {
                    'bitcoin': 'BTC', 'ethereum': 'ETH', 'binance': 'BNB',
                    'cardano': 'ADA', 'solana': 'SOL', 'ripple': 'XRP',
                    'dogecoin': 'DOGE', 'avalanche': 'AVAX', 'polygon': 'MATIC',
                    'chainlink': 'LINK', 'polkadot': 'DOT', 'tron': 'TRX',
                    'litecoin': 'LTC', 'shiba': 'SHIB', 'uniswap': 'UNI'
                }
                mapped_symbol = symbol_map.get(symbol, symbol.upper())
                if mapped_symbol not in symbols:
                    symbols.append(mapped_symbol)
        
        # Also extract $SYMBOL patterns
        symbol_pattern = r'\$([A-Z]{2,10})'
        matches = re.findall(symbol_pattern, text.upper())
        for match in matches:
            if match not in symbols:
                symbols.append(match)
        
        return symbols
    
    def _calculate_relevance_score(
        self, 
        title: str, 
        summary: str, 
        target_symbol: Optional[str],
        symbols_mentioned: List[str]
    ) -> float:
        """Calculate relevance score for news item."""
        if not target_symbol:
            return 0.8  # Default relevance for global news
        
        score = 0.0
        text = (title + " " + summary).lower()
        target_lower = target_symbol.lower()
        
        # Direct symbol mention
        if target_symbol in symbols_mentioned:
            score += 0.8
        elif target_lower in text:
            score += 0.6
        
        # Title vs summary weight
        if target_lower in title.lower():
            score += 0.3
        elif target_lower in summary.lower():
            score += 0.2
        
        # General crypto relevance
        crypto_keywords = ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'trading']
        for keyword in crypto_keywords:
            if keyword in text:
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keywords."""
        text_lower = text.lower()
        
        positive_keywords = [
            'bull', 'bullish', 'rise', 'rising', 'up', 'gain', 'gains', 'profit',
            'pump', 'moon', 'surge', 'rally', 'breakthrough', 'adoption', 'partnership',
            'launch', 'upgrade', 'positive', 'optimistic', 'buy', 'buying'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'fall', 'falling', 'down', 'loss', 'losses', 'crash',
            'dump', 'drop', 'decline', 'sell', 'selling', 'fear', 'panic', 'hack',
            'regulation', 'ban', 'warning', 'concern', 'negative', 'pessimistic'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0  # Neutral
        
        total_count = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total_count
        
        # Scale to -1 to 1 range
        return max(-1.0, min(1.0, sentiment)) 