"""
CoinDesk RSS news client for crypto market news.

This client implements the NewsClient interface to fetch crypto news from CoinDesk
RSS feeds, providing high-quality institutional crypto news coverage.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urljoin

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..base_interfaces import AssetClass, DataQuality, NewsClient, NewsItem
from .caching import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


class CoinDeskClient(NewsClient):
    """CoinDesk RSS news client for crypto market news."""
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize CoinDesk client.
        
        Args:
            cache_manager: Cache manager for response caching
            rate_limiter: Rate limiter for RSS feed requests
        """
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # CoinDesk RSS is public - conservative rate limiting to be respectful
        default_config = RateLimitConfig(requests_per_minute=30)
        self.rate_limiter = rate_limiter or RateLimiter(default_config)
        
        # CoinDesk RSS feed URLs
        self.feed_urls = {
            'all': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'markets': 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tags=markets',
            'tech': 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tags=tech',
            'policy': 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tags=policy',
            'business': 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tags=business'
        }
        
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
        Get crypto news from CoinDesk RSS feeds.
        
        Args:
            symbol: Crypto symbol to filter news (e.g., 'BTC', 'ETH')
            start_date: Start date for news filtering
            end_date: End date for news filtering
            limit: Maximum number of news items to return
            
        Returns:
            List of NewsItem objects
        """
        cache_key = f"coindesk_news_{symbol}_{start_date}_{end_date}_{limit}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached CoinDesk news for {symbol or 'global'}")
            return [NewsItem(**item) for item in cached_result]
        
        news_items = []
        
        try:
            # Fetch from multiple feeds for comprehensive coverage
            for feed_name, feed_url in self.feed_urls.items():
                feed_items = await self._fetch_feed_news(
                    feed_url, feed_name, symbol, start_date, end_date
                )
                news_items.extend(feed_items)
            
            # Deduplicate by URL and sort by date
            seen_urls = set()
            unique_items = []
            for item in news_items:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    unique_items.append(item)
            
            # Sort by publish date (newest first)
            unique_items.sort(key=lambda x: x.published_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Error fetching CoinDesk news: {e}")
            return []
        
        # Apply limit
        result_items = unique_items[:limit]
        
        # Cache the results
        if result_items:
            cache_data = [item.dict() for item in result_items]
            await self.cache_manager.set(cache_key, cache_data, ttl=600)  # 10 minutes
        
        return result_items
    
    async def get_global_news(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        """Get global crypto market news."""
        return await self.get_news(None, start_date, end_date, limit)
    
    async def _fetch_feed_news(
        self,
        feed_url: str,
        feed_name: str,
        symbol: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[NewsItem]:
        """Fetch news from a specific RSS feed."""
        await self.rate_limiter.acquire()
        
        news_items = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries:
                            news_item = await self._parse_rss_entry(entry, feed_name, symbol)
                            if news_item:
                                # Filter by date if specified
                                if start_date and news_item.published_at < start_date:
                                    continue
                                if end_date and news_item.published_at > end_date:
                                    continue
                                news_items.append(news_item)
                                
                    else:
                        logger.warning(f"CoinDesk RSS error for {feed_name}: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching CoinDesk RSS {feed_name}: {e}")
        
        return news_items
    
    async def _parse_rss_entry(
        self, 
        entry, 
        feed_name: str, 
        target_symbol: Optional[str]
    ) -> Optional[NewsItem]:
        """Parse RSS feed entry into NewsItem."""
        try:
            # Parse published date
            published_at = self._parse_date(entry.published)
            if not published_at:
                return None
            
            # Extract content
            title = entry.title
            summary = self._extract_summary(entry)
            
            # Extract symbols from title and summary
            symbols_mentioned = self._extract_symbols_from_text(title + " " + summary)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                title, summary, target_symbol, symbols_mentioned, feed_name
            )
            
            # Simple sentiment analysis
            sentiment_score = self._analyze_sentiment(title + " " + summary)
            
            return NewsItem(
                title=title,
                summary=summary[:500],  # Limit summary length
                url=entry.link,
                published_at=published_at,
                source=f"CoinDesk-{feed_name}",
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                symbols_mentioned=symbols_mentioned,
                asset_class=AssetClass.CRYPTO,
                data_quality=DataQuality.HIGH  # CoinDesk is high-quality institutional source
            )
            
        except Exception as e:
            logger.warning(f"Error parsing CoinDesk RSS entry: {e}")
            return None
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds."""
        try:
            # Try different date formats commonly used in RSS
            formats = [
                "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",       # ISO format
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
            
            # Fallback to feedparser's date parsing
            import time
            time_tuple = feedparser._parse_date(date_string)
            if time_tuple:
                return datetime(*time_tuple[:6])
                
        except Exception as e:
            logger.debug(f"Error parsing date '{date_string}': {e}")
        
        return None
    
    def _extract_summary(self, entry) -> str:
        """Extract clean summary from RSS entry."""
        summary = ""
        
        # Try different fields for content
        if hasattr(entry, 'summary'):
            summary = entry.summary
        elif hasattr(entry, 'description'):
            summary = entry.description
        elif hasattr(entry, 'content'):
            if isinstance(entry.content, list) and entry.content:
                summary = entry.content[0].get('value', '')
            else:
                summary = str(entry.content)
        
        # Clean HTML tags
        if summary:
            soup = BeautifulSoup(summary, 'html.parser')
            summary = soup.get_text().strip()
        
        return summary
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract crypto symbols from text."""
        symbols = []
        text_lower = text.lower()
        
        # Check for crypto names and symbols
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
        
        # Extract $SYMBOL patterns
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
        symbols_mentioned: List[str],
        feed_name: str
    ) -> float:
        """Calculate relevance score for news item."""
        if not target_symbol:
            # For global news, prioritize market/business feeds
            if feed_name in ['markets', 'business']:
                return 0.9
            elif feed_name in ['policy', 'tech']:
                return 0.7
            else:
                return 0.8
        
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
        
        # Feed category bonus
        if feed_name == 'markets':
            score += 0.2  # Markets feed is highly relevant for trading
        elif feed_name in ['business', 'tech']:
            score += 0.1
        
        # General crypto relevance
        crypto_keywords = ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'trading', 'price']
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
            'surge', 'rally', 'breakthrough', 'adoption', 'partnership', 'launch',
            'upgrade', 'positive', 'optimistic', 'buy', 'buying', 'growth', 'soar',
            'milestone', 'record', 'institutional', 'mainstream', 'breakthrough'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'fall', 'falling', 'down', 'loss', 'losses', 'crash',
            'drop', 'decline', 'sell', 'selling', 'fear', 'panic', 'hack', 'security',
            'regulation', 'ban', 'warning', 'concern', 'negative', 'pessimistic',
            'volatility', 'risk', 'uncertainty', 'bubble', 'correction'
        ]
        
        # Weight words in title higher
        title_words = text_lower.split('.')[0] if '.' in text_lower else text_lower[:100]
        
        positive_count = 0
        negative_count = 0
        
        for word in positive_keywords:
            if word in text_lower:
                positive_count += 2 if word in title_words else 1
        
        for word in negative_keywords:
            if word in text_lower:
                negative_count += 2 if word in title_words else 1
        
        if positive_count == 0 and negative_count == 0:
            return 0.0  # Neutral
        
        total_count = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total_count
        
        # Scale to -1 to 1 range with some dampening for institutional news
        return max(-1.0, min(1.0, sentiment * 0.8))  # Slightly more conservative 