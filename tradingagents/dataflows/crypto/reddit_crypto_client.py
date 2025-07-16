"""
Reddit r/CryptoCurrency sentiment client for crypto social sentiment analysis.

This client implements the SocialSentimentClient interface to fetch sentiment data
from Reddit's r/CryptoCurrency and related crypto subreddits.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

from ..base_interfaces import AssetClass, DataQuality, SocialSentimentClient, SentimentData
from .caching import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


class RedditCryptoClient(SocialSentimentClient):
    """Reddit crypto sentiment client for social sentiment analysis."""
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize Reddit crypto client.
        
        Args:
            cache_manager: Cache manager for response caching
            rate_limiter: Rate limiter for Reddit API calls
        """
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # Reddit JSON endpoints have lenient rate limits for read-only access
        default_config = RateLimitConfig(requests_per_minute=60)
        self.rate_limiter = rate_limiter or RateLimiter(default_config)
        
        # Crypto-related subreddits to monitor
        self.crypto_subreddits = [
            'CryptoCurrency',
            'Bitcoin',
            'ethereum', 
            'cardano',
            'solana',
            'dogecoin',
            'CryptoMarkets',
            'altcoin',
            'binance',
            'defi'
        ]
        
        # Symbol to subreddit mapping for targeted searches
        self.symbol_subreddit_map = {
            'BTC': ['Bitcoin', 'btc'],
            'ETH': ['ethereum', 'ethfinance'],
            'ADA': ['cardano'],
            'SOL': ['solana'],
            'DOGE': ['dogecoin'],
            'MATIC': ['0xpolygon', 'polygonnetwork'],
            'LINK': ['Chainlink'],
            'DOT': ['dot'],
            'UNI': ['Uniswap'],
            'AVAX': ['Avax'],
            'ATOM': ['cosmosnetwork']
        }
        
        # Common crypto symbols for mention detection
        self.crypto_symbols = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'bnb', 'cardano', 'ada',
            'solana', 'sol', 'xrp', 'ripple', 'dogecoin', 'doge', 'avalanche', 'avax',
            'polygon', 'matic', 'chainlink', 'link', 'polkadot', 'dot', 'tron', 'trx',
            'litecoin', 'ltc', 'shiba', 'shib', 'uniswap', 'uni', 'cosmos', 'atom'
        }
    
    @property
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        return AssetClass.CRYPTO
    
    async def get_sentiment(self, symbol: str, as_of_date: datetime) -> Optional[SentimentData]:
        """
        Get sentiment data for a crypto symbol from Reddit.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            as_of_date: Date for sentiment analysis
            
        Returns:
            SentimentData object or None if no data available
        """
        cache_key = f"reddit_sentiment_{symbol}_{as_of_date.date()}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached Reddit sentiment for {symbol}")
            return SentimentData(**cached_result)
        
        try:
            # Get posts mentioning the symbol
            posts = await self._fetch_symbol_posts(symbol, as_of_date)
            
            if not posts:
                logger.debug(f"No Reddit posts found for {symbol}")
                return None
            
            # Analyze sentiment from posts
            sentiment_data = self._analyze_posts_sentiment(posts, symbol, as_of_date)
            
            # Cache the result
            if sentiment_data:
                await self.cache_manager.set(cache_key, sentiment_data.dict(), ttl=1800)  # 30 minutes
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            return None
    
    async def get_trending_symbols(self, limit: int = 10) -> List[str]:
        """
        Get trending crypto symbols from Reddit discussions.
        
        Args:
            limit: Maximum number of trending symbols to return
            
        Returns:
            List of trending crypto symbols
        """
        cache_key = f"reddit_trending_{limit}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug("Using cached Reddit trending symbols")
            return cached_result
        
        try:
            # Fetch hot posts from main crypto subreddits
            symbol_mentions = {}
            
            for subreddit in ['CryptoCurrency', 'CryptoMarkets']:
                posts = await self._fetch_subreddit_posts(subreddit, 'hot', 25)
                
                for post in posts:
                    symbols = self._extract_symbols_from_text(post.get('title', '') + ' ' + post.get('selftext', ''))
                    for symbol in symbols:
                        if symbol not in symbol_mentions:
                            symbol_mentions[symbol] = 0
                        # Weight by upvotes and comments
                        weight = (post.get('ups', 0) + post.get('num_comments', 0)) / 2
                        symbol_mentions[symbol] += weight
            
            # Sort by mention weight and return top symbols
            trending = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
            trending_symbols = [symbol for symbol, _ in trending[:limit]]
            
            # Cache the result
            await self.cache_manager.set(cache_key, trending_symbols, ttl=900)  # 15 minutes
            
            return trending_symbols
            
        except Exception as e:
            logger.error(f"Error fetching trending symbols from Reddit: {e}")
            return []
    
    async def _fetch_symbol_posts(self, symbol: str, as_of_date: datetime) -> List[Dict]:
        """Fetch Reddit posts mentioning a specific symbol."""
        posts = []
        
        # Search in relevant subreddits
        subreddits = self.symbol_subreddit_map.get(symbol.upper(), ['CryptoCurrency'])
        
        for subreddit in subreddits:
            # Fetch recent posts
            subreddit_posts = await self._fetch_subreddit_posts(subreddit, 'new', 50)
            
            # Filter posts by date and symbol mentions
            for post in subreddit_posts:
                post_date = datetime.fromtimestamp(post.get('created_utc', 0))
                if abs((post_date.date() - as_of_date.date()).days) <= 1:  # Within 1 day
                    if self._mentions_symbol(post, symbol):
                        posts.append(post)
        
        # Also search in general crypto subreddits
        search_query = f"{symbol} OR ${symbol}"
        general_posts = await self._search_reddit_posts(search_query, ['CryptoCurrency'], as_of_date)
        posts.extend(general_posts)
        
        return posts
    
    async def _fetch_subreddit_posts(self, subreddit: str, sort: str = 'hot', limit: int = 25) -> List[Dict]:
        """Fetch posts from a specific subreddit."""
        await self.rate_limiter.acquire()
        
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {'limit': limit}
        
        posts = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'TradingAgents/1.0 (Crypto Sentiment Analysis Bot)'
                }
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data and 'children' in data['data']:
                            for child in data['data']['children']:
                                if child.get('kind') == 't3':  # Post type
                                    posts.append(child['data'])
                    else:
                        logger.warning(f"Reddit API error for r/{subreddit}: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit}: {e}")
        
        return posts
    
    async def _search_reddit_posts(self, query: str, subreddits: List[str], as_of_date: datetime) -> List[Dict]:
        """Search for posts in subreddits with a specific query."""
        posts = []
        
        for subreddit in subreddits:
            await self.rate_limiter.acquire()
            
            # Use Reddit search API
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': query,
                'restrict_sr': 'on',
                'sort': 'new',
                'limit': 25,
                't': 'week'  # Past week
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'User-Agent': 'TradingAgents/1.0 (Crypto Sentiment Analysis Bot)'
                    }
                    
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'data' in data and 'children' in data['data']:
                                for child in data['data']['children']:
                                    if child.get('kind') == 't3':
                                        post = child['data']
                                        post_date = datetime.fromtimestamp(post.get('created_utc', 0))
                                        if abs((post_date.date() - as_of_date.date()).days) <= 1:
                                            posts.append(post)
                        else:
                            logger.warning(f"Reddit search error for r/{subreddit}: {response.status}")
                            
            except Exception as e:
                logger.error(f"Error searching Reddit in r/{subreddit}: {e}")
        
        return posts
    
    def _mentions_symbol(self, post: Dict, symbol: str) -> bool:
        """Check if post mentions the given symbol."""
        text = (post.get('title', '') + ' ' + post.get('selftext', '')).lower()
        symbol_lower = symbol.lower()
        
        # Check for symbol mention
        if symbol_lower in text or f"${symbol}" in text or f"${symbol_lower}" in text:
            return True
        
        # Check for full name mentions
        name_map = {
            'btc': 'bitcoin', 'eth': 'ethereum', 'ada': 'cardano',
            'sol': 'solana', 'doge': 'dogecoin', 'matic': 'polygon',
            'link': 'chainlink', 'dot': 'polkadot', 'uni': 'uniswap'
        }
        
        full_name = name_map.get(symbol_lower)
        if full_name and full_name in text:
            return True
        
        return False
    
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
    
    def _analyze_posts_sentiment(self, posts: List[Dict], symbol: str, as_of_date: datetime) -> Optional[SentimentData]:
        """Analyze sentiment from Reddit posts."""
        if not posts:
            return None
        
        total_sentiment = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_mentions = len(posts)
        total_upvotes = 0
        total_comments = 0
        
        for post in posts:
            # Analyze sentiment of title and content
            text = post.get('title', '') + ' ' + post.get('selftext', '')
            sentiment = self._analyze_text_sentiment(text)
            
            # Weight by upvotes (higher upvotes = more influence)
            upvotes = max(1, post.get('ups', 1))  # Minimum weight of 1
            weight = min(10, upvotes / 10)  # Cap weight at 10x
            
            total_sentiment += sentiment * weight
            total_upvotes += upvotes
            total_comments += post.get('num_comments', 0)
            
            if sentiment > 0.1:
                positive_count += 1
            elif sentiment < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate weighted average sentiment
        average_sentiment = total_sentiment / sum(max(1, post.get('ups', 1)) for post in posts)
        
        # Calculate social volume (24h activity approximation)
        social_volume_24h = len([p for p in posts if (
            datetime.fromtimestamp(p.get('created_utc', 0)) > 
            as_of_date - timedelta(hours=24)
        )])
        
        return SentimentData(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            as_of_date=as_of_date,
            sentiment_score=max(-1.0, min(1.0, average_sentiment)),
            mention_count=total_mentions,
            positive_mentions=positive_count,
            negative_mentions=negative_count,
            neutral_mentions=neutral_count,
            social_volume_24h=social_volume_24h,
            reddit_sentiment=average_sentiment,
            data_sources=['Reddit'],
            data_quality=DataQuality.MEDIUM
        )
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using keyword-based approach."""
        text_lower = text.lower()
        
        positive_keywords = [
            'bull', 'bullish', 'moon', 'pump', 'buy', 'buying', 'hodl', 'diamond hands',
            'to the moon', 'green', 'profit', 'gains', 'surge', 'rally', 'breakout',
            'adoption', 'partnership', 'upgrade', 'optimistic', 'positive', 'love',
            'amazing', 'great', 'excellent', 'best', 'good', 'strong', 'solid'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'dump', 'crash', 'sell', 'selling', 'red', 'loss',
            'losses', 'drop', 'fall', 'decline', 'fear', 'panic', 'worried',
            'concern', 'risk', 'bad', 'worst', 'terrible', 'scam', 'bubble',
            'overvalued', 'dead', 'rekt', 'paper hands', 'capitulation'
        ]
        
        # Count occurrences with weights
        positive_score = 0
        negative_score = 0
        
        for keyword in positive_keywords:
            if keyword in text_lower:
                positive_score += 1
        
        for keyword in negative_keywords:
            if keyword in text_lower:
                negative_score += 1
        
        # Calculate sentiment score
        if positive_score == 0 and negative_score == 0:
            return 0.0  # Neutral
        
        total_score = positive_score + negative_score
        sentiment = (positive_score - negative_score) / total_score
        
        return sentiment 