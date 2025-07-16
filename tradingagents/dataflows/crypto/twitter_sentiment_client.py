"""
Twitter/X sentiment client for crypto social sentiment analysis.

This client implements the SocialSentimentClient interface to fetch sentiment data
from Twitter/X using bearer token authentication with Nitter scraping as fallback.
"""

import asyncio
import json
import logging
import random
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


class TwitterSentimentClient(SocialSentimentClient):
    """Twitter/X sentiment client for crypto social sentiment analysis."""
    
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
        use_nitter_fallback: bool = True
    ):
        """
        Initialize Twitter sentiment client.
        
        Args:
            bearer_token: Optional Twitter API bearer token for authentication
            cache_manager: Cache manager for response caching
            rate_limiter: Rate limiter for API calls
            use_nitter_fallback: Whether to use Nitter as fallback when API fails
        """
        self.bearer_token = bearer_token
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        self.use_nitter_fallback = use_nitter_fallback
        
        # Twitter API v2 rate limits: 300 requests per 15 min with bearer token
        # Nitter has no official rate limits but be respectful
        if self.bearer_token:
            default_config = RateLimitConfig(requests_per_minute=15)
        else:
            default_config = RateLimitConfig(requests_per_minute=5)
        
        self.rate_limiter = rate_limiter or RateLimiter(default_config)
        
        # Twitter API endpoints
        self.api_base_url = "https://api.twitter.com/2"
        
        # Nitter instances (public instances for fallback)
        self.nitter_instances = [
            "https://nitter.net",
            "https://nitter.it", 
            "https://nitter.fdn.fr",
            "https://nitter.kavin.rocks",
            "https://nitter.unixfox.eu"
        ]
        
        # Crypto-related Twitter accounts to monitor for trending
        self.crypto_accounts = [
            'coindesk', 'cointelegraph', 'decrypt_co', 'thedefiant_io',
            'bitcoinmagazine', 'messaricrypto', 'whale_alert', 'santimentfeed'
        ]
        
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
        Get sentiment data for a crypto symbol from Twitter/X.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            as_of_date: Date for sentiment analysis
            
        Returns:
            SentimentData object or None if no data available
        """
        cache_key = f"twitter_sentiment_{symbol}_{as_of_date.date()}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached Twitter sentiment for {symbol}")
            return SentimentData(**cached_result)
        
        tweets = []
        
        try:
            # Try Twitter API first if bearer token available
            if self.bearer_token:
                tweets = await self._fetch_tweets_api(symbol, as_of_date)
            
            # Fallback to Nitter if API fails or no bearer token
            if not tweets and self.use_nitter_fallback:
                tweets = await self._fetch_tweets_nitter(symbol, as_of_date)
            
            if not tweets:
                logger.debug(f"No tweets found for {symbol}")
                return None
            
            # Analyze sentiment from tweets
            sentiment_data = self._analyze_tweets_sentiment(tweets, symbol, as_of_date)
            
            # Cache the result
            if sentiment_data:
                await self.cache_manager.set(cache_key, sentiment_data.dict(), ttl=900)  # 15 minutes
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment for {symbol}: {e}")
            return None
    
    async def get_trending_symbols(self, limit: int = 10) -> List[str]:
        """
        Get trending crypto symbols from Twitter discussions.
        
        Args:
            limit: Maximum number of trending symbols to return
            
        Returns:
            List of trending crypto symbols
        """
        cache_key = f"twitter_trending_{limit}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug("Using cached Twitter trending symbols")
            return cached_result
        
        try:
            symbol_mentions = {}
            
            # Search for general crypto terms
            crypto_queries = ['#crypto', '#bitcoin', '#ethereum', '#altcoin', '#defi']
            
            for query in crypto_queries:
                if self.bearer_token:
                    tweets = await self._search_tweets_api(query, limit=100)
                elif self.use_nitter_fallback:
                    tweets = await self._search_tweets_nitter(query, limit=50)
                else:
                    continue
                
                for tweet in tweets:
                    symbols = self._extract_symbols_from_text(tweet.get('text', ''))
                    for symbol in symbols:
                        if symbol not in symbol_mentions:
                            symbol_mentions[symbol] = 0
                        # Weight by engagement (likes, retweets, replies)
                        engagement = (
                            tweet.get('public_metrics', {}).get('like_count', 0) +
                            tweet.get('public_metrics', {}).get('retweet_count', 0) * 2 +
                            tweet.get('public_metrics', {}).get('reply_count', 0)
                        )
                        symbol_mentions[symbol] += max(1, engagement / 10)
            
            # Sort by mention weight and return top symbols
            trending = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
            trending_symbols = [symbol for symbol, _ in trending[:limit]]
            
            # Cache the result
            await self.cache_manager.set(cache_key, trending_symbols, ttl=600)  # 10 minutes
            
            return trending_symbols
            
        except Exception as e:
            logger.error(f"Error fetching trending symbols from Twitter: {e}")
            return []
    
    async def _fetch_tweets_api(self, symbol: str, as_of_date: datetime) -> List[Dict]:
        """Fetch tweets using Twitter API v2."""
        await self.rate_limiter.acquire()
        
        # Build search query
        symbol_queries = [f"${symbol}", symbol.lower()]
        
        # Add full name if known
        name_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano',
            'SOL': 'solana', 'DOGE': 'dogecoin', 'MATIC': 'polygon',
            'LINK': 'chainlink', 'DOT': 'polkadot', 'UNI': 'uniswap'
        }
        
        if symbol.upper() in name_map:
            symbol_queries.append(name_map[symbol.upper()])
        
        query = " OR ".join(symbol_queries)
        query += " -is:retweet lang:en"  # Exclude retweets, English only
        
        tweets = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.bearer_token}',
                    'Content-Type': 'application/json'
                }
                
                params = {
                    'query': query,
                    'max_results': 100,
                    'tweet.fields': 'created_at,public_metrics,context_annotations,lang',
                    'start_time': (as_of_date - timedelta(days=1)).isoformat() + 'Z',
                    'end_time': (as_of_date + timedelta(days=1)).isoformat() + 'Z'
                }
                
                url = f"{self.api_base_url}/tweets/search/recent"
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data:
                            tweets = data['data']
                            logger.debug(f"Fetched {len(tweets)} tweets for {symbol} via API")
                    else:
                        logger.warning(f"Twitter API error for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error calling Twitter API for {symbol}: {e}")
        
        return tweets
    
    async def _search_tweets_api(self, query: str, limit: int = 100) -> List[Dict]:
        """Search tweets using Twitter API v2."""
        await self.rate_limiter.acquire()
        
        tweets = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.bearer_token}',
                    'Content-Type': 'application/json'
                }
                
                params = {
                    'query': f"{query} -is:retweet lang:en",
                    'max_results': min(limit, 100),
                    'tweet.fields': 'created_at,public_metrics,context_annotations,lang'
                }
                
                url = f"{self.api_base_url}/tweets/search/recent"
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data:
                            tweets = data['data']
                    else:
                        logger.warning(f"Twitter API search error for '{query}': {response.status}")
                        
        except Exception as e:
            logger.error(f"Error searching Twitter API for '{query}': {e}")
        
        return tweets
    
    async def _fetch_tweets_nitter(self, symbol: str, as_of_date: datetime) -> List[Dict]:
        """Fetch tweets using Nitter instances as fallback."""
        tweets = []
        
        # Try different Nitter instances
        for instance in random.sample(self.nitter_instances, min(3, len(self.nitter_instances))):
            try:
                instance_tweets = await self._fetch_from_nitter_instance(instance, symbol)
                tweets.extend(instance_tweets)
                
                if len(tweets) >= 50:  # Enough tweets collected
                    break
                    
            except Exception as e:
                logger.debug(f"Nitter instance {instance} failed: {e}")
                continue
        
        return tweets
    
    async def _search_tweets_nitter(self, query: str, limit: int = 50) -> List[Dict]:
        """Search tweets using Nitter instances."""
        tweets = []
        
        for instance in random.sample(self.nitter_instances, min(2, len(self.nitter_instances))):
            try:
                await self.rate_limiter.acquire()
                
                search_url = f"{instance}/search"
                params = {'q': query, 'f': 'tweets'}
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    async with session.get(search_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            instance_tweets = self._parse_nitter_html(html)
                            tweets.extend(instance_tweets[:limit])
                            
                            if len(tweets) >= limit:
                                break
                                
            except Exception as e:
                logger.debug(f"Nitter search failed for {instance}: {e}")
                continue
        
        return tweets[:limit]
    
    async def _fetch_from_nitter_instance(self, instance: str, symbol: str) -> List[Dict]:
        """Fetch tweets from a specific Nitter instance."""
        await self.rate_limiter.acquire()
        
        # Search for symbol mentions
        search_url = f"{instance}/search"
        params = {'q': f"${symbol} OR {symbol.lower()}", 'f': 'tweets'}
        
        tweets = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(search_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        tweets = self._parse_nitter_html(html)
                        
        except Exception as e:
            logger.debug(f"Error fetching from Nitter {instance}: {e}")
        
        return tweets
    
    def _parse_nitter_html(self, html: str) -> List[Dict]:
        """Parse tweets from Nitter HTML."""
        tweets = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            tweet_elements = soup.find_all('div', class_='tweet-content')
            
            for element in tweet_elements:
                try:
                    text_elem = element.find('div', class_='tweet-text')
                    if not text_elem:
                        continue
                    
                    text = text_elem.get_text().strip()
                    
                    # Extract basic metrics if available
                    stats_elem = element.find('div', class_='tweet-stats')
                    likes = 0
                    retweets = 0
                    replies = 0
                    
                    if stats_elem:
                        # Try to extract numbers from stats
                        stats_text = stats_elem.get_text()
                        numbers = re.findall(r'\d+', stats_text)
                        if len(numbers) >= 3:
                            replies = int(numbers[0])
                            retweets = int(numbers[1]) 
                            likes = int(numbers[2])
                    
                    tweet = {
                        'text': text,
                        'created_at': datetime.now().isoformat(),  # Approximate
                        'public_metrics': {
                            'like_count': likes,
                            'retweet_count': retweets,
                            'reply_count': replies
                        }
                    }
                    
                    tweets.append(tweet)
                    
                except Exception as e:
                    logger.debug(f"Error parsing tweet element: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing Nitter HTML: {e}")
        
        return tweets
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract crypto symbols from tweet text."""
        symbols = []
        text_lower = text.lower()
        
        # Extract $SYMBOL patterns (common on Twitter)
        symbol_pattern = r'\$([A-Z]{2,10})'
        matches = re.findall(symbol_pattern, text.upper())
        for match in matches:
            if match not in symbols:
                symbols.append(match)
        
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
        
        return symbols
    
    def _analyze_tweets_sentiment(self, tweets: List[Dict], symbol: str, as_of_date: datetime) -> Optional[SentimentData]:
        """Analyze sentiment from tweets."""
        if not tweets:
            return None
        
        total_sentiment = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_mentions = len(tweets)
        total_engagement = 0
        
        for tweet in tweets:
            # Analyze sentiment of tweet text
            text = tweet.get('text', '')
            sentiment = self._analyze_text_sentiment(text)
            
            # Weight by engagement (likes + retweets + replies)
            metrics = tweet.get('public_metrics', {})
            engagement = (
                metrics.get('like_count', 0) +
                metrics.get('retweet_count', 0) * 2 +  # Retweets have more weight
                metrics.get('reply_count', 0)
            )
            weight = max(1, engagement / 5)  # Minimum weight of 1
            
            total_sentiment += sentiment * weight
            total_engagement += engagement
            
            if sentiment > 0.1:
                positive_count += 1
            elif sentiment < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate weighted average sentiment
        total_weight = sum(max(1, tweet.get('public_metrics', {}).get('like_count', 0) + 
                              tweet.get('public_metrics', {}).get('retweet_count', 0) + 
                              tweet.get('public_metrics', {}).get('reply_count', 0)) / 5 for tweet in tweets)
        average_sentiment = total_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Calculate social volume (approximate 24h activity)
        social_volume_24h = len([t for t in tweets if (
            datetime.fromisoformat(t.get('created_at', '').replace('Z', '+00:00')) > 
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
            twitter_sentiment=average_sentiment,
            data_sources=['Twitter'],
            data_quality=DataQuality.HIGH if self.bearer_token else DataQuality.MEDIUM
        )
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of tweet text using crypto-specific keywords."""
        text_lower = text.lower()
        
        positive_keywords = [
            'bull', 'bullish', 'moon', 'pump', 'buy', 'buying', 'hodl', 'diamond hands',
            'to the moon', 'ath', 'all time high', 'green', 'profit', 'gains', 'surge',
            'rally', 'breakout', 'adoption', 'partnership', 'launch', 'upgrade',
            'optimistic', 'positive', 'love', 'amazing', 'great', 'excellent', 'best',
            'good', 'strong', 'solid', 'accumulate', 'accumulating', 'dca'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'dump', 'crash', 'sell', 'selling', 'red', 'loss',
            'losses', 'drop', 'fall', 'decline', 'fear', 'panic', 'worried',
            'concern', 'risk', 'bad', 'worst', 'terrible', 'scam', 'rug', 'rugpull',
            'bubble', 'overvalued', 'dead', 'rekt', 'paper hands', 'capitulation',
            'bear market', 'correction', 'dip', 'blood', 'massacre'
        ]
        
        # Count occurrences with emoji consideration
        positive_score = 0
        negative_score = 0
        
        for keyword in positive_keywords:
            if keyword in text_lower:
                positive_score += 1
        
        for keyword in negative_keywords:
            if keyword in text_lower:
                negative_score += 1
        
        # Check for common crypto emojis
        if '🚀' in text or '🌙' in text or '💎' in text or '🔥' in text:
            positive_score += 1
        if '📉' in text or '💀' in text or '🩸' in text:
            negative_score += 1
        
        # Calculate sentiment score
        if positive_score == 0 and negative_score == 0:
            return 0.0  # Neutral
        
        total_score = positive_score + negative_score
        sentiment = (positive_score - negative_score) / total_score
        
        return sentiment 