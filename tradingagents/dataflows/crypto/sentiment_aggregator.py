"""
Sentiment aggregator for crypto social sentiment analysis.

This module aggregates sentiment data from multiple sources (Reddit, Twitter, news)
and provides normalized, weighted sentiment scores with deduplication.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from statistics import mean, median

from ..base_interfaces import AssetClass, DataQuality, SentimentData, NewsItem
from .caching import CacheManager, CacheConfig

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """Aggregates sentiment data from multiple crypto sources."""
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        source_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize sentiment aggregator.
        
        Args:
            cache_manager: Cache manager for aggregated results
            source_weights: Weights for different sentiment sources
        """
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # Default weights for different sources (0.0 to 1.0)
        self.source_weights = source_weights or {
            'Twitter': 0.4,    # High weight - real-time, broad reach
            'Reddit': 0.35,    # High weight - engaged community discussions
            'CryptoPanic': 0.15,  # Medium weight - news sentiment
            'CoinDesk': 0.1    # Lower weight - institutional news
        }
        
        # Quality score multipliers for different data qualities
        self.quality_multipliers = {
            DataQuality.HIGH: 1.0,
            DataQuality.MEDIUM: 0.8,
            DataQuality.LOW: 0.6,
            DataQuality.UNKNOWN: 0.5
        }
        
        # Time decay factors (how much to weight older vs newer data)
        self.time_decay_hours = 24  # Full weight for past 24 hours
        self.min_weight = 0.3  # Minimum weight for older data
    
    async def aggregate_sentiment(
        self,
        symbol: str,
        sentiment_data: List[SentimentData],
        news_items: Optional[List[NewsItem]] = None,
        as_of_date: Optional[datetime] = None
    ) -> Optional[SentimentData]:
        """
        Aggregate sentiment data from multiple sources.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            sentiment_data: List of SentimentData from different sources
            news_items: Optional news items to include in sentiment calculation
            as_of_date: Reference date for aggregation
            
        Returns:
            Aggregated SentimentData or None if no valid data
        """
        if not sentiment_data and not news_items:
            return None
        
        as_of_date = as_of_date or datetime.now()
        cache_key = f"aggregated_sentiment_{symbol}_{as_of_date.date()}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached aggregated sentiment for {symbol}")
            return SentimentData(**cached_result)
        
        try:
            # Filter and validate sentiment data
            valid_sentiment_data = self._filter_valid_sentiment(sentiment_data, as_of_date)
            
            # Include news sentiment if provided
            if news_items:
                news_sentiment = self._extract_news_sentiment(news_items, symbol, as_of_date)
                if news_sentiment:
                    valid_sentiment_data.append(news_sentiment)
            
            if not valid_sentiment_data:
                logger.debug(f"No valid sentiment data for {symbol}")
                return None
            
            # Calculate aggregated sentiment
            aggregated = self._calculate_aggregated_sentiment(
                symbol, valid_sentiment_data, as_of_date
            )
            
            # Cache the result
            if aggregated:
                await self.cache_manager.set(cache_key, aggregated.dict(), ttl=900)  # 15 minutes
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment for {symbol}: {e}")
            return None
    
    async def get_multi_symbol_sentiment(
        self,
        symbols: List[str],
        sentiment_data_by_symbol: Dict[str, List[SentimentData]],
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, Optional[SentimentData]]:
        """
        Aggregate sentiment for multiple symbols in parallel.
        
        Args:
            symbols: List of crypto symbols
            sentiment_data_by_symbol: Dictionary mapping symbols to their sentiment data
            as_of_date: Reference date for aggregation
            
        Returns:
            Dictionary mapping symbols to aggregated sentiment data
        """
        tasks = []
        for symbol in symbols:
            sentiment_data = sentiment_data_by_symbol.get(symbol, [])
            task = self.aggregate_sentiment(symbol, sentiment_data, None, as_of_date)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        aggregated_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error aggregating sentiment for {symbol}: {result}")
                aggregated_results[symbol] = None
            else:
                aggregated_results[symbol] = result
        
        return aggregated_results
    
    def _filter_valid_sentiment(
        self, 
        sentiment_data: List[SentimentData],
        as_of_date: datetime
    ) -> List[SentimentData]:
        """Filter and validate sentiment data."""
        valid_data = []
        
        for data in sentiment_data:
            # Check if data is recent enough
            age_hours = abs((as_of_date - data.as_of_date).total_seconds()) / 3600
            if age_hours > 48:  # Skip data older than 48 hours
                continue
            
            # Check if data has valid sentiment score
            if data.sentiment_score is None:
                continue
            
            # Check minimum mention count threshold
            min_mentions = 3  # At least 3 mentions required
            if data.mention_count is not None and data.mention_count < min_mentions:
                continue
            
            valid_data.append(data)
        
        return valid_data
    
    def _extract_news_sentiment(
        self,
        news_items: List[NewsItem],
        symbol: str,
        as_of_date: datetime
    ) -> Optional[SentimentData]:
        """Extract sentiment data from news items."""
        if not news_items:
            return None
        
        # Filter news items by symbol relevance and recency
        relevant_news = []
        for item in news_items:
            # Check if news mentions the symbol
            if symbol in item.symbols_mentioned or (
                item.relevance_score is not None and item.relevance_score > 0.3
            ):
                # Check if news is recent
                age_hours = abs((as_of_date - item.published_at).total_seconds()) / 3600
                if age_hours <= 24:  # Only include news from past 24 hours
                    relevant_news.append(item)
        
        if not relevant_news:
            return None
        
        # Calculate aggregated news sentiment
        sentiments = [item.sentiment_score for item in relevant_news if item.sentiment_score is not None]
        if not sentiments:
            return None
        
        # Weight by relevance score
        weighted_sentiments = []
        total_weight = 0
        
        for item in relevant_news:
            if item.sentiment_score is not None:
                weight = item.relevance_score or 0.5
                weighted_sentiments.append(item.sentiment_score * weight)
                total_weight += weight
        
        if total_weight == 0:
            return None
        
        avg_sentiment = sum(weighted_sentiments) / total_weight
        
        # Count positive/negative/neutral news
        positive_count = len([s for s in sentiments if s > 0.1])
        negative_count = len([s for s in sentiments if s < -0.1])
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return SentimentData(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            as_of_date=as_of_date,
            sentiment_score=avg_sentiment,
            mention_count=len(relevant_news),
            positive_mentions=positive_count,
            negative_mentions=negative_count,
            neutral_mentions=neutral_count,
            news_sentiment=avg_sentiment,
            data_sources=['News'],
            data_quality=DataQuality.MEDIUM
        )
    
    def _calculate_aggregated_sentiment(
        self,
        symbol: str,
        sentiment_data: List[SentimentData],
        as_of_date: datetime
    ) -> Optional[SentimentData]:
        """Calculate weighted aggregated sentiment."""
        if not sentiment_data:
            return None
        
        weighted_scores = []
        total_weight = 0
        
        # Aggregate mention counts and source breakdowns
        total_mentions = 0
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        total_volume_24h = 0
        
        source_sentiments = {}
        data_sources = []
        
        for data in sentiment_data:
            if data.sentiment_score is None:
                continue
            
            # Calculate time decay weight
            age_hours = abs((as_of_date - data.as_of_date).total_seconds()) / 3600
            time_weight = self._calculate_time_weight(age_hours)
            
            # Get source weight
            source_weight = self._get_source_weight(data.data_sources)
            
            # Get quality weight
            quality_weight = self.quality_multipliers.get(data.data_quality, 0.5)
            
            # Calculate volume weight (more mentions = higher weight)
            volume_weight = min(2.0, 1.0 + (data.mention_count or 0) / 100)
            
            # Combined weight
            final_weight = time_weight * source_weight * quality_weight * volume_weight
            
            weighted_scores.append(data.sentiment_score * final_weight)
            total_weight += final_weight
            
            # Aggregate counts
            total_mentions += data.mention_count or 0
            total_positive += data.positive_mentions or 0
            total_negative += data.negative_mentions or 0
            total_neutral += data.neutral_mentions or 0
            total_volume_24h += data.social_volume_24h or 0
            
            # Track source-specific sentiments
            for source in data.data_sources:
                if source not in source_sentiments:
                    source_sentiments[source] = []
                source_sentiments[source].append(data.sentiment_score)
                
                if source not in data_sources:
                    data_sources.append(source)
        
        if total_weight == 0:
            return None
        
        # Calculate final aggregated sentiment
        aggregated_sentiment = sum(weighted_scores) / total_weight
        
        # Calculate trending rank (approximation based on volume and sentiment)
        trending_score = (total_volume_24h * abs(aggregated_sentiment)) if total_volume_24h > 0 else 0
        trending_rank = min(100, max(1, int(trending_score / 10))) if trending_score > 0 else None
        
        # Calculate source-specific sentiments
        twitter_sentiment = mean(source_sentiments.get('Twitter', [None])) if 'Twitter' in source_sentiments else None
        reddit_sentiment = mean(source_sentiments.get('Reddit', [None])) if 'Reddit' in source_sentiments else None
        news_sentiment = mean(source_sentiments.get('News', [None])) if 'News' in source_sentiments else None
        
        # Determine overall data quality
        qualities = [data.data_quality for data in sentiment_data]
        if DataQuality.HIGH in qualities:
            overall_quality = DataQuality.HIGH
        elif DataQuality.MEDIUM in qualities:
            overall_quality = DataQuality.MEDIUM
        else:
            overall_quality = DataQuality.LOW
        
        return SentimentData(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            as_of_date=as_of_date,
            sentiment_score=max(-1.0, min(1.0, aggregated_sentiment)),
            mention_count=total_mentions,
            positive_mentions=total_positive,
            negative_mentions=total_negative,
            neutral_mentions=total_neutral,
            social_volume_24h=total_volume_24h,
            trending_rank=trending_rank,
            twitter_sentiment=twitter_sentiment,
            reddit_sentiment=reddit_sentiment,
            news_sentiment=news_sentiment,
            data_sources=data_sources,
            data_quality=overall_quality
        )
    
    def _calculate_time_weight(self, age_hours: float) -> float:
        """Calculate time decay weight for sentiment data."""
        if age_hours <= self.time_decay_hours:
            return 1.0  # Full weight for recent data
        
        # Linear decay after time_decay_hours
        decay_rate = (1.0 - self.min_weight) / (48 - self.time_decay_hours)  # Decay to min_weight over 48 hours
        weight = 1.0 - (age_hours - self.time_decay_hours) * decay_rate
        
        return max(self.min_weight, weight)
    
    def _get_source_weight(self, data_sources: List[str]) -> float:
        """Get weight for sentiment data based on its sources."""
        if not data_sources:
            return 0.1  # Default low weight for unknown sources
        
        # Use the maximum weight among all sources
        weights = [self.source_weights.get(source, 0.1) for source in data_sources]
        return max(weights)
    
    def calculate_sentiment_confidence(self, sentiment_data: SentimentData) -> float:
        """
        Calculate confidence score for aggregated sentiment (0.0 to 1.0).
        
        Args:
            sentiment_data: Aggregated sentiment data
            
        Returns:
            Confidence score (higher = more reliable)
        """
        confidence_factors = []
        
        # Factor 1: Number of mentions (more mentions = higher confidence)
        if sentiment_data.mention_count:
            mention_confidence = min(1.0, sentiment_data.mention_count / 50)
            confidence_factors.append(mention_confidence)
        
        # Factor 2: Data quality
        quality_confidence = self.quality_multipliers.get(sentiment_data.data_quality, 0.5)
        confidence_factors.append(quality_confidence)
        
        # Factor 3: Number of data sources (more sources = higher confidence)
        source_confidence = min(1.0, len(sentiment_data.data_sources) / 3)
        confidence_factors.append(source_confidence)
        
        # Factor 4: Sentiment consistency (if sources agree, higher confidence)
        source_sentiments = [
            sentiment_data.twitter_sentiment,
            sentiment_data.reddit_sentiment,
            sentiment_data.news_sentiment
        ]
        valid_sentiments = [s for s in source_sentiments if s is not None]
        
        if len(valid_sentiments) >= 2:
            # Calculate sentiment agreement (lower variance = higher confidence)
            sentiment_variance = sum((s - sentiment_data.sentiment_score) ** 2 for s in valid_sentiments) / len(valid_sentiments)
            consistency_confidence = max(0.0, 1.0 - sentiment_variance)
            confidence_factors.append(consistency_confidence)
        
        # Factor 5: Social volume (higher activity = higher confidence)
        if sentiment_data.social_volume_24h:
            volume_confidence = min(1.0, sentiment_data.social_volume_24h / 100)
            confidence_factors.append(volume_confidence)
        
        # Calculate weighted average confidence
        return mean(confidence_factors) if confidence_factors else 0.0 