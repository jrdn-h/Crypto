import os

DEFAULT_CONFIG = {
    # Project paths
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    
    # =============================================================================
    # Asset Class Configuration 
    # =============================================================================
    "asset_class": "equity",  # "equity" | "crypto"
    
    # =============================================================================
    # Crypto Provider Configuration
    # =============================================================================
    "crypto_providers": ["coingecko", "binance_public", "cryptocompare"],
    "crypto_quote_currency": "USD",  # Default normalization currency
    "crypto_timeframe": "1d",       # Default OHLCV interval  
    "use_perps": False,             # Enable perpetual futures (Phase 7+)
    
    # Provider-specific settings
    "coingecko_base_url": "https://api.coingecko.com/api/v3",
    "binance_base_url": "https://api.binance.com/api/v3",
    "cryptocompare_base_url": "https://min-api.cryptocompare.com/data",
    
    # =============================================================================
    # Execution & Trading Configuration
    # =============================================================================
    "execution_mode": "paper",      # "paper" | "ccxt" | "hyperliquid"
    
    # Risk management (crypto-specific additions)
    "risk": {
        "max_notional_pct": 0.10,       # Max % of portfolio per position
        "max_leverage": 3,              # Max leverage for crypto positions
        "concentration_limit": 0.25,    # Max % in single token
        "max_drawdown": 0.20,          # Max portfolio drawdown
    },
    
    # =============================================================================
    # Caching Configuration  
    # =============================================================================
    "cache": {
        "ttl_fast": 60,              # Fast data TTL (seconds) - prices, quotes
        "ttl_slow": 300,             # Slow data TTL (seconds) - fundamentals
        "ttl_news": 1800,            # News TTL (30 minutes)
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "use_redis": True,           # Use Redis if available, fallback to filesystem
        "max_cache_size_mb": 500,    # Max filesystem cache size
    },
    
    # =============================================================================
    # LLM Configuration (Existing)
    # =============================================================================
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini", 
    "backend_url": "https://api.openai.com/v1",
    
    # Model cost optimization
    "model_cost_preset": "balanced",  # "cheap" | "balanced" | "premium"
    "model_costs": {
        "cheap": {
            "deep_think_llm": "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini"
        },
        "balanced": {
            "deep_think_llm": "o4-mini", 
            "quick_think_llm": "gpt-4o-mini"
        },
        "premium": {
            "deep_think_llm": "o1-preview",
            "quick_think_llm": "gpt-4o"
        }
    },
    
    # =============================================================================
    # Agent Debate Configuration (Existing)
    # =============================================================================
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    
    # =============================================================================
    # Data Source Configuration
    # =============================================================================
    "online_tools": True,
    "data_quality_threshold": "medium",  # "low" | "medium" | "high"
    "enable_data_fallback": True,        # Auto-fallback between providers
    
    # Provider timeouts and retry
    "api_timeout_seconds": 30,
    "max_retries": 3,
    "retry_delay_seconds": 1,
    
    # =============================================================================
    # Feature Flags (Gradual Rollout)
    # =============================================================================
    "features": {
        "crypto_support": False,         # Enable crypto functionality (default disabled for backward compatibility)
        "multi_asset_portfolio": False,  # Mix equity + crypto (Phase 8+)
        "advanced_risk_metrics": False,  # Enhanced risk calculations
        "social_sentiment": True,        # Social media sentiment analysis
        "news_sentiment": True,          # News sentiment scoring
    },
    
    # =============================================================================
    # Debug & Logging
    # =============================================================================
    "debug_mode": False,
    "log_level": "INFO",  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    "log_api_calls": False,              # Log all API requests (debug)
    "save_raw_data": False,              # Save raw API responses
}

# =============================================================================
# Configuration Validation & Helpers
# =============================================================================

def validate_config(config: dict) -> dict:
    """Validate and normalize configuration."""
    validated = config.copy()
    
    # Validate asset class
    if validated["asset_class"] not in ["equity", "crypto"]:
        raise ValueError(f"Invalid asset_class: {validated['asset_class']}. Must be 'equity' or 'crypto'")
    
    # Apply model cost preset
    cost_preset = validated.get("model_cost_preset", "balanced")
    if cost_preset in validated["model_costs"]:
        preset_models = validated["model_costs"][cost_preset]
        validated.update(preset_models)
    
    # Validate execution mode
    valid_execution = ["paper", "ccxt", "hyperliquid"]
    if validated["execution_mode"] not in valid_execution:
        raise ValueError(f"Invalid execution_mode: {validated['execution_mode']}. Must be one of {valid_execution}")
    
    # Crypto-specific validation
    if validated["asset_class"] == "crypto":
        if not validated["crypto_providers"]:
            raise ValueError("crypto_providers cannot be empty when asset_class is 'crypto'")
            
        # Validate quote currency
        if validated["crypto_quote_currency"] not in ["USD", "USDT", "BTC", "ETH"]:
            print(f"Warning: Unusual quote currency {validated['crypto_quote_currency']}")
    
    # Risk validation
    risk_config = validated["risk"]
    if risk_config["max_notional_pct"] > 1.0:
        raise ValueError("max_notional_pct cannot exceed 1.0 (100%)")
    if risk_config["max_leverage"] < 1:
        raise ValueError("max_leverage must be >= 1")
    
    return validated


def get_crypto_config_template() -> dict:
    """Get a template configuration for crypto trading."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "asset_class": "crypto",
        "crypto_providers": ["coingecko", "binance_public"],
        "model_cost_preset": "cheap",  # Save costs for crypto experimentation
        "max_debate_rounds": 1,        # Faster iterations
        "cache": {
            **config["cache"],
            "ttl_fast": 30,            # Faster cache refresh for volatile crypto
        },
        "features": {
            **config["features"],
            "crypto_support": True,
            "social_sentiment": True,   # Important for crypto
        }
    })
    return validate_config(config)


def get_equity_config_template() -> dict:
    """Get a template configuration for equity trading (backward compatibility)."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "asset_class": "equity",
        "model_cost_preset": "balanced",
        "features": {
            **config["features"],
            "crypto_support": False,    # Disable crypto features
        }
    })
    return validate_config(config)


def merge_user_config(user_config: dict) -> dict:
    """Merge user configuration with defaults."""
    config = DEFAULT_CONFIG.copy()
    
    # Deep merge nested dictionaries
    for key, value in user_config.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    
    return validate_config(config)


# =============================================================================
# Environment Variable Integration
# =============================================================================

def load_env_config() -> dict:
    """Load configuration from environment variables."""
    env_config = {}
    
    # Asset class
    if os.getenv("TRADINGAGENTS_ASSET_CLASS"):
        env_config["asset_class"] = os.getenv("TRADINGAGENTS_ASSET_CLASS")
    
    # Execution mode
    if os.getenv("TRADINGAGENTS_EXECUTION_MODE"):
        env_config["execution_mode"] = os.getenv("TRADINGAGENTS_EXECUTION_MODE")
    
    # Model settings
    if os.getenv("TRADINGAGENTS_MODEL_PRESET"):
        env_config["model_cost_preset"] = os.getenv("TRADINGAGENTS_MODEL_PRESET")
    
    # Debug mode
    if os.getenv("TRADINGAGENTS_DEBUG"):
        env_config["debug_mode"] = os.getenv("TRADINGAGENTS_DEBUG").lower() == "true"
    
    # Online tools
    if os.getenv("TRADINGAGENTS_ONLINE"):
        env_config["online_tools"] = os.getenv("TRADINGAGENTS_ONLINE").lower() == "true"
        
    return env_config


# Auto-load environment config if available
_env_config = load_env_config()
if _env_config:
    DEFAULT_CONFIG = merge_user_config(_env_config)
