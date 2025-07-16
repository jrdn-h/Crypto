import questionary
from typing import List, Optional, Tuple, Dict

from cli.models import AnalystType

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_asset_class() -> str:
    """Select asset class using an interactive selection."""
    
    ASSET_CLASS_OPTIONS = [
        ("Equity - Traditional stocks and shares (Finnhub, YFinance)", "equity"),
        ("Crypto - Digital assets and cryptocurrencies (CoinGecko, Binance)", "crypto"),
    ]
    
    choice = questionary.select(
        "Select Your [Asset Class]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ASSET_CLASS_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]No asset class selected. Exiting...[/red]")
        exit(1)
    
    return choice


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    # Define shallow thinking llm engine options with their corresponding model names
    SHALLOW_AGENT_OPTIONS = {
        "openai": [
            ("GPT-4o-mini - Fast and efficient for quick tasks", "gpt-4o-mini"),
            ("GPT-4.1-nano - Ultra-lightweight model for basic operations", "gpt-4.1-nano"),
            ("GPT-4.1-mini - Compact model with good performance", "gpt-4.1-mini"),
            ("GPT-4o - Standard model with solid capabilities", "gpt-4o"),
        ],
        "anthropic": [
            ("Claude Haiku 3.5 - Fast inference and standard capabilities", "claude-3-5-haiku-latest"),
            ("Claude Sonnet 3.5 - Highly capable standard model", "claude-3-5-sonnet-latest"),
            ("Claude Sonnet 3.7 - Exceptional hybrid reasoning and agentic capabilities", "claude-3-7-sonnet-latest"),
            ("Claude Sonnet 4 - High performance and excellent reasoning", "claude-sonnet-4-0"),
        ],
        "google": [
            ("Gemini 2.0 Flash-Lite - Cost efficiency and low latency", "gemini-2.0-flash-lite"),
            ("Gemini 2.0 Flash - Next generation features, speed, and thinking", "gemini-2.0-flash"),
            ("Gemini 2.5 Flash - Adaptive thinking, cost efficiency", "gemini-2.5-flash-preview-05-20"),
        ],
        "openrouter": [
            ("Meta: Llama 4 Scout", "meta-llama/llama-4-scout:free"),
            ("Meta: Llama 3.3 8B Instruct - A lightweight and ultra-fast variant of Llama 3.3 70B", "meta-llama/llama-3.3-8b-instruct:free"),
            ("google/gemini-2.0-flash-exp:free - Gemini Flash 2.0 offers a significantly faster time to first token", "google/gemini-2.0-flash-exp:free"),
        ],
        "ollama": [
            ("llama3.1 local", "llama3.1"),
            ("llama3.2 local", "llama3.2"),
        ]
    }

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]No shallow thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    # Define deep thinking llm engine options with their corresponding model names
    DEEP_AGENT_OPTIONS = {
        "openai": [
            ("GPT-4.1-nano - Ultra-lightweight model for basic operations", "gpt-4.1-nano"),
            ("GPT-4.1-mini - Compact model with good performance", "gpt-4.1-mini"),
            ("GPT-4o - Standard model with solid capabilities", "gpt-4o"),
            ("o4-mini - Specialized reasoning model (compact)", "o4-mini"),
            ("o3-mini - Advanced reasoning model (lightweight)", "o3-mini"),
            ("o3 - Full advanced reasoning model", "o3"),
            ("o1 - Premier reasoning and problem-solving model", "o1"),
        ],
        "anthropic": [
            ("Claude Haiku 3.5 - Fast inference and standard capabilities", "claude-3-5-haiku-latest"),
            ("Claude Sonnet 3.5 - Highly capable standard model", "claude-3-5-sonnet-latest"),
            ("Claude Sonnet 3.7 - Exceptional hybrid reasoning and agentic capabilities", "claude-3-7-sonnet-latest"),
            ("Claude Sonnet 4 - High performance and excellent reasoning", "claude-sonnet-4-0"),
            ("Claude Opus 4 - Most powerful Anthropic model", "	claude-opus-4-0"),
        ],
        "google": [
            ("Gemini 2.0 Flash-Lite - Cost efficiency and low latency", "gemini-2.0-flash-lite"),
            ("Gemini 2.0 Flash - Next generation features, speed, and thinking", "gemini-2.0-flash"),
            ("Gemini 2.5 Flash - Adaptive thinking, cost efficiency", "gemini-2.5-flash-preview-05-20"),
            ("Gemini 2.5 Pro", "gemini-2.5-pro-preview-06-05"),
        ],
        "openrouter": [
            ("DeepSeek V3 - a 685B-parameter, mixture-of-experts model", "deepseek/deepseek-chat-v3-0324:free"),
            ("Deepseek - latest iteration of the flagship chat model family from the DeepSeek team.", "deepseek/deepseek-chat-v3-0324:free"),
        ],
        "ollama": [
            ("llama3.1 local", "llama3.1"),
            ("qwen3", "qwen3"),
        ]
    }
    
    choice = questionary.select(
        "Select Your [Deep-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in DEEP_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No deep thinking llm engine selected. Exiting...[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str]:
    """Select the OpenAI api url using interactive selection."""
    # Define OpenAI api options with their corresponding endpoints
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Anthropic", "https://api.anthropic.com/"),
        ("Google", "https://generativelanguage.googleapis.com/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://localhost:11434/v1"),        
    ]
    
    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]no OpenAI backend selected. Exiting...[/red]")
        exit(1)
    
    display_name, url = choice
    print(f"You selected: {display_name}\tURL: {url}")
    
    return display_name, url


# =============================================================================
# Phase 9: Enhanced Provider Selection and Cost Management
# =============================================================================

def select_provider_preset() -> str:
    """Select provider preset (free, premium, enterprise)."""
    
    PROVIDER_PRESETS = [
        ("Free Tier - Basic providers with rate limits", "free"),
        ("Premium Tier - Enhanced providers with higher limits", "premium"),
        ("Enterprise Tier - Premium providers with maximum capabilities", "enterprise"),
    ]
    
    choice = questionary.select(
        "Select Your [Provider Preset]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in PROVIDER_PRESETS
        ],
        instruction="\n- Free: CoinGecko (basic), Yahoo Finance, basic LLMs\n- Premium: CoinGecko Pro, Finnhub Premium, GPT-4\n- Enterprise: All premium providers + specialized tools",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]No provider preset selected. Using 'free' tier...[/red]")
        return "free"
    
    return choice


def select_cost_preset() -> str:
    """Select cost optimization preset."""
    
    COST_PRESETS = [
        ("Cheap - Fast models, lower costs (~$0.10/analysis)", "cheap"),
        ("Balanced - Mix of performance and cost (~$0.50/analysis)", "balanced"),
        ("Premium - Best models, higher costs (~$2.00/analysis)", "premium"),
    ]
    
    choice = questionary.select(
        "Select Your [Cost Preset]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in COST_PRESETS
        ],
        instruction="\n- Cheap: GPT-4o-mini, fast inference\n- Balanced: Mix of GPT-4o-mini and GPT-4o\n- Premium: GPT-4o, GPT-o1 for deep analysis",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]No cost preset selected. Using 'balanced'...[/red]")
        return "balanced"
    
    return choice


def select_crypto_providers(provider_preset: str = "free") -> dict:
    """Select specific crypto providers based on preset."""
    
    provider_options = {
        "free": {
            "market_data": ["CoinGecko (Free)", "Binance Public API"],
            "news": ["CryptoPanic (Free)", "CoinDesk RSS"],
            "execution": ["Paper Trading (Simulation)"],
            "risk": ["Basic Risk Manager"]
        },
        "premium": {
            "market_data": ["CoinGecko Pro", "CryptoCompare", "Binance API"],
            "news": ["CryptoPanic Pro", "Twitter Sentiment", "Reddit Analysis"],
            "execution": ["Paper Trading", "CCXT Exchanges", "Binance"],
            "risk": ["Advanced Risk Manager", "Funding Calculator"]
        },
        "enterprise": {
            "market_data": ["All Premium Sources", "Real-time WebSockets"],
            "news": ["All Premium Sources", "Custom News Aggregation"],
            "execution": ["All Exchanges", "Hyperliquid", "Advanced Order Types"],
            "risk": ["Full Risk Suite", "24/7 Monitoring", "Liquidation Alerts"]
        }
    }
    
    selected_providers = provider_options.get(provider_preset, provider_options["free"])
    
    try:
        console.print(f"\n[green]Selected {provider_preset.title()} Tier Crypto Providers:[/green]")
        for category, providers in selected_providers.items():
            console.print(f"  • {category.replace('_', ' ').title()}: {', '.join(providers)}")
    except NameError:
        pass  # Console not available during testing
    
    return selected_providers


def select_equity_providers(provider_preset: str = "free") -> dict:
    """Select specific equity providers based on preset."""
    
    provider_options = {
        "free": {
            "market_data": ["Yahoo Finance", "Alpha Vantage (Free)"],
            "news": ["Yahoo Finance News", "RSS Feeds"],
            "execution": ["Paper Trading (Simulation)"],
            "risk": ["Basic Risk Manager"]
        },
        "premium": {
            "market_data": ["Finnhub", "Alpha Vantage Pro", "Yahoo Finance"],
            "news": ["Finnhub News", "Reddit Analysis", "Twitter Sentiment"],
            "execution": ["Paper Trading", "Broker APIs"],
            "risk": ["Advanced Risk Manager", "Portfolio Analytics"]
        },
        "enterprise": {
            "market_data": ["All Premium Sources", "Real-time Feeds"],
            "news": ["All Premium Sources", "Custom Aggregation"],
            "execution": ["All Brokers", "Advanced Order Management"],
            "risk": ["Full Risk Suite", "Real-time Monitoring"]
        }
    }
    
    selected_providers = provider_options.get(provider_preset, provider_options["free"])
    
    try:
        console.print(f"\n[green]Selected {provider_preset.title()} Tier Equity Providers:[/green]")
        for category, providers in selected_providers.items():
            console.print(f"  • {category.replace('_', ' ').title()}: {', '.join(providers)}")
    except NameError:
        pass  # Console not available during testing
    
    return selected_providers


def apply_cost_preset_to_config(config: dict, cost_preset: str, asset_class: str) -> dict:
    """Apply cost preset settings to configuration."""
    
    cost_configs = {
        "cheap": {
            "shallow_thinker": "gpt-4o-mini",
            "deep_thinker": "gpt-4o-mini",
            "max_debate_rounds": 2,
            "max_risk_discuss_rounds": 2,
            "model_cost_preset": "cheap"
        },
        "balanced": {
            "shallow_thinker": "gpt-4o-mini", 
            "deep_thinker": "gpt-4o",
            "max_debate_rounds": 3,
            "max_risk_discuss_rounds": 3,
            "model_cost_preset": "balanced"
        },
        "premium": {
            "shallow_thinker": "gpt-4o",
            "deep_thinker": "o1-preview",
            "max_debate_rounds": 5,
            "max_risk_discuss_rounds": 4,
            "model_cost_preset": "premium"
        }
    }
    
    # Default to cheap for crypto, balanced for equity
    if cost_preset not in cost_configs:
        cost_preset = "cheap" if asset_class == "crypto" else "balanced"
    
    preset_config = cost_configs[cost_preset]
    
    # Apply preset to config
    for key, value in preset_config.items():
        config[key] = value
    
    # Crypto-specific optimizations
    if asset_class == "crypto":
        # Ensure features dict exists
        if "features" not in config:
            config["features"] = {}
        
        config["features"]["crypto_support"] = True
        config["features"]["24_7_trading"] = True
        config["features"]["funding_analysis"] = True
        
        # Enable cost optimization for crypto
        if cost_preset == "cheap":
            config["max_concurrent_requests"] = 2
            config["rate_limit_delay"] = 1.0
        else:
            config["max_concurrent_requests"] = 5
            config["rate_limit_delay"] = 0.5
    
    # Optional console output (only if console is available)
    try:
        console.print(f"\n[green]Applied {cost_preset.title()} cost preset for {asset_class} analysis[/green]")
    except NameError:
        pass  # Console not available (e.g., during testing)
    
    return config


def validate_provider_configuration(asset_class: str, provider_preset: str) -> bool:
    """Validate that required providers are available for the selected configuration."""
    import os
    
    required_keys = []
    
    if asset_class == "crypto":
        if provider_preset in ["premium", "enterprise"]:
            required_keys.extend(["COINGECKO_API_KEY", "BINANCE_API_KEY"])
        required_keys.append("OPENAI_API_KEY")
    else:  # equity
        if provider_preset in ["premium", "enterprise"]:
            required_keys.extend(["FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"])
        required_keys.append("OPENAI_API_KEY")
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        try:
            console.print(f"\n[yellow]Warning: Missing API keys for {provider_preset} tier:[/yellow]")
            for key in missing_keys:
                console.print(f"  • {key}")
            console.print("\n[dim]You can still proceed with available providers, but some features may be limited.[/dim]")
        except NameError:
            pass  # Console not available during testing
        
        try:
            if questionary.confirm("Continue with limited functionality?").ask():
                return True
            else:
                return False
        except:
            # During testing, assume user would continue
            return True
    
    try:
        console.print(f"\n[green]✅ All required API keys available for {provider_preset} tier[/green]")
    except NameError:
        pass  # Console not available during testing
    return True


def get_provider_recommendations(asset_class: str) -> str:
    """Get provider setup recommendations for asset class."""
    
    if asset_class == "crypto":
        return """
[bold]Crypto Provider Setup Recommendations:[/bold]

[green]Free Tier (No API keys required):[/green]
• CoinGecko public API for market data
• CryptoPanic free tier for news
• Paper trading for risk-free testing

[yellow]Premium Tier (API keys recommended):[/yellow]
• CoinGecko Pro API key: Enhanced rate limits and data
• Binance API key: Real-time trading and market data
• Twitter/Reddit APIs: Advanced sentiment analysis

[red]Enterprise Tier (All API keys):[/red]
• All premium providers
• Hyperliquid for advanced perpetual futures
• Custom exchange connections
• 24/7 monitoring and alerts

[dim]Get API keys at:[/dim]
• CoinGecko: https://coingecko.com/en/api
• Binance: https://binance.com/en/binance-api
• OpenAI: https://platform.openai.com/api-keys
"""
    else:
        return """
[bold]Equity Provider Setup Recommendations:[/bold]

[green]Free Tier (No API keys required):[/green]
• Yahoo Finance for market data
• Alpha Vantage free tier
• Paper trading for testing

[yellow]Premium Tier (API keys recommended):[/yellow]
• Finnhub API key: Professional market data
• Alpha Vantage Pro: Enhanced data access
• Social media APIs for sentiment

[red]Enterprise Tier (All API keys):[/red]
• All premium providers
• Real-time market data feeds
• Professional broker connections
• Advanced analytics

[dim]Get API keys at:[/dim]
• Finnhub: https://finnhub.io
• Alpha Vantage: https://alphavantage.co
• OpenAI: https://platform.openai.com/api-keys
"""


def create_config_template(asset_class: str, provider_preset: str, cost_preset: str) -> dict:
    """Create optimized configuration template."""
    from tradingagents.default_config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.copy()
    
    # Apply basic settings
    config["asset_class"] = asset_class
    config["provider_preset"] = provider_preset
    config["cost_preset"] = cost_preset
    
    # Apply cost preset optimizations
    config = apply_cost_preset_to_config(config, cost_preset, asset_class)
    
    # Asset class specific settings
    # Ensure nested dictionaries exist
    config["features"] = config.get("features", {})
    config["trading"] = config.get("trading", {})
    config["risk"] = config.get("risk", {})
    config["execution"] = config.get("execution", {})
    
    if asset_class == "crypto":
        config["features"]["crypto_support"] = True
        config["trading"]["enable_24_7"] = True
        config["risk"]["enable_funding_analysis"] = True
        config["risk"]["enable_liquidation_monitoring"] = True
        
        if provider_preset == "enterprise":
            config["execution"]["enable_advanced_orders"] = True
            config["risk"]["enable_realtime_monitoring"] = True
    else:
        config["features"]["crypto_support"] = False
        config["trading"]["enable_24_7"] = False
        config["trading"]["market_hours_only"] = True
    
    return config
