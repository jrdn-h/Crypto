"""
Configuration Manager for TradingAgents CLI

Provides comprehensive configuration management including:
- Configuration file templates
- Validation and error checking  
- Environment setup and API key management
- Provider and cost preset configuration
- Asset class specific settings
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfigTemplate:
    """Configuration template with validation."""
    asset_class: str = "equity"
    provider_preset: str = "free"
    cost_preset: str = "balanced"
    
    # LLM Configuration
    llm_provider: str = "OpenAI"
    shallow_thinker: str = "gpt-4o-mini"
    deep_thinker: str = "gpt-4o"
    backend_url: str = "https://api.openai.com/v1"
    
    # Analysis Configuration
    max_debate_rounds: int = 3
    max_risk_discuss_rounds: int = 3
    default_analysts: List[str] = None
    
    # Trading Configuration
    enable_24_7: bool = False
    enable_crypto_support: bool = False
    enable_advanced_orders: bool = False
    
    # Risk Management
    enable_funding_analysis: bool = False
    enable_liquidation_monitoring: bool = False
    enable_realtime_monitoring: bool = False
    
    # Output Configuration
    results_dir: str = "./results"
    debug: bool = False
    
    def __post_init__(self):
        if self.default_analysts is None:
            self.default_analysts = ["market", "news", "fundamentals"]


class ConfigValidator:
    """Validates TradingAgents configuration."""
    
    REQUIRED_FIELDS = [
        "asset_class", "provider_preset", "cost_preset", 
        "llm_provider", "shallow_thinker", "deep_thinker"
    ]
    
    VALID_ASSET_CLASSES = ["equity", "crypto"]
    VALID_PROVIDER_PRESETS = ["free", "premium", "enterprise"]
    VALID_COST_PRESETS = ["cheap", "balanced", "premium"]
    VALID_LLM_PROVIDERS = ["OpenAI", "Anthropic", "Azure"]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate configuration and return errors/warnings."""
        errors = []
        warnings = []
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate asset class
        if config.get("asset_class") not in cls.VALID_ASSET_CLASSES:
            errors.append(f"Invalid asset_class. Must be one of: {cls.VALID_ASSET_CLASSES}")
        
        # Validate provider preset
        if config.get("provider_preset") not in cls.VALID_PROVIDER_PRESETS:
            errors.append(f"Invalid provider_preset. Must be one of: {cls.VALID_PROVIDER_PRESETS}")
        
        # Validate cost preset
        if config.get("cost_preset") not in cls.VALID_COST_PRESETS:
            errors.append(f"Invalid cost_preset. Must be one of: {cls.VALID_COST_PRESETS}")
        
        # Validate LLM provider
        if config.get("llm_provider") not in cls.VALID_LLM_PROVIDERS:
            warnings.append(f"Unusual LLM provider: {config.get('llm_provider')}. Supported: {cls.VALID_LLM_PROVIDERS}")
        
        # Validate crypto-specific settings
        if config.get("asset_class") == "crypto":
            if not config.get("enable_crypto_support"):
                warnings.append("Crypto asset class selected but crypto support not enabled")
            
            if config.get("cost_preset") == "premium" and config.get("provider_preset") == "free":
                warnings.append("Premium cost preset with free provider may have limited functionality")
        
        # Validate directories
        results_dir = config.get("results_dir", "./results")
        try:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create results directory {results_dir}: {e}")
        
        return {"errors": errors, "warnings": warnings}
    
    @classmethod
    def validate_api_keys(cls, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate required API keys based on configuration."""
        errors = []
        warnings = []
        
        asset_class = config.get("asset_class", "equity")
        provider_preset = config.get("provider_preset", "free")
        
        # Always need OpenAI API key for LLM
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY environment variable not set")
        
        # Asset class specific API keys
        if asset_class == "crypto":
            if provider_preset in ["premium", "enterprise"]:
                if not os.getenv("COINGECKO_API_KEY"):
                    warnings.append("COINGECKO_API_KEY not set - using free tier with rate limits")
                
                if not os.getenv("BINANCE_API_KEY"):
                    warnings.append("BINANCE_API_KEY not set - trading functionality limited")
        
        elif asset_class == "equity":
            if provider_preset in ["premium", "enterprise"]:
                if not os.getenv("FINNHUB_API_KEY"):
                    warnings.append("FINNHUB_API_KEY not set - using free data sources")
                
                if not os.getenv("ALPHA_VANTAGE_API_KEY"):
                    warnings.append("ALPHA_VANTAGE_API_KEY not set - limited market data")
        
        return {"errors": errors, "warnings": warnings}


class ConfigManager:
    """Manages TradingAgents configuration files and settings."""
    
    DEFAULT_CONFIG_PATH = Path.home() / ".tradingagents" / "config.json"
    CONFIG_DIR = Path.home() / ".tradingagents"
    
    def __init__(self):
        # Ensure config directory exists
        self.CONFIG_DIR.mkdir(exist_ok=True)
    
    def create_default_config(self, asset_class: str = "equity") -> Dict[str, Any]:
        """Create default configuration for asset class."""
        template = ConfigTemplate(asset_class=asset_class)
        
        # Apply asset class specific defaults
        if asset_class == "crypto":
            template.enable_crypto_support = True
            template.enable_24_7 = True
            template.enable_funding_analysis = True
            template.cost_preset = "cheap"  # Default to cheaper models for crypto
        
        return asdict(template)
    
    def save_config(self, config: Dict[str, Any], config_path: Optional[Path] = None) -> bool:
        """Save configuration to file."""
        try:
            config_path = config_path or self.DEFAULT_CONFIG_PATH
            
            # Add metadata
            config["_metadata"] = {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "tool": "tradingagents-cli"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, config_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            config_path = config_path or self.DEFAULT_CONFIG_PATH
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate loaded config
            validation = ConfigValidator.validate_config(config)
            if validation["errors"]:
                logger.error(f"Configuration validation errors: {validation['errors']}")
                return None
            
            if validation["warnings"]:
                logger.warning(f"Configuration warnings: {validation['warnings']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get current configuration status."""
        config_exists = self.DEFAULT_CONFIG_PATH.exists()
        
        if config_exists:
            config = self.load_config()
            if config:
                validation = ConfigValidator.validate_config(config)
                api_validation = ConfigValidator.validate_api_keys(config)
                
                return {
                    "exists": True,
                    "valid": len(validation["errors"]) == 0,
                    "asset_class": config.get("asset_class"),
                    "provider_preset": config.get("provider_preset"),
                    "cost_preset": config.get("cost_preset"),
                    "validation_errors": validation["errors"],
                    "validation_warnings": validation["warnings"],
                    "api_errors": api_validation["errors"],
                    "api_warnings": api_validation["warnings"],
                    "last_modified": datetime.fromtimestamp(
                        self.DEFAULT_CONFIG_PATH.stat().st_mtime
                    ).isoformat()
                }
        
        return {
            "exists": False,
            "valid": False,
            "message": "No configuration file found"
        }
    
    def create_config_template(self, template_path: Path, asset_class: str = "equity") -> bool:
        """Create a configuration template file."""
        try:
            config = self.create_default_config(asset_class)
            
            # Add helpful comments as a separate metadata section
            config["_help"] = {
                "asset_class": "Choose 'equity' for stocks or 'crypto' for cryptocurrencies",
                "provider_preset": "free=basic providers, premium=enhanced access, enterprise=full features",
                "cost_preset": "cheap=fast models, balanced=mix of speed/quality, premium=best models",
                "llm_provider": "Currently supports OpenAI, Anthropic, Azure",
                "enable_crypto_support": "Enable crypto-specific features (24/7 trading, funding analysis)",
                "results_dir": "Directory where analysis results will be saved"
            }
            
            with open(template_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return False
    
    def migrate_config(self, old_config_path: Path) -> bool:
        """Migrate configuration from older format."""
        try:
            # Load old config
            with open(old_config_path, 'r') as f:
                old_config = json.load(f)
            
            # Create new config with defaults
            new_config = self.create_default_config()
            
            # Migrate known fields
            field_mappings = {
                "asset_class": "asset_class",
                "debug": "debug",
                "results_dir": "results_dir",
                # Add more mappings as needed
            }
            
            for old_field, new_field in field_mappings.items():
                if old_field in old_config:
                    new_config[new_field] = old_config[old_field]
            
            # Save migrated config
            return self.save_config(new_config)
            
        except Exception as e:
            logger.error(f"Failed to migrate configuration: {e}")
            return False
    
    def backup_config(self) -> Optional[Path]:
        """Create a backup of current configuration."""
        try:
            if not self.DEFAULT_CONFIG_PATH.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.CONFIG_DIR / f"config_backup_{timestamp}.json"
            
            import shutil
            shutil.copy2(self.DEFAULT_CONFIG_PATH, backup_path)
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            return None
    
    def reset_config(self, asset_class: str = "equity") -> bool:
        """Reset configuration to defaults."""
        try:
            # Backup existing config if it exists
            if self.DEFAULT_CONFIG_PATH.exists():
                backup_path = self.backup_config()
                if backup_path:
                    logger.info(f"Backed up existing config to {backup_path}")
            
            # Create new default config
            config = self.create_default_config(asset_class)
            return self.save_config(config)
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for troubleshooting."""
        import platform
        import sys
        
        api_keys = {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "COINGECKO_API_KEY": bool(os.getenv("COINGECKO_API_KEY")),
            "BINANCE_API_KEY": bool(os.getenv("BINANCE_API_KEY")),
            "FINNHUB_API_KEY": bool(os.getenv("FINNHUB_API_KEY")),
            "ALPHA_VANTAGE_API_KEY": bool(os.getenv("ALPHA_VANTAGE_API_KEY")),
        }
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "config_dir": str(self.CONFIG_DIR),
            "config_exists": self.DEFAULT_CONFIG_PATH.exists(),
            "api_keys_configured": api_keys,
            "total_api_keys": sum(api_keys.values()),
        }


def create_example_configs():
    """Create example configuration files for different use cases."""
    config_manager = ConfigManager()
    examples_dir = config_manager.CONFIG_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Crypto trader configuration
    crypto_config = config_manager.create_default_config("crypto")
    crypto_config.update({
        "provider_preset": "premium",
        "cost_preset": "balanced",
        "enable_funding_analysis": True,
        "enable_liquidation_monitoring": True,
        "enable_realtime_monitoring": True,
    })
    
    # Equity analyst configuration  
    equity_config = config_manager.create_default_config("equity")
    equity_config.update({
        "provider_preset": "premium",
        "cost_preset": "premium",
        "max_debate_rounds": 5,
        "max_risk_discuss_rounds": 4,
    })
    
    # Save examples
    examples = {
        "crypto_trader.json": crypto_config,
        "equity_analyst.json": equity_config,
    }
    
    for filename, config in examples.items():
        config_path = examples_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    return examples_dir 