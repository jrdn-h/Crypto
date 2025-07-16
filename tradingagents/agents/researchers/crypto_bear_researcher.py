"""
Crypto-enhanced bear researcher with tokenomics, regulatory, and on-chain risk analysis.

This module extends the standard bear researcher with crypto-specific risk analysis including
tokenomics risks, regulatory threats, unlock schedule risks, and on-chain warning signals.
"""

from langchain_core.messages import AIMessage
import time
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


def create_crypto_bear_researcher(llm, memory, config=None):
    """Create crypto-enhanced bear researcher with additional risk analysis capabilities."""
    
    def crypto_bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get symbol/company info
        symbol = state.get("company_of_interest", "")
        
        # Check if this is a crypto asset
        asset_class = config.get("asset_class", "equity") if config else "equity"
        
        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Build base prompt
        if asset_class == "crypto":
            prompt = _build_crypto_bear_prompt(
                symbol, market_research_report, sentiment_report, 
                news_report, fundamentals_report, history, 
                current_response, past_memory_str
            )
        else:
            # Use standard equity-focused prompt
            prompt = _build_standard_bear_prompt(
                market_research_report, sentiment_report, 
                news_report, fundamentals_report, history, 
                current_response, past_memory_str
            )

        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return crypto_bear_node


def _build_crypto_bear_prompt(
    symbol: str,
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    current_response: str,
    past_memory_str: str
) -> str:
    """Build crypto-specific bear researcher prompt."""
    
    return f"""You are a Crypto Bear Analyst making the case against investing in {symbol}. Your goal is to present a well-reasoned argument emphasizing cryptocurrency-specific risks, technological challenges, regulatory threats, and negative indicators. Leverage the provided research and crypto-specific risk analysis to highlight potential downsides and counter bullish arguments effectively.

**CRYPTO-SPECIFIC BEAR ANALYSIS FRAMEWORK:**

**⚠️ Tokenomics & Economic Risks:**
- High inflation rates and excessive token issuance
- Massive unlock events and vesting cliffs
- Poor token utility and lack of value accrual
- Insider concentration and centralization risks
- Treasury mismanagement and token dumping
- Unsustainable yield farming and emissions

**🚨 Regulatory & Compliance Threats:**
- Regulatory uncertainty and potential crackdowns
- Securities classification risks and enforcement actions
- Geographic restrictions and exchange delistings
- Compliance failures and legal challenges
- Government bans and capital controls
- Tax implications and reporting requirements

**📉 Technical & Network Risks:**
- Scalability limitations and congestion issues
- Security vulnerabilities and exploit history
- Centralization of mining/validation
- Fork risks and community splits
- Technology obsolescence threats
- Interoperability challenges

**🐋 On-Chain Warning Signals:**
- Whale distribution and selling pressure
- Exchange inflows indicating selling intent
- Declining network activity and adoption
- Validator/miner centralization
- Low staking ratios and governance participation
- Unusual transaction patterns

**💥 Market & Liquidity Risks:**
- Extreme volatility and price manipulation
- Low liquidity and high slippage
- Market maker concentration risks
- Correlation with broader crypto market
- Limited institutional support
- Regulatory-driven delistings

**🌪️ Macro & Industry Headwinds:**
- Bear market cycle positioning
- Rising interest rates reducing risk appetite
- Regulatory tightening globally
- Competition from CBDCs
- Environmental concerns and ESG issues
- Institutional redemptions and outflows

**KEY BEAR ARGUMENTS TO EMPHASIZE:**

1. **Regulatory Guillotine**: Pending regulations could devastate market access
2. **Token Inflation**: Constant selling pressure from high issuance rates
3. **Whale Manipulation**: Large holders can crash prices at will
4. **Technology Risks**: Unproven tech with potential critical failures
5. **Liquidity Evaporation**: Market depth insufficient for large trades
6. **Utility Fiction**: Token not actually needed for protocol function
7. **Hype Bubble**: Valuation disconnected from fundamental reality
8. **Centralization Drift**: Progressive concentration of power/wealth

**COUNTER-BULL STRATEGY:**
- Challenge adoption metrics with concentration analysis
- Question "institutional adoption" with specific counterexamples
- Expose tokenomics flaws in value accrual mechanisms
- Highlight regulatory risks that bulls minimize
- Use on-chain data to show distribution and selling patterns
- Point out technical limitations and scalability problems

**Bull Counterpoints Guidelines:**
- Use specific tokenomics data to refute "scarcity" arguments
- Counter regulatory optimism with concrete compliance risks
- Challenge growth narratives with network activity decline
- Question sustainability of yield mechanisms
- Expose centralization behind "decentralization" claims
- Present your argument conversationally, directly challenging bull points

**CRYPTO-SPECIFIC RISK FACTORS TO HIGHLIGHT:**

**Tokenomics Disasters:**
- Hyperinflation destroying holder value
- Unlock events causing price collapse
- Lack of real utility driving token to zero
- Insider selling destroying retail confidence

**Regulatory Nuclear Options:**
- Classification as security triggering enforcement
- Exchange delistings limiting market access
- Government bans eliminating use cases
- Tax treatment making holding uneconomical

**Technical Time Bombs:**
- Smart contract exploits draining treasuries
- Consensus mechanism failures
- Scalability walls hitting adoption limits
- Competitor technologies making protocol obsolete

**Market Structure Fragility:**
- Whale concentration enabling manipulation
- Thin liquidity causing extreme volatility
- Market maker withdrawal during stress
- Correlation with crypto market destroying diversification

**Available Research Data:**
Market research report: {market_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Crypto fundamentals report: {fundamentals_report}
Conversation history: {history}
Last bull argument: {current_response}
Past lessons learned: {past_memory_str}

**CRITICAL INSTRUCTIONS:**
- Focus on crypto-native risks and failure modes
- Emphasize 24/7 market risks and weekend gaps
- Use tokenomics analysis to highlight economic flaws
- Address regulatory risks with specific enforcement examples
- Leverage on-chain data for concrete warning signals
- Present arguments in a conversational, challenging debate style
- Learn from past mistakes and apply those lessons to strengthen your position

Build a compelling bear case that demonstrates why {symbol} represents a dangerous investment in the volatile and risky crypto landscape."""


def _build_standard_bear_prompt(
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    current_response: str,
    past_memory_str: str
) -> str:
    """Build standard equity-focused bear researcher prompt."""
    
    return f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}

Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past."""


# Crypto-specific risk analysis integration functions
async def _get_tokenomics_risks(symbol: str) -> str:
    """Get tokenomics risk analysis for the symbol."""
    try:
        from ...dataflows.crypto.tokenomics_analyzer import get_tokenomics_analysis
        analysis = await get_tokenomics_analysis(symbol)
        # Extract bearish points from tokenomics analysis
        return f"**TOKENOMICS RISK ANALYSIS:**\n{analysis}"
    except ImportError:
        logger.warning("Tokenomics analyzer not available")
        return "Tokenomics risk analysis not available - crypto modules not loaded"
    except Exception as e:
        logger.error(f"Error getting tokenomics risks: {e}")
        return f"Error analyzing tokenomics risks: {str(e)}"


async def _get_regulatory_risks(symbol: str) -> str:
    """Get regulatory risk analysis for the symbol.""" 
    try:
        from ...dataflows.crypto.regulatory_analyzer import get_regulatory_analysis
        analysis = await get_regulatory_analysis(symbol)
        return f"**REGULATORY RISK ANALYSIS:**\n{analysis}"
    except ImportError:
        logger.warning("Regulatory analyzer not available")
        return "Regulatory risk analysis not available - crypto modules not loaded"
    except Exception as e:
        logger.error(f"Error getting regulatory risks: {e}")
        return f"Error analyzing regulatory risks: {str(e)}"


async def _get_whale_distribution_risks(symbol: str) -> str:
    """Get whale distribution and selling risk analysis."""
    try:
        from ...dataflows.crypto.whale_flow_tracker import WhaleFlowTracker
        tracker = WhaleFlowTracker()
        analysis = await tracker.get_whale_flow_summary(symbol, '7d')  # Longer timeframe for patterns
        return f"**WHALE DISTRIBUTION RISK ANALYSIS:**\n{analysis}"
    except ImportError:
        logger.warning("Whale flow tracker not available")
        return "Whale distribution analysis not available - crypto modules not loaded"
    except Exception as e:
        logger.error(f"Error getting whale distribution risks: {e}")
        return f"Error analyzing whale distribution: {str(e)}"


# Enhanced crypto bear researcher with integrated risk analysis
def create_enhanced_crypto_bear_researcher(llm, memory, config=None):
    """Create enhanced crypto bear researcher with integrated crypto risk analysis."""
    
    async def enhanced_crypto_bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        symbol = state.get("company_of_interest", "")
        asset_class = config.get("asset_class", "equity") if config else "equity"
        
        # Enhanced risk analysis for crypto assets
        additional_risk_analysis = ""
        if asset_class == "crypto":
            try:
                # Get comprehensive crypto risk analysis
                tokenomics_risks = await _get_tokenomics_risks(symbol)
                regulatory_risks = await _get_regulatory_risks(symbol)
                whale_risks = await _get_whale_distribution_risks(symbol)
                
                additional_risk_analysis = f"""

**ENHANCED CRYPTO RISK ANALYSIS DATA:**

{tokenomics_risks}

{regulatory_risks}

{whale_risks}

**INSTRUCTION**: Use this enhanced crypto risk analysis to strengthen your bear arguments with specific tokenomics flaws, regulatory threats, and on-chain warning signals."""
                
            except Exception as e:
                logger.warning(f"Error getting enhanced crypto risk analysis: {e}")
                additional_risk_analysis = "\n**Note**: Enhanced crypto risk analysis temporarily unavailable."

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}{additional_risk_analysis}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Build comprehensive prompt
        if asset_class == "crypto":
            prompt = _build_crypto_bear_prompt(
                symbol, market_research_report, sentiment_report, 
                news_report, fundamentals_report + additional_risk_analysis, 
                history, current_response, past_memory_str
            )
        else:
            prompt = _build_standard_bear_prompt(
                market_research_report, sentiment_report, 
                news_report, fundamentals_report, history, 
                current_response, past_memory_str
            )

        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    # Return sync wrapper for async function
    def sync_wrapper(state):
        return asyncio.run(enhanced_crypto_bear_node(state))
    
    return sync_wrapper 