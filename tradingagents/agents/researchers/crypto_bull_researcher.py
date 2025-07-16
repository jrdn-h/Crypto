"""
Crypto-enhanced bull researcher with tokenomics, regulatory, and on-chain analysis.

This module extends the standard bull researcher with crypto-specific analysis including
tokenomics, regulatory environment, unlock schedules, and on-chain metrics.
"""

from langchain_core.messages import AIMessage
import time
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


def create_crypto_bull_researcher(llm, memory, config=None):
    """Create crypto-enhanced bull researcher with additional analysis capabilities."""
    
    def crypto_bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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
            prompt = _build_crypto_bull_prompt(
                symbol, market_research_report, sentiment_report, 
                news_report, fundamentals_report, history, 
                current_response, past_memory_str
            )
        else:
            # Use standard equity-focused prompt
            prompt = _build_standard_bull_prompt(
                market_research_report, sentiment_report, 
                news_report, fundamentals_report, history, 
                current_response, past_memory_str
            )

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return crypto_bull_node


def _build_crypto_bull_prompt(
    symbol: str,
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    current_response: str,
    past_memory_str: str
) -> str:
    """Build crypto-specific bull researcher prompt."""
    
    return f"""You are a Crypto Bull Analyst advocating for investing in {symbol}. Your task is to build a strong, evidence-based case emphasizing growth potential, technological advantages, adoption trends, and positive market indicators specific to cryptocurrency markets. Leverage the provided research and crypto-specific analysis to address concerns and counter bearish arguments effectively.

**CRYPTO-SPECIFIC BULL ANALYSIS FRAMEWORK:**

**🚀 Growth & Adoption Potential:**
- Network growth metrics (active addresses, transaction volume, developer activity)
- Institutional adoption and mainstream integration trends
- Real-world utility and use case expansion
- Ecosystem development and partnerships
- Total Addressable Market (TAM) in crypto/DeFi space

**🏗️ Technological & Competitive Advantages:**
- Protocol innovation and technical superiority
- Scalability solutions and performance metrics
- Security track record and audit results
- Interoperability and cross-chain capabilities
- Developer ecosystem and community strength

**💰 Tokenomics & Economic Model:**
- Deflationary mechanisms and token burns
- Staking rewards and yield opportunities
- Supply scarcity and emission schedules
- Value accrual mechanisms and fee capture
- Treasury management and protocol revenue

**⚖️ Regulatory & Compliance Positioning:**
- Regulatory clarity and compliance status
- Proactive regulatory engagement
- Favorable legal precedents
- Geographic diversification benefits
- Compliance-first approach advantages

**📊 On-Chain & Market Indicators:**
- Whale accumulation patterns
- Exchange outflows suggesting long-term holding
- Network health and decentralization metrics
- Staking ratios and governance participation
- Market sentiment and social momentum

**🌍 Macro & Industry Trends:**
- Crypto market cycle positioning
- Institutional investment flows
- Traditional finance integration (ETFs, custody)
- Central Bank Digital Currency (CBDC) implications
- Global monetary policy impacts

**KEY BULL ARGUMENTS TO EMPHASIZE:**

1. **Network Effects**: Growing user base creates exponential value
2. **First-Mover Advantage**: Established position in emerging markets
3. **Institutional Validation**: Major corporations and funds investing
4. **Technological Moats**: Unique capabilities competitors can't replicate
5. **Scarcity Premium**: Limited supply with growing demand
6. **Yield Generation**: Staking/DeFi opportunities provide income
7. **Global Accessibility**: 24/7 markets and cross-border utility
8. **Inflation Hedge**: Store of value properties in uncertain times

**COUNTER-BEAR STRATEGY:**
- Address regulatory FUD with concrete compliance examples
- Counter volatility concerns with long-term adoption trends
- Refute "bubble" claims with fundamental utility and cash flows
- Challenge centralization claims with decentralization metrics
- Counter environmental concerns with sustainability initiatives

**Bear Counterpoints Guidelines:**
- Use specific on-chain data and metrics to refute bear claims
- Highlight recent positive developments in regulation, adoption, or technology
- Compare {symbol} favorably to other crypto assets and traditional investments
- Address concerns with concrete examples and case studies
- Present your argument conversationally, engaging directly with bear points

**Available Research Data:**
Market research report: {market_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Crypto fundamentals report: {fundamentals_report}
Conversation history: {history}
Last bear argument: {current_response}
Past lessons learned: {past_memory_str}

**CRITICAL INSTRUCTIONS:**
- Focus on crypto-native metrics and analysis frameworks
- Emphasize 24/7 market dynamics and global accessibility
- Use tokenomics analysis to support value appreciation thesis
- Address regulatory risks head-on with compliance positioning
- Leverage on-chain data for concrete evidence
- Present arguments in a conversational, engaging debate style
- Learn from past mistakes and apply those lessons to strengthen your position

Build a compelling bull case that demonstrates why {symbol} represents a strong investment opportunity in the evolving crypto landscape."""

def _build_standard_bull_prompt(
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    current_response: str,
    past_memory_str: str
) -> str:
    """Build standard equity-focused bull researcher prompt."""
    
    return f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {market_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}

Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past."""


# Crypto-specific analysis integration functions
async def _get_tokenomics_analysis(symbol: str) -> str:
    """Get tokenomics analysis for the symbol."""
    try:
        from ...dataflows.crypto.tokenomics_analyzer import get_tokenomics_analysis
        return await get_tokenomics_analysis(symbol)
    except ImportError:
        logger.warning("Tokenomics analyzer not available")
        return "Tokenomics analysis not available - crypto modules not loaded"
    except Exception as e:
        logger.error(f"Error getting tokenomics analysis: {e}")
        return f"Error analyzing tokenomics: {str(e)}"


async def _get_regulatory_analysis(symbol: str) -> str:
    """Get regulatory analysis for the symbol.""" 
    try:
        from ...dataflows.crypto.regulatory_analyzer import get_regulatory_analysis
        return await get_regulatory_analysis(symbol)
    except ImportError:
        logger.warning("Regulatory analyzer not available")
        return "Regulatory analysis not available - crypto modules not loaded"
    except Exception as e:
        logger.error(f"Error getting regulatory analysis: {e}")
        return f"Error analyzing regulatory environment: {str(e)}"


async def _get_whale_flow_analysis(symbol: str) -> str:
    """Get whale flow analysis for the symbol."""
    try:
        from ...dataflows.crypto.whale_flow_tracker import WhaleFlowTracker
        tracker = WhaleFlowTracker()
        return await tracker.get_whale_flow_summary(symbol, '24h')
    except ImportError:
        logger.warning("Whale flow tracker not available")
        return "Whale flow analysis not available - crypto modules not loaded"
    except Exception as e:
        logger.error(f"Error getting whale flow analysis: {e}")
        return f"Error analyzing whale flows: {str(e)}"


# Enhanced crypto bull researcher with integrated analysis
def create_enhanced_crypto_bull_researcher(llm, memory, config=None):
    """Create enhanced crypto bull researcher with integrated crypto analysis."""
    
    async def enhanced_crypto_bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        symbol = state.get("company_of_interest", "")
        asset_class = config.get("asset_class", "equity") if config else "equity"
        
        # Enhanced analysis for crypto assets
        additional_analysis = ""
        if asset_class == "crypto":
            try:
                # Get comprehensive crypto analysis
                tokenomics_analysis = await _get_tokenomics_analysis(symbol)
                regulatory_analysis = await _get_regulatory_analysis(symbol)
                whale_analysis = await _get_whale_flow_analysis(symbol)
                
                additional_analysis = f"""

**ENHANCED CRYPTO ANALYSIS DATA:**

{tokenomics_analysis}

{regulatory_analysis}

{whale_analysis}

**INSTRUCTION**: Use this enhanced crypto analysis to strengthen your bull arguments with specific tokenomics, regulatory, and on-chain data points."""
                
            except Exception as e:
                logger.warning(f"Error getting enhanced crypto analysis: {e}")
                additional_analysis = "\n**Note**: Enhanced crypto analysis temporarily unavailable."

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}{additional_analysis}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Build comprehensive prompt
        if asset_class == "crypto":
            prompt = _build_crypto_bull_prompt(
                symbol, market_research_report, sentiment_report, 
                news_report, fundamentals_report + additional_analysis, 
                history, current_response, past_memory_str
            )
        else:
            prompt = _build_standard_bull_prompt(
                market_research_report, sentiment_report, 
                news_report, fundamentals_report, history, 
                current_response, past_memory_str
            )

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    # Return sync wrapper for async function
    def sync_wrapper(state):
        return asyncio.run(enhanced_crypto_bull_node(state))
    
    return sync_wrapper 