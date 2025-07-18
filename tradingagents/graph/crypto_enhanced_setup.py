"""
Crypto-enhanced graph setup that uses crypto-specific researchers for crypto assets.

This module extends the standard graph setup to use crypto-enhanced bull and bear
researchers when the asset class is set to 'crypto'.
"""

import logging
from typing import Dict, List
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from ..agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_social_media_analyst,
    create_research_manager,
    create_trader,
    create_risky_debator,
    create_safe_debator,
    create_neutral_debator,
    create_risk_manager,
    create_msg_delete,
    AgentState,
    
    # Standard researchers
    create_bear_researcher,
    create_bull_researcher,
    
    # Crypto-enhanced researchers
    create_enhanced_crypto_bear_researcher,
    create_enhanced_crypto_bull_researcher
)

logger = logging.getLogger(__name__)


class CryptoEnhancedGraphSetup:
    """Enhanced graph setup with crypto-specific researcher support."""
    
    def __init__(
        self,
        quick_thinking_llm,
        deep_thinking_llm,
        toolkit,
        tool_nodes,
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic,
        config: Dict = None
    ):
        """Initialize crypto-enhanced graph setup."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.config = config or {}
    
    def setup_crypto_enhanced_graph(
        self, 
        selected_analysts: List[str] = ["market", "social", "news", "fundamentals"]
    ):
        """Set up crypto-enhanced agent workflow graph."""
        
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")
        
        # Create analyst nodes (same as standard setup)
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher nodes - enhanced for crypto
        asset_class = self.config.get("asset_class", "equity")
        
        if asset_class == "crypto":
            logger.info("Using crypto-enhanced researchers for crypto asset analysis")
            bull_researcher_node = create_enhanced_crypto_bull_researcher(
                self.quick_thinking_llm, self.bull_memory, self.config
            )
            bear_researcher_node = create_enhanced_crypto_bear_researcher(
                self.quick_thinking_llm, self.bear_memory, self.config
            )
        else:
            logger.info("Using standard researchers for equity asset analysis")
            bull_researcher_node = create_bull_researcher(
                self.quick_thinking_llm, self.bull_memory
            )
            bear_researcher_node = create_bear_researcher(
                self.quick_thinking_llm, self.bear_memory
            )
        
        # Create other nodes (same as standard)
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst = create_safe_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add researcher and other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges (same as standard setup)
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        return workflow.compile()


def create_crypto_enhanced_graph_setup(
    quick_thinking_llm,
    deep_thinking_llm,
    toolkit,
    tool_nodes,
    bull_memory,
    bear_memory,
    trader_memory,
    invest_judge_memory,
    risk_manager_memory,
    conditional_logic,
    config: Dict = None
):
    """Create crypto-enhanced graph setup instance."""
    return CryptoEnhancedGraphSetup(
        quick_thinking_llm=quick_thinking_llm,
        deep_thinking_llm=deep_thinking_llm,
        toolkit=toolkit,
        tool_nodes=tool_nodes,
        bull_memory=bull_memory,
        bear_memory=bear_memory,
        trader_memory=trader_memory,
        invest_judge_memory=invest_judge_memory,
        risk_manager_memory=risk_manager_memory,
        conditional_logic=conditional_logic,
        config=config
    ) 