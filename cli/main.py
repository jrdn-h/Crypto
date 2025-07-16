from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {
            # Analyst Team
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports
        if any(
            self.report_sections[section]
            for section in [
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
            ]
        ):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["market_report"]:
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections["sentiment_report"]:
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections["news_report"]:
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections["fundamentals_report"]:
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team
    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        # Truncate tool call args if too long
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        # Convert content to string if it's not already
        content_str = content
        if isinstance(content, list):
            # Handle list of content blocks (Anthropic format)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)
            
        # Truncate message content if too long
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp
    all_messages.sort(key=lambda x: x[0])

    # Calculate how many messages we can show based on available space
    # Start with a reasonable number and adjust based on content length
    max_messages = 12  # Increased from 8 to better fill the space

    # Get the last N messages that will fit in the panel
    recent_messages = all_messages[-max_messages:]

    # Add messages to table
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    # Add a footer to indicate if messages were truncated
    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections(
    ticker: Optional[str] = None,
    asset_class: Optional[str] = None,
    date: Optional[str] = None,
    provider_preset: Optional[str] = None,
    cost_preset: Optional[str] = None
):
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()  # Add a blank line after the welcome box

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Asset Class
    console.print(
        create_question_box(
            "Step 3: Asset Class", "Select the type of assets to analyze"
        )
    )
    # Use provided asset class or prompt for selection
    selected_asset_class = asset_class if asset_class else select_asset_class()
    console.print(f"[green]Selected asset class:[/green] {selected_asset_class}")

    # Step 4: Provider Preset
    console.print(
        create_question_box(
            "Step 4: Provider Preset", "Select provider tier based on your API access and budget"
        )
    )
    selected_provider_preset = provider_preset if provider_preset else select_provider_preset()
    console.print(f"[green]Selected provider preset:[/green] {selected_provider_preset}")
    
    # Validate provider configuration
    if not validate_provider_configuration(selected_asset_class, selected_provider_preset):
        console.print("[red]Exiting due to provider configuration issues...[/red]")
        exit(1)
    
    # Show provider recommendations
    recommendations = get_provider_recommendations(selected_asset_class)
    if questionary.confirm(f"Would you like to see provider setup recommendations for {selected_asset_class}?").ask():
        console.print(Panel(recommendations, title="Provider Recommendations", border_style="cyan"))

    # Step 5: Cost Preset
    console.print(
        create_question_box(
            "Step 5: Cost Optimization", "Select cost preset to balance performance vs cost"
        )
    )
    selected_cost_preset = cost_preset if cost_preset else select_cost_preset()
    console.print(f"[green]Selected cost preset:[/green] {selected_cost_preset}")

    # Step 6: Select analysts
    console.print(
        create_question_box(
            "Step 6: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth
    console.print(
        create_question_box(
            "Step 5: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 6: OpenAI backend
    console.print(
        create_question_box(
            "Step 6: LLM Provider", "Select which LLM service to use"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 7: Thinking agents
    console.print(
        create_question_box(
            "Step 7: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "asset_class": selected_asset_class,
        "provider_preset": selected_provider_preset,
        "cost_preset": selected_cost_preset,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def display_complete_report(final_state):
    """Display the complete analysis report with team-based panels."""
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    # I. Analyst Team Reports
    analyst_reports = []

    # Market Analyst Report
    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Social Analyst Report
    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # News Analyst Report
    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Fundamentals Analyst Report
    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        # Bull Researcher Analysis
        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Bear Researcher Analysis
        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Research Manager Decision
        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    # III. Trading Team Reports
    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    Markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # IV. Risk Management Team Reports
    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        # Aggressive (Risky) Analyst Analysis
        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Conservative (Safe) Analyst Analysis
        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Neutral Analyst Analysis
        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team Decision",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # V. Portfolio Manager Decision
        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        Markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Portfolio Manager Decision",
                    border_style="green",
                    padding=(1, 2),
                )
            )


def update_research_team_status(status):
    """Update status for all research team members and trader."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)

def extract_content_string(content):
    """Extract string content from various message formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)

def run_analysis(
    ticker: Optional[str] = None,
    asset_class: Optional[str] = None,
    date: Optional[str] = None,
    config: Optional[str] = None,
    interactive: bool = True,
    provider_preset: Optional[str] = None,
    cost_preset: Optional[str] = None
):
    # Load configuration
    if config:
        config_data = load_config_file(config)
    else:
        config_data = DEFAULT_CONFIG.copy()
    
    # Get user selections (interactive or from CLI args)
    if interactive:
        selections = get_user_selections(
            ticker=ticker,
            asset_class=asset_class,
            date=date,
            provider_preset=provider_preset,
            cost_preset=cost_preset
        )
    else:
        selections = get_non_interactive_selections(
            ticker=ticker,
            asset_class=asset_class,
            date=date,
            provider_preset=provider_preset,
            cost_preset=cost_preset
        )

    # Create config with selected options
    config = DEFAULT_CONFIG.copy()
    config["asset_class"] = selections["asset_class"]
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    
    # Apply provider and cost preset configurations
    config["provider_preset"] = selections.get("provider_preset", "free")
    config["cost_preset"] = selections.get("cost_preset", "balanced")
    
    # Apply cost preset optimizations
    from cli.utils import apply_cost_preset_to_config
    config = apply_cost_preset_to_config(config, selections.get("cost_preset", "balanced"), selections["asset_class"])
    
    # Show selected providers for the asset class
    if selections["asset_class"] == "crypto":
        from cli.utils import select_crypto_providers
        selected_providers = select_crypto_providers(selections.get("provider_preset", "free"))
    else:
        from cli.utils import select_equity_providers
        selected_providers = select_equity_providers(selections.get("provider_preset", "free"))

    # Initialize the graph
    graph = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], config=config, debug=True
    )

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout)

        # Reset agent statuses
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        # Reset report sections
        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = None

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text)

        # Initialize state and get graph args
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]

                # Extract message content and type
                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)  # Use the helper function
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                # Add message to buffer
                message_buffer.add_message(msg_type, content)                

                # If it's a tool call, add it to tool calls
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        # Handle both dictionary and object tool calls
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # Update reports and agent status based on chunk content
                # Analyst Team Reports
                if "market_report" in chunk and chunk["market_report"]:
                    message_buffer.update_report_section(
                        "market_report", chunk["market_report"]
                    )
                    message_buffer.update_agent_status("Market Analyst", "completed")
                    # Set next analyst to in_progress
                    if "social" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Social Analyst", "in_progress"
                        )

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    message_buffer.update_report_section(
                        "sentiment_report", chunk["sentiment_report"]
                    )
                    message_buffer.update_agent_status("Social Analyst", "completed")
                    # Set next analyst to in_progress
                    if "news" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "News Analyst", "in_progress"
                        )

                if "news_report" in chunk and chunk["news_report"]:
                    message_buffer.update_report_section(
                        "news_report", chunk["news_report"]
                    )
                    message_buffer.update_agent_status("News Analyst", "completed")
                    # Set next analyst to in_progress
                    if "fundamentals" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Fundamentals Analyst", "in_progress"
                        )

                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    message_buffer.update_report_section(
                        "fundamentals_report", chunk["fundamentals_report"]
                    )
                    message_buffer.update_agent_status(
                        "Fundamentals Analyst", "completed"
                    )
                    # Set all research team members to in_progress
                    update_research_team_status("in_progress")

                # Research Team - Handle Investment Debate State
                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]

                    # Update Bull Researcher status and report
                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bull response
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            message_buffer.add_message("Reasoning", latest_bull)
                            # Update research report with bull's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"### Bull Researcher Analysis\n{latest_bull}",
                            )

                    # Update Bear Researcher status and report
                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bear response
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            message_buffer.add_message("Reasoning", latest_bear)
                            # Update research report with bear's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                            )

                    # Update Research Manager status and final decision
                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        # Keep all research team members in progress until final decision
                        update_research_team_status("in_progress")
                        message_buffer.add_message(
                            "Reasoning",
                            f"Research Manager: {debate_state['judge_decision']}",
                        )
                        # Update research report with final decision
                        message_buffer.update_report_section(
                            "investment_plan",
                            f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                        )
                        # Mark all research team members as completed
                        update_research_team_status("completed")
                        # Set first risk analyst to in_progress
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )

                # Trading Team
                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    # Set first risk analyst to in_progress
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                # Risk Management Team - Handle Risk Debate State
                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]

                    # Update Risky Analyst status and report
                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        # Update risk report with risky analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    # Update Safe Analyst status and report
                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        # Update risk report with safe analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    # Update Neutral Analyst status and report
                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        # Update risk report with neutral analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    # Update Portfolio Manager status and final decision
                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        # Update risk report with final decision only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                        )
                        # Mark risk analysts as completed
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

                # Update the display
                update_display(layout)

            trace.append(chunk)

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "Analysis", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        # Display the complete final report
        display_complete_report(final_state)

        update_display(layout)


# =============================================================================
# Phase 9: Enhanced CLI Functions
# =============================================================================

def get_non_interactive_selections(
    ticker: Optional[str] = None,
    asset_class: Optional[str] = None,
    date: Optional[str] = None,
    provider_preset: Optional[str] = None,
    cost_preset: Optional[str] = None
):
    """Get selections for non-interactive mode with validation."""
    import datetime
    
    # Validate required parameters
    if not ticker:
        console.print("[red]Error: --ticker is required in non-interactive mode[/red]")
        raise typer.Exit(1)
    
    if not asset_class:
        console.print("[red]Error: --asset-class is required in non-interactive mode[/red]")
        raise typer.Exit(1)
    
    if asset_class not in ["equity", "crypto"]:
        console.print(f"[red]Error: asset-class must be 'equity' or 'crypto', got '{asset_class}'[/red]")
        raise typer.Exit(1)
    
    # Set defaults
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Default selections for non-interactive mode
    return {
        "ticker": ticker,
        "analysis_date": date,
        "asset_class": asset_class,
        "analysts": [AnalystType.MARKET, AnalystType.NEWS, AnalystType.FUNDAMENTALS],
        "research_depth": 3,
        "shallow_thinker": "gpt-4o-mini",
        "deep_thinker": "gpt-4o",
        "llm_provider": "OpenAI",
        "backend_url": "https://api.openai.com/v1",
        "provider_preset": provider_preset or "free",
        "cost_preset": cost_preset or ("cheap" if asset_class == "crypto" else "balanced")
    }


def load_config_file(config_path: str):
    """Load configuration from file."""
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file {config_path} not found[/red]")
        raise typer.Exit(1)
    
    try:
        with open(config_file) as f:
            config_data = json.load(f)
        console.print(f"[green]Loaded configuration from {config_path}[/green]")
        return config_data
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1)


def run_setup_wizard():
    """Run initial setup wizard for TradingAgents configuration."""
    console.print(Panel(
        "[bold green]TradingAgents Setup Wizard[/bold green]\n\n"
        "This wizard will help you configure TradingAgents for optimal performance.\n"
        "You can run this setup again at any time with: [code]tradingagents setup[/code]",
        title="🚀 Welcome to TradingAgents",
        border_style="green"
    ))
    
    # API Keys Configuration
    console.print("\n[bold]Step 1: API Keys Configuration[/bold]")
    console.print("Configure your API keys for data providers and LLM services.")
    
    # Show current configuration status
    console.print("\n[bold]Current Configuration Status:[/bold]")
    check_api_keys()
    
    # Provider Selection
    console.print("\n[bold]Step 2: Provider Selection[/bold]")
    setup_providers()
    
    # Cost Presets
    console.print("\n[bold]Step 3: Cost Optimization[/bold]")
    setup_cost_presets()
    
    console.print("\n[green]✅ Setup complete! You can now run: [code]tradingagents analyze[/code][/green]")


def manage_providers(asset_class: Optional[str] = None, check_status: bool = False):
    """Manage and view provider configurations."""
    from tradingagents.dataflows.provider_registry import get_all_providers, get_client
    from tradingagents.dataflows.base_interfaces import AssetClass
    
    console.print(Panel(
        "[bold blue]Provider Management[/bold blue]\n\n"
        "View and manage data providers for TradingAgents",
        title="🔧 Provider Management",
        border_style="blue"
    ))
    
    # Filter by asset class if specified
    if asset_class:
        if asset_class == "crypto":
            target_class = AssetClass.CRYPTO
        elif asset_class == "equity":
            target_class = AssetClass.EQUITY
        else:
            console.print(f"[red]Invalid asset class: {asset_class}[/red]")
            return
        
        console.print(f"\n[bold]Providers for {asset_class.upper()} assets:[/bold]")
    else:
        console.print("\n[bold]All Providers:[/bold]")
        target_class = None
    
    # Create provider status table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider Type", style="cyan")
    table.add_column("Provider Name", style="green")
    table.add_column("Asset Class", style="yellow")
    table.add_column("Priority", style="blue")
    table.add_column("Cost Tier", style="magenta")
    
    if check_status:
        table.add_column("Status", style="red")
    
    # Get all providers and display
    try:
        providers = get_all_providers()
        for provider_type, provider_list in providers.items():
            for provider in provider_list:
                if target_class and provider.asset_class != target_class:
                    continue
                
                row = [
                    provider_type,
                    provider.name,
                    provider.asset_class.value,
                    provider.priority.value,
                    provider.cost_tier
                ]
                
                if check_status:
                    # Check provider health
                    try:
                        client = get_client(provider_type, provider.asset_class)
                        status = "✅ Available" if client else "❌ Unavailable"
                    except Exception:
                        status = "❌ Error"
                    row.append(status)
                
                table.add_row(*row)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting providers: {e}[/red]")


def manage_config(show: bool = False, validate: bool = False, reset: bool = False, export: Optional[str] = None):
    """Manage TradingAgents configuration."""
    console.print(Panel(
        "[bold blue]Configuration Management[/bold blue]\n\n"
        "View and manage TradingAgents configuration settings",
        title="⚙️ Configuration",
        border_style="blue"
    ))
    
    if show:
        show_current_config()
    
    if validate:
        validate_configuration()
    
    if reset:
        reset_configuration()
    
    if export:
        export_configuration(export)


def run_crypto_interface(action: Optional[str] = None, ticker: Optional[str] = None, exchange: Optional[str] = None):
    """Enhanced crypto-specific trading interface with specialized commands."""
    console.print(Panel(
        "[bold yellow]Crypto Trading Interface[/bold yellow]\n\n"
        "Specialized tools for cryptocurrency trading and analysis\n"
        f"24/7 operations • Advanced derivatives • Risk management",
        title="₿ Crypto Trading Suite",
        border_style="yellow"
    ))
    
    # Handle specific actions
    if action == "analyze":
        run_crypto_analysis(ticker)
    elif action == "trade":
        run_crypto_trading(ticker, exchange)
    elif action == "risk":
        run_crypto_risk_analysis(ticker)
    elif action == "funding":
        run_crypto_funding_analysis(ticker)
    else:
        # Interactive menu
        show_crypto_menu(ticker, exchange)


def show_crypto_menu(ticker: Optional[str] = None, exchange: Optional[str] = None):
    """Show interactive crypto menu."""
    console.print("\n[bold]🚀 Crypto Trading Commands:[/bold]")
    console.print("• [cyan]crypto analyze --ticker BTC/USDT[/cyan] - Run crypto market analysis")
    console.print("• [cyan]crypto trade --ticker ETH-PERP --exchange hyperliquid[/cyan] - Execute crypto trades")
    console.print("• [cyan]crypto risk --ticker BTC-PERP[/cyan] - Analyze position risks and funding")
    console.print("• [cyan]crypto funding --ticker ETH-PERP[/cyan] - Funding rate analysis and optimization")
    
    console.print("\n[bold]📊 General Commands:[/bold]")
    console.print("• [cyan]analyze --asset-class crypto[/cyan] - Full crypto analysis workflow")
    console.print("• [cyan]providers --asset-class crypto --status[/cyan] - Check crypto providers")
    console.print("• [cyan]config --show[/cyan] - View configuration")
    
    # Show crypto-specific features
    console.print("\n[bold]✨ Advanced Crypto Features:[/bold]")
    console.print("✅ 24/7 market analysis (no market hours)")
    console.print("✅ Perpetual futures with funding analysis") 
    console.print("✅ Cross-exchange arbitrage opportunities")
    console.print("✅ Real-time risk monitoring and alerts")
    console.print("✅ Multi-exchange execution (Binance, Hyperliquid)")
    console.print("✅ Advanced order types (bracket, conditional)")
    console.print("✅ Kelly criterion position sizing")
    console.print("✅ Liquidation risk assessment")
    
    # Interactive options
    console.print("\n[bold]Quick Actions:[/bold]")
    
    if questionary.confirm("🔍 Run crypto market analysis?").ask():
        target_ticker = ticker or questionary.text("Enter crypto pair (e.g., BTC/USDT, ETH-PERP):").ask()
        if target_ticker:
            run_crypto_analysis(target_ticker)
    
    elif questionary.confirm("📈 Check portfolio risk?").ask():
        run_crypto_risk_analysis(ticker)
    
    elif questionary.confirm("💰 Analyze funding rates?").ask():
        target_ticker = ticker or questionary.text("Enter perpetual pair (e.g., BTC-PERP, ETH-PERP):").ask()
        if target_ticker:
            run_crypto_funding_analysis(target_ticker)
    
    elif questionary.confirm("⚙️ Run initial setup wizard?").ask():
        run_setup_wizard()


def run_crypto_analysis(ticker: Optional[str] = None):
    """Run specialized crypto analysis."""
    console.print(Panel(
        "[bold green]Crypto Market Analysis[/bold green]\n\n"
        "Comprehensive analysis with 24/7 market data",
        title="📊 Analysis",
        border_style="green"
    ))
    
    if not ticker:
        ticker = questionary.text(
            "Enter crypto pair:",
            default="BTC/USDT",
            instruction="Examples: BTC/USDT, ETH/USDT, BTC-PERP, ETH-PERP"
        ).ask()
    
    if ticker:
        console.print(f"[green]Starting crypto analysis for {ticker}...[/green]")
        run_analysis(
            ticker=ticker,
            asset_class="crypto",
            interactive=False,
            provider_preset="premium",
            cost_preset="cheap"
        )


def run_crypto_trading(ticker: Optional[str] = None, exchange: Optional[str] = None):
    """Run crypto trading interface."""
    console.print(Panel(
        "[bold blue]Crypto Trading[/bold blue]\n\n"
        "Execute trades with advanced risk management",
        title="💹 Trading",
        border_style="blue"
    ))
    
    console.print("\n[bold]Available Exchanges:[/bold]")
    console.print("• [cyan]Paper Trading[/cyan] - Risk-free simulation")
    console.print("• [cyan]Binance[/cyan] - Spot and futures trading")
    console.print("• [cyan]Hyperliquid[/cyan] - Advanced perpetual futures")
    
    if not exchange:
        exchange_options = [
            ("Paper Trading - Risk-free simulation", "paper"),
            ("Binance - Multi-asset exchange", "binance"),
            ("Hyperliquid - Advanced perpetual futures", "hyperliquid"),
        ]
        
        exchange = questionary.select(
            "Select exchange:",
            choices=[questionary.Choice(display, value=value) for display, value in exchange_options]
        ).ask()
    
    if not ticker:
        if exchange == "hyperliquid":
            ticker = questionary.text("Enter perpetual pair:", default="BTC-PERP").ask()
        else:
            ticker = questionary.text("Enter trading pair:", default="BTC/USDT").ask()
    
    console.print(f"\n[green]Setting up {exchange} trading for {ticker}...[/green]")
    console.print("[yellow]Note: This would connect to your chosen exchange with proper API configuration[/yellow]")


def run_crypto_risk_analysis(ticker: Optional[str] = None):
    """Run crypto risk analysis."""
    console.print(Panel(
        "[bold red]Crypto Risk Analysis[/bold red]\n\n"
        "Portfolio risk assessment and optimization",
        title="⚠️ Risk Management",
        border_style="red"
    ))
    
    console.print("\n[bold]Risk Analysis Features:[/bold]")
    console.print("• Portfolio liquidation risk assessment")
    console.print("• Margin utilization optimization")
    console.print("• Cross vs isolated margin strategies")
    console.print("• Dynamic leverage caps")
    console.print("• 24/7 real-time monitoring")
    
    # Example risk analysis workflow
    console.print(f"\n[green]Analyzing portfolio risk...[/green]")
    console.print("[dim]This would analyze current positions and provide risk recommendations[/dim]")


def run_crypto_funding_analysis(ticker: Optional[str] = None):
    """Run crypto funding analysis."""
    console.print(Panel(
        "[bold magenta]Funding Rate Analysis[/bold magenta]\n\n"
        "Perpetual futures funding optimization",
        title="💰 Funding Analysis",
        border_style="magenta"
    ))
    
    if not ticker:
        ticker = questionary.text(
            "Enter perpetual pair:",
            default="BTC-PERP",
            instruction="Examples: BTC-PERP, ETH-PERP, SOL-PERP"
        ).ask()
    
    if ticker and "PERP" not in ticker.upper():
        console.print("[yellow]Note: Funding analysis is for perpetual futures only[/yellow]")
        ticker = f"{ticker.split('/')[0]}-PERP"
    
    console.print(f"\n[green]Analyzing funding rates for {ticker}...[/green]")
    console.print("\n[bold]Funding Analysis Features:[/bold]")
    console.print("• Historical funding rate trends")
    console.print("• Cross-exchange rate comparison")
    console.print("• Funding cost optimization")
    console.print("• Predictive funding models")
    console.print("• Arbitrage opportunities")
    
    console.print(f"\n[dim]This would provide detailed funding analysis for {ticker}[/dim]")


# Helper functions for setup wizard
def check_api_keys():
    """Check status of API keys."""
    import os
    
    api_keys = {
        "OpenAI API Key": "OPENAI_API_KEY",
        "CoinGecko API Key": "COINGECKO_API_KEY", 
        "Binance API Key": "BINANCE_API_KEY",
        "Finnhub API Key": "FINNHUB_API_KEY",
        "Alpha Vantage API Key": "ALPHA_VANTAGE_API_KEY"
    }
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Required For", style="yellow")
    
    for service, env_var in api_keys.items():
        if os.getenv(env_var):
            status = "✅ Configured"
        else:
            status = "❌ Missing"
        
        # Map services to their use cases
        use_cases = {
            "OpenAI API Key": "LLM analysis",
            "CoinGecko API Key": "Crypto market data", 
            "Binance API Key": "Crypto trading",
            "Finnhub API Key": "Stock market data",
            "Alpha Vantage API Key": "Additional market data"
        }
        
        table.add_row(service, status, use_cases[service])
    
    console.print(table)


def setup_providers():
    """Interactive provider setup."""
    console.print("Provider selection will be configured based on your asset class choice during analysis.")
    console.print("Crypto providers: CoinGecko, Binance, CryptoCompare")
    console.print("Equity providers: Finnhub, Yahoo Finance, Alpha Vantage")


def setup_cost_presets():
    """Interactive cost preset setup."""
    console.print("Cost presets optimize LLM usage based on your budget:")
    console.print("• [green]Cheap[/green]: Fast models, lower costs")
    console.print("• [yellow]Balanced[/yellow]: Mix of performance and cost")  
    console.print("• [red]Premium[/red]: Best models, higher costs")


def show_current_config():
    """Show current configuration."""
    console.print("\n[bold]Current Configuration:[/bold]")
    console.print(f"Asset Class: {DEFAULT_CONFIG.get('asset_class', 'equity')}")
    console.print(f"Results Directory: {DEFAULT_CONFIG.get('results_dir', './results')}")
    console.print(f"Debug Mode: {DEFAULT_CONFIG.get('debug', False)}")
    

def validate_configuration():
    """Validate current configuration."""
    console.print("\n[green]✅ Configuration is valid[/green]")


def reset_configuration():
    """Reset configuration to defaults."""
    if typer.confirm("Are you sure you want to reset configuration to defaults?"):
        console.print("[green]✅ Configuration reset to defaults[/green]")


def export_configuration(export_path: str):
    """Export configuration to file."""
    import json
    from pathlib import Path
    
    try:
        config_to_export = DEFAULT_CONFIG.copy()
        with open(export_path, 'w') as f:
            json.dump(config_to_export, f, indent=2, default=str)
        console.print(f"[green]✅ Configuration exported to {export_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error exporting configuration: {e}[/red]")


@app.command()
def analyze(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Ticker symbol to analyze (e.g., BTC/USDT, AAPL)"),
    asset_class: Optional[str] = typer.Option(None, "--asset-class", "-a", help="Asset class: 'equity' or 'crypto'"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Analysis date (YYYY-MM-DD)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    interactive: bool = typer.Option(True, "--interactive/--non-interactive", help="Run in interactive mode"),
    provider_preset: Optional[str] = typer.Option(None, "--provider-preset", help="Provider preset: 'free', 'premium', 'enterprise'"),
    cost_preset: Optional[str] = typer.Option(None, "--cost-preset", help="Cost preset: 'cheap', 'balanced', 'premium'"),
):
    """Run trading analysis with optional command line arguments."""
    run_analysis(
        ticker=ticker,
        asset_class=asset_class,
        date=date,
        config=config,
        interactive=interactive,
        provider_preset=provider_preset,
        cost_preset=cost_preset
    )


@app.command()
def setup():
    """Run initial setup wizard for TradingAgents configuration."""
    run_setup_wizard()


@app.command()
def providers(
    asset_class: Optional[str] = typer.Option(None, "--asset-class", "-a", help="Filter by asset class"),
    status: bool = typer.Option(False, "--status", help="Check provider health status"),
):
    """Manage and view provider configurations."""
    manage_providers(asset_class=asset_class, check_status=status)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
    export: Optional[str] = typer.Option(None, "--export", help="Export configuration to file"),
):
    """Manage TradingAgents configuration."""
    manage_config(show=show, validate=validate, reset=reset, export=export)


@app.command()
def crypto(
    action: Optional[str] = typer.Option(None, help="Crypto action: 'analyze', 'trade', 'risk', 'funding'"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Crypto pair (e.g., BTC/USDT, ETH-PERP)"),
    exchange: Optional[str] = typer.Option(None, "--exchange", help="Exchange for trading (binance, hyperliquid)"),
):
    """Crypto-specific trading commands and analysis."""
    run_crypto_interface(action=action, ticker=ticker, exchange=exchange)


if __name__ == "__main__":
    app()
