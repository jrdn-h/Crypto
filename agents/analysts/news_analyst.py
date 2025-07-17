from tradingagents.dataflows import Toolkit

# Define the tools for the news analyst
# Use online tools if available, otherwise fall back to offline tools
tools = [toolkit.get_global_news_openai, toolkit.get_news]
# Initialize the NewsAnalyst with the selected tools
news_analyst_agent = NewsAnalyst(
    llm=llm,
    toolkit=toolkit.get_news,
    # Pass the runnable directly
    thought_process_runnable=news_analyst_runnable,
)
news_analyst_node = news_analyst_agent.get_node() 