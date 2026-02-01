"""
Pattern 7: Tool Agent
=====================

An agent with external tools using direct primitive composition.

This example shows how to BUILD a tool-using agent by composing primitives:
    - Agent: LLM with tool execution capability
    - tool decorator: defines callable tools

Architecture:
    User Query → Agent.run() → LLM decides tools → Execute tools → Response

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Agent, tool


# =============================================================================
# STEP 1: Define tools using the @tool decorator primitive
# =============================================================================

@tool(
    "get_weather",
    "Get current weather for a location",
    {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)
def get_weather(location: str) -> str:
    """Simulated weather API."""
    weather_data = {
        "new york": {"temp": 72, "condition": "Sunny"},
        "london": {"temp": 58, "condition": "Cloudy"},
        "tokyo": {"temp": 68, "condition": "Clear"},
    }
    data = weather_data.get(location.lower(), {"temp": 70, "condition": "Unknown"})
    return f"Weather in {location}: {data['temp']}F, {data['condition']}"


@tool(
    "calculate",
    "Perform mathematical calculations",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
)
def calculate(expression: str) -> str:
    """Safe calculator."""
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Error: Invalid expression"


@tool(
    "get_time",
    "Get current date and time",
    {"type": "object", "properties": {}}
)
def get_time() -> str:
    """Get current time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool(
    "search_web",
    "Search the web for information",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
)
def search_web(query: str) -> str:
    """Simulated web search."""
    return f"Search results for '{query}':\n1. [Result about {query}]\n2. [More info on {query}]"


# =============================================================================
# PATTERN 7: TOOL AGENT WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def tool_agent_pattern():
    """
    Build a tool-using agent by composing Agent + tool primitives.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 7: TOOL AGENT")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 2: Create Agent primitive with tools
    # =========================================================================

    # COMPOSE: Agent + tools
    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_weather, calculate, get_time, search_web],
        temperature=0.7
    )

    print("Agent created with tools:")
    print("  - get_weather: Get weather for a location")
    print("  - calculate: Perform math calculations")
    print("  - get_time: Get current time")
    print("  - search_web: Search the web")
    print()

    # =========================================================================
    # STEP 3: Run queries - Agent decides which tools to use
    # =========================================================================

    queries = [
        "What's the weather like in Tokyo?",
        "Calculate 145 * 23 + 50",
        "What time is it right now?",
        "Search for information about Python programming",
    ]

    for query in queries:
        print(f"User: {query}")
        print("-" * 40)

        # COMPOSE: Agent.run() handles tool selection and execution
        response = await agent.run(
            query,
            system="You are a helpful assistant with access to tools. Use them when needed."
        )

        print(f"Agent: {response.content}")

        # Show which tools were used
        if response.tool_calls:
            tools_used = [tc.get("function", {}).get("name", "unknown")
                         for tc in response.tool_calls if isinstance(tc, dict)]
            if tools_used:
                print(f"  (Tools used: {', '.join(tools_used)})")
        print()

    print("=" * 60)


# =============================================================================
# SIMPLE TOOL AGENT - Minimal composition example
# =============================================================================

async def simple_tool_agent(query: str, tools: list) -> str:
    """
    Minimal tool agent in just a few lines - pure primitive composition.

    Args:
        query: User's query
        tools: List of tool functions decorated with @tool

    Returns:
        Agent's response
    """
    # Create Agent with tools
    agent = Agent(model="gpt-4o-mini", tools=tools)

    # Run - agent decides and executes tools automatically
    response = await agent.run(query)

    return response.content


# =============================================================================
# MULTI-TOOL COMPOSITION - Complex query handling
# =============================================================================

async def multi_tool_demo():
    """
    Demonstrate agent handling queries that need multiple tools.
    """
    print("\n" + "=" * 60)
    print("   MULTI-TOOL COMPOSITION")
    print("   Agent + Multiple Tools")
    print("=" * 60 + "\n")

    # COMPOSE: Agent with multiple tools
    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_weather, calculate, get_time]
    )

    # Complex query that might use multiple tools
    query = "What's the weather in New York, and what's 100 divided by the temperature in Fahrenheit?"

    print(f"User: {query}")
    print("-" * 40)

    response = await agent.run(
        query,
        system="You can use multiple tools to answer complex questions. Think step by step."
    )

    print(f"Agent: {response.content}")
    print()


# =============================================================================
# CUSTOM TOOL COMPOSITION - Define tools inline
# =============================================================================

async def custom_tool_demo():
    """
    Show how to define and compose custom tools on the fly.
    """
    print("\n" + "=" * 60)
    print("   CUSTOM TOOL COMPOSITION")
    print("   Define tools inline")
    print("=" * 60 + "\n")

    # Define a custom tool
    @tool(
        "get_stock_price",
        "Get stock price for a ticker symbol",
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    )
    def get_stock_price(ticker: str) -> str:
        """Simulated stock price."""
        prices = {"AAPL": 175.50, "GOOGL": 140.25, "MSFT": 380.00}
        price = prices.get(ticker.upper(), 100.00)
        return f"{ticker.upper()}: ${price}"

    # COMPOSE: Agent with custom tool
    agent = Agent(model="gpt-4o-mini", tools=[get_stock_price, calculate])

    query = "What's the price of AAPL stock, and if I have 10 shares, what's my total value?"

    print(f"User: {query}")
    print("-" * 40)

    response = await agent.run(query)
    print(f"Agent: {response.content}")
    print()


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all tool agent pattern demonstrations."""
    await tool_agent_pattern()
    await multi_tool_demo()
    await custom_tool_demo()


if __name__ == "__main__":
    asyncio.run(demo())
