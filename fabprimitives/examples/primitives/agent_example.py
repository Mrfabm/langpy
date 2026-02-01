"""
Agent Primitive
===============
An LLM that can use tools to accomplish tasks.

Agents extend Pipes by adding:
    - Tool execution capabilities
    - Multi-step reasoning
    - External system interaction

Architecture:
    Prompt → LLM → Tool Call? → Execute Tool → LLM → Response

    ┌──────────────────────────────────────┐
    │              Agent                   │
    │  ┌─────┐   ┌────────┐   ┌────────┐   │
    │  │ LLM │ ↔ │ Tools  │ → │Execute │   │
    │  │     │   │ Array  │   │        │   │
    │  └─────┘   └────────┘   └────────┘   │
    │      ↓                      ↓        │
    │         ← ← ← ← ← ← ← ← ←           │
    │              Response                │
    └──────────────────────────────────────┘
"""

import asyncio
import io
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Agent, tool


# =============================================================================
# DEFINE TOOLS
# =============================================================================

@tool(
    "get_current_time",
    "Get the current date and time",
    {"type": "object", "properties": {}}
)
def get_current_time() -> str:
    """Get current time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool(
    "calculate",
    "Perform mathematical calculations",
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '15 * 7 + 23')"
            }
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
            return f"Result: {expression} = {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Error: Invalid expression (only numbers and basic operators allowed)"


@tool(
    "search_knowledge",
    "Search the knowledge base for information",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
)
def search_knowledge(query: str) -> str:
    """Simulated knowledge base search."""
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a versatile programming language created by Guido van Rossum in 1991.",
        "langpy": "LangPy is a framework with 9 primitives for building AI applications.",
        "agent": "Agents are LLMs that can use tools to interact with external systems.",
        "memory": "Memory stores documents using vector embeddings for semantic search.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Found: {value}"
    return f"No results found for '{query}'"


@tool(
    "get_weather",
    "Get weather for a location",
    {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str) -> str:
    """Simulated weather service."""
    weather_data = {
        "New York": "72°F, Sunny",
        "London": "58°F, Cloudy",
        "Tokyo": "68°F, Rainy",
        "Paris": "65°F, Partly Cloudy",
    }
    weather = weather_data.get(location, "65°F, Clear")
    return f"Weather in {location}: {weather}"


# =============================================================================
# BASIC AGENT USAGE
# =============================================================================

async def basic_agent_demo():
    """Demonstrate basic Agent with tools."""
    print("=" * 60)
    print("   BASIC AGENT - LLM with Tools")
    print("=" * 60)
    print()

    # Create agent with tools
    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_current_time, calculate],
        temperature=0.7
    )

    print(f"Agent created with {len(agent.tools)} tools:")
    for t in agent.tools:
        print(f"   - {t.name}: {t.description}")
    print()

    # Simple prompts that trigger tool use
    prompts = [
        "What time is it right now?",
        "Calculate 15 times 7 plus 23",
        "What's 100 divided by 4?",
    ]

    print("Agent interactions:")
    print("-" * 40)

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        response = await agent.quick(prompt)
        print(f"Agent: {response}")

    print()


# =============================================================================
# MULTI-TOOL AGENT
# =============================================================================

async def multi_tool_demo():
    """Demonstrate agent with multiple tools."""
    print("=" * 60)
    print("   MULTI-TOOL AGENT - Complex Tasks")
    print("=" * 60)
    print()

    # Agent with all tools
    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_current_time, calculate, search_knowledge, get_weather]
    )

    print(f"Agent has {len(agent.tools)} tools available")
    print()

    # Complex prompts requiring multiple tools or decisions
    prompts = [
        "What's the weather in Tokyo?",
        "What is Python and when was it created?",
        "What time is it and what's the weather in New York?",
    ]

    print("Multi-tool interactions:")
    print("-" * 40)

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        response = await agent.quick(prompt)
        print(f"Agent: {response}")

    print()


# =============================================================================
# AGENT WITH SYSTEM PROMPT
# =============================================================================

async def system_prompt_demo():
    """Demonstrate agent with custom system prompt."""
    print("=" * 60)
    print("   AGENT PERSONAS - Custom System Prompts")
    print("=" * 60)
    print()

    # Math tutor agent
    math_tutor = Agent(
        model="gpt-4o-mini",
        tools=[calculate]
    )

    print("1. Math Tutor Agent:")
    print("-" * 40)

    response = await math_tutor.quick(
        "Can you help me understand what 15% of 80 is?",
        system="You are a patient math tutor. Explain calculations step by step."
    )
    print(f"   User: Can you help me understand what 15% of 80 is?")
    print(f"   Tutor: {response}")
    print()

    # Research assistant agent
    research_agent = Agent(
        model="gpt-4o-mini",
        tools=[search_knowledge, get_current_time]
    )

    print("2. Research Assistant Agent:")
    print("-" * 40)

    response = await research_agent.quick(
        "Tell me about LangPy",
        system="You are a technical research assistant. Provide accurate, sourced information."
    )
    print(f"   User: Tell me about LangPy")
    print(f"   Assistant: {response}")
    print()


# =============================================================================
# ADDING TOOLS DYNAMICALLY
# =============================================================================

async def dynamic_tools_demo():
    """Demonstrate adding tools to an agent dynamically."""
    print("=" * 60)
    print("   DYNAMIC TOOLS - Runtime Tool Addition")
    print("=" * 60)
    print()

    # Start with basic agent
    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_current_time]
    )

    print(f"Initial tools: {len(agent.tools)}")

    # Add more tools
    agent.add_tool(calculate)
    agent.add_tool(get_weather)

    print(f"After adding tools: {len(agent.tools)}")
    print()

    # Now agent can use all tools
    print("Using dynamically added tools:")
    print("-" * 40)

    response = await agent.quick("What's 25 + 17 and what's the weather in Paris?")
    print(f"   Response: {response}")
    print()


# =============================================================================
# TOOL EXECUTION FLOW
# =============================================================================

async def tool_flow_demo():
    """Demonstrate and explain tool execution flow."""
    print("=" * 60)
    print("   TOOL EXECUTION FLOW - Under the Hood")
    print("=" * 60)
    print()

    print("How Agent tool execution works:")
    print("-" * 40)
    print("""
    1. User sends prompt to Agent
    2. Agent (LLM) decides if tools are needed
    3. If yes: LLM generates tool call(s) with arguments
    4. Agent executes the tool handler function
    5. Tool result is sent back to LLM
    6. LLM generates final response using tool results
    7. Response returned to user

    Example flow for "What's 15 * 7?":

    ┌──────────────────────────────────────────────┐
    │ User: "What's 15 * 7?"                       │
    └─────────────────────┬────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────┐
    │ LLM decides: use calculate tool              │
    │ Tool call: calculate(expression="15 * 7")    │
    └─────────────────────┬────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────┐
    │ Handler executes: eval("15 * 7")             │
    │ Returns: "Result: 15 * 7 = 105"              │
    └─────────────────────┬────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────┐
    │ LLM receives result, generates response      │
    │ Response: "15 times 7 equals 105"            │
    └──────────────────────────────────────────────┘
    """)

    # Demonstrate with actual agent
    agent = Agent(
        model="gpt-4o-mini",
        tools=[calculate]
    )

    print("Live example:")
    print("-" * 40)
    response = await agent.quick("What's 15 * 7?")
    print(f"   User: What's 15 * 7?")
    print(f"   Agent: {response}")
    print()


# =============================================================================
# AGENT VS PIPE COMPARISON
# =============================================================================

async def agent_vs_pipe_demo():
    """Compare Agent and Pipe primitives."""
    print("=" * 60)
    print("   AGENT vs PIPE - When to Use Each")
    print("=" * 60)
    print()

    print("Comparison:")
    print("-" * 40)
    print("""
    ┌────────────────────────────────────────────────────────┐
    │                  PIPE                                  │
    ├────────────────────────────────────────────────────────┤
    │  - Single LLM call                                     │
    │  - No tools                                            │
    │  - Fast and simple                                     │
    │  - Good for: Q&A, text generation, classification      │
    │  - Example: "Summarize this text"                      │
    └────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────┐
    │                  AGENT                                 │
    ├────────────────────────────────────────────────────────┤
    │  - Multiple LLM calls (reasoning loop)                 │
    │  - Has tools for external actions                      │
    │  - More powerful but slower                            │
    │  - Good for: Tasks requiring actions, calculations,    │
    │              API calls, database queries               │
    │  - Example: "Book a meeting for tomorrow at 3pm"       │
    └────────────────────────────────────────────────────────┘
    """)

    print("Rule of thumb:")
    print("   - Use Pipe when you just need text in → text out")
    print("   - Use Agent when you need to DO something")
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Agent demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 17 + "AGENT PRIMITIVE DEMO" + " " * 18 + "*")
    print("*" * 60)
    print()

    await basic_agent_demo()
    await multi_tool_demo()
    await system_prompt_demo()
    await dynamic_tools_demo()
    await tool_flow_demo()
    await agent_vs_pipe_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
