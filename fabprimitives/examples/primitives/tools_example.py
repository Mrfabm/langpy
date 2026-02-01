"""
Tools Primitive
===============
Define tools that AI agents can use to interact with external systems.

Tools enable agents to:
    - Execute functions based on LLM decisions
    - Access external APIs and data
    - Perform calculations and lookups
    - Take actions in the real world

Architecture:
    LLM → Tool Call → Handler Function → Result → LLM

    ┌──────────────────────────────────────┐
    │             Tool System              │
    │  ┌──────┐   ┌──────────┐   ┌──────┐  │
    │  │ LLM  │ → │ Tool Call│ → │Handler│ │
    │  │      │ ← │          │ ← │       │ │
    │  └──────┘   └──────────┘   └──────┘  │
    └──────────────────────────────────────┘
"""

import asyncio
import io
import os
import sys
from datetime import datetime
import json

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import tool, ToolDef


# =============================================================================
# BASIC TOOL DEFINITION
# =============================================================================

def basic_tools_demo():
    """Demonstrate basic @tool decorator usage."""
    print("=" * 60)
    print("   BASIC TOOLS - The @tool Decorator")
    print("=" * 60)
    print()

    # Simple tool with no parameters
    @tool(
        "get_current_time",
        "Get the current date and time",
        {"type": "object", "properties": {}}
    )
    def get_current_time() -> str:
        """Get current time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("1. Simple tool (no parameters):")
    print("-" * 40)
    print(f"   Name: {get_current_time.name}")
    print(f"   Description: {get_current_time.description}")
    print(f"   Result: {get_current_time.handler()}")
    print()

    # Tool with parameters
    @tool(
        "greet_user",
        "Greet a user by name",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The user's name"},
                "formal": {"type": "boolean", "description": "Use formal greeting"}
            },
            "required": ["name"]
        }
    )
    def greet_user(name: str, formal: bool = False) -> str:
        """Greet a user."""
        if formal:
            return f"Good day, {name}. How may I assist you?"
        return f"Hey {name}! How's it going?"

    print("2. Tool with parameters:")
    print("-" * 40)
    print(f"   Name: {greet_user.name}")
    print(f"   Parameters: {json.dumps(greet_user.parameters, indent=6)}")
    print(f"   Result (casual): {greet_user.handler('Alice')}")
    print(f"   Result (formal): {greet_user.handler('Bob', formal=True)}")
    print()


# =============================================================================
# TOOL PARAMETER SCHEMAS
# =============================================================================

def parameter_schemas_demo():
    """Demonstrate JSON Schema for tool parameters."""
    print("=" * 60)
    print("   PARAMETER SCHEMAS - JSON Schema Definition")
    print("=" * 60)
    print()

    # Calculator with enum parameter
    @tool(
        "calculate",
        "Perform a mathematical calculation",
        {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The math operation to perform"
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"}
            },
            "required": ["operation", "a", "b"]
        }
    )
    def calculate(operation: str, a: float, b: float) -> str:
        """Safe calculator."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: division by zero"
        }
        if operation not in operations:
            return f"Error: unknown operation {operation}"
        result = operations[operation](a, b)
        return f"{a} {operation} {b} = {result}"

    print("1. Enum parameter (operation):")
    print("-" * 40)
    print(f"   Allowed values: {calculate.parameters['properties']['operation']['enum']}")
    print(f"   Result: {calculate.handler('multiply', 7, 6)}")
    print()

    # Tool with array parameter
    @tool(
        "analyze_numbers",
        "Analyze a list of numbers",
        {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers to analyze"
                },
                "stats": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["mean", "min", "max", "sum"]
                    },
                    "description": "Statistics to calculate"
                }
            },
            "required": ["numbers"]
        }
    )
    def analyze_numbers(numbers: list, stats: list = None) -> str:
        """Analyze numbers."""
        if not numbers:
            return "Error: empty list"
        stats = stats or ["mean", "min", "max"]
        results = {}
        if "mean" in stats:
            results["mean"] = sum(numbers) / len(numbers)
        if "min" in stats:
            results["min"] = min(numbers)
        if "max" in stats:
            results["max"] = max(numbers)
        if "sum" in stats:
            results["sum"] = sum(numbers)
        return json.dumps(results)

    print("2. Array parameters:")
    print("-" * 40)
    print(f"   Schema: {json.dumps(analyze_numbers.parameters['properties']['numbers'], indent=6)}")
    print(f"   Result: {analyze_numbers.handler([1, 2, 3, 4, 5], ['mean', 'sum'])}")
    print()

    # Tool with nested object
    @tool(
        "create_event",
        "Create a calendar event",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time in HH:MM format"},
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"}
                        },
                        "required": ["email"]
                    },
                    "description": "List of attendees"
                }
            },
            "required": ["title", "date"]
        }
    )
    def create_event(title: str, date: str, time: str = "09:00", attendees: list = None) -> str:
        """Create calendar event."""
        event = {
            "title": title,
            "datetime": f"{date} {time}",
            "attendees": attendees or []
        }
        return f"Created event: {json.dumps(event)}"

    print("3. Nested object parameters:")
    print("-" * 40)
    attendee_schema = analyze_numbers.parameters['properties'].get('numbers', {})
    print("   Attendee schema: object with name (optional) and email (required)")
    result = create_event.handler(
        "Team Meeting",
        "2024-03-15",
        "14:00",
        [{"name": "Alice", "email": "alice@example.com"}]
    )
    print(f"   Result: {result}")
    print()


# =============================================================================
# ASYNC TOOL HANDLERS
# =============================================================================

async def async_tools_demo():
    """Demonstrate async tool handlers."""
    print("=" * 60)
    print("   ASYNC TOOLS - Asynchronous Handlers")
    print("=" * 60)
    print()

    # Async tool for simulated API call
    @tool(
        "fetch_weather",
        "Get weather for a city",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    )
    async def fetch_weather(city: str) -> str:
        """Simulated weather API call."""
        # Simulate network delay
        await asyncio.sleep(0.1)

        # Simulated weather data
        weather_data = {
            "New York": {"temp": 72, "condition": "Sunny"},
            "London": {"temp": 58, "condition": "Cloudy"},
            "Tokyo": {"temp": 68, "condition": "Rainy"},
        }
        data = weather_data.get(city, {"temp": 70, "condition": "Unknown"})
        return f"{city}: {data['temp']}°F, {data['condition']}"

    print("1. Async tool execution:")
    print("-" * 40)

    cities = ["New York", "London", "Tokyo"]
    for city in cities:
        result = await fetch_weather.handler(city)
        print(f"   {result}")
    print()

    # Parallel async execution
    @tool(
        "search_database",
        "Search a database",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "table": {"type": "string", "description": "Table to search"}
            },
            "required": ["query"]
        }
    )
    async def search_database(query: str, table: str = "users") -> str:
        """Simulated database search."""
        await asyncio.sleep(0.1)
        return f"Found 5 results for '{query}' in {table}"

    print("2. Parallel async tools:")
    print("-" * 40)

    # Execute multiple async tools in parallel
    results = await asyncio.gather(
        fetch_weather.handler("New York"),
        search_database.handler("John"),
        fetch_weather.handler("Tokyo")
    )

    for result in results:
        print(f"   {result}")
    print()


# =============================================================================
# DYNAMIC TOOL CREATION
# =============================================================================

def dynamic_tools_demo():
    """Demonstrate creating tools dynamically."""
    print("=" * 60)
    print("   DYNAMIC TOOLS - Runtime Tool Creation")
    print("=" * 60)
    print()

    # Create tool from ToolDef directly
    print("1. Creating ToolDef directly:")
    print("-" * 40)

    def add_numbers(a: float, b: float) -> str:
        return f"{a} + {b} = {a + b}"

    add_tool = ToolDef(
        name="add_numbers",
        description="Add two numbers together",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        handler=add_numbers
    )

    print(f"   Tool: {add_tool.name}")
    print(f"   Result: {add_tool.handler(5, 3)}")
    print()

    # Factory function for creating tools
    print("2. Tool factory pattern:")
    print("-" * 40)

    def create_comparison_tool(operation: str, comparator):
        """Factory to create comparison tools."""

        def handler(a: float, b: float) -> str:
            result = comparator(a, b)
            return f"{a} {operation} {b} = {result}"

        return ToolDef(
            name=f"compare_{operation}",
            description=f"Check if first number is {operation} second",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            },
            handler=handler
        )

    # Create multiple tools dynamically
    tools = [
        create_comparison_tool("greater_than", lambda a, b: a > b),
        create_comparison_tool("less_than", lambda a, b: a < b),
        create_comparison_tool("equal_to", lambda a, b: a == b),
    ]

    for t in tools:
        print(f"   {t.name}: {t.handler(10, 5)}")
    print()


# =============================================================================
# TOOL VALIDATION
# =============================================================================

def validation_demo():
    """Demonstrate tool input validation."""
    print("=" * 60)
    print("   TOOL VALIDATION - Input Checking")
    print("=" * 60)
    print()

    @tool(
        "send_email",
        "Send an email to a recipient",
        {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$",
                    "description": "Email address"
                },
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "default": "normal",
                    "description": "Email priority"
                }
            },
            "required": ["to", "subject", "body"]
        }
    )
    def send_email(to: str, subject: str, body: str, priority: str = "normal") -> str:
        """Send email with validation."""
        # The handler can add runtime validation
        if not to or "@" not in to:
            return "Error: Invalid email address"
        if not subject.strip():
            return "Error: Subject cannot be empty"
        return f"Email sent to {to} with priority {priority}"

    print("1. Schema-based validation:")
    print("-" * 40)
    print("   Email pattern: regex for valid emails")
    print("   Subject: minLength=1, maxLength=200")
    print("   Priority: enum [low, normal, high]")
    print()

    print("2. Runtime validation in handler:")
    print("-" * 40)
    print(f"   Valid: {send_email.handler('test@example.com', 'Hello', 'Body')}")
    print(f"   Invalid email: {send_email.handler('invalid', 'Hello', 'Body')}")
    print()


# =============================================================================
# TOOLS FOR AGENTS
# =============================================================================

def agent_tools_demo():
    """Show tools ready for Agent usage."""
    print("=" * 60)
    print("   TOOLS FOR AGENTS - Ready to Use")
    print("=" * 60)
    print()

    # Collection of tools for an agent
    @tool(
        "get_time",
        "Get the current time",
        {"type": "object", "properties": {}}
    )
    def get_time() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @tool(
        "calculate",
        "Perform a calculation",
        {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '2 + 2'"
                }
            },
            "required": ["expression"]
        }
    )
    def calculate(expression: str) -> str:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            try:
                return f"Result: {eval(expression)}"
            except Exception as e:
                return f"Error: {e}"
        return "Error: Invalid expression"

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
        return f"[Simulated] Top results for '{query}': ..."

    # Collect tools for agent
    agent_tools = [get_time, calculate, search_web]

    print("Tools ready for Agent:")
    print("-" * 40)
    for t in agent_tools:
        print(f"   - {t.name}: {t.description}")
    print()

    print("Usage with Agent:")
    print("-" * 40)
    print("""
    from langpy_sdk import Agent

    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_time, calculate, search_web]
    )

    response = await agent.run("What time is it and what's 15 * 7?")
    """)
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Tools demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 17 + "TOOLS PRIMITIVE DEMO" + " " * 18 + "*")
    print("*" * 60)
    print()

    basic_tools_demo()
    parameter_schemas_demo()
    await async_tools_demo()
    dynamic_tools_demo()
    validation_demo()
    agent_tools_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
