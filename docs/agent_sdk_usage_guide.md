# Agent Primitive SDK Usage Guide

This guide explains how to use the Agent primitive with the LangPy SDK installable package, including all available methods and examples.

## Installation

First, install the LangPy package:

```bash
pip install langpy
```

## Basic Usage

### 1. Direct Import (Recommended)

```python
from langpy.sdk import agent

# Create agent instance
agent_instance = agent()

# Run agent (requires backend setup)
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="What is the capital of France?",
    apiKey="sk-your-api-key-here"
)
```

### 2. Full SDK Client

```python
from langpy.sdk import Primitives

# Initialize with backend
primitives = Primitives(
    async_backend=your_async_llm_function,
    sync_backend=your_sync_llm_function,  # Optional
    tools=[your_tools]  # Optional
)

# Use agent
response = await primitives.agent.run(
    model="openai:gpt-4o",
    input="What is the capital of France?",
    apiKey="sk-your-api-key-here"
)
```

### 3. Individual Interface Import

```python
from langpy.sdk.agent_interface import AgentInterface

agent_interface = AgentInterface(
    async_backend=your_async_llm_function,
    sync_backend=your_sync_llm_function,  # Optional
    tools=[your_tools]  # Optional
)
```

## Available Methods

### 1. `run()` - Async Method

**Signature:**
```python
async def run(self, **kwargs) -> AgentRunResponse
```

**Parameters:**
- `model` (str): Model name (provider-qualified, e.g., "openai:gpt-4o")
- `input` (str | List[InputMessage]): Prompt or message array
- `instructions` (str, optional): System-level guidance
- `stream` (bool, optional): Enable streaming (default: False)
- `tools` (List[Tool], optional): List of Tool objects
- `tool_choice` (str | ToolChoice, optional): Tool selection mode ('auto', 'required', or ToolChoice)
- `parallel_tool_calls` (bool, optional): Run tool calls concurrently (default: False)
- `temperature` (float, optional): Sampling temperature (default: 1.0)
- `top_p` (float, optional): Nucleus sampling parameter (default: 1.0)
- `max_tokens` (int, optional): Maximum tokens to generate
- `stop` (List[str], optional): List of stop sequences
- `presence_penalty` (float, optional): Penalize new tokens based on presence
- `frequency_penalty` (float, optional): Penalize new tokens based on frequency
- `apiKey` (str): API key (required)

**Example:**
```python
# Basic usage
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="What is the weather like in New York?",
    apiKey="sk-your-api-key-here"
)

# With instructions
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="Tell me a joke",
    instructions="You are a friendly comedian. Keep jokes family-friendly.",
    apiKey="sk-your-api-key-here"
)

# With streaming
async for chunk in await agent_instance.run(
    model="openai:gpt-4o",
    input="Write a short story",
    stream=True,
    apiKey="sk-your-api-key-here"
):
    print(chunk.choices[0].delta.content, end="")
```

### 2. `run_sync()` - Synchronous Method

**Signature:**
```python
def run_sync(
    self,
    *,
    model: str,
    input: Union[str, List[InputMessage]],
    instructions: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    tool_choice: Optional[Union[str, ToolChoice]] = None,
    parallel_tool_calls: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    customModelParams: Optional[dict] = None,
    apiKey: str,
) -> AgentRunResponse
```

**Example:**
```python
response = agent_instance.run_sync(
    model="openai:gpt-4o",
    input="What is 2 + 2?",
    apiKey="sk-your-api-key-here"
)
```

### 3. `with_tools()` - Tool Management

**Signature:**
```python
def with_tools(self, tools: List[Tool]) -> "AgentInterface"
```

**Example:**
```python
from agent import Tool, ToolFunction

# Define a tool
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

weather_tool = Tool(
    type="function",
    function=ToolFunction(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    ),
    callable=get_weather
)

# Create agent with tools
agent_with_tools = agent_instance.with_tools([weather_tool])

# Run with tool execution
response = await agent_with_tools.run(
    model="openai:gpt-4o",
    input="What's the weather like in Paris?",
    apiKey="sk-your-api-key-here"
)
```

## Tool Integration

### Creating Tools

```python
from agent import Tool, ToolFunction

# Simple tool
def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

calc_tool = Tool(
    type="function",
    function=ToolFunction(
        name="calculate",
        description="Calculate mathematical expressions",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    ),
    callable=calculate
)

# Complex tool with multiple parameters
def search_database(query: str, limit: int = 10, category: str = None) -> str:
    # Your database search logic here
    return f"Found {limit} results for '{query}' in category '{category}'"

search_tool = Tool(
    type="function",
    function=ToolFunction(
        name="search_database",
        description="Search the knowledge base",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "category": {"type": "string"}
            },
            "required": ["query"]
        }
    ),
    callable=search_database
)
```

### Using Tools with Agent

```python
# Initialize agent with tools
agent_with_tools = agent_instance.with_tools([calc_tool, search_tool])

# Run with tool execution
response = await agent_with_tools.run(
    model="openai:gpt-4o",
    input="Calculate 15 * 23 and then search for 'machine learning'",
    tool_choice="auto",  # Let the model decide when to use tools
    apiKey="sk-your-api-key-here"
)

# Force tool usage
response = await agent_with_tools.run(
    model="openai:gpt-4o",
    input="Calculate 15 * 23",
    tool_choice="required",  # Force the model to use tools
    apiKey="sk-your-api-key-here"
)
```

## Message Format

### String Input
```python
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="Hello, how are you?",
    apiKey="sk-your-api-key-here"
)
```

### Message Array Input
```python
from agent import InputMessage

messages = [
    InputMessage(role="system", content="You are a helpful assistant."),
    InputMessage(role="user", content="What is Python?"),
    InputMessage(role="assistant", content="Python is a programming language."),
    InputMessage(role="user", content="Tell me more about it.")
]

response = await agent_instance.run(
    model="openai:gpt-4o",
    input=messages,
    apiKey="sk-your-api-key-here"
)
```

## Response Handling

### AgentRunResponse Structure

```python
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="What is AI?",
    apiKey="sk-your-api-key-here"
)

# Access response properties
print(f"Response ID: {response.id}")
print(f"Model: {response.model}")
print(f"Created: {response.created}")

# Access the main content
if response.choices:
    main_choice = response.choices[0]
    print(f"Finish reason: {main_choice.finish_reason}")
    print(f"Content: {main_choice.message.content}")

# Access usage information
if response.usage:
    print(f"Prompt tokens: {response.usage.get('prompt_tokens')}")
    print(f"Completion tokens: {response.usage.get('completion_tokens')}")
    print(f"Total tokens: {response.usage.get('total_tokens')}")

# Check for errors
if response.error:
    print(f"Error: {response.error}")
```

### Streaming Response

```python
async for chunk in await agent_instance.run(
    model="openai:gpt-4o",
    input="Write a poem about AI",
    stream=True,
    apiKey="sk-your-api-key-here"
):
    if chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            print(delta.content, end="")
    
    # Check for completion
    if chunk.choices and chunk.choices[0].finish_reason:
        print(f"\nFinished: {chunk.choices[0].finish_reason}")
        break
```

## Error Handling

```python
try:
    response = await agent_instance.run(
        model="openai:gpt-4o",
        input="Hello",
        apiKey="invalid-key"
    )
    
    if response.error:
        print(f"API Error: {response.error}")
    else:
        print("Success!")
        
except RuntimeError as e:
    print(f"Runtime error: {e}")
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Configuration

### Temperature and Sampling

```python
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="Write a creative story",
    temperature=0.8,  # More creative
    top_p=0.9,        # Nucleus sampling
    max_tokens=500,   # Limit response length
    apiKey="sk-your-api-key-here"
)
```

### Stop Sequences

```python
response = await agent_instance.run(
    model="openai:gpt-4o",
    input="List the first 5 programming languages:",
    stop=["6.", "7.", "8."],  # Stop at these sequences
    apiKey="sk-your-api-key-here"
)
```

### Parallel Tool Calls

```python
response = await agent_with_tools.run(
    model="openai:gpt-4o",
    input="Get weather for both Paris and London",
    parallel_tool_calls=True,  # Execute tools concurrently
    apiKey="sk-your-api-key-here"
)
```

## Complete Example

```python
import asyncio
from langpy.sdk import agent
from agent import Tool, ToolFunction

# Define tools
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

def get_time(timezone: str = "UTC") -> str:
    from datetime import datetime
    return f"Current time in {timezone}: {datetime.now().strftime('%H:%M:%S')}"

weather_tool = Tool(
    type="function",
    function=ToolFunction(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    ),
    callable=get_weather
)

time_tool = Tool(
    type="function",
    function=ToolFunction(
        name="get_time",
        description="Get current time in a timezone",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "default": "UTC"}
            }
        }
    ),
    callable=get_time
)

async def main():
    # Create agent with tools
    agent_instance = agent()
    agent_with_tools = agent_instance.with_tools([weather_tool, time_tool])
    
    # Run agent
    response = await agent_with_tools.run(
        model="openai:gpt-4o",
        input="What's the weather in Tokyo and what time is it there?",
        instructions="You are a helpful assistant that can check weather and time.",
        tool_choice="auto",
        temperature=0.7,
        apiKey="sk-your-api-key-here"
    )
    
    # Print response
    if response.choices:
        print(response.choices[0].message.content)
    
    if response.error:
        print(f"Error: {response.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Backend Setup

The agent requires a backend function to handle LLM calls. Here's how to set it up:

```python
import openai
from typing import Dict, Any

# Async backend function
async def openai_backend(payload: Dict[str, Any]):
    client = openai.AsyncOpenAI(api_key=payload["apiKey"])
    
    response = await client.chat.completions.create(
        model=payload["model"].replace("openai:", ""),
        messages=payload["input"],
        temperature=payload.get("temperature", 1.0),
        max_tokens=payload.get("max_tokens"),
        stream=payload.get("stream", False)
    )
    
    return response

# Sync backend function
def openai_sync_backend(payload: Dict[str, Any]):
    client = openai.OpenAI(api_key=payload["apiKey"])
    
    response = client.chat.completions.create(
        model=payload["model"].replace("openai:", ""),
        messages=payload["input"],
        temperature=payload.get("temperature", 1.0),
        max_tokens=payload.get("max_tokens"),
        stream=payload.get("stream", False)
    )
    
    return response

# Use with agent
agent_instance = agent(
    async_backend=openai_backend,
    sync_backend=openai_sync_backend
)
```

This guide covers all the essential methods and usage patterns for the Agent primitive with the LangPy SDK. The agent provides a powerful interface for running LLM-powered conversations with tool execution capabilities. 