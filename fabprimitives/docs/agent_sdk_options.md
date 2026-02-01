# Agent SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Agent SDK in LangPy.

## Table of Contents

1. [Agent Creation](#agent-creation)
2. [Agent Configuration](#agent-configuration)
3. [Agent Operations](#agent-operations)
4. [Tool System](#tool-system)
5. [Streaming Support](#streaming-support)
6. [Advanced Features](#advanced-features)
7. [Response Format](#response-format)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

## Agent Creation

### Factory Method (Recommended)

```python
from sdk import agent

# Create an agent interface
agent_interface = agent()

# Create an agent with options
agent_interface = agent(
    async_backend=async_backend_func,        # Async backend callable
    sync_backend=sync_backend_func,          # Sync backend callable
    tools=[tool_definitions]                 # Default tools
)
```

### Direct Creation Methods

```python
# Using AgentInterface directly
from sdk.agent_interface import AgentInterface

agent_interface = AgentInterface(
    async_backend=async_backend_func,
    sync_backend=sync_backend_func,
    tools=[tool_definitions]
)

# Using AsyncAgent directly
from agent import AsyncAgent, Tool

agent = AsyncAgent(
    async_llm=async_backend_func,
    tools=[tool_definitions]
)

# Using SyncAgent directly
from agent import SyncAgent, Tool

agent = SyncAgent(
    sync_llm=sync_backend_func,
    tools=[tool_definitions]
)
```

### Agent Instance Creation

```python
# Create a specific agent instance
agent_instance = agent_interface.create_agent(
    name="my_agent",                         # Agent name
    instructions="You are a helpful assistant",  # System instructions
    input="Hello, world!",                   # Initial input
    tools=[tool_definitions],                # Tools for this agent
    tool_choice="auto",                      # Tool choice strategy
    model="gpt-4o-mini",                     # Model to use
    api_key="sk-...",                        # API key
    **kwargs                                 # Additional options
)
```

## Agent Configuration

### Basic Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | `str` | **Required** | Agent name identifier |
| `instructions` | `str` | **Required** | System-level guidance for the agent |
| `input` | `str` or `List[InputMessage]` | **Required** | Initial input or message array |
| `model` | `str` | `"gpt-4o-mini"` | Model name (provider-qualified) |
| `apiKey` | `str` | **Required** | API key for the model provider |

### Generation Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | `float` | `1.0` | Sampling temperature (0-2) |
| `top_p` | `float` | `1.0` | Nucleus sampling parameter (0-1) |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `stop` | `List[str]` | `None` | Stop sequences |
| `presence_penalty` | `float` | `None` | Presence penalty (-2 to 2) |
| `frequency_penalty` | `float` | `None` | Frequency penalty (-2 to 2) |

### Tool Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tools` | `List[Tool]` | `[]` | List of tool objects |
| `tool_choice` | `str` or `ToolChoice` | `"auto"` | Tool selection mode: `"auto"`, `"required"`, or specific tool |
| `parallel_tool_calls` | `bool` | `False` | Execute multiple tool calls concurrently |

### Streaming Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stream` | `bool` | `False` | Enable streaming responses |

## Agent Operations

### Run Operation

```python
# Run agent with full configuration
response = await agent_interface.run(
    model="gpt-4o-mini",                     # Model name (required)
    input="Hello, world!",                   # Input or message array (required)
    apiKey="sk-...",                         # API key (required)
    instructions="You are helpful",          # System instructions (optional)
    tools=[tool_definitions],                # Tools (optional)
    tool_choice="auto",                      # Tool choice strategy (optional)
    parallel_tool_calls=False,              # Parallel tool execution (optional)
    temperature=1.0,                         # Sampling temperature (optional)
    top_p=1.0,                              # Nucleus sampling (optional)
    max_tokens=None,                        # Max tokens (optional)
    stop=None,                              # Stop sequences (optional)
    presence_penalty=None,                  # Presence penalty (optional)
    frequency_penalty=None,                 # Frequency penalty (optional)
)

# Access response content
print(response.output)  # First message content
```

### Sync Run Operation

```python
# Run agent synchronously
response = agent_interface.run_sync(
    model="gpt-4o-mini",
    input="Hello, world!",
    apiKey="sk-...",
    instructions="You are helpful",
    tools=[tool_definitions],
    tool_choice="auto",
    parallel_tool_calls=False,
    temperature=1.0,
    top_p=1.0,
    max_tokens=None,
    stop=None,
    presence_penalty=None,
    frequency_penalty=None
)
```

### Stream Operation

```python
# Stream agent responses
async for chunk in agent_interface.stream(
    model="gpt-4o-mini",
    input="Tell me a story",
    apiKey="sk-...",
    instructions="You are a storyteller",
    stream=True
):
    print(chunk.choices[0].delta.get("content", ""), end="", flush=True)
```

### Agent Instance Operations

```python
# Run agent instance
response = await agent_instance.run()

# Run with overrides
response = await agent_instance.run(
    model="gpt-4o",                          # Override model
    temperature=0.7,                         # Override temperature
    tools=[additional_tools]                 # Override tools
)

# Run synchronously
response = agent_instance.run_sync()

# Stream responses
async for chunk in agent_instance.stream():
    print(chunk.choices[0].delta.get("content", ""), end="")

# Update configuration
agent_instance.update_config(
    temperature=0.5,
    max_tokens=1000
)
```

## Tool System

### Tool Definition

```python
from agent import Tool, ToolFunction

# Define a tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

# Create tool definition
tool_definition = Tool(
    type="function",
    function=ToolFunction(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    ),
    callable=search_web  # Attach the callable function
)
```

### Tool Choice Options

```python
# Automatic tool selection (default)
tool_choice = "auto"

# Force tool usage
tool_choice = "required"

# Specific tool selection
from agent import ToolChoice, ToolFunction

tool_choice = ToolChoice(
    type="function",
    function=ToolFunction(name="search_web")
)
```

### Tool Helper Functions

```python
from sdk.helpers import create_tool

# Create tool using helper
tool = create_tool(
    name="calculate",
    func=lambda x, y: x + y,
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"}
        },
        "required": ["x", "y"]
    }
)
```

### Multiple Tools

```python
# Define multiple tools
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="search_web",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        ),
        callable=search_web
    ),
    Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Perform calculations",
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
]

# Use tools with agent
response = await agent_interface.run(
    model="gpt-4o-mini",
    input="Search for Python tutorials and calculate 15 * 7",
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=True,  # Execute tools in parallel
    apiKey="sk-..."
)
```

## Streaming Support

### Basic Streaming

```python
# Enable streaming
stream = agent_interface.stream(
    model="gpt-4o-mini",
    input="Tell me about artificial intelligence",
    apiKey="sk-...",
    stream=True
)

# Process stream chunks
async for chunk in stream:
    if chunk.choices and chunk.choices[0].delta:
        content = chunk.choices[0].delta.get("content", "")
        if content:
            print(content, end="", flush=True)
```

### Streaming with Tools

```python
# Stream with tool execution
async for chunk in agent_interface.stream(
    model="gpt-4o-mini",
    input="Search for recent AI news and summarize",
    tools=[search_tool],
    tool_choice="auto",
    apiKey="sk-...",
    stream=True
):
    # Handle different chunk types
    if chunk.choices:
        for choice in chunk.choices:
            if choice.delta:
                # Regular content
                if "content" in choice.delta:
                    print(choice.delta["content"], end="", flush=True)
                # Tool calls
                if "tool_calls" in choice.delta:
                    print(f"\n[Tool call: {choice.delta['tool_calls']}]")
```

### Stream Helper

```python
from sdk.helpers import print_streaming_response

# Use helper for automatic printing
full_response = await print_streaming_response(
    agent_interface.stream(
        model="gpt-4o-mini",
        input="Write a short story",
        apiKey="sk-...",
        stream=True
    ),
    prefix="Story: "
)
```

## Advanced Features

### Message Arrays

```python
from agent import InputMessage

# Create message array
messages = [
    InputMessage(role="system", content="You are a helpful assistant"),
    InputMessage(role="user", content="Hello"),
    InputMessage(role="assistant", content="Hi there!"),
    InputMessage(role="user", content="How are you?")
]

# Use message array
response = await agent_interface.run(
    model="gpt-4o-mini",
    input=messages,
    apiKey="sk-..."
)
```

### Tool Messages

```python
from agent import InputMessage

# Include tool messages in conversation
messages = [
    InputMessage(role="user", content="What's the weather like?"),
    InputMessage(role="assistant", content=None, tool_calls=[
        {"id": "call_1", "name": "get_weather", "arguments": {"location": "NYC"}}
    ]),
    InputMessage(role="tool", content="Sunny, 75°F", tool_call_id="call_1"),
    InputMessage(role="user", content="Great! What about tomorrow?")
]

response = await agent_interface.run(
    model="gpt-4o-mini",
    input=messages,
    tools=[weather_tool],
    apiKey="sk-..."
)
```

### Extended Agent Interface

```python
# Create agent with additional tools
extended_agent = agent_interface.with_tools([
    additional_tool_1,
    additional_tool_2
])

# Use extended agent
response = await extended_agent.run(
    model="gpt-4o-mini",
    input="Use the additional tools",
    apiKey="sk-..."
)
```

## Response Format

### AgentRunResponse

```python
# Response structure
class AgentRunResponse:
    id: str                    # Response ID
    object: str               # Response type ("agent.run")
    created: int              # Timestamp
    model: str                # Model used
    choices: List[RunChoice]   # Response choices
    usage: Optional[JsonDict] # Token usage info
    error: Optional[Dict]     # Error info (if any)
    systemFingerprint: Optional[str]  # System fingerprint
    
    # Convenience properties
    output: str               # First message content
    
    # Methods
    def json(self) -> dict    # Parse output as JSON
```

### RunChoice

```python
class RunChoice:
    message: JsonDict         # Message content
    finish_reason: str        # Completion reason
    index: int               # Choice index
```

### AgentStreamChunk

```python
class AgentStreamChunk:
    id: str                   # Chunk ID
    object: str              # Chunk type ("agent.chunk")
    created: int             # Timestamp
    model: str               # Model used
    choices: List[StreamDelta]  # Stream deltas
    usage: Optional[JsonDict]   # Token usage info
    error: Optional[Dict]       # Error info (if any)
    systemFingerprint: Optional[str]  # System fingerprint
```

### StreamDelta

```python
class StreamDelta:
    delta: JsonDict          # Delta content
    index: int              # Choice index
    finish_reason: Optional[str]  # Completion reason
```

## Error Handling

### Exception Handling

```python
try:
    response = await agent_interface.run(
        model="gpt-4o-mini",
        input="Hello",
        apiKey="invalid-key"
    )
except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Error Responses

```python
# Errors are wrapped in structured responses
response = await agent_interface.run(
    model="gpt-4o-mini",
    input="Hello",
    apiKey="invalid-key"
)

# Check for errors
if response.error:
    print(f"Error: {response.error['message']}")
    print(f"Error type: {response.error['type']}")
else:
    print(response.output)
```

### Validation Errors

```python
# Empty API key
try:
    response = await agent_interface.run(
        model="gpt-4o-mini",
        input="Hello",
        apiKey=""  # Empty API key
    )
except ValueError as e:
    print(f"API key validation failed: {e}")

# Invalid tool choice
try:
    response = await agent_interface.run(
        model="gpt-4o-mini",
        input="Hello",
        tool_choice="invalid_choice",  # Invalid choice
        apiKey="sk-..."
    )
except ValueError as e:
    print(f"Tool choice validation failed: {e}")
```

## Examples

### Basic Agent Usage

```python
from sdk import agent

# Create agent interface
agent_interface = agent()

# Simple agent run
response = await agent_interface.run(
    model="gpt-4o-mini",
    input="What is the capital of France?",
    instructions="You are a geography expert.",
    apiKey="sk-..."
)

print(response.output)  # "The capital of France is Paris."
```

### Agent with Tools

```python
from sdk import agent
from agent import Tool, ToolFunction

# Define tools
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 75°F"

def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error in calculation"

# Create tool definitions
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name"}
                },
                "required": ["location"]
            }
        ),
        callable=get_weather
    ),
    Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Calculate mathematical expressions",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        ),
        callable=calculate
    )
]

# Create agent with tools
agent_interface = agent(tools=tools)

# Run with tool execution
response = await agent_interface.run(
    model="gpt-4o-mini",
    input="What's the weather in New York and what's 15 * 7?",
    instructions="Use the available tools to help the user.",
    tool_choice="auto",
    parallel_tool_calls=True,
    apiKey="sk-..."
)

print(response.output)
```

### Streaming Agent

```python
from sdk import agent

agent_interface = agent()

# Stream response
print("Agent: ", end="", flush=True)
async for chunk in agent_interface.stream(
    model="gpt-4o-mini",
    input="Tell me a short story about a robot",
    instructions="You are a creative storyteller.",
    apiKey="sk-..."
):
    if chunk.choices and chunk.choices[0].delta:
        content = chunk.choices[0].delta.get("content", "")
        if content:
            print(content, end="", flush=True)
print()  # New line after streaming
```

### Agent Instance Pattern

```python
from sdk import agent

# Create agent interface
agent_interface = agent()

# Create specific agent instance
storyteller = agent_interface.create_agent(
    name="storyteller",
    instructions="You are a creative storyteller who writes engaging tales.",
    input="",  # Will be set per run
    model="gpt-4o-mini",
    api_key="sk-...",
    temperature=0.8,
    max_tokens=500
)

# Use agent instance multiple times
story1 = await storyteller.run(
    input="Tell me a story about a dragon"
)

story2 = await storyteller.run(
    input="Tell me a story about a space explorer"
)

# Update configuration
storyteller.update_config(temperature=1.2)

story3 = await storyteller.run(
    input="Tell me a funny story"
)
```

### Multi-Turn Conversation

```python
from sdk import agent
from agent import InputMessage

agent_interface = agent()

# Build conversation
conversation = [
    InputMessage(role="system", content="You are a helpful assistant."),
    InputMessage(role="user", content="Hello!"),
]

# First turn
response1 = await agent_interface.run(
    model="gpt-4o-mini",
    input=conversation,
    apiKey="sk-..."
)

# Add response to conversation
conversation.append(InputMessage(
    role="assistant", 
    content=response1.output
))

# Add user's next message
conversation.append(InputMessage(
    role="user", 
    content="Can you help me with Python?"
))

# Second turn
response2 = await agent_interface.run(
    model="gpt-4o-mini",
    input=conversation,
    apiKey="sk-..."
)

print(response2.output)
```

### Agent with Custom Backend

```python
from sdk import agent

# Define custom backend
async def custom_backend(payload):
    """Custom backend implementation."""
    # Process payload and return response
    return {
        "id": "custom-response",
        "object": "agent.run",
        "created": 1234567890,
        "model": payload["model"],
        "choices": [{
            "message": {"content": f"Custom response to: {payload['input']}"},
            "finish_reason": "stop",
            "index": 0
        }],
        "usage": {"total_tokens": 50}
    }

# Create agent with custom backend
agent_interface = agent(async_backend=custom_backend)

# Use custom backend
response = await agent_interface.run(
    model="custom-model",
    input="Hello from custom backend",
    apiKey="not-used"
)

print(response.output)
```

### Error Handling Example

```python
from sdk import agent

agent_interface = agent()

async def safe_agent_run(input_text: str, api_key: str):
    """Safe agent run with comprehensive error handling."""
    try:
        response = await agent_interface.run(
            model="gpt-4o-mini",
            input=input_text,
            apiKey=api_key
        )
        
        # Check for errors in response
        if response.error:
            print(f"Agent error: {response.error['message']}")
            return None
        
        return response.output
    
    except ValueError as e:
        print(f"Validation error: {e}")
        return None
    
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Use safe function
result = await safe_agent_run("Hello", "sk-...")
if result:
    print(f"Success: {result}")
else:
    print("Failed to get response")
```

### Advanced Tool Usage

```python
from sdk import agent
from agent import Tool, ToolFunction
import asyncio

# Define async tool
async def fetch_data(url: str) -> str:
    """Fetch data from URL (async)."""
    await asyncio.sleep(1)  # Simulate network delay
    return f"Data from {url}: Sample content"

# Define sync tool
def process_text(text: str) -> str:
    """Process text (sync)."""
    return f"Processed: {text.upper()}"

# Create tool definitions
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="fetch_data",
            description="Fetch data from URL",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        ),
        callable=fetch_data  # Async callable
    ),
    Tool(
        type="function",
        function=ToolFunction(
            name="process_text",
            description="Process text content",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to process"}
                },
                "required": ["text"]
            }
        ),
        callable=process_text  # Sync callable
    )
]

# Create agent with mixed async/sync tools
agent_interface = agent(tools=tools)

# Run with parallel tool execution
response = await agent_interface.run(
    model="gpt-4o-mini",
    input="Fetch data from https://example.com and process the text 'hello world'",
    instructions="Use the available tools to help the user.",
    tool_choice="auto",
    parallel_tool_calls=True,  # Execute tools in parallel
    apiKey="sk-..."
)

print(response.output)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `GOOGLE_API_KEY` | Google API key | None |
| `MISTRAL_API_KEY` | Mistral API key | None |
| `GROQ_API_KEY` | Groq API key | None |

## Best Practices

1. **Use appropriate models**: Choose models based on task complexity and cost
2. **Validate inputs**: Always validate API keys and input parameters
3. **Handle errors gracefully**: Implement comprehensive error handling
4. **Use tools effectively**: Design tools with clear descriptions and parameters
5. **Optimize for streaming**: Use streaming for long-form content generation
6. **Manage conversations**: Keep track of conversation history for multi-turn interactions
7. **Test tool execution**: Verify tool functions work correctly before using with agents
8. **Monitor token usage**: Track token consumption for cost optimization
9. **Use parallel tools**: Enable parallel tool execution for better performance
10. **Secure API keys**: Use environment variables for API keys

## Troubleshooting

### Common Issues

1. **API key errors**: Verify API key format and permissions
2. **Model not found**: Check model name and provider availability
3. **Tool execution failures**: Verify tool function signatures and implementations
4. **Streaming interruptions**: Handle stream errors and connection issues
5. **Memory errors**: Monitor memory usage with large conversations
6. **Rate limiting**: Implement backoff strategies for API limits
7. **Tool timeout**: Set appropriate timeouts for tool execution
8. **Conversation length**: Manage conversation length to avoid token limits
9. **Parallel tool issues**: Debug tool execution order and dependencies
10. **Response parsing**: Handle malformed responses gracefully

### Performance Optimization

1. **Model selection**: Use appropriate models for tasks
2. **Tool optimization**: Optimize tool function performance
3. **Parallel execution**: Use parallel tool calls when possible
4. **Caching**: Implement caching for repeated operations
5. **Streaming**: Use streaming for better user experience
6. **Memory management**: Clean up large conversation histories
7. **Connection pooling**: Reuse connections when possible
8. **Batch operations**: Process multiple requests efficiently

---

## New in v2.0: Composable Pipeline API

LangPy 2.0 introduces a composable pipeline architecture. Agent now implements the `IPrimitive` interface and can be composed with other primitives using the `|` (sequential) and `&` (parallel) operators.

### The `process()` Method

```python
from langpy.core import Context
from langpy_sdk import Agent, tool

@tool("get_weather", "Get weather for a location", {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"]
})
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

agent = Agent(model="gpt-4o-mini", tools=[get_weather])

# Create context
ctx = Context(query="What's the weather in Tokyo?")

# Process and get Result
result = await agent.process(ctx)

if result.is_success():
    response_ctx = result.unwrap()
    print(response_ctx.response)
    print(f"Tokens: {response_ctx.token_usage.total_tokens}")
    print(f"Cost: ${response_ctx.cost.total_cost:.4f}")
else:
    print(f"Error: {result.error.message}")
```

### Pipeline Composition with `|` Operator

Chain Agent with other primitives:

```python
from langpy.core import Context
from langpy_sdk import Memory, Agent, Pipe, tool

# Define tools
@tool("search_docs", "Search documentation", {...})
def search_docs(query: str) -> str: ...

# Create primitives
memory = Memory(name="context", k=3)
agent = Agent(model="gpt-4o-mini", tools=[search_docs])
validator = Pipe(model="gpt-4o-mini", system_prompt="Verify the answer is accurate.")

# Compose pipeline: retrieve context → agent processes → validate
pipeline = memory | agent | validator

result = await pipeline.process(Context(query="How do I configure logging?"))
```

### Agent with Memory Context

Agent receives documents from upstream Memory:

```python
from langpy.core import Context
from langpy_sdk import Memory, Agent, tool

@tool("clarify", "Ask for clarification", {...})
def clarify(question: str) -> str: ...

memory = Memory(name="docs", k=5)
agent = Agent(
    model="gpt-4o-mini",
    tools=[clarify],
    system_prompt="Use the provided context to answer. Use tools if needed."
)

# Agent receives documents from memory in context
rag_agent = memory | agent

result = await rag_agent.process(Context(query="How do I deploy?"))
```

### Parallel Agents with `&` Operator

Run multiple specialized agents in parallel:

```python
from langpy.core import Context, parallel
from langpy_sdk import Agent, Pipe, tool

# Specialized agents
@tool("search_web", "Search the web", {...})
def search_web(query: str) -> str: ...

@tool("search_docs", "Search documentation", {...})
def search_docs(query: str) -> str: ...

web_agent = Agent(model="gpt-4o-mini", tools=[search_web])
docs_agent = Agent(model="gpt-4o-mini", tools=[search_docs])

# Run both in parallel, then synthesize
synthesizer = Pipe(model="gpt-4o-mini", system_prompt="Combine both results.")

pipeline = (web_agent & docs_agent) | synthesizer

result = await pipeline.process(Context(query="Latest Python best practices"))
```

### Agent Constructor Options for Composable API

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | LLM model to use |
| `tools` | `List` | `[]` | Tools available to the agent |
| `system_prompt` | `str` | `None` | System instructions |
| `name` | `str` | `"Agent"` | Name for tracing/logging |
| `tool_choice` | `str` | `"auto"` | Tool selection mode |
| `parallel_tool_calls` | `bool` | `False` | Execute tools in parallel |

### Context Flow Through Pipeline

When Agent processes a Context:

1. **Input**: Context with `query` and optionally `documents`
2. **Process**: Agent uses LLM + tools to generate response
3. **Output**: Context with `response` field populated

```python
# Context flows through pipeline
ctx = Context(query="What's 2+2 and what's the weather?")

# Agent may call multiple tools, then responds
result = await agent.process(ctx)
ctx = result.unwrap()
# ctx.response = "2+2 equals 4, and the weather is sunny."
```

### Error Handling with Result Types

```python
from langpy.core import Context, ErrorCode

result = await agent.process(ctx)

if result.is_failure():
    error = result.error
    if error.code == ErrorCode.LLM_API_ERROR:
        print(f"API error: {error.message}")
    elif error.code == ErrorCode.LLM_RATE_LIMIT:
        print("Rate limited, retrying...")
    else:
        print(f"Error: {error.message}")
```

### Recovery and Retry in Pipelines

```python
from langpy.core import Context, retry, recover
from langpy_sdk import Agent, Pipe

agent = Agent(model="gpt-4o-mini", tools=[my_tools])
fallback = Pipe(model="gpt-4o-mini", system_prompt="Provide a basic answer.")

# Retry agent 3 times, then fall back to simple pipe
robust_pipeline = recover(
    retry(agent, max_attempts=3, delay=1.0),
    handler=lambda err, ctx: ctx.set("used_fallback", True)
) | fallback

result = await robust_pipeline.process(ctx)
```

### Cost and Token Tracking

```python
result = await agent.process(ctx)
if result.is_success():
    ctx = result.unwrap()

    # Token usage (includes tool call tokens)
    print(f"Input tokens: {ctx.token_usage.input_tokens}")
    print(f"Output tokens: {ctx.token_usage.output_tokens}")
    print(f"Total tokens: {ctx.token_usage.total_tokens}")

    # Cost tracking
    print(f"Total cost: ${ctx.cost.total_cost:.4f}")
```

### Backward Compatibility

The original Agent API continues to work:

```python
# Original API (still supported)
response = await agent.run("What's the weather in Tokyo?")
print(response.content)

# New composable API
result = await agent.process(Context(query="What's the weather in Tokyo?"))
print(result.unwrap().response)
```

### Testing Agent Pipelines

```python
from langpy.testing import MockPrimitive, mock_llm_response, assert_success
from langpy.core import Context

# Mock agent that returns predictable responses
mock_agent = MockPrimitive(
    name="mock_agent",
    responses=["The weather in Tokyo is sunny, 72°F."]
)

# Test pipeline
result = await mock_agent.process(Context(query="Weather in Tokyo?"))
ctx = assert_success(result)
assert "Tokyo" in ctx.response
```

### Complete Agent Pipeline Example

```python
from langpy.core import Context, retry, recover
from langpy_sdk import Memory, Agent, Pipe, tool

# Define tools
@tool("calculate", "Perform calculations", {
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"]
})
def calculate(expression: str) -> str:
    return str(eval(expression))

@tool("get_date", "Get current date", {"type": "object", "properties": {}})
def get_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# Build pipeline
memory = Memory(name="context", k=3)
agent = Agent(
    model="gpt-4o-mini",
    tools=[calculate, get_date],
    system_prompt="Use tools when needed. Be concise."
)
validator = Pipe(
    model="gpt-4o-mini",
    system_prompt="Verify the response is accurate and complete."
)

# Compose: context → agent → validate
pipeline = memory | retry(agent, max_attempts=2) | validator

# Execute
result = await pipeline.process(Context(
    query="What's today's date and what's 15% of 250?"
))

if result.is_success():
    ctx = result.unwrap()
    print(f"Answer: {ctx.response}")
    print(f"Total cost: ${ctx.cost.total_cost:.4f}")
```

See [CORE_API.md](CORE_API.md) and [TESTING.md](TESTING.md) for complete documentation.