# How the Agent Code Runs - Complete Execution Flow

## Overview
The agent follows a **Think-Act-Observe-Reflect** loop, similar to Langbase agents. Here's how it works:

## 1. Initialization Phase

```python
# User creates agent
agent = AsyncAgent(
    async_llm=openai_backend.call_async,  # LLM backend
    tools=[tool1, tool2, ...]             # Available tools
)
```

**What happens:**
- Agent stores the LLM backend function
- Registers tools in `_tool_registry` for later execution
- Validates tool definitions

## 2. Agent Run Phase

```python
response = await agent.run(
    model="gpt-4o-mini",
    input="What's the weather in NYC?",
    instructions="You are a helpful assistant.",
    stream=False,
    apiKey="sk-..."
)
```

### Step-by-Step Execution:

#### Step 1: Input Validation & Normalization
```python
# Validate API key
self._validate_api_key(apiKey)

# Normalize input to OpenAI format
norm_input = self._normalize_input(input)
# "Hello" → [{"role": "user", "content": "Hello"}]
```

#### Step 2: Payload Preparation
```python
payload = {
    "model": model,
    "input": norm_input,
    "instructions": instructions,
    "stream": stream,
    "tools": tools,
    "temperature": temperature,
    "apiKey": apiKey,
    # ... other parameters
}
```

#### Step 3: LLM Call
```python
# Call the LLM backend (OpenAI, Anthropic, etc.)
resp = await self._async_llm(payload)
```

**What the LLM does:**
- Sends request to OpenAI API
- Receives response with potential tool calls
- Returns structured response

#### Step 4: Tool Call Detection & Execution
```python
# Check if LLM wants to use tools
resp = await self._tool_call_execution_loop(
    resp, tools, parallel_tool_calls, self._tool_registry
)
```

**Tool Execution Loop:**
1. **Extract tool calls** from LLM response
2. **Execute tools** (if any):
   ```python
   async def exec_tool(tool_call):
       name = tool_call.get("name")
       args = tool_call.get("arguments", {})
       func = self._tool_registry.get(name)
       result = await func(**args)  # or func(**args) for sync
       return {"role": "tool", "content": result, "tool_call_id": tool_call.get("id")}
   ```
3. **Add tool results** to conversation
4. **Re-query LLM** with updated context
5. **Repeat** until no more tool calls

#### Step 5: Response Formatting
```python
# Convert to standard format
if not isinstance(resp, AgentRunResponse):
    resp = AgentRunResponse(**resp)
return resp
```

## 3. Streaming Mode

When `stream=True`:

```python
async for chunk in agent.run(stream=True, ...):
    # Process each chunk as it arrives
    print(chunk.choices[0].delta.content, end='')
```

**Streaming Flow:**
1. **Create recursive stream generator**
2. **Yield chunks** as they arrive from LLM
3. **Detect tool calls** in streaming chunks
4. **Execute tools** and re-query if needed
5. **Continue streaming** until complete

## 4. Tool Execution Details

### Tool Registration
```python
# During initialization
for tool in self._tools:
    if tool.function.name and tool.callable:
        self._tool_registry[tool.function.name] = tool.callable
```

### Tool Execution
```python
# When LLM calls a tool
tool_calls = self._extract_tool_calls(llm_response)

for tool_call in tool_calls:
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    func = self._tool_registry.get(name)
    
    if asyncio.iscoroutinefunction(func):
        result = await func(**args)  # Async tool
    else:
        result = func(**args)        # Sync tool
    
    # Add result to conversation
    messages.append({
        "role": "tool",
        "content": result,
        "tool_call_id": tool_call.get("id")
    })
```

## 5. Error Handling

```python
try:
    # Main execution
    resp = await self._async_llm(payload)
    # ... tool execution
except Exception as e:
    return self._wrap_error(e, stream=stream, object_type="agent.run", payload=payload)
```

**Error Response Format:**
```python
{
    "id": "uuid",
    "object": "agent.run",
    "created": timestamp,
    "model": "error",
    "choices": [],
    "error": {
        "type": "ExceptionType",
        "message": "Error message"
    }
}
```

## 6. Complete Example Flow

```python
# 1. User asks: "What's the weather in NYC?"
input = "What's the weather in NYC?"

# 2. Agent normalizes input
messages = [{"role": "user", "content": "What's the weather in NYC?"}]

# 3. LLM responds with tool call
llm_response = {
    "choices": [{
        "message": {
            "content": "Let me check the weather for you.",
            "tool_calls": [{
                "name": "get_weather",
                "arguments": {"city": "New York"}
            }]
        }
    }]
}

# 4. Agent executes tool
weather_result = await get_weather(city="New York")
# Returns: "75°F, sunny"

# 5. Agent adds tool result to conversation
messages.append({
    "role": "tool",
    "content": "75°F, sunny",
    "tool_call_id": "call_123"
})

# 6. Agent re-queries LLM with context
final_response = await llm(messages)
# LLM responds: "The weather in NYC is 75°F and sunny."

# 7. Agent returns final response
return AgentRunResponse(
    choices=[{
        "message": {"content": "The weather in NYC is 75°F and sunny."},
        "finish_reason": "stop"
    }]
)
```

## Key Features

1. **Recursive Tool Execution**: Can call multiple tools in sequence
2. **Parallel Tool Calls**: Execute multiple tools simultaneously
3. **Streaming Support**: Real-time response streaming
4. **Error Recovery**: Graceful error handling and reporting
5. **Multi-LLM Support**: Works with OpenAI, Anthropic, Mistral, etc.
6. **Tool Registry**: Dynamic tool registration and execution

## Performance Characteristics

- **Latency**: Depends on LLM API response time + tool execution time
- **Throughput**: Limited by LLM rate limits and tool execution speed
- **Memory**: Keeps conversation context in memory during execution
- **Scalability**: Can handle multiple concurrent agent instances 