# Pipe SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Pipe SDK in LangPy, including the new RAG capabilities.

## Table of Contents

1. [Pipe Creation](#pipe-creation)
2. [Basic Pipe Operations](#basic-pipe-operations)
3. [RAG Operations](#rag-operations)
4. [Integration Options](#integration-options)
5. [Generation Parameters](#generation-parameters)
6. [Advanced Features](#advanced-features)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Pipe Creation

### Factory Method (Recommended)

```python
from sdk import pipe

# Create a pipe factory
pipe_factory = pipe()

# Create pipe with options
pipe_instance = pipe_factory.create(
    default_model="gpt-4o-mini",           # Default model (optional)
    **defaults                             # Additional default settings
)
```

### Direct Creation

```python
from pipe.async_pipe import AsyncPipe

# Create pipe instance
pipe = AsyncPipe(
    default_model="gpt-4o-mini",           # Default model
    **defaults                             # Additional default settings
)
```

## Basic Pipe Operations

### Simple Completion

```python
result = await pipe.run(
    apiKey="your_openai_key",              # OpenAI API key (required)
    input="Your prompt here",              # Input text (required)
    model="gpt-4o-mini",                   # Model (optional)
    stream=False,                          # Enable streaming (optional)
    **kwargs                              # Additional parameters
)
```

### Message-based Completion

```python
result = await pipe.run(
    apiKey="your_openai_key",              # OpenAI API key (required)
    messages=[                             # Messages array (required)
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    model="gpt-4o-mini",                   # Model (optional)
    stream=False,                          # Enable streaming (optional)
    **kwargs                              # Additional parameters
)
```

## RAG Operations

The Pipe SDK provides powerful RAG capabilities with **simple RAG process** but **full Pipe integrations**.

### Simple RAG with Memory

```python
response = await pipe.rag_with_memory(
    query="What is the child policy?",     # User question (required)
    memory=memory_instance,                # Memory instance (required)
    k=5,                                  # Number of chunks to retrieve (optional)
    apiKey="your_openai_key",             # OpenAI API key (required)
    model="gpt-4o-mini",                  # Model (optional)
    **pipe_kwargs                         # ALL Pipe options supported!
)
```

**Key Features:**
- **Simple RAG process** - Basic memory query (no advanced features)
- **Full Pipe integrations** - Thread, tools, generation params, etc.
- **All Pipe options supported** - Everything from `pipe.run()` works

### RAG Context Extraction

```python
context = await pipe.extract_context(
    query="user question",                 # Search query (required)
    memory=memory_instance,                # Memory instance (required)
    k=5                                   # Number of chunks (optional)
)
```

**Returns:** Combined context string from memory chunks

### RAG Response Generation

```python
response = await pipe.extract_response(
    query="user question",                 # User question (required)
    context="retrieved context",          # Context from memory (required)
    apiKey="your_openai_key",             # OpenAI API key (required)
    **pipe_kwargs                         # ALL Pipe options supported!
)
```

**Parameters:** All `pipe.run()` parameters are supported

### Complete RAG Workflow

```python
# Manual RAG workflow
context = await pipe.extract_context("What is the child policy?", mem, k=5)
response = await pipe.extract_response(
    query="What is the child policy?",
    context=context,
    apiKey=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.3
)

# Or use the convenience method
response = await pipe.rag_with_memory(
    query="What is the child policy?",
    memory=mem,
    k=5,
    apiKey=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.3
)
```

## Integration Options

### Thread Integration

```python
# Set up thread interface
from thread.async_thread import AsyncThread
thread_manager = AsyncThread()
thread = await thread_manager.create_thread(name="conversation")
pipe.set_thread_interface(thread_manager)

# Use thread in RAG
response = await pipe.rag_with_memory(
    query="What about the rates for children?",
    memory=mem,
    thread=thread,                        # Thread integration
    apiKey=os.getenv("OPENAI_API_KEY")
)
```

**Benefits:**
- **Conversation history** maintained across queries
- **Context from previous questions** available
- **Thread storage** of all interactions

### Memory Integration

```python
# Set up memory interface
pipe.set_memory_interface(memory_instance)

# Memory is automatically used in pipe.run()
result = await pipe.run(
    messages=[{"role": "user", "content": "What is the child policy?"}],
    memory=memory_instance,               # Memory integration
    apiKey=os.getenv("OPENAI_API_KEY")
)
```

### Tools Integration

```python
# Register tools
def get_current_time():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

pipe.register_tool("get_time", get_current_time)

# Use tools in RAG
response = await pipe.rag_with_memory(
    query="What time is it and what's the child policy?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY")
    # Tools are automatically available
)
```

## Generation Parameters

All generation parameters from `pipe.run()` are supported in RAG operations:

### Basic Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | LLM model to use |
| `temperature` | `float` | `0.7` | Creativity level (0-2) |
| `max_tokens` | `int` | `1000` | Maximum response length |
| `top_p` | `float` | `1.0` | Nucleus sampling parameter |

### Advanced Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `presence_penalty` | `float` | `0.0` | Penalty for new topics (-2 to 2) |
| `frequency_penalty` | `float` | `0.0` | Penalty for repetition (-2 to 2) |
| `stop` | `List[str]` | `None` | Stop sequences |

### Example with Generation Parameters

```python
response = await pipe.rag_with_memory(
    query="How many cottages are there?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,                      # Low temperature for consistency
    max_tokens=50,                        # Short response
    top_p=0.9,                           # Nucleus sampling
    presence_penalty=0.1,                 # Encourage new topics
    frequency_penalty=0.1                 # Reduce repetition
)
```

## Advanced Features

### Few-shot Examples

```python
few_shot_examples = [
    {"role": "user", "content": "What is the check-in time?"},
    {"role": "assistant", "content": "The check-in time is 14:00 hours."},
    {"role": "user", "content": "What is the child policy?"},
    {"role": "assistant", "content": "Children aged 7-11 pay 50% of adult rate."}
]

response = await pipe.rag_with_memory(
    query="What are the room rates?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    few_shot=few_shot_examples,           # Few-shot examples
    temperature=0.3
)
```

### Safety Prompts

```python
response = await pipe.rag_with_memory(
    query="Tell me about the location",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    safety_prompt="You are a helpful assistant for Amakoro Lodge. Always be polite and professional.",
    temperature=0.2
)
```

### Content Moderation

```python
response = await pipe.rag_with_memory(
    query="What activities are available?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    moderate=True,                        # Enable content moderation
    temperature=0.1
)
```

### JSON Output

```python
response = await pipe.rag_with_memory(
    query="What are the check-in times?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    json_output=True,                     # Request JSON output
    response_format={                     # Define JSON schema
        "type": "object",
        "properties": {
            "check_in_time": {"type": "string"},
            "check_out_time": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["check_in_time"]
    }
)
```

### Streaming

```python
# Streaming with RAG
result = await pipe.rag_with_memory(
    query="What activities are available?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    stream=True,                          # Enable streaming
    temperature=0.7
)

async for chunk in result:
    if hasattr(chunk, 'choices') and chunk.choices:
        content = chunk.choices[0].delta.get('content', '')
        if content:
            print(content, end='', flush=True)
```

## Error Handling

### Retry Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `retry_delay` | `float` | `1.0` | Base delay between retries |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

### Example with Error Handling

```python
response = await pipe.rag_with_memory(
    query="What are the room rates?",
    memory=mem,
    apiKey=os.getenv("OPENAI_API_KEY"),
    max_retries=2,                        # Custom retry settings
    retry_delay=0.5,                      # Faster retries
    timeout=15.0                          # Shorter timeout
)
```

## Examples

### Complete RAG with All Integrations

```python
from sdk import pipe, memory
from thread.async_thread import AsyncThread
import os

# Create instances
pipe_factory = pipe()
mem_factory = memory()
thread_manager = AsyncThread()

# Initialize
pipe_instance = pipe_factory.create(default_model="gpt-4o-mini")
mem_instance = mem_factory.create(
    name="amakoro",
    backend="pgvector",
    dsn=os.getenv("POSTGRES_DSN"),
    embedding_model="openai:text-embedding-3-small"
)
thread = await thread_manager.create_thread(name="conversation")

# Set up integrations
pipe_instance.set_memory_interface(mem_instance)
pipe_instance.set_thread_interface(thread_manager)

# Register tools
def get_current_time():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

pipe_instance.register_tool("get_time", get_current_time)

# Complete RAG with all features
response = await pipe_instance.rag_with_memory(
    query="What time is it and what's the child policy?",
    memory=mem_instance,
    thread=thread,                        # Thread integration
    k=5,                                 # Memory chunks
    apiKey=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,                      # Generation params
    max_tokens=150,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    safety_prompt="Be concise and accurate.",  # Safety prompt
    moderate=True                         # Content moderation
)

print(f"Response: {response}")
```

### Interactive RAG Loop

```python
while True:
    question = input("Ask anything: ")
    if question.lower() == "exit":
        break
    
    response = await pipe_instance.rag_with_memory(
        query=question,
        memory=mem_instance,
        thread=thread,
        apiKey=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )
    
    print(f"Answer: {response}")
```

## Key Benefits

### Simple RAG Process
- **Basic memory query** - No advanced features (reranker/BM25)
- **Fast and clean** - Just embedding search
- **Simple context extraction**

### Full Pipe Integrations
- **Thread integration** - Conversation history
- **Tools integration** - Function calling
- **Generation parameters** - All LLM options
- **Advanced features** - Few-shot, safety prompts, moderation
- **Streaming support** - Real-time responses
- **JSON output** - Structured responses

### Best of Both Worlds
- **Simplicity** in RAG process
- **Sophistication** in Pipe integrations
- **Flexibility** to use any Pipe feature with RAG

---

## New in v2.0: Composable Pipeline API

LangPy 2.0 introduces a composable pipeline architecture. Pipe now implements the `IPrimitive` interface and can be composed with other primitives using the `|` (sequential) and `&` (parallel) operators.

### The `process()` Method

```python
from langpy.core import Context
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini", system_prompt="You are helpful.")

# Create context
ctx = Context(query="What is Python?")

# Process and get Result
result = await pipe.process(ctx)

if result.is_success():
    response_ctx = result.unwrap()
    print(response_ctx.response)
    print(f"Tokens: {response_ctx.token_usage.total_tokens}")
    print(f"Cost: ${response_ctx.cost.total_cost:.4f}")
else:
    print(f"Error: {result.error.message}")
```

### Pipeline Composition with `|` Operator

Chain primitives sequentially:

```python
from langpy.core import Context
from langpy_sdk import Memory, Pipe

# Create primitives
memory = Memory(name="docs", k=5)
answerer = Pipe(model="gpt-4o-mini", system_prompt="Answer using the context.")

# Compose with | operator
rag_pipeline = memory | answerer

# Execute pipeline
result = await rag_pipeline.process(Context(query="What is LangPy?"))
if result.is_success():
    print(result.unwrap().response)
```

### Parallel Composition with `&` Operator

Run primitives in parallel and merge results:

```python
from langpy.core import Context, parallel
from langpy_sdk import Pipe

# Create specialized pipes
optimist = Pipe(model="gpt-4o-mini", system_prompt="Find positive aspects.")
pessimist = Pipe(model="gpt-4o-mini", system_prompt="Find negative aspects.")
synthesizer = Pipe(model="gpt-4o-mini", system_prompt="Synthesize both views.")

# Parallel then sequential
pipeline = (optimist & pessimist) | synthesizer

result = await pipeline.process(Context(query="Analyze AI in healthcare"))
```

### Pipe Constructor Options for Composable API

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | LLM model to use |
| `system_prompt` | `str` | `None` | System prompt for the model |
| `name` | `str` | `"Pipe"` | Name for tracing/logging |
| `include_documents` | `bool` | `True` | Include context documents in prompt |
| `document_template` | `str` | See below | Template for formatting documents |

Default document template:
```
Context:
{documents}

Question: {query}
```

### Using Context Variables

```python
from langpy.core import Context
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini")

# Context with variables
ctx = Context(
    query="Summarize the document",
    variables={"style": "bullet points", "max_length": 100}
)

result = await pipe.process(ctx)
```

### Cost and Token Tracking

```python
result = await pipe.process(ctx)
if result.is_success():
    ctx = result.unwrap()

    # Token usage
    print(f"Input tokens: {ctx.token_usage.input_tokens}")
    print(f"Output tokens: {ctx.token_usage.output_tokens}")
    print(f"Total tokens: {ctx.token_usage.total_tokens}")

    # Cost tracking
    print(f"Input cost: ${ctx.cost.input_cost:.4f}")
    print(f"Output cost: ${ctx.cost.output_cost:.4f}")
    print(f"Total cost: ${ctx.cost.total_cost:.4f}")
```

### Backward Compatibility

The original `pipe.run()` API continues to work:

```python
# Original API (still supported)
response = await pipe.run("What is Python?")
print(response.content)

# New composable API
result = await pipe.process(Context(query="What is Python?"))
print(result.unwrap().response)
```

### Testing with Mock Primitives

```python
from langpy.testing import mock_llm_response, assert_success
from langpy.core import Context

# Create mock pipe
mock_pipe = mock_llm_response("This is a test response.")

# Test
result = await mock_pipe.process(Context(query="Test"))
ctx = assert_success(result)
assert ctx.response == "This is a test response."
```

See [CORE_API.md](CORE_API.md) and [TESTING.md](TESTING.md) for complete documentation.