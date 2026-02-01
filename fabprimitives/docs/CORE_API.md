# LangPy Core API Reference

The `langpy.core` module provides the foundation for LangPy's "True Lego Blocks" architecture, enabling composable AI primitives with unified context, explicit error handling, and built-in observability.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Context](#context)
- [Result Types](#result-types)
- [Primitives](#primitives)
- [Pipeline Composition](#pipeline-composition)
- [Observability](#observability)
- [Provider Configuration](#provider-configuration)
- [Examples](#examples)

---

## Installation

The core module is included with LangPy:

```bash
pip install langpy
```

Import the core types:

```python
from langpy.core import (
    # Context and data types
    Context, Message, Document, TokenUsage, CostInfo,

    # Result types
    Result, Success, Failure, Ok, Err, PrimitiveError, ErrorCode,

    # Primitives
    IPrimitive, BasePrimitive, primitive,

    # Pipeline composition
    pipeline, parallel, when, recover, retry, branch,

    # Observability
    CostCalculator, MetricsCollector, calculate_cost,

    # Configuration
    configure, get_config,
)
```

---

## Quick Start

```python
from langpy.core import Context, pipeline
from langpy_sdk import Memory, Pipe

# Create primitives
memory = Memory(name="knowledge", k=5)
answerer = Pipe(system_prompt="Answer questions using the provided context.")

# Compose with | operator
rag = memory | answerer

# Execute
ctx = Context(query="What is machine learning?")
result = await rag.process(ctx)

if result.is_success():
    response_ctx = result.unwrap()
    print(response_ctx.response)
    print(f"Cost: ${response_ctx.cost.total_cost:.4f}")
    print(f"Tokens: {response_ctx.token_usage.total_tokens}")
else:
    print(f"Error: {result.error}")
```

---

## Context

The `Context` class is the unified data structure that flows between all primitives.

### Creating a Context

```python
from langpy.core import Context, Message, Document

# Simple context with query
ctx = Context(query="What is Python?")

# Context with conversation history
ctx = Context(
    query="Tell me more",
    messages=[
        Message.user("What is Python?"),
        Message.assistant("Python is a programming language.")
    ]
)

# Context with RAG documents
ctx = Context(
    query="Summarize the key points",
    documents=[
        Document(content="Point 1: ...", score=0.95),
        Document(content="Point 2: ...", score=0.87)
    ]
)
```

### Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | `Optional[str]` | Input question/prompt |
| `response` | `Optional[str]` | Output response |
| `messages` | `List[Message]` | Conversation history |
| `documents` | `List[Document]` | Retrieved documents (RAG) |
| `trace_id` | `str` | Request tracing ID |
| `token_usage` | `TokenUsage` | Accumulated token usage |
| `cost` | `CostInfo` | Accumulated cost |
| `spans` | `List[TraceSpan]` | Execution trace |
| `variables` | `Dict[str, Any]` | Custom extensible data |
| `errors` | `List[str]` | Error log |

### Context Methods

```python
# Immutable updates (return new Context)
ctx = ctx.with_query("New query")
ctx = ctx.with_response("Response text")
ctx = ctx.with_documents([doc1, doc2])
ctx = ctx.add_message(Message.user("Hello"))
ctx = ctx.add_document(Document(content="..."))
ctx = ctx.set("custom_key", "custom_value")

# Accessors
value = ctx.get("custom_key", default=None)
prompt = ctx.build_prompt()  # Combines query + documents
formatted = ctx.format_documents()  # Documents as string
messages_list = ctx.format_messages()  # Messages as dicts for API

# State checks
if ctx.has_errors:
    print(ctx.last_error)
```

### Message Class

```python
from langpy.core import Message, MessageRole

# Factory methods
msg = Message.user("Hello!")
msg = Message.assistant("Hi there!")
msg = Message.system("You are a helpful assistant.")
msg = Message.tool("Result", tool_call_id="call_123")

# Direct creation
msg = Message(role=MessageRole.USER, content="Hello", metadata={"source": "web"})

# Convert for API
api_msg = msg.to_dict()  # {"role": "user", "content": "Hello"}
```

### Document Class

```python
from langpy.core import Document

doc = Document(
    content="Paris is the capital of France.",
    score=0.95,  # Relevance score (0-1)
    metadata={"source": "wikipedia", "page": 42},
    id="doc_abc123"
)
```

---

## Result Types

LangPy uses Result types for explicit error handling - no silent failures.

### Success and Failure

```python
from langpy.core import Result, Success, Failure, Ok, Err, ErrorCode

# Creating results
success_result = Ok(ctx)  # or Success(ctx)
failure_result = Err(ErrorCode.LLM_API_ERROR, "API call failed")

# Checking results
if result.is_success():
    ctx = result.unwrap()
    print(ctx.response)
else:
    error = result.error
    print(f"{error.code}: {error.message}")

# Safe unwrapping
ctx = result.unwrap_or(default_ctx)
ctx = result.unwrap_or_else(lambda err: handle_error(err))
```

### Error Codes

```python
from langpy.core import ErrorCode

# General errors
ErrorCode.UNKNOWN
ErrorCode.TIMEOUT
ErrorCode.CANCELLED

# LLM errors
ErrorCode.LLM_API_ERROR
ErrorCode.LLM_RATE_LIMIT
ErrorCode.LLM_CONTEXT_LENGTH
ErrorCode.LLM_INVALID_RESPONSE

# Memory/RAG errors
ErrorCode.MEMORY_NOT_FOUND
ErrorCode.MEMORY_CONNECTION_ERROR
ErrorCode.EMBEDDING_ERROR

# Validation errors
ErrorCode.VALIDATION_ERROR
ErrorCode.MISSING_REQUIRED
ErrorCode.INVALID_INPUT

# Pipeline errors
ErrorCode.PIPELINE_ERROR
ErrorCode.PRIMITIVE_NOT_FOUND
```

### Result Transformations

```python
# Map over success value
new_result = result.map(lambda ctx: ctx.with_response(ctx.response.upper()))

# Chain operations
final_result = result.flat_map(lambda ctx: another_primitive.process(ctx))

# Transform errors
result = result.map_error(lambda err: err.with_primitive("MyPrimitive"))

# Recovery
result = result.recover(lambda err: Context(response="Fallback response"))

# Side effects
result.on_success(lambda ctx: print(f"Success: {ctx.response}"))
result.on_failure(lambda err: log_error(err))
```

---

## Primitives

### IPrimitive Protocol

All primitives implement this interface:

```python
from langpy.core import IPrimitive

class IPrimitive(Protocol):
    @property
    def name(self) -> str: ...

    async def process(self, ctx: Context) -> Result[Context]: ...
```

### Creating Custom Primitives

**Using BasePrimitive (recommended):**

```python
from langpy.core import BasePrimitive, Context, Result, Success, Failure, ErrorCode

class MyPrimitive(BasePrimitive):
    def __init__(self, config_value: str):
        super().__init__("MyPrimitive")
        self.config_value = config_value

    async def _process(self, ctx: Context) -> Result[Context]:
        try:
            # Your logic here
            new_response = f"{self.config_value}: {ctx.query}"
            return Success(ctx.with_response(new_response))
        except Exception as e:
            return Failure(PrimitiveError.from_exception(e, self.name))
```

**Using the @primitive decorator:**

```python
from langpy.core import primitive, Context, Result, Ok

@primitive("add_greeting")
async def add_greeting(ctx: Context) -> Result[Context]:
    return Ok(ctx.with_response(f"Hello! {ctx.response or ''}"))
```

**Using FunctionPrimitive:**

```python
from langpy.core import FunctionPrimitive, Context, Ok

def add_prefix(ctx: Context):
    return Ok(ctx.with_response(f"PREFIX: {ctx.response}"))

prefix_primitive = FunctionPrimitive("add_prefix", add_prefix, is_async=False)
```

### Built-in Utility Primitives

```python
from langpy.core import (
    IdentityPrimitive,      # Returns context unchanged
    TransformPrimitive,     # Transform context with a function
    ValidatorPrimitive,     # Validate context, fail if invalid
)

# Identity - useful as placeholder
identity = IdentityPrimitive()

# Transform
transform = TransformPrimitive(
    "uppercase",
    lambda ctx: ctx.with_response(ctx.response.upper())
)

# Validator
validator = ValidatorPrimitive(
    "check_response",
    predicate=lambda ctx: ctx.response is not None and len(ctx.response) > 0,
    error_message="Response cannot be empty"
)
```

---

## Pipeline Composition

### Sequential Composition (|)

Execute primitives in order, passing context from one to the next:

```python
from langpy.core import pipeline
from langpy_sdk import Memory, Pipe

# Using | operator
rag = Memory(name="docs") | Pipe(system_prompt="Answer using context.")

# Using pipeline() function
rag = pipeline(
    Memory(name="docs"),
    Pipe(system_prompt="Answer using context."),
    name="RAG Pipeline"
)

# Execute
result = await rag.process(Context(query="What is Python?"))
```

### Parallel Composition (&)

Execute primitives simultaneously and merge results:

```python
from langpy.core import parallel
from langpy_sdk import Pipe

# Using & operator
perspectives = (
    Pipe(system_prompt="Find positive aspects.") &
    Pipe(system_prompt="Find negative aspects.") &
    Pipe(system_prompt="Find neutral observations.")
)

# Using parallel() function
perspectives = parallel(
    Pipe(system_prompt="Find positive aspects.", name="Optimist"),
    Pipe(system_prompt="Find negative aspects.", name="Pessimist"),
    merge_strategy="concat"  # or "first", "list"
)

# Execute
result = await perspectives.process(Context(query="Analyze AI ethics"))
# result.unwrap().response contains all three responses concatenated
```

### Conditional Execution

```python
from langpy.core import when

conditional = when(
    condition=lambda ctx: ctx.get("needs_search", False),
    then_do=search_primitive,
    else_do=direct_answer_primitive
)
```

### Error Recovery

```python
from langpy.core import recover

safe_pipeline = recover(
    risky_primitive,
    handler=lambda err, ctx: ctx.set("fallback", True).with_response("Fallback response")
)
```

### Retry Logic

```python
from langpy.core import retry

reliable = retry(
    flaky_primitive,
    max_attempts=3,
    delay=1.0,
    backoff_multiplier=2.0
)
```

### Branching

```python
from langpy.core import branch

router = branch(
    router=lambda ctx: "fast" if ctx.get("urgent") else "thorough",
    routes={
        "fast": quick_llm,
        "thorough": detailed_llm
    },
    default=quick_llm
)
```

### Loop

```python
from langpy.core import loop_while

refine = loop_while(
    condition=lambda ctx: ctx.get("quality_score", 0) < 0.9,
    body=refine_primitive,
    max_iterations=5
)
```

---

## Observability

### Cost Calculation

```python
from langpy.core import CostCalculator, TokenUsage, calculate_cost

# Using the calculator
calculator = CostCalculator()
usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
cost = calculator.calculate("gpt-4o-mini", usage)

print(f"Prompt cost: ${cost.prompt_cost:.4f}")
print(f"Completion cost: ${cost.completion_cost:.4f}")
print(f"Total cost: ${cost.total_cost:.4f}")

# Using the convenience function
cost = calculate_cost("gpt-4o", usage)

# Cost estimation before call
estimated = calculator.estimate_cost("gpt-4o", prompt_tokens=1000, estimated_completion_tokens=500)
```

### Supported Models with Pricing

- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1-preview, o1-mini
- Anthropic: claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku
- Google: gemini-1.5-pro, gemini-1.5-flash, gemini-pro
- Mistral: mistral-large, mistral-medium, mistral-small, mixtral-8x7b
- Groq: llama-3.1-70b-versatile, llama-3.1-8b-instant

### Metrics Collection

```python
from langpy.core import MetricsCollector, record_metrics, get_metrics_summary, clear_metrics

# Record metrics from a context
collector = MetricsCollector()
collector.record(ctx, model="gpt-4o-mini")

# Get summary
summary = collector.get_summary()
print(f"Total calls: {summary.total_calls}")
print(f"Total tokens: {summary.total_tokens}")
print(f"Total cost: ${summary.total_cost:.4f}")
print(f"Average duration: {summary.avg_duration_ms:.2f}ms")
print(f"By primitive: {summary.by_primitive}")
print(f"By model: {summary.by_model}")

# Using global functions
record_metrics(ctx, "gpt-4o-mini")
summary = get_metrics_summary()
clear_metrics()
```

### Tracing

```python
from langpy.core import TracingMiddleware

tracer = TracingMiddleware(service_name="my-app")

# Start trace
ctx = tracer.start_trace(Context(query="Hello"))

# Execute pipeline (spans are created automatically by primitives)
result = await pipeline.process(ctx)

# End trace
ctx = tracer.end_trace(result.unwrap())

# Get trace summary
summary = tracer.get_trace_summary(ctx)
print(f"Trace ID: {summary['trace_id']}")
print(f"Duration: {summary['duration_ms']:.2f}ms")
print(f"Spans: {len(summary['spans'])}")
```

---

## Provider Configuration

### Global Configuration

```python
from langpy.core import configure, get_config

# Configure globally
configure(
    default_model="gpt-4o",
    api_keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-..."
    },
    timeout=60.0,
    max_retries=3
)

# Get current config
config = get_config()
```

### Model Aliases

Use convenient aliases instead of full model names:

```python
from langpy.core import resolve_model

# Built-in aliases
model = resolve_model("fast")      # -> gpt-4o-mini
model = resolve_model("smart")     # -> gpt-4o
model = resolve_model("cheap")     # -> gpt-3.5-turbo
model = resolve_model("local")     # -> ollama llama3
```

### Register Custom Models

```python
from langpy.core import register_model

register_model(
    alias="my-fine-tuned",
    name="ft:gpt-4o-mini:my-org:custom:abc123",
    provider="openai",
    context_window=128000
)
```

### List Available Models

```python
from langpy.core import list_models

all_models = list_models()
openai_models = list_models(provider="openai")
```

---

## Examples

### Basic RAG Pipeline

```python
from langpy.core import Context
from langpy_sdk import Memory, Pipe

# Setup
memory = Memory(name="knowledge", k=5)
await memory.add("Python is a programming language created by Guido van Rossum.")
await memory.add("Python emphasizes code readability and simplicity.")

answerer = Pipe(
    model="gpt-4o-mini",
    system_prompt="Answer questions using the provided context. Be concise."
)

# Compose
rag = memory | answerer

# Execute
result = await rag.process(Context(query="Who created Python?"))
print(result.unwrap().response)
# Output: "Python was created by Guido van Rossum."
```

### Multi-Perspective Analysis

```python
from langpy.core import Context, parallel
from langpy_sdk import Pipe

optimist = Pipe(system_prompt="Analyze from an optimistic perspective.", name="Optimist")
pessimist = Pipe(system_prompt="Analyze from a pessimistic perspective.", name="Pessimist")
synthesizer = Pipe(system_prompt="Synthesize multiple perspectives into a balanced view.")

# Parallel then sequential
pipeline = parallel(optimist, pessimist) | synthesizer

result = await pipeline.process(Context(query="What are the implications of AI in healthcare?"))
print(result.unwrap().response)
```

### Error Handling and Recovery

```python
from langpy.core import Context, recover, retry
from langpy_sdk import Pipe

# Primary model with retry
primary = retry(
    Pipe(model="gpt-4o", name="Primary"),
    max_attempts=3,
    delay=1.0
)

# Fallback model
fallback = Pipe(model="gpt-3.5-turbo", name="Fallback")

# Recovery pipeline
safe_pipeline = recover(
    primary,
    handler=lambda err, ctx: ctx.set("used_fallback", True)
) | fallback

result = await safe_pipeline.process(Context(query="Hello"))
if result.unwrap().get("used_fallback"):
    print("Used fallback model")
```

### With Observability

```python
from langpy.core import Context, TracingMiddleware, record_metrics, get_metrics_summary
from langpy_sdk import Memory, Pipe

# Setup tracing
tracer = TracingMiddleware(service_name="rag-service")

# Create pipeline
rag = Memory(name="docs") | Pipe(system_prompt="Answer using context.")

# Execute with tracing
ctx = tracer.start_trace(Context(query="What is Python?"))
result = await rag.process(ctx)
ctx = tracer.end_trace(result.unwrap())

# Record metrics
record_metrics(ctx, model="gpt-4o-mini")

# View metrics
summary = get_metrics_summary()
print(f"Total cost so far: ${summary.total_cost:.4f}")
```

---

## Migration from v1.x

### Old API (still works)

```python
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini")
response = await pipe.run("Hello!")
print(response.content)  # PipeResponse.content
```

### New API (recommended)

```python
from langpy.core import Context
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini")
result = await pipe.process(Context(query="Hello!"))
print(result.unwrap().response)  # Context.response
```

### Key Differences

| Aspect | Old API | New API |
|--------|---------|---------|
| Input | String or messages | `Context` object |
| Output | `PipeResponse` | `Result[Context]` |
| Error handling | Exceptions | `Result` type |
| Composition | Manual | `\|` and `&` operators |
| Observability | Manual | Built-in |
| Type safety | Runtime | Protocol-based |
