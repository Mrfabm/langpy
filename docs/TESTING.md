# LangPy Testing Guide

The `langpy.testing` module provides utilities for testing LangPy pipelines without hitting real APIs. This enables fast, deterministic, and cost-free testing of your AI applications.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Mock Primitives](#mock-primitives)
- [Fixtures](#fixtures)
- [Assertions](#assertions)
- [Testing Patterns](#testing-patterns)
- [Examples](#examples)

---

## Installation

The testing module is included with LangPy:

```python
from langpy.testing import (
    # Mock primitives
    MockPrimitive,
    DeterministicLLM,
    RecordingPrimitive,
    SequencePrimitive,
    FailingPrimitive,
    DelayPrimitive,

    # Context factories
    mock_context,
    mock_conversation,
    mock_rag_context,

    # Primitive factories
    mock_llm_response,
    mock_llm_responses,
    mock_memory_results,
    mock_failure,
    mock_pattern_llm,

    # Assertions
    assert_success,
    assert_failure,
    assert_response_contains,
    assert_documents_count,
    assert_variable_equals,
)
```

---

## Quick Start

```python
import pytest
from langpy.core import Context
from langpy.testing import mock_llm_response, mock_memory_results, assert_success

@pytest.mark.asyncio
async def test_rag_pipeline():
    # Create mock primitives
    mock_memory = mock_memory_results([
        "Python is a programming language.",
        "Python was created by Guido van Rossum."
    ])
    mock_llm = mock_llm_response("Python is a popular programming language created by Guido van Rossum.")

    # Compose pipeline
    pipeline = mock_memory | mock_llm

    # Test
    result = await pipeline.process(Context(query="What is Python?"))

    # Assert
    ctx = assert_success(result)
    assert "Python" in ctx.response
    assert len(ctx.documents) == 2
```

---

## Mock Primitives

### MockPrimitive

A general-purpose mock that returns configurable responses:

```python
from langpy.testing import MockPrimitive
from langpy.core import Context, Document

# Basic mock
mock = MockPrimitive(
    name="mock_llm",
    responses=["Hello!", "How can I help?"],
    cycle=True  # Cycle through responses
)

# Mock with documents (for memory simulation)
mock = MockPrimitive(
    name="mock_memory",
    responses=[""],
    documents=[
        Document(content="Fact 1", score=0.95),
        Document(content="Fact 2", score=0.85)
    ]
)

# Mock with variables
mock = MockPrimitive(
    name="mock_enricher",
    responses=["Enriched"],
    variables={"confidence": 0.9, "source": "test"}
)

# Mock with delay (for timeout testing)
mock = MockPrimitive(
    name="slow_mock",
    responses=["Delayed response"],
    delay=2.0  # 2 second delay
)

# Inspecting calls
result = await mock.process(ctx)
print(f"Called {mock.call_count} times")
print(f"Last query: {mock.last_call.query}")
mock.assert_called()
mock.assert_called_times(1)
mock.assert_called_with_query("test query")
mock.reset()  # Reset call history
```

### DeterministicLLM

Pattern-based responses for complex testing scenarios:

```python
from langpy.testing import DeterministicLLM

llm = DeterministicLLM(
    patterns={
        r"hello|hi|hey": "Hello! How can I help you?",
        r"weather": "The weather is sunny today.",
        r"python": "Python is a programming language.",
        r".*": "I don't understand that question."  # Default
    },
    case_sensitive=False
)

# Test different inputs
result = await llm.process(Context(query="Hello there"))
# Response: "Hello! How can I help you?"

result = await llm.process(Context(query="Tell me about Python"))
# Response: "Python is a programming language."
```

### RecordingPrimitive

Wrap any primitive to record all calls:

```python
from langpy.testing import RecordingPrimitive
from langpy_sdk import Pipe

# Wrap a real primitive
real_pipe = Pipe(model="gpt-4o-mini")
recorder = RecordingPrimitive(real_pipe)

# Use in tests
result = await recorder.process(ctx)

# Inspect calls
print(f"Total calls: {recorder.call_count}")
for call in recorder.calls:
    print(f"Input query: {call.input_ctx.query}")
    print(f"Output response: {call.output_ctx.response}")
    print(f"Duration: {call.duration_ms}ms")
    if call.error:
        print(f"Error: {call.error}")

recorder.clear()  # Clear recorded calls
```

### SequencePrimitive

Return different results on each call (useful for testing retries):

```python
from langpy.testing import SequencePrimitive
from langpy.core import Success, Failure, PrimitiveError, ErrorCode, Context

# Fail twice, then succeed
seq = SequencePrimitive([
    Failure(PrimitiveError(ErrorCode.LLM_RATE_LIMIT, "Rate limited")),
    Failure(PrimitiveError(ErrorCode.LLM_RATE_LIMIT, "Rate limited")),
    lambda ctx: Success(ctx.with_response("Finally worked!"))
])

# Test retry logic
from langpy.core import retry
retried = retry(seq, max_attempts=3)
result = await retried.process(Context(query="test"))
assert result.is_success()
```

### FailingPrimitive

Always fails with a specific error:

```python
from langpy.testing import FailingPrimitive
from langpy.core import ErrorCode

failing = FailingPrimitive(
    error_code=ErrorCode.LLM_API_ERROR,
    message="API is unavailable",
    name="FailingAPI"
)

result = await failing.process(ctx)
assert result.is_failure()
assert result.error.code == ErrorCode.LLM_API_ERROR
```

### DelayPrimitive

Adds artificial delay (for timeout testing):

```python
from langpy.testing import DelayPrimitive
import asyncio

slow = DelayPrimitive(
    delay_seconds=5.0,
    response="Slow response"
)

# Test with timeout
try:
    result = await asyncio.wait_for(slow.process(ctx), timeout=1.0)
except asyncio.TimeoutError:
    print("Timed out as expected")
```

---

## Fixtures

### Context Factories

```python
from langpy.testing import mock_context, mock_conversation, mock_rag_context

# Simple context
ctx = mock_context(query="What is Python?")

# Context with pre-set response
ctx = mock_context(
    query="Hello",
    response="Hi there!",
    variables={"user_id": "123"}
)

# Context with documents
ctx = mock_context(
    query="Summarize",
    documents=[
        {"content": "Point 1", "score": 0.9},
        {"content": "Point 2", "score": 0.8}
    ]
)

# Context with conversation history
ctx = mock_conversation([
    ("user", "Hello!"),
    ("assistant", "Hi! How can I help?"),
    ("user", "Tell me about AI")
])

# Pre-populated RAG context
ctx = mock_rag_context(
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI.",
        "ML algorithms learn from data."
    ],
    scores=[0.95, 0.87]
)
```

### Primitive Factories

```python
from langpy.testing import (
    mock_llm_response,
    mock_llm_responses,
    mock_memory_results,
    mock_failure,
    mock_pattern_llm
)

# Single response LLM
llm = mock_llm_response("Hello, world!")

# Multiple responses (cycles through)
llm = mock_llm_responses(["First", "Second", "Third"])

# Memory that returns documents
memory = mock_memory_results([
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris."
])

# Memory with custom scores
memory = mock_memory_results(
    contents=["Doc 1", "Doc 2"],
    scores=[0.95, 0.80]
)

# Failing primitive
failing = mock_failure(
    error_code=ErrorCode.LLM_RATE_LIMIT,
    message="Rate limited"
)

# Pattern-based LLM
llm = mock_pattern_llm({
    r"hello": "Hi there!",
    r"bye": "Goodbye!",
    r".*": "I don't understand."
})
```

---

## Assertions

### assert_success

```python
from langpy.testing import assert_success

result = await pipeline.process(ctx)
ctx = assert_success(result)  # Raises AssertionError if failure
print(ctx.response)
```

### assert_failure

```python
from langpy.testing import assert_failure
from langpy.core import ErrorCode

result = await failing_pipeline.process(ctx)
error = assert_failure(result)  # Raises AssertionError if success
print(error.message)

# With expected error code
error = assert_failure(result, expected_code=ErrorCode.LLM_RATE_LIMIT)
```

### assert_response_contains

```python
from langpy.testing import assert_response_contains

result = await pipeline.process(ctx)
ctx = assert_response_contains(result, "Python")  # Asserts success AND substring
```

### assert_documents_count

```python
from langpy.testing import assert_documents_count

result = await memory.process(ctx)
ctx = assert_documents_count(result, 5)  # Asserts exactly 5 documents
```

### assert_variable_equals

```python
from langpy.testing import assert_variable_equals

result = await pipeline.process(ctx)
ctx = assert_variable_equals(result, "confidence", 0.95)
```

---

## Testing Patterns

### Testing RAG Pipelines

```python
import pytest
from langpy.core import Context
from langpy.testing import (
    mock_memory_results,
    mock_llm_response,
    assert_success,
    assert_response_contains
)

@pytest.mark.asyncio
async def test_rag_pipeline_uses_context():
    # Arrange
    memory = mock_memory_results([
        "The sky is blue due to Rayleigh scattering.",
        "Sunlight is scattered by the atmosphere."
    ])
    llm = mock_llm_response("The sky appears blue because of Rayleigh scattering of sunlight.")

    pipeline = memory | llm

    # Act
    result = await pipeline.process(Context(query="Why is the sky blue?"))

    # Assert
    ctx = assert_success(result)
    assert len(ctx.documents) == 2
    assert "Rayleigh" in ctx.response
```

### Testing Error Handling

```python
import pytest
from langpy.core import Context, recover
from langpy.testing import mock_failure, mock_llm_response, assert_success

@pytest.mark.asyncio
async def test_fallback_on_error():
    # Arrange
    primary = mock_failure(ErrorCode.LLM_API_ERROR, "API down")
    fallback = mock_llm_response("Fallback response")

    pipeline = recover(
        primary,
        handler=lambda err, ctx: ctx.set("used_fallback", True)
    ) | fallback

    # Act
    result = await pipeline.process(Context(query="Hello"))

    # Assert
    ctx = assert_success(result)
    assert ctx.get("used_fallback") == True
    assert ctx.response == "Fallback response"
```

### Testing Retry Logic

```python
import pytest
from langpy.core import Context, retry
from langpy.testing import SequencePrimitive, assert_success
from langpy.core import Success, Failure, PrimitiveError, ErrorCode

@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    # Arrange - fail twice, then succeed
    flaky = SequencePrimitive([
        Failure(PrimitiveError(ErrorCode.LLM_RATE_LIMIT, "Rate limited")),
        Failure(PrimitiveError(ErrorCode.LLM_RATE_LIMIT, "Rate limited")),
        lambda ctx: Success(ctx.with_response("Success!"))
    ])

    retried = retry(flaky, max_attempts=3, delay=0.01)

    # Act
    result = await retried.process(Context(query="test"))

    # Assert
    ctx = assert_success(result)
    assert ctx.response == "Success!"
```

### Testing Parallel Pipelines

```python
import pytest
from langpy.core import Context, parallel
from langpy.testing import mock_llm_response, assert_success

@pytest.mark.asyncio
async def test_parallel_execution():
    # Arrange
    fast = mock_llm_response("Fast response")
    slow = mock_llm_response("Slow response")

    pipeline = parallel(fast, slow, merge_strategy="concat")

    # Act
    result = await pipeline.process(Context(query="Analyze"))

    # Assert
    ctx = assert_success(result)
    assert "Fast response" in ctx.response
    assert "Slow response" in ctx.response
```

### Testing with Recording

```python
import pytest
from langpy.core import Context
from langpy.testing import MockPrimitive, RecordingPrimitive, assert_success

@pytest.mark.asyncio
async def test_pipeline_calls():
    # Arrange
    mock = MockPrimitive(responses=["Response"])
    recorder = RecordingPrimitive(mock)

    # Act
    await recorder.process(Context(query="First"))
    await recorder.process(Context(query="Second"))

    # Assert
    assert recorder.call_count == 2
    assert recorder.calls[0].input_ctx.query == "First"
    assert recorder.calls[1].input_ctx.query == "Second"
```

---

## Examples

### Complete Test Suite Example

```python
import pytest
from langpy.core import Context, pipeline, parallel, recover, retry
from langpy.testing import (
    mock_context,
    mock_llm_response,
    mock_memory_results,
    mock_failure,
    MockPrimitive,
    DeterministicLLM,
    assert_success,
    assert_failure,
    assert_response_contains,
)
from langpy.core import ErrorCode


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_basic_rag(self):
        memory = mock_memory_results(["Fact 1", "Fact 2"])
        llm = mock_llm_response("Answer based on facts.")

        rag = memory | llm
        result = await rag.process(mock_context(query="Question"))

        ctx = assert_success(result)
        assert len(ctx.documents) == 2
        assert ctx.response == "Answer based on facts."

    @pytest.mark.asyncio
    async def test_rag_with_empty_memory(self):
        memory = mock_memory_results([])
        llm = mock_llm_response("No context available.")

        rag = memory | llm
        result = await rag.process(mock_context(query="Question"))

        ctx = assert_success(result)
        assert len(ctx.documents) == 0

    @pytest.mark.asyncio
    async def test_rag_with_llm_failure(self):
        memory = mock_memory_results(["Fact"])
        llm = mock_failure(ErrorCode.LLM_API_ERROR, "API error")

        rag = memory | llm
        result = await rag.process(mock_context(query="Question"))

        assert_failure(result, ErrorCode.LLM_API_ERROR)


class TestParallelPipelines:
    @pytest.mark.asyncio
    async def test_multi_perspective(self):
        optimist = mock_llm_response("This is great!")
        pessimist = mock_llm_response("This is concerning.")

        multi = parallel(optimist, pessimist)
        result = await multi.process(mock_context(query="Analyze AI"))

        ctx = assert_success(result)
        assert "great" in ctx.response
        assert "concerning" in ctx.response

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        success = mock_llm_response("Success!")
        failure = mock_failure(ErrorCode.LLM_API_ERROR, "Failed")

        multi = parallel(success, failure)
        result = await multi.process(mock_context(query="Test"))

        # Parallel should succeed if at least one succeeds
        ctx = assert_success(result)
        assert "Success!" in ctx.response
        assert len(ctx.errors) > 0  # Error recorded


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_recovery(self):
        primary = mock_failure(ErrorCode.LLM_API_ERROR, "Primary failed")

        safe = recover(
            primary,
            handler=lambda err, ctx: ctx.with_response("Recovered")
        )

        result = await safe.process(mock_context(query="Test"))
        ctx = assert_success(result)
        assert ctx.response == "Recovered"

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        always_fails = mock_failure(ErrorCode.LLM_RATE_LIMIT, "Rate limited")
        retried = retry(always_fails, max_attempts=3, delay=0.01)

        result = await retried.process(mock_context(query="Test"))
        assert_failure(result, ErrorCode.LLM_RATE_LIMIT)
```

### Pytest Fixtures

```python
# conftest.py
import pytest
from langpy.testing import mock_llm_response, mock_memory_results

@pytest.fixture
def mock_llm():
    return mock_llm_response("Default test response")

@pytest.fixture
def mock_memory():
    return mock_memory_results([
        "Test document 1",
        "Test document 2"
    ])

@pytest.fixture
def test_context():
    from langpy.core import Context
    return Context(query="Test query")

# test_my_pipeline.py
@pytest.mark.asyncio
async def test_with_fixtures(mock_memory, mock_llm, test_context):
    pipeline = mock_memory | mock_llm
    result = await pipeline.process(test_context)
    assert result.is_success()
```
