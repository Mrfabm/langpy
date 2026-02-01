"""
LangPy Testing Mocks - Mock primitives for testing pipelines.

Provides deterministic primitives for unit testing without hitting real APIs.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Union, Pattern
import re
from dataclasses import dataclass, field

from ..core.context import Context, Document, TokenUsage, CostInfo
from ..core.result import Result, Success, Failure, PrimitiveError, ErrorCode
from ..core.primitive import BasePrimitive


class MockPrimitive(BasePrimitive):
    """
    A mock primitive with configurable responses.

    Useful for testing pipelines without hitting real APIs.

    Example:
        mock = MockPrimitive(
            "mock_llm",
            responses=["Hello!", "How can I help?"],
            cycle=True
        )

        result = await mock.process(ctx)  # Returns "Hello!"
        result = await mock.process(ctx)  # Returns "How can I help?"
        result = await mock.process(ctx)  # Returns "Hello!" (cycles)
    """

    def __init__(
        self,
        name: str = "MockPrimitive",
        responses: Optional[List[str]] = None,
        documents: Optional[List[Document]] = None,
        variables: Optional[Dict[str, Any]] = None,
        error: Optional[PrimitiveError] = None,
        cycle: bool = True,
        delay: float = 0.0
    ):
        """
        Create a mock primitive.

        Args:
            name: Primitive name
            responses: List of responses to return
            documents: List of documents to add to context
            variables: Variables to add to context
            error: Error to return (if set, always fails)
            cycle: Whether to cycle through responses
            delay: Artificial delay in seconds
        """
        super().__init__(name)
        self._responses = responses or ["Mock response"]
        self._documents = documents or []
        self._variables = variables or {}
        self._error = error
        self._cycle = cycle
        self._delay = delay
        self._call_count = 0
        self._call_history: List[Context] = []

    async def _process(self, ctx: Context) -> Result[Context]:
        """Process with mock response."""
        import asyncio

        # Record the call
        self._call_history.append(ctx.clone())
        self._call_count += 1

        # Apply delay if set
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        # Return error if configured
        if self._error:
            return Failure(self._error)

        # Get response
        if self._cycle:
            idx = (self._call_count - 1) % len(self._responses)
        else:
            idx = min(self._call_count - 1, len(self._responses) - 1)
        response = self._responses[idx]

        # Build result context
        result_ctx = ctx.with_response(response)

        # Add documents
        for doc in self._documents:
            result_ctx = result_ctx.add_document(doc)

        # Add variables
        for k, v in self._variables.items():
            result_ctx = result_ctx.set(k, v)

        return Success(result_ctx)

    @property
    def call_count(self) -> int:
        """Number of times this mock was called."""
        return self._call_count

    @property
    def call_history(self) -> List[Context]:
        """List of contexts passed to this mock."""
        return self._call_history

    @property
    def last_call(self) -> Optional[Context]:
        """The last context passed to this mock."""
        return self._call_history[-1] if self._call_history else None

    def reset(self) -> None:
        """Reset call count and history."""
        self._call_count = 0
        self._call_history = []

    def assert_called(self) -> None:
        """Assert that the mock was called at least once."""
        assert self._call_count > 0, f"{self._name} was never called"

    def assert_called_times(self, n: int) -> None:
        """Assert that the mock was called exactly n times."""
        assert self._call_count == n, f"{self._name} was called {self._call_count} times, expected {n}"

    def assert_called_with_query(self, query: str) -> None:
        """Assert that the mock was called with a specific query."""
        assert any(
            ctx.query == query for ctx in self._call_history
        ), f"{self._name} was never called with query '{query}'"


class DeterministicLLM(BasePrimitive):
    """
    A deterministic LLM mock that responds based on patterns.

    Useful for testing complex pipelines with predictable behavior.

    Example:
        llm = DeterministicLLM(
            patterns={
                r"hello|hi": "Hello there!",
                r"weather": "The weather is sunny.",
                r".*": "I don't understand."  # Default
            }
        )

        ctx = Context(query="hello")
        result = await llm.process(ctx)
        # Returns "Hello there!"
    """

    def __init__(
        self,
        name: str = "DeterministicLLM",
        patterns: Optional[Dict[str, str]] = None,
        default_response: str = "No matching pattern found.",
        case_sensitive: bool = False
    ):
        """
        Create a pattern-based deterministic LLM.

        Args:
            name: Primitive name
            patterns: Dict of regex patterns to responses
            default_response: Response when no pattern matches
            case_sensitive: Whether patterns are case-sensitive
        """
        super().__init__(name)
        self._patterns = patterns or {}
        self._default_response = default_response
        self._case_sensitive = case_sensitive
        self._compiled_patterns: List[tuple[Pattern, str]] = []

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        for pattern, response in self._patterns.items():
            self._compiled_patterns.append(
                (re.compile(pattern, flags), response)
            )

    async def _process(self, ctx: Context) -> Result[Context]:
        """Match query against patterns and return response."""
        query = ctx.query or ""

        for pattern, response in self._compiled_patterns:
            if pattern.search(query):
                return Success(ctx.with_response(response))

        return Success(ctx.with_response(self._default_response))


class RecordingPrimitive(BasePrimitive):
    """
    A primitive that records all calls and delegates to another primitive.

    Useful for testing and debugging.

    Example:
        recorder = RecordingPrimitive(actual_llm)
        result = await recorder.process(ctx)

        # Check what was passed
        print(recorder.calls[0].input_ctx.query)
        print(recorder.calls[0].output_ctx.response)
    """

    @dataclass
    class CallRecord:
        """Record of a single call."""
        input_ctx: Context
        output_ctx: Optional[Context]
        error: Optional[PrimitiveError]
        duration_ms: float

    def __init__(
        self,
        wrapped: BasePrimitive,
        name: Optional[str] = None
    ):
        """
        Create a recording primitive.

        Args:
            wrapped: Primitive to wrap and record calls for
            name: Optional name override
        """
        super().__init__(name or f"Recording({wrapped.name})")
        self._wrapped = wrapped
        self._calls: List[RecordingPrimitive.CallRecord] = []

    async def _process(self, ctx: Context) -> Result[Context]:
        """Delegate to wrapped primitive and record the call."""
        import time

        start = time.time()
        result = await self._wrapped.process(ctx)
        duration_ms = (time.time() - start) * 1000

        if result.is_success():
            self._calls.append(self.CallRecord(
                input_ctx=ctx,
                output_ctx=result.unwrap(),
                error=None,
                duration_ms=duration_ms
            ))
        else:
            self._calls.append(self.CallRecord(
                input_ctx=ctx,
                output_ctx=None,
                error=result.error,
                duration_ms=duration_ms
            ))

        return result

    @property
    def calls(self) -> List[CallRecord]:
        """List of all recorded calls."""
        return self._calls

    @property
    def call_count(self) -> int:
        """Number of calls recorded."""
        return len(self._calls)

    def clear(self) -> None:
        """Clear all recorded calls."""
        self._calls = []


class SequencePrimitive(BasePrimitive):
    """
    A primitive that returns different results on each call.

    Useful for testing retry logic and error handling.

    Example:
        # Fail twice then succeed
        seq = SequencePrimitive([
            Failure(PrimitiveError(ErrorCode.LLM_API_ERROR, "Rate limited")),
            Failure(PrimitiveError(ErrorCode.LLM_API_ERROR, "Rate limited")),
            Success(ctx.with_response("Finally worked!"))
        ])
    """

    def __init__(
        self,
        results: List[Union[Result[Context], Callable[[Context], Result[Context]]]],
        name: str = "SequencePrimitive",
        repeat_last: bool = True
    ):
        """
        Create a sequence primitive.

        Args:
            results: List of results to return in order
            name: Primitive name
            repeat_last: If True, repeat the last result; if False, fail
        """
        super().__init__(name)
        self._results = results
        self._repeat_last = repeat_last
        self._index = 0

    async def _process(self, ctx: Context) -> Result[Context]:
        """Return the next result in sequence."""
        if self._index >= len(self._results):
            if self._repeat_last:
                self._index = len(self._results) - 1
            else:
                return Failure(PrimitiveError(
                    code=ErrorCode.UNKNOWN,
                    message="Sequence exhausted",
                    primitive=self._name
                ))

        result = self._results[self._index]
        self._index += 1

        if callable(result):
            return result(ctx)
        return result

    def reset(self) -> None:
        """Reset the sequence to the beginning."""
        self._index = 0


class FailingPrimitive(BasePrimitive):
    """
    A primitive that always fails.

    Useful for testing error handling and recovery.

    Example:
        failing = FailingPrimitive(
            ErrorCode.LLM_API_ERROR,
            "API is down"
        )

        result = await failing.process(ctx)
        assert result.is_failure()
    """

    def __init__(
        self,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        message: str = "Intentional failure",
        name: str = "FailingPrimitive"
    ):
        """
        Create a failing primitive.

        Args:
            error_code: Error code to return
            message: Error message
            name: Primitive name
        """
        super().__init__(name)
        self._error_code = error_code
        self._message = message

    async def _process(self, ctx: Context) -> Result[Context]:
        """Always fail."""
        return Failure(PrimitiveError(
            code=self._error_code,
            message=self._message,
            primitive=self._name
        ))


class DelayPrimitive(BasePrimitive):
    """
    A primitive that adds artificial delay.

    Useful for testing timeouts and async behavior.

    Example:
        slow = DelayPrimitive(2.0)  # 2 second delay

        # With timeout
        from langpy.core import retry
        with_timeout = retry(slow, max_attempts=1, timeout=1.0)
    """

    def __init__(
        self,
        delay_seconds: float,
        response: str = "Delayed response",
        name: str = "DelayPrimitive"
    ):
        """
        Create a delay primitive.

        Args:
            delay_seconds: Delay in seconds
            response: Response to return after delay
            name: Primitive name
        """
        super().__init__(name)
        self._delay = delay_seconds
        self._response = response

    async def _process(self, ctx: Context) -> Result[Context]:
        """Wait then return response."""
        import asyncio
        await asyncio.sleep(self._delay)
        return Success(ctx.with_response(self._response))
