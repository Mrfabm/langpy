"""
LangPy Result - Result types for explicit error handling.

No silent failures - Result types force explicit error handling.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Optional, Callable, Union, Any, List
from dataclasses import dataclass, field
from enum import Enum
import traceback


T = TypeVar("T")
U = TypeVar("U")


class ErrorCode(str, Enum):
    """Standard error codes for primitives."""
    # General errors
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

    # LLM errors
    LLM_API_ERROR = "LLM_API_ERROR"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_CONTEXT_LENGTH = "LLM_CONTEXT_LENGTH"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"

    # Memory/RAG errors
    MEMORY_NOT_FOUND = "MEMORY_NOT_FOUND"
    MEMORY_CONNECTION_ERROR = "MEMORY_CONNECTION_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"

    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MISSING_REQUIRED = "MISSING_REQUIRED"
    INVALID_INPUT = "INVALID_INPUT"

    # Pipeline errors
    PIPELINE_ERROR = "PIPELINE_ERROR"
    PRIMITIVE_NOT_FOUND = "PRIMITIVE_NOT_FOUND"


@dataclass
class PrimitiveError:
    """
    Detailed error information from a primitive.

    Attributes:
        code: Error code for categorization
        message: Human-readable error message
        primitive: Name of the primitive that failed
        details: Additional error details
        cause: Original exception (if any)
        traceback: Stack trace (if available)
    """
    code: ErrorCode
    message: str
    primitive: Optional[str] = None
    details: dict = field(default_factory=dict)
    cause: Optional[Exception] = None
    traceback: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.primitive}] " if self.primitive else ""
        return f"{prefix}{self.code.value}: {self.message}"

    def __repr__(self) -> str:
        return f"PrimitiveError(code={self.code.value!r}, message={self.message!r}, primitive={self.primitive!r})"

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        primitive: Optional[str] = None,
        code: Optional[ErrorCode] = None
    ) -> "PrimitiveError":
        """Create a PrimitiveError from an exception."""
        # Try to infer error code from exception type
        if code is None:
            exc_name = type(exc).__name__.lower()
            if "timeout" in exc_name:
                code = ErrorCode.TIMEOUT
            elif "rate" in exc_name or "ratelimit" in exc_name:
                code = ErrorCode.LLM_RATE_LIMIT
            elif "context" in exc_name or "length" in exc_name:
                code = ErrorCode.LLM_CONTEXT_LENGTH
            elif "validation" in exc_name:
                code = ErrorCode.VALIDATION_ERROR
            else:
                code = ErrorCode.UNKNOWN

        return cls(
            code=code,
            message=str(exc),
            primitive=primitive,
            cause=exc,
            traceback=traceback.format_exc()
        )

    def with_primitive(self, primitive: str) -> "PrimitiveError":
        """Return a copy with the primitive name set."""
        return PrimitiveError(
            code=self.code,
            message=self.message,
            primitive=primitive,
            details=self.details,
            cause=self.cause,
            traceback=self.traceback
        )


@dataclass
class Success(Generic[T]):
    """
    Represents a successful result.

    Attributes:
        value: The success value
    """
    value: T

    def is_success(self) -> bool:
        """Returns True."""
        return True

    def is_failure(self) -> bool:
        """Returns False."""
        return False

    def unwrap(self) -> T:
        """Return the success value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return the success value."""
        return self.value

    def unwrap_or_else(self, f: Callable[[PrimitiveError], T]) -> T:
        """Return the success value."""
        return self.value

    @property
    def error(self) -> None:
        """Returns None for Success."""
        return None

    def map(self, f: Callable[[T], U]) -> "Result[U]":
        """Apply a function to the success value."""
        return Success(f(self.value))

    def flat_map(self, f: Callable[[T], "Result[U]"]) -> "Result[U]":
        """Apply a function that returns a Result."""
        return f(self.value)

    def map_error(self, f: Callable[[PrimitiveError], PrimitiveError]) -> "Result[T]":
        """No-op for Success."""
        return self

    def recover(self, f: Callable[[PrimitiveError], T]) -> "Result[T]":
        """No-op for Success."""
        return self

    def on_success(self, f: Callable[[T], None]) -> "Result[T]":
        """Execute a side effect on success."""
        f(self.value)
        return self

    def on_failure(self, f: Callable[[PrimitiveError], None]) -> "Result[T]":
        """No-op for Success."""
        return self

    def __repr__(self) -> str:
        return f"Success({self.value!r})"


@dataclass
class Failure(Generic[T]):
    """
    Represents a failed result.

    Attributes:
        error: The error information
    """
    _error: PrimitiveError

    def is_success(self) -> bool:
        """Returns False."""
        return False

    def is_failure(self) -> bool:
        """Returns True."""
        return True

    def unwrap(self) -> T:
        """Raise an exception with the error."""
        raise ValueError(f"Cannot unwrap Failure: {self._error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default

    def unwrap_or_else(self, f: Callable[[PrimitiveError], T]) -> T:
        """Apply the function to the error and return the result."""
        return f(self._error)

    @property
    def error(self) -> PrimitiveError:
        """Return the error."""
        return self._error

    def map(self, f: Callable[[T], U]) -> "Result[U]":
        """Return self (propagate the error)."""
        return Failure(self._error)

    def flat_map(self, f: Callable[[T], "Result[U]"]) -> "Result[U]":
        """Return self (propagate the error)."""
        return Failure(self._error)

    def map_error(self, f: Callable[[PrimitiveError], PrimitiveError]) -> "Result[T]":
        """Apply a function to transform the error."""
        return Failure(f(self._error))

    def recover(self, f: Callable[[PrimitiveError], T]) -> "Result[T]":
        """Try to recover from the error."""
        try:
            return Success(f(self._error))
        except Exception as e:
            return Failure(PrimitiveError.from_exception(e))

    def on_success(self, f: Callable[[T], None]) -> "Result[T]":
        """No-op for Failure."""
        return self

    def on_failure(self, f: Callable[[PrimitiveError], None]) -> "Result[T]":
        """Execute a side effect on failure."""
        f(self._error)
        return self

    def __repr__(self) -> str:
        return f"Failure({self._error!r})"


# Type alias for Result
Result = Union[Success[T], Failure[T]]


def Ok(value: T) -> Success[T]:
    """Create a Success result."""
    return Success(value)


def Err(
    code: ErrorCode,
    message: str,
    primitive: Optional[str] = None,
    **details
) -> Failure[Any]:
    """Create a Failure result."""
    return Failure(PrimitiveError(
        code=code,
        message=message,
        primitive=primitive,
        details=details
    ))


def try_result(f: Callable[[], T], primitive: Optional[str] = None) -> Result[T]:
    """
    Execute a function and wrap the result in a Result type.

    Args:
        f: Function to execute
        primitive: Optional primitive name for error context

    Returns:
        Success if function succeeds, Failure if it raises

    Example:
        result = try_result(lambda: int("42"))  # Success(42)
        result = try_result(lambda: int("bad"))  # Failure(...)
    """
    try:
        return Success(f())
    except Exception as e:
        return Failure(PrimitiveError.from_exception(e, primitive))


async def try_result_async(
    f: Callable[[], Any],
    primitive: Optional[str] = None
) -> Result[T]:
    """
    Execute an async function and wrap the result in a Result type.

    Args:
        f: Async function to execute
        primitive: Optional primitive name for error context

    Returns:
        Success if function succeeds, Failure if it raises
    """
    try:
        result = await f()
        return Success(result)
    except Exception as e:
        return Failure(PrimitiveError.from_exception(e, primitive))


def collect_results(results: List[Result[T]]) -> Result[List[T]]:
    """
    Collect a list of Results into a Result of list.

    If any result is a Failure, returns the first Failure.
    Otherwise, returns Success with all values.

    Args:
        results: List of Result objects

    Returns:
        Result[List[T]]

    Example:
        results = [Success(1), Success(2), Success(3)]
        collected = collect_results(results)  # Success([1, 2, 3])

        results = [Success(1), Failure(...), Success(3)]
        collected = collect_results(results)  # Failure(...)
    """
    values = []
    for result in results:
        if result.is_failure():
            return result
        values.append(result.unwrap())
    return Success(values)


def partition_results(results: List[Result[T]]) -> tuple[List[T], List[PrimitiveError]]:
    """
    Partition a list of Results into successes and failures.

    Args:
        results: List of Result objects

    Returns:
        Tuple of (success_values, errors)

    Example:
        results = [Success(1), Failure(err1), Success(2), Failure(err2)]
        successes, errors = partition_results(results)
        # successes = [1, 2]
        # errors = [err1, err2]
    """
    successes = []
    failures = []
    for result in results:
        if result.is_success():
            successes.append(result.unwrap())
        else:
            failures.append(result.error)
    return successes, failures
