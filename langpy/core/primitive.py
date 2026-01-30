"""
LangPy Primitive - Base protocol and class for all primitives.

All primitives implement ONE method: process(ctx: Context) -> Result[Context]
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context
    from .result import Result


@runtime_checkable
class IPrimitive(Protocol):
    """
    Protocol that all primitives must implement.

    Every primitive has a single method: process()
    This enables true Lego-like composition with pipeline operators.

    Example:
        class MyPrimitive:
            async def process(self, ctx: Context) -> Result[Context]:
                # Do something with context
                return Ok(ctx.with_response("Hello!"))
    """

    @property
    def name(self) -> str:
        """Return the primitive name for tracing."""
        ...

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process the context and return a Result.

        Args:
            ctx: Input context

        Returns:
            Result[Context] - Success with modified context or Failure with error

        This is THE method that all primitives must implement.
        It reads from the context, does its work, and returns a new context.
        """
        ...


class BasePrimitive(ABC):
    """
    Base class for primitives with common functionality.

    Provides:
    - Automatic tracing (span creation)
    - Error handling wrapper
    - Pipeline operator support (| and &)

    Subclasses implement _process() instead of process().

    Example:
        class MyPrimitive(BasePrimitive):
            def __init__(self):
                super().__init__("my_primitive")

            async def _process(self, ctx: Context) -> Result[Context]:
                return Ok(ctx.with_response("Hello!"))
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the primitive.

        Args:
            name: Primitive name for tracing. Defaults to class name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Return the primitive name."""
        return self._name

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process the context with automatic tracing and error handling.

        This method wraps _process() with:
        - Span creation for tracing
        - Error handling and Result wrapping

        Args:
            ctx: Input context

        Returns:
            Result[Context]
        """
        from .context import Context
        from .result import Result, Success, Failure, PrimitiveError, ErrorCode

        # Start a span for this primitive
        ctx = ctx.start_span(self._name)

        try:
            result = await self._process(ctx)

            if result.is_success():
                # End span successfully
                result_ctx = result.unwrap()
                result_ctx = result_ctx.end_span("ok")
                return Success(result_ctx)
            else:
                # End span with error
                error = result.error
                ctx = ctx.end_span("error", str(error))
                return Failure(error.with_primitive(self._name))

        except Exception as e:
            # Handle unexpected exceptions
            ctx = ctx.end_span("error", str(e))
            return Failure(PrimitiveError.from_exception(e, self._name))

    @abstractmethod
    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Internal process method to be implemented by subclasses.

        Args:
            ctx: Input context

        Returns:
            Result[Context]
        """
        ...

    def __or__(self, other: "IPrimitive") -> "IPrimitive":
        """
        Sequential composition with the | operator.

        Example:
            pipeline = primitive1 | primitive2 | primitive3
        """
        from .pipeline import Pipeline
        return Pipeline([self, other])

    def __and__(self, other: "IPrimitive") -> "IPrimitive":
        """
        Parallel composition with the & operator.

        Example:
            parallel = primitive1 & primitive2 & primitive3
        """
        from .pipeline import ParallelPrimitives
        return ParallelPrimitives([self, other])

    def __rshift__(self, other: "IPrimitive") -> "IPrimitive":
        """
        Alternative sequential composition with >> operator.

        Example:
            pipeline = primitive1 >> primitive2 >> primitive3
        """
        return self.__or__(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


class FunctionPrimitive(BasePrimitive):
    """
    A primitive created from a function.

    Useful for quick inline primitives without creating a class.

    Example:
        def add_greeting(ctx: Context) -> Result[Context]:
            return Ok(ctx.with_response(f"Hello! {ctx.response or ''}"))

        greeting = FunctionPrimitive("greeting", add_greeting)
    """

    def __init__(
        self,
        name: str,
        func: Any,  # Callable[[Context], Result[Context]] or async version
        is_async: bool = True
    ):
        """
        Create a function-based primitive.

        Args:
            name: Primitive name
            func: Function to execute
            is_async: Whether the function is async (default: True)
        """
        super().__init__(name)
        self._func = func
        self._is_async = is_async

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute the wrapped function."""
        if self._is_async:
            return await self._func(ctx)
        else:
            return self._func(ctx)


def primitive(name: str):
    """
    Decorator to create a primitive from a function.

    Example:
        @primitive("greeting")
        async def add_greeting(ctx: Context) -> Result[Context]:
            return Ok(ctx.with_response(f"Hello! {ctx.response or ''}"))

        # Use it in a pipeline
        pipeline = memory | add_greeting | llm
    """
    import asyncio

    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        return FunctionPrimitive(name, func, is_async)

    return decorator


class IdentityPrimitive(BasePrimitive):
    """
    A primitive that returns the context unchanged.

    Useful as a placeholder or no-op in conditional pipelines.
    """

    def __init__(self):
        super().__init__("identity")

    async def _process(self, ctx: "Context") -> "Result[Context]":
        from .result import Success
        return Success(ctx)


class TransformPrimitive(BasePrimitive):
    """
    A primitive that transforms the context using a function.

    Example:
        # Add a prefix to the response
        transform = TransformPrimitive(
            "add_prefix",
            lambda ctx: ctx.with_response(f"PREFIX: {ctx.response}")
        )
    """

    def __init__(self, name: str, transform_func):
        """
        Create a transform primitive.

        Args:
            name: Primitive name
            transform_func: Function (Context) -> Context
        """
        super().__init__(name)
        self._transform = transform_func

    async def _process(self, ctx: "Context") -> "Result[Context]":
        from .result import Success, Failure, PrimitiveError, ErrorCode
        try:
            new_ctx = self._transform(ctx)
            return Success(new_ctx)
        except Exception as e:
            return Failure(PrimitiveError.from_exception(e, self._name))


class ValidatorPrimitive(BasePrimitive):
    """
    A primitive that validates the context.

    Returns Failure if validation fails, Success otherwise.

    Example:
        # Ensure response is not empty
        validator = ValidatorPrimitive(
            "response_validator",
            lambda ctx: ctx.response is not None and len(ctx.response) > 0,
            "Response cannot be empty"
        )
    """

    def __init__(
        self,
        name: str,
        predicate,
        error_message: str = "Validation failed"
    ):
        """
        Create a validator primitive.

        Args:
            name: Primitive name
            predicate: Function (Context) -> bool
            error_message: Error message if validation fails
        """
        super().__init__(name)
        self._predicate = predicate
        self._error_message = error_message

    async def _process(self, ctx: "Context") -> "Result[Context]":
        from .result import Success, Failure, PrimitiveError, ErrorCode
        try:
            if self._predicate(ctx):
                return Success(ctx)
            else:
                return Failure(PrimitiveError(
                    code=ErrorCode.VALIDATION_ERROR,
                    message=self._error_message,
                    primitive=self._name
                ))
        except Exception as e:
            return Failure(PrimitiveError.from_exception(e, self._name))
