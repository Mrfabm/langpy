"""
LangPy Primitive - Base protocol and class for all primitives.

All primitives implement TWO methods:
1. run(**options) -> Response  (Langbase-compatible API)
2. process(ctx: Context) -> Result[Context]  (Pipeline composition)

This enables both Langbase parity AND true Lego-like composition.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Optional, Dict, Any, TypeVar, Generic, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pydantic import BaseModel

if TYPE_CHECKING:
    from .context import Context
    from .result import Result


# ============================================================================
# Response Types (Langbase-compatible)
# ============================================================================

class BaseResponse(BaseModel):
    """Base response class for all primitives."""
    success: bool = True
    error: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields


class RunOptions(BaseModel):
    """Base options for run() method."""
    class Config:
        extra = "allow"


# Generic type for response
T = TypeVar('T', bound=BaseResponse)


# ============================================================================
# Protocol Definitions
# ============================================================================

@runtime_checkable
class IPrimitive(Protocol):
    """
    Protocol that all primitives must implement.

    Every primitive has TWO methods:
    1. run(**options) -> Response  (Langbase API style)
    2. process(ctx) -> Result[Context]  (Pipeline composition)

    Example:
        class MyPrimitive:
            async def run(self, **options) -> MyResponse:
                # Langbase-style API
                return MyResponse(output="Hello!")

            async def process(self, ctx: Context) -> Result[Context]:
                # Pipeline composition
                return Ok(ctx.with_response("Hello!"))
    """

    @property
    def name(self) -> str:
        """Return the primitive name for tracing."""
        ...

    async def run(self, **options: Any) -> BaseResponse:
        """
        Run the primitive with Langbase-compatible options.

        This is the primary API for direct usage:
            response = await primitive.run(input="Hello", model="gpt-4")

        Args:
            **options: Primitive-specific options

        Returns:
            Response object with primitive output
        """
        ...

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process the context and return a Result.

        This enables pipeline composition:
            pipeline = primitive1 | primitive2 | primitive3
            result = await pipeline.process(ctx)

        Args:
            ctx: Input context

        Returns:
            Result[Context] - Success with modified context or Failure with error
        """
        ...


class BasePrimitive(ABC):
    """
    Base class for primitives with common functionality.

    Provides:
    - Langbase-compatible run() API
    - Pipeline composition via process()
    - Automatic tracing (span creation)
    - Error handling wrapper
    - Pipeline operator support (| and &)

    Subclasses implement:
    - _run(**options) for Langbase API
    - _process(ctx) for pipeline composition

    Example:
        class MyPrimitive(BasePrimitive):
            def __init__(self):
                super().__init__("my_primitive")

            async def _run(self, input: str, **options) -> MyResponse:
                return MyResponse(output=f"Hello {input}!")

            async def _process(self, ctx: Context) -> Result[Context]:
                response = await self._run(input=ctx.query)
                return Ok(ctx.with_response(response.output))
    """

    def __init__(self, name: Optional[str] = None, client: Any = None):
        """
        Initialize the primitive.

        Args:
            name: Primitive name for tracing. Defaults to class name.
            client: Parent Langpy client instance (for shared config).
        """
        self._name = name or self.__class__.__name__
        self._client = client

    @property
    def name(self) -> str:
        """Return the primitive name."""
        return self._name

    # ========================================================================
    # Langbase-compatible API: run()
    # ========================================================================

    async def run(self, **options: Any) -> BaseResponse:
        """
        Run the primitive with Langbase-compatible options.

        This is the primary API for direct usage:
            response = await lb.agent.run(
                model="openai:gpt-4",
                input="Hello",
                instructions="Be helpful"
            )

        Args:
            **options: Primitive-specific options

        Returns:
            Response object with primitive output
        """
        try:
            return await self._run(**options)
        except Exception as e:
            return BaseResponse(success=False, error=str(e))

    async def _run(self, **options: Any) -> BaseResponse:
        """
        Internal run method to be implemented by subclasses.

        Override this method to implement the Langbase-compatible API.

        Args:
            **options: Primitive-specific options

        Returns:
            Response object
        """
        # Default implementation: convert to process() call
        from .context import Context
        from .result import Success

        ctx = Context(
            query=options.get("input") or options.get("query"),
            variables=options
        )
        result = await self.process(ctx)

        if result.is_success():
            output_ctx = result.unwrap()
            return BaseResponse(
                success=True,
                output=output_ctx.response,
                **output_ctx.variables
            )
        else:
            return BaseResponse(success=False, error=str(result.error))

    # ========================================================================
    # Pipeline API: process()
    # ========================================================================

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process the context with automatic tracing and error handling.

        This enables pipeline composition:
            pipeline = memory | agent | thread
            result = await pipeline.process(ctx)

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

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Internal process method to be implemented by subclasses.

        Override this method to implement pipeline composition.

        Default implementation: convert run() response to context.

        Args:
            ctx: Input context

        Returns:
            Result[Context]
        """
        from .result import Success, Failure, PrimitiveError, ErrorCode

        # Default implementation: call _run() and convert to context
        try:
            response = await self._run(
                input=ctx.query,
                **ctx.variables
            )

            if response.success:
                # Extract output from response
                output = getattr(response, 'output', None) or getattr(response, 'response', None)
                new_ctx = ctx.with_response(output) if output else ctx
                return Success(new_ctx)
            else:
                return Failure(PrimitiveError(
                    code=ErrorCode.PRIMITIVE_ERROR,
                    message=response.error or "Unknown error",
                    primitive=self._name
                ))
        except Exception as e:
            return Failure(PrimitiveError.from_exception(e, self._name))

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


# ============================================================================
# Primitive-Specific Response Types (Langbase-compatible)
# ============================================================================

class AgentResponse(BaseResponse):
    """Response from Agent primitive."""
    output: Optional[str] = None
    messages: Optional[list] = None
    tool_calls: Optional[list] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class PipeResponse(BaseResponse):
    """Response from Pipe primitive."""
    output: Optional[str] = None
    messages: Optional[list] = None
    variables: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseResponse):
    """Response from Memory primitive."""
    documents: Optional[list] = None
    count: Optional[int] = None
    action: Optional[str] = None  # "retrieve", "add", "delete"


class ThreadResponse(BaseResponse):
    """Response from Thread primitive."""
    thread_id: Optional[str] = None
    messages: Optional[list] = None
    action: Optional[str] = None  # "create", "append", "list", "delete"


class WorkflowResponse(BaseResponse):
    """Response from Workflow primitive."""
    outputs: Optional[Dict[str, Any]] = None
    steps: Optional[list] = None
    status: Optional[str] = None  # "completed", "failed", "partial"


class ParserResponse(BaseResponse):
    """Response from Parser primitive."""
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    format: Optional[str] = None
    pages: Optional[int] = None


class ChunkerResponse(BaseResponse):
    """Response from Chunker primitive."""
    chunks: Optional[list] = None
    count: Optional[int] = None
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None


class EmbedResponse(BaseResponse):
    """Response from Embed primitive."""
    embeddings: Optional[list] = None
    model: Optional[str] = None
    dimensions: Optional[int] = None
    count: Optional[int] = None


class ToolResponse(BaseResponse):
    """Response from Tools primitive."""
    output: Optional[Any] = None
    tool: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Export all public symbols
# ============================================================================

__all__ = [
    # Protocols
    "IPrimitive",
    # Base classes
    "BasePrimitive",
    "BaseResponse",
    "RunOptions",
    # Helper primitives
    "FunctionPrimitive",
    "IdentityPrimitive",
    "TransformPrimitive",
    "ValidatorPrimitive",
    # Decorator
    "primitive",
    # Response types
    "AgentResponse",
    "PipeResponse",
    "MemoryResponse",
    "ThreadResponse",
    "WorkflowResponse",
    "ParserResponse",
    "ChunkerResponse",
    "EmbedResponse",
    "ToolResponse",
]
