"""
LangPy Pipeline - Pipeline operators for composing primitives.

Provides | (sequential) and & (parallel) operators for true Lego-like composition.
"""

from __future__ import annotations
import asyncio
from typing import List, Optional, Callable, Any, Union, TYPE_CHECKING

from .primitive import BasePrimitive, IPrimitive

if TYPE_CHECKING:
    from .context import Context
    from .result import Result


class Pipeline(BasePrimitive):
    """
    Sequential pipeline that processes primitives in order.

    Created using the | operator or the pipeline() function.

    Example:
        # Using | operator
        pipeline = memory | llm | validator

        # Using pipeline()
        pipeline = pipeline(memory, llm, validator)

        # Execute
        result = await pipeline.process(Context(query="Hello"))
    """

    def __init__(self, primitives: List[IPrimitive], name: Optional[str] = None):
        """
        Create a sequential pipeline.

        Args:
            primitives: List of primitives to execute in order
            name: Optional pipeline name
        """
        super().__init__(name or "Pipeline")
        self._primitives = primitives

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute primitives sequentially, stopping on first failure."""
        from .result import Success, Failure

        current_ctx = ctx

        for prim in self._primitives:
            result = await prim.process(current_ctx)

            if result.is_failure():
                return result

            current_ctx = result.unwrap()

        return Success(current_ctx)

    def __or__(self, other: IPrimitive) -> "Pipeline":
        """Add another primitive to the pipeline."""
        if isinstance(other, Pipeline):
            return Pipeline(self._primitives + other._primitives, self._name)
        return Pipeline(self._primitives + [other], self._name)

    def __repr__(self) -> str:
        names = [p.name for p in self._primitives]
        return f"Pipeline({' | '.join(names)})"


class ParallelPrimitives(BasePrimitive):
    """
    Parallel execution of primitives.

    Created using the & operator or the parallel() function.
    All primitives receive the same input context.
    Results are merged into a single context.

    Example:
        # Using & operator
        parallel = optimist & pessimist & pragmatist

        # Using parallel()
        parallel = parallel(optimist, pessimist, pragmatist)

        # Execute
        result = await parallel.process(Context(query="Analyze this"))
    """

    def __init__(
        self,
        primitives: List[IPrimitive],
        name: Optional[str] = None,
        merge_strategy: str = "concat"
    ):
        """
        Create a parallel primitive executor.

        Args:
            primitives: List of primitives to execute in parallel
            name: Optional name
            merge_strategy: How to merge results - "concat", "first", or "list"
        """
        super().__init__(name or "Parallel")
        self._primitives = primitives
        self._merge_strategy = merge_strategy

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute all primitives in parallel and merge results."""
        from .result import Success, Failure, PrimitiveError, ErrorCode
        from .context import Document

        # Execute all in parallel
        tasks = [prim.process(ctx.clone()) for prim in self._primitives]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results and errors
        successful_contexts = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{self._primitives[i].name}: {str(result)}")
            elif result.is_failure():
                errors.append(f"{self._primitives[i].name}: {result.error.message}")
            else:
                successful_contexts.append(result.unwrap())

        # If all failed, return failure
        if not successful_contexts:
            return Failure(PrimitiveError(
                code=ErrorCode.PIPELINE_ERROR,
                message=f"All parallel primitives failed: {'; '.join(errors)}",
                primitive=self._name
            ))

        # Merge results based on strategy
        merged_ctx = ctx.clone()

        if self._merge_strategy == "concat":
            # Concatenate all responses
            responses = [c.response for c in successful_contexts if c.response]
            merged_ctx = merged_ctx.with_response("\n\n---\n\n".join(responses))

            # Merge all documents
            all_docs = []
            for c in successful_contexts:
                all_docs.extend(c.documents)
            merged_ctx = merged_ctx.with_documents(all_docs)

            # Merge variables
            for c in successful_contexts:
                for k, v in c.variables.items():
                    merged_ctx.variables[k] = v

        elif self._merge_strategy == "first":
            # Use first successful result
            merged_ctx = successful_contexts[0]

        elif self._merge_strategy == "list":
            # Store all responses in a list variable
            responses = [c.response for c in successful_contexts if c.response]
            merged_ctx = merged_ctx.set("parallel_responses", responses)

            # Store all contexts for further processing
            merged_ctx = merged_ctx.set("parallel_contexts", successful_contexts)

        # Accumulate token usage and cost from all
        for c in successful_contexts:
            merged_ctx = merged_ctx.add_usage(c.token_usage)
            merged_ctx = merged_ctx.add_cost(c.cost)

        # Add errors to context if any
        for err in errors:
            merged_ctx = merged_ctx.add_error(err)

        return Success(merged_ctx)

    def __and__(self, other: IPrimitive) -> "ParallelPrimitives":
        """Add another primitive to parallel execution."""
        if isinstance(other, ParallelPrimitives):
            return ParallelPrimitives(
                self._primitives + other._primitives,
                self._name,
                self._merge_strategy
            )
        return ParallelPrimitives(
            self._primitives + [other],
            self._name,
            self._merge_strategy
        )

    def __repr__(self) -> str:
        names = [p.name for p in self._primitives]
        return f"Parallel({' & '.join(names)})"


class ConditionalPrimitive(BasePrimitive):
    """
    Conditional execution based on context state.

    Example:
        conditional = when(
            condition=lambda ctx: ctx.get("needs_search"),
            then_do=search_primitive,
            else_do=skip_primitive
        )
    """

    def __init__(
        self,
        condition: Callable[["Context"], bool],
        then_primitive: IPrimitive,
        else_primitive: Optional[IPrimitive] = None,
        name: Optional[str] = None
    ):
        """
        Create a conditional primitive.

        Args:
            condition: Function (Context) -> bool
            then_primitive: Primitive to execute if condition is True
            else_primitive: Primitive to execute if condition is False (optional)
            name: Optional name
        """
        super().__init__(name or "Conditional")
        self._condition = condition
        self._then_primitive = then_primitive
        self._else_primitive = else_primitive

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute based on condition."""
        from .result import Success
        from .primitive import IdentityPrimitive

        if self._condition(ctx):
            return await self._then_primitive.process(ctx)
        elif self._else_primitive:
            return await self._else_primitive.process(ctx)
        else:
            return Success(ctx)


class RecoveryPrimitive(BasePrimitive):
    """
    Error recovery wrapper.

    Wraps a primitive and provides a recovery handler for failures.

    Example:
        safe = recover(
            risky_primitive,
            handler=lambda err, ctx: ctx.set("fallback", True)
        )
    """

    def __init__(
        self,
        primitive: IPrimitive,
        handler: Callable[["PrimitiveError", "Context"], "Context"],
        name: Optional[str] = None
    ):
        """
        Create a recovery primitive.

        Args:
            primitive: Primitive to wrap
            handler: Recovery function (error, context) -> context
            name: Optional name
        """
        super().__init__(name or f"Recover({primitive.name})")
        self._primitive = primitive
        self._handler = handler

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute and recover on failure."""
        from .result import Success, Failure, PrimitiveError

        result = await self._primitive.process(ctx)

        if result.is_success():
            return result

        # Try to recover
        try:
            recovered_ctx = self._handler(result.error, ctx)
            return Success(recovered_ctx)
        except Exception as e:
            return Failure(PrimitiveError.from_exception(e, self._name))


class RetryPrimitive(BasePrimitive):
    """
    Retry wrapper for primitives.

    Retries a primitive on failure with configurable backoff.

    Example:
        retried = retry(
            flaky_primitive,
            max_attempts=3,
            delay=1.0
        )
    """

    def __init__(
        self,
        primitive: IPrimitive,
        max_attempts: int = 3,
        delay: float = 0.5,
        backoff_multiplier: float = 2.0,
        name: Optional[str] = None
    ):
        """
        Create a retry primitive.

        Args:
            primitive: Primitive to wrap
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries (seconds)
            backoff_multiplier: Multiply delay by this on each retry
            name: Optional name
        """
        super().__init__(name or f"Retry({primitive.name})")
        self._primitive = primitive
        self._max_attempts = max_attempts
        self._delay = delay
        self._backoff_multiplier = backoff_multiplier

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute with retries."""
        from .result import Failure

        last_result = None
        current_delay = self._delay

        for attempt in range(self._max_attempts):
            result = await self._primitive.process(ctx)

            if result.is_success():
                return result

            last_result = result

            # Wait before retry (except on last attempt)
            if attempt < self._max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= self._backoff_multiplier

        return last_result


class BranchPrimitive(BasePrimitive):
    """
    Branch execution based on a router function.

    Routes to different primitives based on context.

    Example:
        router = branch(
            lambda ctx: "search" if ctx.get("needs_search") else "direct",
            routes={
                "search": search_pipeline,
                "direct": direct_pipeline
            }
        )
    """

    def __init__(
        self,
        router: Callable[["Context"], str],
        routes: dict[str, IPrimitive],
        default: Optional[IPrimitive] = None,
        name: Optional[str] = None
    ):
        """
        Create a branch primitive.

        Args:
            router: Function (Context) -> route_key
            routes: Dict mapping route keys to primitives
            default: Default primitive if route not found
            name: Optional name
        """
        super().__init__(name or "Branch")
        self._router = router
        self._routes = routes
        self._default = default

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Route to appropriate primitive."""
        from .result import Success, Failure, PrimitiveError, ErrorCode

        route_key = self._router(ctx)

        if route_key in self._routes:
            return await self._routes[route_key].process(ctx)
        elif self._default:
            return await self._default.process(ctx)
        else:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_NOT_FOUND,
                message=f"No route found for key: {route_key}",
                primitive=self._name
            ))


class LoopPrimitive(BasePrimitive):
    """
    Loop execution while a condition is true.

    Example:
        loop = loop_while(
            condition=lambda ctx: ctx.get("iterations", 0) < 3,
            body=process_step
        )
    """

    def __init__(
        self,
        condition: Callable[["Context"], bool],
        body: IPrimitive,
        max_iterations: int = 100,
        name: Optional[str] = None
    ):
        """
        Create a loop primitive.

        Args:
            condition: Function (Context) -> bool to continue looping
            body: Primitive to execute each iteration
            max_iterations: Safety limit on iterations
            name: Optional name
        """
        super().__init__(name or "Loop")
        self._condition = condition
        self._body = body
        self._max_iterations = max_iterations

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute loop."""
        from .result import Success, Failure, PrimitiveError, ErrorCode

        current_ctx = ctx
        iterations = 0

        while self._condition(current_ctx) and iterations < self._max_iterations:
            result = await self._body.process(current_ctx)

            if result.is_failure():
                return result

            current_ctx = result.unwrap()
            iterations += 1

        if iterations >= self._max_iterations:
            current_ctx = current_ctx.add_error(
                f"Loop reached max iterations ({self._max_iterations})"
            )

        return Success(current_ctx)


# ============================================================================
# Helper Functions
# ============================================================================

def pipeline(*primitives: IPrimitive, name: Optional[str] = None) -> Pipeline:
    """
    Create a sequential pipeline from primitives.

    Example:
        p = pipeline(memory, llm, validator)
        result = await p.process(ctx)
    """
    return Pipeline(list(primitives), name)


def parallel(
    *primitives: IPrimitive,
    merge_strategy: str = "concat",
    name: Optional[str] = None
) -> ParallelPrimitives:
    """
    Create a parallel executor from primitives.

    Args:
        primitives: Primitives to execute in parallel
        merge_strategy: How to merge results - "concat", "first", or "list"
        name: Optional name

    Example:
        p = parallel(optimist, pessimist, pragmatist)
        result = await p.process(ctx)
    """
    return ParallelPrimitives(list(primitives), name, merge_strategy)


def when(
    condition: Callable[["Context"], bool],
    then_do: IPrimitive,
    else_do: Optional[IPrimitive] = None,
    name: Optional[str] = None
) -> ConditionalPrimitive:
    """
    Create a conditional primitive.

    Example:
        cond = when(
            condition=lambda ctx: ctx.get("needs_search"),
            then_do=search,
            else_do=skip
        )
    """
    return ConditionalPrimitive(condition, then_do, else_do, name)


def recover(
    primitive: IPrimitive,
    handler: Callable[["PrimitiveError", "Context"], "Context"],
    name: Optional[str] = None
) -> RecoveryPrimitive:
    """
    Wrap a primitive with error recovery.

    Example:
        safe = recover(
            risky_call,
            handler=lambda err, ctx: ctx.set("fallback", True)
        )
    """
    return RecoveryPrimitive(primitive, handler, name)


def retry(
    primitive: IPrimitive,
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff_multiplier: float = 2.0,
    name: Optional[str] = None
) -> RetryPrimitive:
    """
    Wrap a primitive with retry logic.

    Example:
        retried = retry(flaky_api, max_attempts=3, delay=1.0)
    """
    return RetryPrimitive(primitive, max_attempts, delay, backoff_multiplier, name)


def branch(
    router: Callable[["Context"], str],
    routes: dict[str, IPrimitive],
    default: Optional[IPrimitive] = None,
    name: Optional[str] = None
) -> BranchPrimitive:
    """
    Create a branching primitive.

    Example:
        router = branch(
            lambda ctx: "fast" if ctx.get("urgent") else "thorough",
            routes={"fast": quick_llm, "thorough": detailed_llm}
        )
    """
    return BranchPrimitive(router, routes, default, name)


def loop_while(
    condition: Callable[["Context"], bool],
    body: IPrimitive,
    max_iterations: int = 100,
    name: Optional[str] = None
) -> LoopPrimitive:
    """
    Create a loop primitive.

    Example:
        refine = loop_while(
            condition=lambda ctx: ctx.get("score", 0) < 0.9,
            body=refine_answer,
            max_iterations=5
        )
    """
    return LoopPrimitive(condition, body, max_iterations, name)
