"""
LangPy Core - Unified primitives architecture for true Lego-like composition.

This module provides the core building blocks for composable AI primitives:

- Context: Unified context type that flows between all primitives
- Result: Explicit error handling with Success/Failure types
- IPrimitive: Protocol for all primitives
- Pipeline: Sequential (|) and parallel (&) composition operators
- Observability: Cost calculation, metrics, and tracing

Example:
    from langpy.core import Context, pipeline, parallel
    from langpy.primitives import Pipe, MemorySearch

    # Create a RAG pipeline
    memory = MemorySearch(store=my_store, k=5)
    answerer = Pipe(system_prompt="Answer using context.")

    # Compose with | operator
    rag = memory | answerer

    # Execute
    ctx = Context(query="What is LangPy?")
    result = await rag.process(ctx)

    if result.is_success():
        print(result.unwrap().response)
        print(f"Cost: ${result.unwrap().cost.total_cost:.4f}")
"""

# Context and data types
from .context import (
    Context,
    Message,
    MessageRole,
    Document,
    TokenUsage,
    CostInfo,
    TraceSpan,
)

# Result types
from .result import (
    Result,
    Success,
    Failure,
    Ok,
    Err,
    PrimitiveError,
    ErrorCode,
    try_result,
    try_result_async,
    collect_results,
    partition_results,
)

# Primitive base classes and protocols
from .primitive import (
    IPrimitive,
    BasePrimitive,
    FunctionPrimitive,
    IdentityPrimitive,
    TransformPrimitive,
    ValidatorPrimitive,
    primitive,
)

# Pipeline composition
from .pipeline import (
    Pipeline,
    ParallelPrimitives,
    ConditionalPrimitive,
    RecoveryPrimitive,
    RetryPrimitive,
    BranchPrimitive,
    LoopPrimitive,
    pipeline,
    parallel,
    when,
    recover,
    retry,
    branch,
    loop_while,
)

# Observability
from .observability import (
    CostCalculator,
    MetricsCollector,
    MetricsSummary,
    TracingMiddleware,
    MODEL_PRICING,
    MODEL_ALIASES,
    calculate_cost,
    record_metrics,
    get_metrics_summary,
    clear_metrics,
)

# Provider configuration
from .providers import (
    Provider,
    ProviderConfig,
    ModelConfig,
    configure,
    get_config,
    register_model,
    resolve_model,
    get_client,
    list_models,
)

__all__ = [
    # Context and data types
    "Context",
    "Message",
    "MessageRole",
    "Document",
    "TokenUsage",
    "CostInfo",
    "TraceSpan",

    # Result types
    "Result",
    "Success",
    "Failure",
    "Ok",
    "Err",
    "PrimitiveError",
    "ErrorCode",
    "try_result",
    "try_result_async",
    "collect_results",
    "partition_results",

    # Primitive base classes
    "IPrimitive",
    "BasePrimitive",
    "FunctionPrimitive",
    "IdentityPrimitive",
    "TransformPrimitive",
    "ValidatorPrimitive",
    "primitive",

    # Pipeline composition
    "Pipeline",
    "ParallelPrimitives",
    "ConditionalPrimitive",
    "RecoveryPrimitive",
    "RetryPrimitive",
    "BranchPrimitive",
    "LoopPrimitive",
    "pipeline",
    "parallel",
    "when",
    "recover",
    "retry",
    "branch",
    "loop_while",

    # Observability
    "CostCalculator",
    "MetricsCollector",
    "MetricsSummary",
    "TracingMiddleware",
    "MODEL_PRICING",
    "MODEL_ALIASES",
    "calculate_cost",
    "record_metrics",
    "get_metrics_summary",
    "clear_metrics",

    # Provider configuration
    "Provider",
    "ProviderConfig",
    "ModelConfig",
    "configure",
    "get_config",
    "register_model",
    "resolve_model",
    "get_client",
    "list_models",
]
