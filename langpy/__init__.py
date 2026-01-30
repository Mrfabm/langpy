"""
LangPy - True Lego Blocks Architecture for AI Primitives.

LangPy provides composable AI primitives with:
- Unified Context type that flows between all primitives
- Pipeline operators (| and &) for real Lego-like composition
- Result types for proper error handling
- Built-in observability and testing support

Example:
    from langpy.core import Context, pipeline
    from langpy.primitives import Pipe, MemorySearch

    # Compose primitives
    rag = MemorySearch(store=my_store) | Pipe(system_prompt="Answer using context.")

    # Execute
    result = await rag.process(Context(query="What is LangPy?"))
    print(result.unwrap().response)
"""

__version__ = "2.0.0"

# Re-export core types for convenience
from .core import (
    # Context
    Context,
    Message,
    Document,
    TokenUsage,
    CostInfo,

    # Result
    Result,
    Success,
    Failure,
    Ok,
    Err,
    PrimitiveError,
    ErrorCode,

    # Primitives
    IPrimitive,
    BasePrimitive,
    primitive,

    # Pipeline
    pipeline,
    parallel,
    when,
    recover,
    retry,
    branch,

    # Configuration
    configure,
    get_config,
)

__all__ = [
    "__version__",
    # Context
    "Context",
    "Message",
    "Document",
    "TokenUsage",
    "CostInfo",
    # Result
    "Result",
    "Success",
    "Failure",
    "Ok",
    "Err",
    "PrimitiveError",
    "ErrorCode",
    # Primitives
    "IPrimitive",
    "BasePrimitive",
    "primitive",
    # Pipeline
    "pipeline",
    "parallel",
    "when",
    "recover",
    "retry",
    "branch",
    # Configuration
    "configure",
    "get_config",
]
