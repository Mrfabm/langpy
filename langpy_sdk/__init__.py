"""
LangPy SDK - Clean, simple interface for AI primitives.

Usage:
    # Old API (still works for backward compatibility)
    from langpy_sdk import Agent, Memory, Thread, Pipe, Workflow

    agent = Agent(model="gpt-4o-mini")
    response = await agent.run("Hello!")

    # New API (recommended) - True Lego Blocks Architecture
    from langpy.core import Context, pipeline
    from langpy_sdk import Pipe, Memory

    # Compose primitives
    rag = Memory(name="docs") | Pipe(model="gpt-4o-mini")
    result = await rag.process(Context(query="What is LangPy?"))

    if result.is_success():
        print(result.unwrap().response)
"""

__version__ = "2.0.0"

# Original SDK primitives (backward compatible)
from .agent import Agent, ToolDef, tool, AgentResponse
from .memory import Memory, SearchResult, MemoryStats
from .thread import Thread, Message, ThreadInfo
from .pipe import Pipe, PipeResponse
from .workflow import Workflow, Step, WorkflowResult

# Skills (Claude Agent Skills format)
from .skill import Skill, SkillManager

# New core types for composable architecture
try:
    from langpy.core import (
        # Context and data types
        Context,
        Document,
        TokenUsage,
        CostInfo,
        TraceSpan,

        # Result types
        Result,
        Success,
        Failure,
        Ok,
        Err,
        PrimitiveError,
        ErrorCode,

        # Primitive base classes
        IPrimitive,
        BasePrimitive,
        primitive,

        # Pipeline composition
        pipeline,
        parallel,
        when,
        recover,
        retry,
        branch,

        # Observability
        CostCalculator,
        MetricsCollector,
        calculate_cost,
        record_metrics,

        # Configuration
        configure,
        get_config,
    )
    _NEW_API_AVAILABLE = True
except ImportError:
    _NEW_API_AVAILABLE = False

__all__ = [
    # Version
    "__version__",

    # Original SDK (backward compatible)
    # Agent
    "Agent",
    "ToolDef",
    "tool",
    "AgentResponse",
    # Memory
    "Memory",
    "SearchResult",
    "MemoryStats",
    # Thread
    "Thread",
    "Message",
    "ThreadInfo",
    # Pipe
    "Pipe",
    "PipeResponse",
    # Workflow
    "Workflow",
    "Step",
    "WorkflowResult",
    # Skills (Claude Agent Skills format)
    "Skill",
    "SkillManager",

    # New Core Types (if available)
    "Context",
    "Document",
    "TokenUsage",
    "CostInfo",
    "TraceSpan",
    "Result",
    "Success",
    "Failure",
    "Ok",
    "Err",
    "PrimitiveError",
    "ErrorCode",
    "IPrimitive",
    "BasePrimitive",
    "primitive",
    "pipeline",
    "parallel",
    "when",
    "recover",
    "retry",
    "branch",
    "CostCalculator",
    "MetricsCollector",
    "calculate_cost",
    "record_metrics",
    "configure",
    "get_config",
]
