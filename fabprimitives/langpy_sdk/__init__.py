"""
LangPy SDK - DEPRECATED: Use langpy.Langpy instead.

===========================================================================
MIGRATION NOTICE
===========================================================================

This module (langpy_sdk) is deprecated. Please migrate to the new unified
Langpy client for Langbase-compatible API:

OLD (deprecated):
    from langpy_sdk import Agent, Memory, Pipe
    agent = Agent(model="gpt-4o-mini")
    response = await agent.run("Hello!")

NEW (recommended):
    from langpy import Langpy

    lb = Langpy(api_key="...")

    # All 9 primitives available
    response = await lb.agent.run(model="openai:gpt-4", input="Hello!")
    results = await lb.memory.retrieve(query="search term")

    # Pipeline composition
    rag = lb.memory | lb.pipe
    result = await rag.process(Context(query="What is Python?"))

    # Workflow orchestration
    wf = lb.workflow(name="my-agent")
    wf.step(id="retrieve", primitive=lb.memory)
    wf.step(id="generate", primitive=lb.agent, after=["retrieve"])
    result = await wf.run(query="Hello")

===========================================================================
"""

import warnings

warnings.warn(
    "langpy_sdk is deprecated. Use 'from langpy import Langpy' instead. "
    "See langpy_sdk/__init__.py for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

__version__ = "2.0.0"  # Deprecated

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
