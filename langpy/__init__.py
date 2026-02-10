"""
LangPy - Langbase-compatible AI Primitives for Python.

LangPy is a Python implementation of Langbase's AI primitives,
providing a unified SDK for building AI agents and applications.

Quick Start:
    from langpy import Langpy

    # Initialize (like Langbase SDK)
    lb = Langpy(api_key="your-api-key")

    # Use primitives
    response = await lb.agent.run(
        model="openai:gpt-4",
        input="Hello!",
        instructions="Be helpful"
    )

    # RAG with memory
    await lb.memory.add(documents=[...])
    results = await lb.memory.retrieve(query="search")

    # Compose with pipelines
    rag = lb.memory | lb.pipe
    result = await rag.process(Context(query="What is Python?"))

    # Orchestrate with workflows
    wf = lb.workflow(name="my-agent")
    wf.step(id="retrieve", primitive=lb.memory)
    wf.step(id="generate", primitive=lb.agent, after=["retrieve"])
    result = await wf.run(query="Hello")
"""

__version__ = "2.0.0"


# ============================================================================
# Unified Langpy Client (Langbase SDK parity)
# ============================================================================

import os
from typing import Optional


class Langpy:
    """
    Unified LangPy client - The single entry point for all primitives.

    This mirrors the Langbase SDK design:
        const langbase = new Langbase({ apiKey: "..." })
        langbase.agent.run(...)
        langbase.memory.retrieve(...)

    In Python:
        lb = Langpy(api_key="...")
        await lb.agent.run(...)
        await lb.memory.retrieve(...)

    All 9 Langbase primitives are available:
    - lb.agent    - Unified LLM API (100+ models)
    - lb.pipe     - Single LLM call with templates
    - lb.memory   - Vector storage and RAG
    - lb.thread   - Conversation history
    - lb.workflow - Multi-step orchestration
    - lb.parser   - Document text extraction
    - lb.chunker  - Text segmentation
    - lb.embed    - Text to vectors
    - lb.tools    - Pre-built and custom tools
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **config
    ):
        """
        Initialize the Langpy client.

        Args:
            api_key: Default API key for LLM providers.
                     Can also be set via LANGPY_API_KEY env var.
            **config: Additional configuration options.
        """
        self._api_key = api_key or os.getenv("LANGPY_API_KEY")
        self._config = config

        # Lazy-initialized primitives
        self._agent = None
        self._pipe = None
        self._memory = None
        self._thread = None
        self._workflow_builder = None
        self._parser = None
        self._chunker = None
        self._embed = None
        self._tools = None

    # ========================================================================
    # Primitive Properties (lazy initialization)
    # ========================================================================

    @property
    def agent(self):
        """Agent primitive - Unified LLM API for 100+ models."""
        if self._agent is None:
            from .primitives import Agent
            self._agent = Agent(client=self)
        return self._agent

    @property
    def pipe(self):
        """Pipe primitive - Single LLM call with templates."""
        if self._pipe is None:
            from .primitives import Pipe
            self._pipe = Pipe(client=self)
        return self._pipe

    @property
    def memory(self):
        """Memory primitive - Vector storage and RAG."""
        if self._memory is None:
            from .primitives import Memory
            self._memory = Memory(
                client=self,
                backend=self._config.get("memory_backend", "faiss")
            )
        return self._memory

    @property
    def thread(self):
        """Thread primitive - Conversation history management."""
        if self._thread is None:
            from .primitives import Thread
            self._thread = Thread(client=self)
        return self._thread

    @property
    def parser(self):
        """Parser primitive - Document text extraction."""
        if self._parser is None:
            from .primitives import Parser
            self._parser = Parser(client=self)
        return self._parser

    @property
    def chunker(self):
        """Chunker primitive - Text segmentation."""
        if self._chunker is None:
            from .primitives import Chunker
            self._chunker = Chunker(client=self)
        return self._chunker

    @property
    def embed(self):
        """Embed primitive - Text to vector embeddings."""
        if self._embed is None:
            from .primitives import Embed
            self._embed = Embed(client=self)
        return self._embed

    @property
    def tools(self):
        """Tools primitive - Pre-built and custom tools."""
        if self._tools is None:
            from .primitives import Tools
            self._tools = Tools(client=self)
        return self._tools

    # ========================================================================
    # Workflow Factory
    # ========================================================================

    def workflow(self, name: str = None, debug: bool = False):
        """
        Create a new workflow builder.

        Args:
            name: Workflow name
            debug: Enable debug logging

        Returns:
            Workflow primitive for step configuration

        Example:
            wf = lb.workflow(name="rag-agent")
            wf.step(id="retrieve", primitive=lb.memory)
            wf.step(id="generate", primitive=lb.agent, after=["retrieve"])
            result = await wf.run(query="Hello")
        """
        from .primitives import Workflow
        return Workflow(client=self, name=name or "workflow", debug=debug)

    # ========================================================================
    # Configuration
    # ========================================================================

    def configure(self, **options):
        """Update configuration."""
        self._config.update(options)
        return self

    @property
    def api_key(self) -> Optional[str]:
        """Get the configured API key."""
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        """Set the API key."""
        self._api_key = value


# Convenience function
def create_client(api_key: str = None, **config) -> Langpy:
    """Create a new Langpy client."""
    return Langpy(api_key=api_key, **config)

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
    loop_while,
    map_over,
    reduce,

    # Configuration
    configure,
    get_config,
)

__all__ = [
    "__version__",
    # Unified Client (Langbase SDK parity)
    "Langpy",
    "create_client",
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
    "loop_while",
    "map_over",
    "reduce",
    # Configuration
    "configure",
    "get_config",
]
