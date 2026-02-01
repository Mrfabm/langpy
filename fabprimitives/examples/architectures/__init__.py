"""
LangPy Agent Architecture Patterns
==================================

This module demonstrates the 8 reference agent architectures built by
DIRECTLY COMPOSING LangPy primitives - no wrapper classes, just primitives
working together like Lego blocks.

Architectures (all use direct primitive composition):

1. augmented_llm      - Agent + Memory + Thread + tools composed together
2. prompt_chaining    - Multiple Pipe.run() calls chained sequentially
3. agent_routing      - Pipe classifies, routes to specialized Pipes
4. parallelization    - asyncio.gather runs multiple Pipes in parallel
5. orchestrator       - Pipe decomposes tasks, Pipes execute in parallel
6. evaluator_optimizer - Generator Pipe + Evaluator Pipe in loop
7. tool_agent         - Agent + @tool decorator composed together
8. memory_agent       - Memory + Pipe + Thread for RAG

Philosophy:
    These examples show that complex agent behaviors emerge from COMPOSING
    simple primitives, not from creating wrapper classes. Each pattern
    demonstrates direct primitive usage.

Usage:
    # Run any pattern demo
    python -m examples.architectures.memory_agent
    python -m examples.architectures.tool_agent
    python -m examples.architectures.prompt_chaining

    # Or import the demo functions
    from examples.architectures.memory_agent import memory_agent_pattern
    await memory_agent_pattern()
"""

# Export the main demo functions for each pattern
from .augmented_llm import augmented_llm_pattern
from .prompt_chaining import prompt_chain_pattern
from .agent_routing import agent_routing_pattern
from .parallelization import parallelization_pattern
from .orchestrator import orchestrator_pattern
from .evaluator_optimizer import evaluator_optimizer_pattern
from .tool_agent import tool_agent_pattern
from .memory_agent import memory_agent_pattern

# Also export the simple utility functions
from .prompt_chaining import simple_chain
from .agent_routing import simple_router
from .parallelization import simple_parallel
from .orchestrator import simple_orchestrator
from .evaluator_optimizer import simple_optimizer
from .tool_agent import simple_tool_agent
from .memory_agent import simple_rag

__all__ = [
    # Main pattern demonstrations
    "augmented_llm_pattern",
    "prompt_chain_pattern",
    "agent_routing_pattern",
    "parallelization_pattern",
    "orchestrator_pattern",
    "evaluator_optimizer_pattern",
    "tool_agent_pattern",
    "memory_agent_pattern",

    # Simple utility functions
    "simple_chain",
    "simple_router",
    "simple_parallel",
    "simple_orchestrator",
    "simple_optimizer",
    "simple_tool_agent",
    "simple_rag",
]
