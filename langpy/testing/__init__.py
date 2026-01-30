"""
LangPy Testing - Test utilities for LangPy primitives.

Provides mock primitives, fixtures, and assertion helpers for testing
pipelines without hitting real APIs.

Example:
    from langpy.testing import (
        mock_context,
        mock_llm_response,
        mock_memory_results,
        assert_success,
        assert_response_contains
    )

    # Create test fixtures
    mock_mem = mock_memory_results(["fact 1", "fact 2"])
    mock_llm = mock_llm_response("Test response")

    # Test the pipeline
    result = await (mock_mem | mock_llm).process(mock_context(query="test"))

    # Assert results
    ctx = assert_success(result)
    assert_response_contains(result, "Test")
"""

# Mock primitives
from .mocks import (
    MockPrimitive,
    DeterministicLLM,
    RecordingPrimitive,
    SequencePrimitive,
    FailingPrimitive,
    DelayPrimitive,
)

# Fixtures and helpers
from .fixtures import (
    # Context factories
    mock_context,
    mock_conversation,
    mock_rag_context,

    # Primitive factories
    mock_llm_response,
    mock_llm_responses,
    mock_memory_results,
    mock_failure,
    mock_pattern_llm,

    # Assertions
    assert_success,
    assert_failure,
    assert_response_contains,
    assert_documents_count,
    assert_variable_equals,
)

__all__ = [
    # Mock primitives
    "MockPrimitive",
    "DeterministicLLM",
    "RecordingPrimitive",
    "SequencePrimitive",
    "FailingPrimitive",
    "DelayPrimitive",

    # Context factories
    "mock_context",
    "mock_conversation",
    "mock_rag_context",

    # Primitive factories
    "mock_llm_response",
    "mock_llm_responses",
    "mock_memory_results",
    "mock_failure",
    "mock_pattern_llm",

    # Assertions
    "assert_success",
    "assert_failure",
    "assert_response_contains",
    "assert_documents_count",
    "assert_variable_equals",
]
