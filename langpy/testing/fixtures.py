"""
LangPy Testing Fixtures - Helper functions for creating test data.

Provides convenient factory functions for creating mock primitives and test contexts.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

from ..core.context import Context, Document, Message, TokenUsage, CostInfo
from ..core.result import Result, Success, Failure, PrimitiveError, ErrorCode
from ..core.primitive import BasePrimitive
from .mocks import MockPrimitive, DeterministicLLM, FailingPrimitive


def mock_context(
    query: Optional[str] = None,
    response: Optional[str] = None,
    documents: Optional[List[Dict[str, Any]]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    variables: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Context:
    """
    Create a Context for testing.

    Args:
        query: Input query
        response: Pre-set response
        documents: List of document dicts with 'content' and optional 'score', 'metadata'
        messages: List of message dicts with 'role' and 'content'
        variables: Custom variables to set
        **kwargs: Additional Context fields

    Returns:
        Context instance

    Example:
        ctx = mock_context(
            query="What is Python?",
            documents=[
                {"content": "Python is a programming language.", "score": 0.95},
                {"content": "Python was created by Guido.", "score": 0.85}
            ]
        )
    """
    ctx = Context(query=query, response=response, **kwargs)

    if documents:
        doc_objects = []
        for doc in documents:
            doc_objects.append(Document(
                content=doc["content"],
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
                id=doc.get("id", "")
            ))
        ctx = ctx.with_documents(doc_objects)

    if messages:
        msg_objects = []
        for msg in messages:
            msg_objects.append(Message(
                role=msg["role"],
                content=msg["content"],
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id")
            ))
        ctx = ctx.with_messages(msg_objects)

    if variables:
        for k, v in variables.items():
            ctx = ctx.set(k, v)

    return ctx


def mock_llm_response(
    response: str,
    name: str = "MockLLM",
    token_usage: Optional[Dict[str, int]] = None
) -> MockPrimitive:
    """
    Create a mock LLM primitive that returns a fixed response.

    Args:
        response: The response to return
        name: Primitive name
        token_usage: Optional token usage to simulate

    Returns:
        MockPrimitive

    Example:
        mock = mock_llm_response("Hello, world!")
        result = await mock.process(ctx)
        assert result.unwrap().response == "Hello, world!"
    """
    return MockPrimitive(
        name=name,
        responses=[response]
    )


def mock_llm_responses(
    responses: List[str],
    name: str = "MockLLM",
    cycle: bool = True
) -> MockPrimitive:
    """
    Create a mock LLM that cycles through responses.

    Args:
        responses: List of responses
        name: Primitive name
        cycle: Whether to cycle (or stop at last)

    Returns:
        MockPrimitive

    Example:
        mock = mock_llm_responses(["First", "Second", "Third"])
    """
    return MockPrimitive(
        name=name,
        responses=responses,
        cycle=cycle
    )


def mock_memory_results(
    contents: List[str],
    scores: Optional[List[float]] = None,
    name: str = "MockMemory"
) -> MockPrimitive:
    """
    Create a mock memory/RAG primitive that returns documents.

    Args:
        contents: Document contents to return
        scores: Optional relevance scores (defaults to descending from 1.0)
        name: Primitive name

    Returns:
        MockPrimitive that adds documents to context

    Example:
        mock = mock_memory_results([
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris."
        ])
    """
    if scores is None:
        scores = [1.0 - (i * 0.1) for i in range(len(contents))]

    documents = [
        Document(content=content, score=score)
        for content, score in zip(contents, scores)
    ]

    return MockPrimitive(
        name=name,
        responses=[""],  # Memory doesn't set response, just documents
        documents=documents
    )


def mock_failure(
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    message: str = "Mock failure",
    name: str = "MockFailure"
) -> FailingPrimitive:
    """
    Create a primitive that always fails.

    Args:
        error_code: Error code
        message: Error message
        name: Primitive name

    Returns:
        FailingPrimitive

    Example:
        failing = mock_failure(ErrorCode.LLM_RATE_LIMIT, "Rate limited")
        result = await failing.process(ctx)
        assert result.is_failure()
    """
    return FailingPrimitive(
        error_code=error_code,
        message=message,
        name=name
    )


def mock_pattern_llm(
    patterns: Dict[str, str],
    default: str = "No match found",
    name: str = "PatternLLM"
) -> DeterministicLLM:
    """
    Create a pattern-matching LLM for testing.

    Args:
        patterns: Dict of regex patterns to responses
        default: Default response if no pattern matches
        name: Primitive name

    Returns:
        DeterministicLLM

    Example:
        llm = mock_pattern_llm({
            r"hello|hi": "Hello!",
            r"bye|goodbye": "Goodbye!",
            r"weather": "The weather is sunny."
        })
    """
    return DeterministicLLM(
        name=name,
        patterns=patterns,
        default_response=default
    )


def mock_conversation(
    messages: List[tuple[str, str]],
    name: str = "MockConversation"
) -> Context:
    """
    Create a context with conversation history.

    Args:
        messages: List of (role, content) tuples
        name: Not used, kept for API consistency

    Returns:
        Context with messages

    Example:
        ctx = mock_conversation([
            ("user", "Hello!"),
            ("assistant", "Hi there!"),
            ("user", "How are you?")
        ])
    """
    ctx = Context()
    for role, content in messages:
        ctx = ctx.add_message(Message(role=role, content=content))
    return ctx


def mock_rag_context(
    query: str,
    documents: List[str],
    scores: Optional[List[float]] = None
) -> Context:
    """
    Create a context pre-populated with RAG results.

    Args:
        query: The search query
        documents: Document contents
        scores: Optional relevance scores

    Returns:
        Context ready for LLM processing

    Example:
        ctx = mock_rag_context(
            "What is Python?",
            ["Python is a language.", "Python was created in 1991."]
        )
    """
    if scores is None:
        scores = [1.0 - (i * 0.1) for i in range(len(documents))]

    ctx = Context(query=query)

    for content, score in zip(documents, scores):
        ctx = ctx.add_document(Document(content=content, score=score))

    return ctx


def assert_success(result: Result) -> Context:
    """
    Assert that a result is successful and return the context.

    Args:
        result: Result to check

    Returns:
        The unwrapped context

    Raises:
        AssertionError: If result is a failure

    Example:
        ctx = assert_success(await pipeline.process(input_ctx))
        assert ctx.response == "Expected response"
    """
    assert result.is_success(), f"Expected success but got: {result.error}"
    return result.unwrap()


def assert_failure(result: Result, expected_code: Optional[ErrorCode] = None) -> PrimitiveError:
    """
    Assert that a result is a failure and return the error.

    Args:
        result: Result to check
        expected_code: Optional expected error code

    Returns:
        The error

    Raises:
        AssertionError: If result is a success or wrong error code

    Example:
        error = assert_failure(result, ErrorCode.LLM_RATE_LIMIT)
        assert "rate" in error.message.lower()
    """
    assert result.is_failure(), f"Expected failure but got success: {result.unwrap()}"
    error = result.error

    if expected_code:
        assert error.code == expected_code, f"Expected {expected_code} but got {error.code}"

    return error


def assert_response_contains(result: Result, substring: str) -> Context:
    """
    Assert that result is successful and response contains substring.

    Args:
        result: Result to check
        substring: Expected substring in response

    Returns:
        The context

    Example:
        ctx = assert_response_contains(result, "Hello")
    """
    ctx = assert_success(result)
    assert ctx.response is not None, "Response is None"
    assert substring in ctx.response, f"'{substring}' not found in response: {ctx.response}"
    return ctx


def assert_documents_count(result: Result, expected_count: int) -> Context:
    """
    Assert that result has expected number of documents.

    Args:
        result: Result to check
        expected_count: Expected document count

    Returns:
        The context

    Example:
        ctx = assert_documents_count(result, 5)
    """
    ctx = assert_success(result)
    actual_count = len(ctx.documents)
    assert actual_count == expected_count, f"Expected {expected_count} documents but got {actual_count}"
    return ctx


def assert_variable_equals(result: Result, key: str, expected_value: Any) -> Context:
    """
    Assert that a context variable has expected value.

    Args:
        result: Result to check
        key: Variable key
        expected_value: Expected value

    Returns:
        The context

    Example:
        ctx = assert_variable_equals(result, "confidence", 0.95)
    """
    ctx = assert_success(result)
    actual_value = ctx.get(key)
    assert actual_value == expected_value, f"Variable '{key}' is {actual_value}, expected {expected_value}"
    return ctx
