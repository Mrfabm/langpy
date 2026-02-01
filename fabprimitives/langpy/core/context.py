"""
LangPy Context - Unified context type that flows between all primitives.

The Context object is the core data structure that ALL primitives read from
and write to, enabling true Lego-like composition.
"""

from __future__ import annotations
import uuid
import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """Standard message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    A single message in a conversation.

    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message content
        name: Optional name (for tool messages)
        tool_call_id: Optional tool call ID (for tool responses)
        metadata: Additional metadata
    """
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM API calls."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        """Create a tool response message."""
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class Document:
    """
    A retrieved document (for RAG pipelines).

    Attributes:
        content: Document text content
        score: Relevance score (0-1, higher is better)
        metadata: Document metadata (source, page, etc.)
        id: Document identifier
    """
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        return self.content


@dataclass
class TokenUsage:
    """
    Token usage tracking.

    Attributes:
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        total_tokens: Total tokens used
        cached_tokens: Tokens served from cache (if applicable)
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens
        )

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "TokenUsage":
        """Create from a dictionary."""
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            cached_tokens=d.get("cached_tokens", 0)
        )


@dataclass
class CostInfo:
    """
    Cost tracking for API calls.

    Attributes:
        prompt_cost: Cost for prompt tokens
        completion_cost: Cost for completion tokens
        total_cost: Total cost
        currency: Currency code (default: USD)
    """
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def __add__(self, other: "CostInfo") -> "CostInfo":
        """Add two CostInfo objects together."""
        return CostInfo(
            prompt_cost=self.prompt_cost + other.prompt_cost,
            completion_cost=self.completion_cost + other.completion_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency
        )


@dataclass
class TraceSpan:
    """
    A single span in the execution trace.

    Attributes:
        name: Span name (primitive name)
        span_id: Unique span identifier
        parent_id: Parent span ID (for nested spans)
        start_time: Start timestamp
        end_time: End timestamp (None if still running)
        attributes: Additional span attributes
        status: Span status (ok, error)
        error_message: Error message if status is error
    """
    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: str = "ok", error_message: Optional[str] = None) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()
        self.status = status
        self.error_message = error_message


@dataclass
class Context:
    """
    Unified Context type that flows between all primitives.

    This is the core data structure enabling true Lego-like composition.
    All primitives read from and write to this context.

    Attributes:
        query: Input question/prompt
        response: Output response
        messages: Conversation history
        documents: Retrieved context (for RAG)
        trace_id: Request tracing ID
        token_usage: Accumulated token usage
        cost: Accumulated cost
        spans: Execution trace spans
        variables: Custom extensible data
        errors: Error log
        metadata: Additional metadata

    Example:
        ctx = Context(query="What is Python?")
        result = await pipeline.process(ctx)
        print(result.unwrap().response)
    """
    # Core data
    query: Optional[str] = None
    response: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)

    # Observability
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostInfo = field(default_factory=CostInfo)
    spans: List[TraceSpan] = field(default_factory=list)

    # Extensible
    variables: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal tracking
    _current_span_id: Optional[str] = field(default=None, repr=False)

    def clone(self) -> "Context":
        """Create a shallow copy of the context."""
        return Context(
            query=self.query,
            response=self.response,
            messages=list(self.messages),
            documents=list(self.documents),
            trace_id=self.trace_id,
            token_usage=self.token_usage,
            cost=self.cost,
            spans=list(self.spans),
            variables=dict(self.variables),
            errors=list(self.errors),
            metadata=dict(self.metadata),
            _current_span_id=self._current_span_id
        )

    def with_query(self, query: str) -> "Context":
        """Return a new context with the given query."""
        ctx = self.clone()
        ctx.query = query
        return ctx

    def with_response(self, response: str) -> "Context":
        """Return a new context with the given response."""
        ctx = self.clone()
        ctx.response = response
        return ctx

    def with_documents(self, documents: List[Document]) -> "Context":
        """Return a new context with the given documents."""
        ctx = self.clone()
        ctx.documents = documents
        return ctx

    def with_messages(self, messages: List[Message]) -> "Context":
        """Return a new context with the given messages."""
        ctx = self.clone()
        ctx.messages = messages
        return ctx

    def add_message(self, message: Message) -> "Context":
        """Return a new context with the message added."""
        ctx = self.clone()
        ctx.messages = list(ctx.messages) + [message]
        return ctx

    def add_document(self, document: Document) -> "Context":
        """Return a new context with the document added."""
        ctx = self.clone()
        ctx.documents = list(ctx.documents) + [document]
        return ctx

    def set(self, key: str, value: Any) -> "Context":
        """Set a variable and return the context."""
        ctx = self.clone()
        ctx.variables[key] = value
        return ctx

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable value."""
        return self.variables.get(key, default)

    def add_error(self, error: str) -> "Context":
        """Add an error to the error log."""
        ctx = self.clone()
        ctx.errors = list(ctx.errors) + [error]
        return ctx

    def add_usage(self, usage: TokenUsage) -> "Context":
        """Add token usage to the accumulated total."""
        ctx = self.clone()
        ctx.token_usage = ctx.token_usage + usage
        return ctx

    def add_cost(self, cost: CostInfo) -> "Context":
        """Add cost to the accumulated total."""
        ctx = self.clone()
        ctx.cost = ctx.cost + cost
        return ctx

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Context":
        """Start a new trace span."""
        ctx = self.clone()
        span = TraceSpan(
            name=name,
            parent_id=ctx._current_span_id,
            attributes=attributes or {}
        )
        ctx.spans = list(ctx.spans) + [span]
        ctx._current_span_id = span.span_id
        return ctx

    def end_span(self, status: str = "ok", error_message: Optional[str] = None) -> "Context":
        """End the current trace span."""
        ctx = self.clone()
        if ctx._current_span_id and ctx.spans:
            # Find and finish the current span
            for span in ctx.spans:
                if span.span_id == ctx._current_span_id:
                    span.finish(status, error_message)
                    ctx._current_span_id = span.parent_id
                    break
        return ctx

    def format_documents(self, separator: str = "\n\n---\n\n") -> str:
        """Format documents as a string for LLM context."""
        if not self.documents:
            return ""
        return separator.join(doc.content for doc in self.documents)

    def format_messages(self) -> List[Dict[str, str]]:
        """Format messages as a list of dicts for LLM API calls."""
        return [msg.to_dict() for msg in self.messages]

    def build_prompt(self, template: Optional[str] = None) -> str:
        """
        Build a prompt string combining query and documents.

        Args:
            template: Optional template with {query} and {context} placeholders.
                     Default: "Context:\n{context}\n\nQuestion: {query}"

        Returns:
            Formatted prompt string
        """
        if template is None:
            template = "Context:\n{context}\n\nQuestion: {query}"

        context_str = self.format_documents()
        return template.format(
            query=self.query or "",
            context=context_str
        )

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self.errors[-1] if self.errors else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "messages": [m.to_dict() for m in self.messages],
            "documents": [{"content": d.content, "score": d.score, "metadata": d.metadata, "id": d.id} for d in self.documents],
            "trace_id": self.trace_id,
            "token_usage": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens
            },
            "cost": {
                "prompt_cost": self.cost.prompt_cost,
                "completion_cost": self.cost.completion_cost,
                "total_cost": self.cost.total_cost,
                "currency": self.cost.currency
            },
            "variables": self.variables,
            "errors": self.errors,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        """Create a context from a dictionary."""
        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id")
            )
            for m in data.get("messages", [])
        ]

        documents = [
            Document(
                content=d["content"],
                score=d.get("score", 0.0),
                metadata=d.get("metadata", {}),
                id=d.get("id", "")
            )
            for d in data.get("documents", [])
        ]

        token_usage = TokenUsage.from_dict(data.get("token_usage", {}))

        cost_data = data.get("cost", {})
        cost = CostInfo(
            prompt_cost=cost_data.get("prompt_cost", 0.0),
            completion_cost=cost_data.get("completion_cost", 0.0),
            total_cost=cost_data.get("total_cost", 0.0),
            currency=cost_data.get("currency", "USD")
        )

        return cls(
            query=data.get("query"),
            response=data.get("response"),
            messages=messages,
            documents=documents,
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            token_usage=token_usage,
            cost=cost,
            variables=data.get("variables", {}),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {})
        )
