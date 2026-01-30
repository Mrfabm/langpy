"""
LangPy Observability - Cost calculation, metrics collection, and tracing.

Provides built-in observability for all primitives.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

from .context import Context, TokenUsage, CostInfo, TraceSpan


# ============================================================================
# Model Pricing (per 1M tokens, as of 2024)
# ============================================================================

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "o1-preview": {"prompt": 15.00, "completion": 60.00},
    "o1-mini": {"prompt": 3.00, "completion": 12.00},

    # Anthropic
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-opus": {"prompt": 15.00, "completion": 75.00},
    "claude-3-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},

    # Google
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-pro": {"prompt": 0.50, "completion": 1.50},

    # Mistral
    "mistral-large": {"prompt": 4.00, "completion": 12.00},
    "mistral-medium": {"prompt": 2.70, "completion": 8.10},
    "mistral-small": {"prompt": 1.00, "completion": 3.00},
    "mixtral-8x7b": {"prompt": 0.70, "completion": 0.70},

    # Groq (fast inference)
    "llama-3.1-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},

    # Embedding models (per 1M tokens)
    "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.13, "completion": 0.0},
    "text-embedding-ada-002": {"prompt": 0.10, "completion": 0.0},
}

# Model aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    # Speed-optimized
    "fast": "gpt-4o-mini",
    "fastest": "llama-3.1-8b-instant",

    # Quality-optimized
    "smart": "gpt-4o",
    "smartest": "claude-3-opus",

    # Cost-optimized
    "cheap": "gpt-3.5-turbo",
    "cheapest": "llama-3.1-8b-instant",

    # Balanced
    "balanced": "gpt-4o-mini",

    # Provider-specific defaults
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet",
    "google": "gemini-1.5-flash",
    "mistral": "mistral-small",
    "groq": "llama-3.1-70b-versatile",
}


class CostCalculator:
    """
    Calculate costs for LLM API calls.

    Example:
        calc = CostCalculator()
        cost = calc.calculate("gpt-4o-mini", TokenUsage(prompt_tokens=1000, completion_tokens=500))
        print(f"Total cost: ${cost.total_cost:.4f}")
    """

    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the cost calculator.

        Args:
            custom_pricing: Optional custom pricing to override defaults
        """
        self._pricing = {**MODEL_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def resolve_model(self, model: str) -> str:
        """Resolve a model alias to the actual model name."""
        return MODEL_ALIASES.get(model, model)

    def get_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """Get pricing for a model."""
        resolved = self.resolve_model(model)

        # Try exact match
        if resolved in self._pricing:
            return self._pricing[resolved]

        # Try prefix match (for versioned models)
        for key in self._pricing:
            if resolved.startswith(key) or key.startswith(resolved):
                return self._pricing[key]

        return None

    def calculate(self, model: str, usage: TokenUsage) -> CostInfo:
        """
        Calculate the cost for a given model and token usage.

        Args:
            model: Model name or alias
            usage: Token usage

        Returns:
            CostInfo with calculated costs
        """
        pricing = self.get_pricing(model)

        if pricing is None:
            # Unknown model - return zero cost with warning
            return CostInfo(
                prompt_cost=0.0,
                completion_cost=0.0,
                total_cost=0.0,
                currency="USD"
            )

        # Calculate costs (pricing is per 1M tokens)
        prompt_cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1_000_000) * pricing["completion"]

        return CostInfo(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=prompt_cost + completion_cost,
            currency="USD"
        )

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        estimated_completion_tokens: int
    ) -> CostInfo:
        """
        Estimate cost before making a call.

        Args:
            model: Model name or alias
            prompt_tokens: Number of prompt tokens
            estimated_completion_tokens: Estimated completion tokens

        Returns:
            Estimated CostInfo
        """
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=prompt_tokens + estimated_completion_tokens
        )
        return self.calculate(model, usage)


@dataclass
class MetricsSummary:
    """Summary of collected metrics."""
    total_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    errors: int = 0
    by_primitive: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate metrics from pipeline executions.

    Example:
        collector = MetricsCollector()
        collector.record(ctx)  # After pipeline execution

        summary = collector.get_summary()
        print(f"Total cost: ${summary.total_cost:.4f}")
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._cost_calculator = CostCalculator()

    def record(self, ctx: Context, model: Optional[str] = None) -> None:
        """
        Record metrics from a context.

        Args:
            ctx: Context with execution data
            model: Optional model name for cost calculation
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": ctx.trace_id,
            "token_usage": {
                "prompt_tokens": ctx.token_usage.prompt_tokens,
                "completion_tokens": ctx.token_usage.completion_tokens,
                "total_tokens": ctx.token_usage.total_tokens
            },
            "cost": {
                "prompt_cost": ctx.cost.prompt_cost,
                "completion_cost": ctx.cost.completion_cost,
                "total_cost": ctx.cost.total_cost
            },
            "spans": [],
            "errors": list(ctx.errors),
            "model": model
        }

        # Record span data
        for span in ctx.spans:
            record["spans"].append({
                "name": span.name,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "error": span.error_message
            })

        self._records.append(record)

    def get_summary(self) -> MetricsSummary:
        """Get aggregated metrics summary."""
        summary = MetricsSummary()

        for record in self._records:
            summary.total_calls += 1
            summary.total_tokens += record["token_usage"]["total_tokens"]
            summary.total_prompt_tokens += record["token_usage"]["prompt_tokens"]
            summary.total_completion_tokens += record["token_usage"]["completion_tokens"]
            summary.total_cost += record["cost"]["total_cost"]

            if record["errors"]:
                summary.errors += len(record["errors"])

            # Aggregate by primitive
            for span in record["spans"]:
                name = span["name"]
                if name not in summary.by_primitive:
                    summary.by_primitive[name] = {
                        "calls": 0,
                        "total_duration_ms": 0.0,
                        "errors": 0
                    }
                summary.by_primitive[name]["calls"] += 1
                if span["duration_ms"]:
                    summary.by_primitive[name]["total_duration_ms"] += span["duration_ms"]
                    summary.total_duration_ms += span["duration_ms"]
                if span["status"] == "error":
                    summary.by_primitive[name]["errors"] += 1

            # Aggregate by model
            model = record.get("model")
            if model:
                if model not in summary.by_model:
                    summary.by_model[model] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost": 0.0
                    }
                summary.by_model[model]["calls"] += 1
                summary.by_model[model]["tokens"] += record["token_usage"]["total_tokens"]
                summary.by_model[model]["cost"] += record["cost"]["total_cost"]

        # Calculate average duration
        if summary.total_calls > 0:
            summary.avg_duration_ms = summary.total_duration_ms / summary.total_calls

        return summary

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._records = []

    def export(self) -> List[Dict[str, Any]]:
        """Export all records as a list of dicts."""
        return list(self._records)


class TracingMiddleware:
    """
    Middleware for adding tracing to contexts.

    Can be used to automatically add trace IDs and parent span context.

    Example:
        tracer = TracingMiddleware(service_name="my-app")
        ctx = tracer.start_trace(Context(query="Hello"))
        result = await pipeline.process(ctx)
        tracer.end_trace(result.unwrap())
    """

    def __init__(
        self,
        service_name: str = "langpy",
        on_span_start: Optional[Callable[[TraceSpan], None]] = None,
        on_span_end: Optional[Callable[[TraceSpan], None]] = None
    ):
        """
        Initialize the tracing middleware.

        Args:
            service_name: Name of the service for tracing
            on_span_start: Callback when a span starts
            on_span_end: Callback when a span ends
        """
        self.service_name = service_name
        self._on_span_start = on_span_start
        self._on_span_end = on_span_end
        self._active_traces: Dict[str, float] = {}

    def start_trace(self, ctx: Context) -> Context:
        """
        Start a new trace for the context.

        Args:
            ctx: Context to trace

        Returns:
            Context with trace metadata
        """
        self._active_traces[ctx.trace_id] = time.time()

        # Add service metadata
        ctx = ctx.clone()
        ctx.metadata["service_name"] = self.service_name
        ctx.metadata["trace_start"] = datetime.now().isoformat()

        return ctx

    def end_trace(self, ctx: Context) -> Context:
        """
        End a trace.

        Args:
            ctx: Context to finish tracing

        Returns:
            Context with trace end metadata
        """
        if ctx.trace_id in self._active_traces:
            start_time = self._active_traces.pop(ctx.trace_id)
            duration_ms = (time.time() - start_time) * 1000

            ctx = ctx.clone()
            ctx.metadata["trace_end"] = datetime.now().isoformat()
            ctx.metadata["trace_duration_ms"] = duration_ms

            # Call end callbacks for all spans
            if self._on_span_end:
                for span in ctx.spans:
                    if span.end_time is not None:
                        self._on_span_end(span)

        return ctx

    def get_trace_summary(self, ctx: Context) -> Dict[str, Any]:
        """
        Get a summary of the trace.

        Args:
            ctx: Context with trace data

        Returns:
            Dict with trace summary
        """
        spans_summary = []
        for span in ctx.spans:
            spans_summary.append({
                "name": span.name,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "attributes": span.attributes
            })

        return {
            "trace_id": ctx.trace_id,
            "service_name": ctx.metadata.get("service_name", self.service_name),
            "start_time": ctx.metadata.get("trace_start"),
            "end_time": ctx.metadata.get("trace_end"),
            "duration_ms": ctx.metadata.get("trace_duration_ms"),
            "spans": spans_summary,
            "token_usage": {
                "prompt_tokens": ctx.token_usage.prompt_tokens,
                "completion_tokens": ctx.token_usage.completion_tokens,
                "total_tokens": ctx.token_usage.total_tokens
            },
            "cost": {
                "total_cost": ctx.cost.total_cost,
                "currency": ctx.cost.currency
            },
            "errors": ctx.errors
        }


# Global instances for convenience
_default_calculator = CostCalculator()
_default_collector = MetricsCollector()


def calculate_cost(model: str, usage: TokenUsage) -> CostInfo:
    """Calculate cost using the default calculator."""
    return _default_calculator.calculate(model, usage)


def record_metrics(ctx: Context, model: Optional[str] = None) -> None:
    """Record metrics using the default collector."""
    _default_collector.record(ctx, model)


def get_metrics_summary() -> MetricsSummary:
    """Get metrics summary from the default collector."""
    return _default_collector.get_summary()


def clear_metrics() -> None:
    """Clear the default metrics collector."""
    _default_collector.clear()
