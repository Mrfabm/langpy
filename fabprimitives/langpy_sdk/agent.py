"""
LangPy Agent - Clean SDK wrapper for the Agent primitive.

Simple, intuitive interface for creating AI agents with tool support.

Supports both the original API and the new composable architecture:

Original API:
    from langpy_sdk import Agent, tool

    @tool("greet", "Greet someone", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    agent = Agent(model="gpt-4o-mini", tools=[greet])
    response = await agent.run("Greet Alice")
    print(response.content)

New Composable API:
    from langpy.core import Context
    from langpy_sdk import Agent, tool

    agent = Agent(model="gpt-4o-mini", tools=[greet])
    result = await agent.process(Context(query="Greet Alice"))
    print(result.unwrap().response)
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator, TYPE_CHECKING
from dataclasses import dataclass, field

# Import core types for the new architecture
try:
    from langpy.core.context import Context, TokenUsage, CostInfo
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode
    from langpy.core.primitive import BasePrimitive
    from langpy.core.observability import CostCalculator
    _NEW_ARCH_AVAILABLE = True
except ImportError:
    _NEW_ARCH_AVAILABLE = False
    Context = Any
    Result = Any


@dataclass
class ToolDef:
    """
    Simple tool definition.

    Attributes:
        name: Tool name (used by the LLM to call it)
        description: What the tool does
        parameters: JSON Schema for parameters
        handler: Python function to execute
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def _to_internal(self) -> Any:
        """Convert to internal Tool format."""
        # Lazy import to avoid circular dependencies
        from agent.async_agent import Tool, ToolFunction

        return Tool(
            type="function",
            function=ToolFunction(
                name=self.name,
                description=self.description,
                parameters=self.parameters
            ),
            callable=self.handler
        )


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Callable[[Callable], ToolDef]:
    """
    Decorator to create a tool from a function.

    Args:
        name: Tool name
        description: What the tool does
        parameters: JSON Schema for parameters

    Example:
        @tool(
            "calculate",
            "Calculate a math expression",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        )
        def calculate(expression: str) -> str:
            return str(eval(expression))
    """
    def decorator(func: Callable) -> ToolDef:
        return ToolDef(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            handler=func
        )
    return decorator


@dataclass
class AgentResponse:
    """
    Response from an agent run.

    Attributes:
        content: The text response
        model: Model used
        usage: Token usage stats
        tool_calls: Any tool calls made
        raw: Raw response object
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    raw: Any = None

    def __str__(self) -> str:
        return self.content


class Agent:
    """
    Clean, simple Agent interface.

    Creates an AI agent that can respond to prompts and use tools.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-4o")
        tools: Optional list of ToolDef objects
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 1000)
        provider: LLM provider (default: "openai")

    Example:
        # Simple usage
        agent = Agent(model="gpt-4o-mini")
        response = await agent.run("What is 2+2?")
        print(response.content)

        # With tools
        agent = Agent(model="gpt-4o-mini", tools=[my_tool])
        response = await agent.run("Use the calculator to compute 15 * 7")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        tools: Optional[List[ToolDef]] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        provider: str = "openai"
    ):
        self.model = model
        self.tools = tools or []
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self._agent = None
        self._adapter = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of internal agent."""
        if self._agent is not None:
            return

        # Lazy imports to avoid circular dependencies
        from agent.async_agent import AsyncAgent
        from pipe.adapters import get_adapter

        # Convert tools to internal format
        internal_tools = [t._to_internal() for t in self.tools] if self.tools else []

        # Get the adapter for the provider
        self._adapter = get_adapter(self.provider)

        # Create the internal agent
        self._agent = AsyncAgent(
            async_llm=self._adapter,
            tools=internal_tools
        )

    async def quick(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
    ) -> str:
        """
        Quick helper for simple prompts. Returns just the response string.

        This is the recommended way for simple, non-streaming agent calls.
        For composition with other primitives, use .process(ctx) instead.

        Args:
            prompt: User prompt string
            system: Optional system prompt

        Returns:
            Response string

        Example:
            agent = Agent(model="gpt-4o-mini", tools=[calculator])
            response = await agent.quick("What is 15 * 7?")
            print(response)  # "105"
        """
        if not _NEW_ARCH_AVAILABLE:
            # Fallback to legacy run if new architecture not available
            response = await self._legacy_run(prompt, system=system)
            return response.content

        from langpy.core import Context

        ctx = Context(query=prompt)
        if system:
            # Store system in context messages
            from langpy.core.context import Message
            ctx = ctx.add_message(Message(role="system", content=system))

        result = await self.process(ctx)
        if result.is_success():
            return result.unwrap().response or ""
        else:
            raise RuntimeError(str(result.error))

    async def run(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[AgentResponse, AsyncGenerator[str, None]]:
        """
        DEPRECATED: Use .quick() for simple prompts or .process() for composition.

        Run the agent with a prompt.

        Args:
            prompt: User prompt string or list of messages
            stream: Whether to stream the response (default: False)
            system: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            AgentResponse object (or async generator if streaming)
        """
        import warnings
        warnings.warn(
            "Agent.run() is deprecated. Use .quick() for simple prompts or .process() for composition.",
            DeprecationWarning,
            stacklevel=2
        )
        return await self._legacy_run(
            prompt, stream=stream, system=system,
            temperature=temperature, max_tokens=max_tokens
        )

    async def _legacy_run(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[AgentResponse, AsyncGenerator[str, None]]:
        """
        Internal implementation of run (kept for backward compatibility).
        """
        self._ensure_initialized()

        # Build messages
        if isinstance(prompt, str):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = list(prompt)
            if system:
                messages = [{"role": "system", "content": system}] + messages

        # Determine model string
        model_str = self.model
        if ":" not in model_str:
            model_str = f"{self.provider}:{self.model}"

        # Run the agent
        result = await self._agent.run(
            model=model_str,
            input=messages,
            apiKey=self.api_key,
            stream=stream,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        # Handle streaming
        if stream:
            async def stream_generator():
                async for chunk in result:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if isinstance(delta, dict) and "content" in delta:
                            yield delta["content"]
            return stream_generator()

        # Parse non-streaming response
        content = ""
        tool_calls = None
        usage = None

        if hasattr(result, 'choices') and result.choices:
            msg = result.choices[0].message
            if isinstance(msg, dict):
                # Handle None content (happens with tool calls)
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls")
            else:
                content = str(msg) if msg else ""

        if hasattr(result, 'usage') and result.usage:
            usage = result.usage

        if hasattr(result, 'error') and result.error:
            content = f"Error: {result.error.get('message', 'Unknown error')}"

        return AgentResponse(
            content=content,
            model=self.model,
            usage=usage,
            tool_calls=tool_calls,
            raw=result
        )

    def add_tool(self, tool_def: ToolDef) -> "Agent":
        """
        Add a tool to the agent.

        Args:
            tool_def: Tool definition to add

        Returns:
            Self for chaining
        """
        self.tools.append(tool_def)
        # Reset agent so it gets reinitialized with new tools
        self._agent = None
        return self

    def __repr__(self) -> str:
        return f"Agent(model='{self.model}', tools={len(self.tools)})"

    # ========================================================================
    # New Composable Architecture Methods
    # ========================================================================

    @property
    def name(self) -> str:
        """Return the primitive name for tracing."""
        return f"Agent({self.model})"

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process a context (IPrimitive interface).

        This method enables composable pipelines with the | operator.
        Unlike Pipe, Agent can use tools to accomplish tasks.

        Args:
            ctx: Input context with query and optional documents

        Returns:
            Result[Context] - Success with response or Failure with error

        Example:
            from langpy.core import Context

            agent = Agent(model="gpt-4o-mini", tools=[calculator])
            ctx = Context(query="What is 15 * 7?")
            result = await agent.process(ctx)

            if result.is_success():
                print(result.unwrap().response)
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError(
                "New architecture not available. "
                "Make sure langpy.core is properly installed."
            )

        from langpy.core.context import Context, TokenUsage, CostInfo
        from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode
        from langpy.core.observability import CostCalculator

        # Start span for tracing
        ctx = ctx.start_span(self.name, {"model": self.model, "tools": len(self.tools)})

        try:
            # Build the prompt
            if ctx.documents:
                # RAG mode: include documents in prompt
                context_str = ctx.format_documents()
                prompt = f"Context:\n{context_str}\n\nQuestion: {ctx.query or ''}"
            else:
                prompt = ctx.query or ""

            # Build messages from context
            messages = []
            for msg in ctx.messages:
                messages.append(msg.to_dict())
            messages.append({"role": "user", "content": prompt})

            # Run the agent
            response = await self._legacy_run(messages)

            # Extract content (handle tool call case)
            content = response.content or ""

            # Track token usage
            usage = TokenUsage()
            if response.usage:
                if isinstance(response.usage, dict):
                    usage = TokenUsage(
                        prompt_tokens=response.usage.get("prompt_tokens", 0),
                        completion_tokens=response.usage.get("completion_tokens", 0),
                        total_tokens=response.usage.get("total_tokens", 0)
                    )

            # Calculate cost
            calculator = CostCalculator()
            cost = calculator.calculate(self.model, usage)

            # Build result context
            result_ctx = ctx.with_response(content)
            result_ctx = result_ctx.add_usage(usage)
            result_ctx = result_ctx.add_cost(cost)

            # Store metadata
            result_ctx = result_ctx.set("_last_model", self.model)
            result_ctx = result_ctx.set("_tool_calls", response.tool_calls)

            # End span successfully
            result_ctx = result_ctx.end_span("ok")

            return Success(result_ctx)

        except Exception as e:
            # End span with error
            ctx = ctx.end_span("error", str(e))

            # Determine error code
            error_name = type(e).__name__.lower()
            if "rate" in error_name or "ratelimit" in error_name:
                code = ErrorCode.LLM_RATE_LIMIT
            elif "timeout" in error_name:
                code = ErrorCode.TIMEOUT
            else:
                code = ErrorCode.LLM_API_ERROR

            return Failure(PrimitiveError(
                code=code,
                message=str(e),
                primitive=self.name,
                cause=e
            ))

    def __or__(self, other) -> "Any":
        """
        Sequential composition with the | operator.

        Example:
            pipeline = memory | agent | validator
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import Pipeline
        return Pipeline([self, other])

    def __and__(self, other) -> "Any":
        """
        Parallel composition with the & operator.

        Example:
            parallel = agent1 & agent2
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import ParallelPrimitives
        return ParallelPrimitives([self, other])

    def __rshift__(self, other) -> "Any":
        """
        Alternative sequential composition with >> operator.
        """
        return self.__or__(other)
