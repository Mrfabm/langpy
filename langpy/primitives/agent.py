"""
Agent Primitive - Langbase-compatible Agent API.

The Agent primitive provides a unified LLM API for 100+ models,
supporting tool calling, streaming, and structured outputs.

Usage:
    # Direct API (Langbase-compatible)
    response = await agent.run(
        model="openai:gpt-4",
        input="What is Python?",
        instructions="Be concise",
        stream=False
    )

    # Pipeline composition
    pipeline = memory | agent | thread
    result = await pipeline.process(ctx)
"""

from __future__ import annotations
import os
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Union, TYPE_CHECKING

from langpy.core.primitive import BasePrimitive, AgentResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


class Agent(BasePrimitive):
    """
    Agent primitive - Langbase-compatible unified LLM API.

    Provides a consistent interface for interacting with 100+ LLMs
    across all major providers (OpenAI, Anthropic, Google, Mistral, etc.).

    Features:
    - Unified API across providers
    - Tool calling support
    - Streaming responses
    - Structured outputs (JSON mode)
    - Vision capabilities
    - Automatic retry and error handling

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Simple usage
        response = await lb.agent.run(
            model="openai:gpt-4",
            input="Hello, who are you?",
            instructions="You are a helpful assistant."
        )
        print(response.output)

        # With tools
        response = await lb.agent.run(
            model="openai:gpt-4",
            input="What's the weather in Tokyo?",
            tools=[weather_tool],
            tool_choice="auto"
        )

        # Streaming
        async for chunk in await lb.agent.run(
            model="openai:gpt-4",
            input="Tell me a story",
            stream=True
        ):
            print(chunk.output, end="")
    """

    def __init__(self, client: Any = None, name: str = "agent"):
        """
        Initialize the Agent primitive.

        Args:
            client: Parent Langpy client (for shared config)
            name: Primitive name for tracing
        """
        super().__init__(name=name, client=client)
        self._async_agent = None  # Lazy initialization

    def _get_async_agent(self):
        """Get or create the underlying AsyncAgent."""
        if self._async_agent is None:
            from agent.async_agent import AsyncAgent
            self._async_agent = AsyncAgent()
        return self._async_agent

    def _get_adapter(self, model: str):
        """Get the LLM adapter for the given model."""
        # Parse model string: "provider:model_name"
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            provider = "openai"
            model_name = model

        # Create a simple adapter wrapper for function-based adapters
        class AdapterWrapper:
            def __init__(self, run_func):
                self.run_func = run_func

            async def run(self, payload):
                return await self.run_func(payload)

        # Import the appropriate adapter function
        try:
            if provider == "openai":
                from pipe.adapters.openai import run as openai_run
                return AdapterWrapper(openai_run)
            elif provider == "anthropic":
                from pipe.adapters.anthropic import run as anthropic_run
                return AdapterWrapper(anthropic_run)
            elif provider == "gemini" or provider == "google":
                from pipe.adapters.gemini import run as gemini_run
                return AdapterWrapper(gemini_run)
            elif provider == "mistral":
                from pipe.adapters.mistral import run as mistral_run
                return AdapterWrapper(mistral_run)
            elif provider == "groq":
                from pipe.adapters.groq import run as groq_run
                return AdapterWrapper(groq_run)
            elif provider == "ollama":
                from pipe.adapters.ollama import run as ollama_run
                return AdapterWrapper(ollama_run)
            else:
                # Try to dynamically load adapter
                adapter_module = __import__(
                    f"pipe.adapters.{provider}",
                    fromlist=["run"]
                )
                return AdapterWrapper(adapter_module.run)
        except ImportError as e:
            raise ValueError(f"Unsupported provider: {provider} - {str(e)}")

    # ========================================================================
    # Langbase-compatible API: run()
    # ========================================================================

    async def _run(
        self,
        model: str = "openai:gpt-4",
        input: Union[str, List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Run the agent with Langbase-compatible parameters.

        Args:
            model: LLM model identifier (e.g., "openai:gpt-4", "anthropic:claude-3")
            input: User prompt or message array
            instructions: System-level instructions
            stream: Enable streaming responses
            tools: List of tool definitions
            tool_choice: Tool selection mode ("auto", "required", or specific tool)
            parallel_tool_calls: Run tool calls concurrently
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            api_key: LLM provider API key (or use env var)

        Returns:
            AgentResponse with output and metadata
        """
        from agent.async_agent import AsyncAgent, Tool, ToolFunction

        # Get API key from param, client, or environment
        resolved_api_key = api_key
        if not resolved_api_key and self._client:
            resolved_api_key = getattr(self._client, '_api_key', None)
        if not resolved_api_key:
            # Try provider-specific env vars
            provider = model.split(":")[0] if ":" in model else "openai"
            env_var = f"{provider.upper()}_API_KEY"
            resolved_api_key = os.getenv(env_var) or os.getenv("LANGPY_API_KEY")

        if not resolved_api_key:
            return AgentResponse(
                success=False,
                error="API key required. Set via api_key param or environment variable."
            )

        # Get adapter for the model
        try:
            adapter = self._get_adapter(model)

            async def llm_callable(payload):
                return await adapter.run(payload)

            # Convert tools to Tool objects if provided
            tool_objects = None
            if tools:
                tool_objects = [
                    Tool(
                        type="function",
                        function=ToolFunction(
                            name=t.get("name") or t.get("function", {}).get("name"),
                            description=t.get("description") or t.get("function", {}).get("description"),
                            parameters=t.get("parameters") or t.get("function", {}).get("parameters")
                        ),
                        callable=t.get("callable")
                    )
                    for t in tools
                ]

            # Create agent with adapter
            agent = AsyncAgent(
                async_llm=llm_callable,
                tools=tool_objects
            )

            # Run the agent
            response = await agent.run(
                model=model,
                input=input,
                instructions=instructions,
                stream=stream,
                tools=tool_objects,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                apiKey=resolved_api_key
            )

            # Handle streaming response
            if stream:
                # Return async generator wrapped in response
                return AgentResponse(
                    success=True,
                    output=None,  # Will be streamed
                    model=model,
                    _stream=response  # Store generator for iteration
                )

            # Handle regular response
            if hasattr(response, 'error') and response.error:
                error_msg = str(response.error)
                if isinstance(response.error, dict):
                    error_msg = response.error.get('message', str(response.error))
                return AgentResponse(
                    success=False,
                    error=error_msg,
                    model=model
                )

            # Extract output from choices
            output = None
            messages = []
            tool_calls = []

            if hasattr(response, 'choices') and response.choices:
                for choice in response.choices:
                    if hasattr(choice, 'message'):
                        msg = choice.message
                        if isinstance(msg, dict):
                            content = msg.get('content')
                            if content:
                                output = content
                            messages.append(msg)
                            if 'tool_calls' in msg and msg['tool_calls']:
                                tool_calls.extend(msg['tool_calls'])
                        elif hasattr(msg, 'content'):
                            # Handle object-style message
                            output = msg.content
                            messages.append({"role": "assistant", "content": msg.content})

            return AgentResponse(
                success=True,
                output=output,
                messages=messages,
                tool_calls=tool_calls if tool_calls else None,
                usage=response.usage if hasattr(response, 'usage') else None,
                model=model
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=str(e),
                model=model
            )

    # ========================================================================
    # Pipeline API: process()
    # ========================================================================

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Process context for pipeline composition.

        Reads from ctx.query (or ctx.messages) and writes to ctx.response.

        Args:
            ctx: Input context with query/messages

        Returns:
            Result[Context] with response populated
        """
        # Extract parameters from context
        model = ctx.get("model", "openai:gpt-4")
        instructions = ctx.get("instructions") or ctx.get("system_prompt")
        tools = ctx.get("tools")
        api_key = ctx.get("api_key")

        # Build input from context
        input_data = ctx.query
        if ctx.messages:
            input_data = [msg.to_dict() for msg in ctx.messages]

        # Include documents in context if available (RAG)
        if ctx.documents and instructions:
            doc_context = ctx.format_documents()
            instructions = f"{instructions}\n\nContext:\n{doc_context}"

        # Run the agent
        response = await self._run(
            model=model,
            input=input_data,
            instructions=instructions,
            tools=tools,
            api_key=api_key,
            **ctx.variables
        )

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error or "Agent execution failed",
                primitive=self._name
            ))

        # Update context with response
        new_ctx = ctx.with_response(response.output)

        # Add usage info if available
        if response.usage:
            from langpy.core.context import TokenUsage
            usage = TokenUsage(
                prompt_tokens=response.usage.get('prompt_tokens', 0),
                completion_tokens=response.usage.get('completion_tokens', 0),
                total_tokens=response.usage.get('total_tokens', 0)
            )
            new_ctx = new_ctx.add_usage(usage)

        # Store full response in variables for downstream access
        new_ctx = new_ctx.set("agent_response", response.model_dump())

        return Success(new_ctx)

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    async def chat(
        self,
        message: str,
        model: str = "openai:gpt-4",
        instructions: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simple chat interface.

        Args:
            message: User message
            model: LLM model
            instructions: System instructions
            **kwargs: Additional options

        Returns:
            Response text
        """
        response = await self.run(
            model=model,
            input=message,
            instructions=instructions,
            **kwargs
        )
        return response.output if response.success else f"Error: {response.error}"

    async def complete(
        self,
        prompt: str,
        model: str = "openai:gpt-4",
        **kwargs
    ) -> str:
        """
        Simple completion interface.

        Args:
            prompt: Prompt text
            model: LLM model
            **kwargs: Additional options

        Returns:
            Completion text
        """
        return await self.chat(prompt, model=model, **kwargs)
