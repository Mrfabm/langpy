"""
SyncAgent - Synchronous wrapper around AsyncAgent.

Uses asyncio.run() to delegate all operations to AsyncAgent,
eliminating code duplication while maintaining the same API.
"""

import asyncio
from typing import Callable, Union, Optional, List, Dict, Any

from .async_agent import (
    AsyncAgent,
    AgentRunResponse,
    AgentStreamChunk,
    Tool,
    ToolChoice,
    InputMessage,
    JsonDict,
)


class SyncAgent:
    """
    SyncAgent implements the Langbase Agent Run API (sync variant).

    This class wraps AsyncAgent using asyncio.run() to provide blocking
    functionality for running agents with tool execution.

    Note: Streaming is not supported in sync mode and will raise an error.
    """

    def __init__(
        self,
        *,
        sync_llm: Optional[Callable[[JsonDict], AgentRunResponse]] = None,
        async_llm: Optional[Callable[[JsonDict], Any]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> None:
        """
        Initialize SyncAgent.

        Args:
            sync_llm: Sync backend callable (will be wrapped for async use)
            async_llm: Async backend callable (used directly)
            tools: Optional list of Tool objects with callables
        """
        # If sync_llm provided, wrap it for async use
        effective_async_llm = async_llm
        if sync_llm is not None and async_llm is None:
            effective_async_llm = self._wrap_sync_llm(sync_llm)

        self._async_agent = AsyncAgent(
            async_llm=effective_async_llm,
            tools=tools,
        )
        self._sync_llm = sync_llm
        self._tools = tools or []

    def _wrap_sync_llm(self, sync_llm: Callable[[JsonDict], AgentRunResponse]) -> Callable:
        """Wrap a sync LLM callable for use with AsyncAgent."""
        async def async_wrapper(payload: JsonDict) -> AgentRunResponse:
            return sync_llm(payload)
        return async_wrapper

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "SyncAgent.run() cannot be called from within an async context. "
                "Use AsyncAgent instead."
            )

        return asyncio.run(coro)

    def run(
        self,
        *,
        model: str,
        input: Union[str, List[InputMessage]],
        instructions: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        parallel_tool_calls: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        apiKey: str,
        **kwargs,
    ) -> AgentRunResponse:
        """
        Run the agent with the given parameters (blocking).

        Args:
            model: Model name (provider-qualified)
            input: Prompt or message array
            instructions: System-level guidance
            stream: Must be False (streaming not supported in sync mode)
            tools: List of Tool objects
            tool_choice: Tool selection mode ('auto', 'required', or ToolChoice)
            parallel_tool_calls: Run tool calls concurrently if True
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            presence_penalty: Penalize new tokens based on presence
            frequency_penalty: Penalize new tokens based on frequency
            apiKey: API key (required)

        Returns:
            AgentRunResponse

        Raises:
            RuntimeError: If stream=True (not supported)
            ValueError: If apiKey is empty or tool_choice is invalid
        """
        if stream:
            raise RuntimeError(
                "SyncAgent does not support streaming. "
                "Use AsyncAgent for streaming functionality."
            )

        return self._run_async(
            self._async_agent.run(
                model=model,
                input=input,
                instructions=instructions,
                stream=False,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                apiKey=apiKey,
                **kwargs,
            )
        )
