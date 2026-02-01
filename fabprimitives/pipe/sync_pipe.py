"""
SyncPipe - Synchronous wrapper around AsyncPipe.

Uses asyncio.run() to delegate all operations to AsyncPipe,
eliminating code duplication while maintaining the same API.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Callable

from .async_pipe import AsyncPipe, PipeRunMetadata
from agent.async_agent import AgentRunResponse


class SyncPipe:
    """
    SyncPipe provides a synchronous interface to the pipe primitive.

    This class wraps AsyncPipe using asyncio.run() to provide blocking
    functionality for making LLM calls.

    Note: Streaming is not supported in sync mode and will raise an error.
    """

    def __init__(self, *, default_model: str = "openai:gpt-4o", **defaults):
        """
        Initialize SyncPipe.

        Args:
            default_model: Default model to use (e.g., 'openai:gpt-4o')
            **defaults: Additional default parameters
        """
        self._async_pipe = AsyncPipe(default_model=default_model, **defaults)
        self.default_model = default_model
        self.defaults = defaults

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "SyncPipe methods cannot be called from within an async context. "
                "Use AsyncPipe instead."
            )

        return asyncio.run(coro)

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool function."""
        self._async_pipe.register_tool(name, func)

    def set_memory_interface(self, memory_interface):
        """Set memory interface for pipe."""
        self._async_pipe.set_memory_interface(memory_interface)

    def set_thread_interface(self, thread_interface):
        """Set thread interface for pipe."""
        self._async_pipe.set_thread_interface(thread_interface)

    def set_agent_interface(self, agent_interface):
        """Set agent interface for pipe."""
        self._async_pipe.set_agent_interface(agent_interface)

    def run(
        self,
        *,
        name: Optional[str] = None,
        apiKey: str,
        messages: Optional[List] = None,
        input: Any = None,
        model: Optional[str] = None,
        stream: bool = False,
        memory=None,
        thread=None,
        agent=None,
        tools=None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        json_output: bool = False,
        response_format=None,
        few_shot=None,
        safety_prompt=None,
        moderate: bool = False,
        store: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        **overrides,
    ) -> AgentRunResponse:
        """
        Run the pipe with the given parameters (blocking).

        Args:
            name: Optional preset ID
            apiKey: API key (required)
            messages: List of messages
            input: Simple input (alias for messages)
            model: Model to use
            stream: Must be False (streaming not supported in sync mode)
            memory: Memory integration config
            thread: Thread integration config
            agent: Agent integration config
            tools: Tool definitions
            tool_choice: Tool selection mode
            parallel_tool_calls: Run tool calls concurrently
            json_output: Force JSON output
            response_format: Response format specification
            few_shot: Few-shot examples
            safety_prompt: Safety prompt
            moderate: Enable moderation
            store: Store run metadata
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            timeout: Request timeout
            **overrides: Additional parameters

        Returns:
            AgentRunResponse

        Raises:
            RuntimeError: If stream=True (not supported)
        """
        if stream:
            raise RuntimeError(
                "SyncPipe does not support streaming. "
                "Use AsyncPipe for streaming functionality."
            )

        return self._run_async(
            self._async_pipe.run(
                name=name,
                apiKey=apiKey,
                messages=messages,
                input=input,
                model=model,
                stream=False,
                memory=memory,
                thread=thread,
                agent=agent,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                json_output=json_output,
                response_format=response_format,
                few_shot=few_shot,
                safety_prompt=safety_prompt,
                moderate=moderate,
                store=store,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                **overrides,
            )
        )

    def get_run_logs(self) -> List[PipeRunMetadata]:
        """Get run logs from the async pipe."""
        return self._async_pipe._run_logs
