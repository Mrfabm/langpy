"""
Thread Primitive - Langbase-compatible Thread/Conversation API.

The Thread primitive manages conversation history and context.

Usage:
    # Direct API
    thread = await lb.thread.create(metadata={"user": "john"})
    await lb.thread.append(thread_id=thread.thread_id, messages=[...])
    history = await lb.thread.list(thread_id=thread.thread_id)

    # Pipeline composition
    pipeline = thread.loader | memory | pipe | thread.saver
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langpy.core.primitive import BasePrimitive, ThreadResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context, Message
    from langpy.core.result import Result


class Thread(BasePrimitive):
    """
    Thread primitive - Conversation history management.

    Threads store and manage conversation history, enabling:
    - Multi-turn conversations
    - Context persistence
    - Conversation branching
    - History search

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Create a thread
        thread = await lb.thread.create(metadata={"user_id": "123"})

        # Add messages
        await lb.thread.append(
            thread_id=thread.thread_id,
            messages=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )

        # Get history
        history = await lb.thread.list(thread_id=thread.thread_id)
    """

    def __init__(self, client: Any = None, name: str = "thread"):
        """
        Initialize the Thread primitive.

        Args:
            client: Parent Langpy client
            name: Primitive name for tracing
        """
        super().__init__(name=name, client=client)
        self._async_thread = None
        self._threads: Dict[str, Dict[str, Any]] = {}  # In-memory storage

    def _get_async_thread(self):
        """Get or create the underlying AsyncThread."""
        if self._async_thread is None:
            try:
                from thread.async_thread import AsyncThread
                self._async_thread = AsyncThread()
            except ImportError:
                pass  # Use in-memory fallback
        return self._async_thread

    # ========================================================================
    # Langbase-compatible API
    # ========================================================================

    async def _run(
        self,
        action: str = "list",
        thread_id: str = None,
        messages: List[Dict[str, str]] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> ThreadResponse:
        """
        Run thread operation.

        Args:
            action: "create", "append", "list", "delete"
            thread_id: Thread identifier
            messages: Messages to append
            metadata: Thread metadata

        Returns:
            ThreadResponse
        """
        if action == "create":
            return await self.create(metadata=metadata, **kwargs)
        elif action == "append":
            return await self.append(thread_id=thread_id, messages=messages, **kwargs)
        elif action == "list":
            return await self.list(thread_id=thread_id, **kwargs)
        elif action == "delete":
            return await self.delete(thread_id=thread_id, **kwargs)
        else:
            return ThreadResponse(success=False, error=f"Unknown action: {action}")

    async def create(
        self,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> ThreadResponse:
        """
        Create a new thread.

        Args:
            metadata: Thread metadata

        Returns:
            ThreadResponse with thread_id
        """
        try:
            thread_id = str(uuid.uuid4())

            # Try underlying implementation
            async_thread = self._get_async_thread()
            if async_thread:
                result = await async_thread.create(metadata=metadata)
                if hasattr(result, 'id'):
                    thread_id = result.id

            # Store in memory
            self._threads[thread_id] = {
                "id": thread_id,
                "messages": [],
                "metadata": metadata or {},
                "created_at": __import__("time").time()
            }

            return ThreadResponse(
                success=True,
                thread_id=thread_id,
                action="create",
                messages=[]
            )

        except Exception as e:
            return ThreadResponse(success=False, error=str(e), action="create")

    async def append(
        self,
        thread_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ThreadResponse:
        """
        Append messages to a thread.

        Args:
            thread_id: Thread identifier
            messages: Messages to append

        Returns:
            ThreadResponse
        """
        try:
            if not thread_id:
                return ThreadResponse(
                    success=False,
                    error="thread_id required",
                    action="append"
                )

            # Try underlying implementation
            async_thread = self._get_async_thread()
            if async_thread:
                await async_thread.add_message(thread_id, messages)

            # Update in-memory store
            if thread_id not in self._threads:
                self._threads[thread_id] = {
                    "id": thread_id,
                    "messages": [],
                    "metadata": {}
                }

            self._threads[thread_id]["messages"].extend(messages)

            return ThreadResponse(
                success=True,
                thread_id=thread_id,
                action="append",
                messages=self._threads[thread_id]["messages"]
            )

        except Exception as e:
            return ThreadResponse(success=False, error=str(e), action="append")

    async def list(
        self,
        thread_id: str = None,
        limit: int = 100,
        **kwargs
    ) -> ThreadResponse:
        """
        List messages in a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum messages to return

        Returns:
            ThreadResponse with messages
        """
        try:
            if not thread_id:
                # List all threads
                threads = list(self._threads.keys())
                return ThreadResponse(
                    success=True,
                    action="list",
                    messages=[{"thread_id": tid} for tid in threads]
                )

            # Get thread messages
            if thread_id in self._threads:
                messages = self._threads[thread_id]["messages"][-limit:]
            else:
                # Try underlying implementation
                async_thread = self._get_async_thread()
                if async_thread:
                    result = await async_thread.get_messages(thread_id, limit=limit)
                    messages = result if isinstance(result, list) else []
                else:
                    messages = []

            return ThreadResponse(
                success=True,
                thread_id=thread_id,
                action="list",
                messages=messages
            )

        except Exception as e:
            return ThreadResponse(success=False, error=str(e), action="list")

    async def delete(
        self,
        thread_id: str,
        **kwargs
    ) -> ThreadResponse:
        """
        Delete a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            ThreadResponse
        """
        try:
            if thread_id in self._threads:
                del self._threads[thread_id]

            # Try underlying implementation
            async_thread = self._get_async_thread()
            if async_thread and hasattr(async_thread, 'delete'):
                await async_thread.delete(thread_id)

            return ThreadResponse(
                success=True,
                thread_id=thread_id,
                action="delete"
            )

        except Exception as e:
            return ThreadResponse(success=False, error=str(e), action="delete")

    # ========================================================================
    # Pipeline API: process()
    # ========================================================================

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Process context - loads thread history into context.messages.

        Args:
            ctx: Input context with thread_id in variables

        Returns:
            Result[Context] with messages populated
        """
        thread_id = ctx.get("thread_id")

        if not thread_id:
            # No thread - pass through
            return Success(ctx)

        response = await self.list(thread_id=thread_id)

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error or "Thread load failed",
                primitive=self._name
            ))

        # Convert to context messages
        from langpy.core.context import Message

        messages = []
        for msg in (response.messages or []):
            messages.append(Message(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))

        new_ctx = ctx.with_messages(messages)
        new_ctx = new_ctx.set("thread_id", thread_id)

        return Success(new_ctx)

    # ========================================================================
    # Pipeline Components
    # ========================================================================

    @property
    def loader(self) -> "ThreadLoader":
        """Get a loader primitive that loads thread history."""
        return ThreadLoader(self)

    @property
    def saver(self) -> "ThreadSaver":
        """Get a saver primitive that saves to thread."""
        return ThreadSaver(self)


class ThreadLoader(BasePrimitive):
    """Loads thread history into context."""

    def __init__(self, thread: Thread):
        super().__init__("thread_loader")
        self._thread = thread

    async def _process(self, ctx: "Context") -> "Result[Context]":
        return await self._thread._process(ctx)


class ThreadSaver(BasePrimitive):
    """Saves context messages to thread."""

    def __init__(self, thread: Thread):
        super().__init__("thread_saver")
        self._thread = thread

    async def _process(self, ctx: "Context") -> "Result[Context]":
        thread_id = ctx.get("thread_id")

        if not thread_id:
            return Success(ctx)

        # Save query and response as messages
        messages = []
        if ctx.query:
            messages.append({"role": "user", "content": ctx.query})
        if ctx.response:
            messages.append({"role": "assistant", "content": ctx.response})

        if messages:
            await self._thread.append(thread_id=thread_id, messages=messages)

        return Success(ctx)
