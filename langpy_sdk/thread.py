"""
LangPy Thread - Clean SDK wrapper for conversation management.

Simple, intuitive interface for managing conversation threads.

Supports both the original API and the new composable architecture:

Original API:
    from langpy_sdk import Thread

    thread = Thread()
    thread_id = await thread.create("Customer Support Chat")
    await thread.add_message(thread_id, "user", "Hello!")
    messages = await thread.get_messages(thread_id)

New Composable API:
    from langpy.core import Context
    from langpy_sdk import Thread, Pipe

    # Thread as a primitive that loads conversation history
    thread_loader = Thread.as_primitive(thread_id="abc123")
    pipe = Pipe(model="gpt-4o-mini")

    # Load history, process, and save
    pipeline = thread_loader | pipe | thread_loader.saver()
    result = await pipeline.process(Context(query="Hello!"))
"""

from __future__ import annotations
import os
import json
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

# Import core types for the new architecture
try:
    from langpy.core.context import Context, Message as CoreMessage
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode
    from langpy.core.primitive import BasePrimitive
    _NEW_ARCH_AVAILABLE = True
except ImportError:
    _NEW_ARCH_AVAILABLE = False
    Context = Any
    Result = Any


@dataclass
class Message:
    """
    A single message in a conversation.

    Attributes:
        role: Message role (user, assistant, system, tool)
        content: Message content
        id: Unique message ID
        created_at: Unix timestamp
        metadata: Additional metadata
    """
    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", int(time.time())),
            metadata=data.get("metadata", {})
        )


@dataclass
class ThreadInfo:
    """
    Information about a conversation thread.

    Attributes:
        id: Thread ID
        name: Thread name
        message_count: Number of messages
        created_at: Creation timestamp
        updated_at: Last update timestamp
        tags: Thread tags
        metadata: Additional metadata
    """
    id: str
    name: str
    message_count: int
    created_at: int
    updated_at: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Thread:
    """
    Clean, simple Thread interface for conversation management.

    Manages conversation threads with message persistence.

    Args:
        storage_path: Where to store threads (default: ~/.langpy_sdk/threads)

    Example:
        thread = Thread()

        # Create a new conversation
        thread_id = await thread.create("My Chat", tags=["support"])

        # Add messages
        await thread.add_message(thread_id, "user", "Hello!")
        await thread.add_message(thread_id, "assistant", "Hi there!")

        # Get messages for LLM
        messages = await thread.get_messages(thread_id)
        # Returns: [{"role": "user", "content": "Hello!"}, ...]

        # List all threads
        threads = await thread.list()
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".langpy_sdk" / "threads"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}

    async def create(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation thread.

        Args:
            name: Optional thread name
            tags: Optional list of tags
            metadata: Optional metadata

        Returns:
            Thread ID

        Example:
            thread_id = await thread.create(
                "Support Chat",
                tags=["support", "billing"],
                metadata={"customer_id": "123"}
            )
        """
        thread_id = str(uuid.uuid4())
        now = int(time.time())

        thread_data = {
            "id": thread_id,
            "name": name or f"Thread {thread_id[:8]}",
            "messages": [],
            "tags": tags or [],
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now
        }

        self._cache[thread_id] = thread_data
        await self._save(thread_id)

        return thread_id

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a thread.

        Args:
            thread_id: Thread ID
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional message metadata

        Returns:
            Created Message object

        Example:
            msg = await thread.add_message(
                thread_id,
                "user",
                "What's the weather like?",
                metadata={"source": "web"}
            )
        """
        thread_data = await self._load(thread_id)

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )

        thread_data["messages"].append(message.to_dict())
        thread_data["updated_at"] = int(time.time())

        self._cache[thread_id] = thread_data
        await self._save(thread_id)

        return message

    async def get_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        as_dicts: bool = True
    ) -> List[Any]:
        """
        Get messages from a thread.

        Args:
            thread_id: Thread ID
            limit: Optional limit on number of messages (returns most recent)
            as_dicts: If True, return as dicts for LLM input (default: True)

        Returns:
            List of messages (dicts or Message objects)

        Example:
            # Get messages for LLM
            messages = await thread.get_messages(thread_id)
            response = await agent.run(messages)

            # Get last 10 messages
            recent = await thread.get_messages(thread_id, limit=10)
        """
        thread_data = await self._load(thread_id)
        messages = thread_data["messages"]

        if limit:
            messages = messages[-limit:]

        if as_dicts:
            return [{"role": m["role"], "content": m["content"]} for m in messages]
        else:
            return [Message.from_dict(m) for m in messages]

    async def get(self, thread_id: str) -> ThreadInfo:
        """
        Get thread information.

        Args:
            thread_id: Thread ID

        Returns:
            ThreadInfo object

        Example:
            info = await thread.get(thread_id)
            print(f"Thread: {info.name}, Messages: {info.message_count}")
        """
        thread_data = await self._load(thread_id)

        return ThreadInfo(
            id=thread_data["id"],
            name=thread_data["name"],
            message_count=len(thread_data["messages"]),
            created_at=thread_data["created_at"],
            updated_at=thread_data["updated_at"],
            tags=thread_data.get("tags", []),
            metadata=thread_data.get("metadata", {})
        )

    async def list(
        self,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[ThreadInfo]:
        """
        List all threads.

        Args:
            tags: Filter by tags (returns threads with any matching tag)
            limit: Maximum number of threads to return

        Returns:
            List of ThreadInfo objects, sorted by most recent

        Example:
            # List all threads
            threads = await thread.list()

            # Filter by tags
            support_threads = await thread.list(tags=["support"])
        """
        threads = []

        for file in self.storage_path.glob("*.json"):
            thread_id = file.stem
            try:
                thread_data = await self._load(thread_id)

                # Filter by tags
                if tags:
                    thread_tags = thread_data.get("tags", [])
                    if not any(t in thread_tags for t in tags):
                        continue

                threads.append(ThreadInfo(
                    id=thread_data["id"],
                    name=thread_data["name"],
                    message_count=len(thread_data["messages"]),
                    created_at=thread_data["created_at"],
                    updated_at=thread_data["updated_at"],
                    tags=thread_data.get("tags", []),
                    metadata=thread_data.get("metadata", {})
                ))
            except Exception:
                continue

        # Sort by updated_at descending
        threads.sort(key=lambda t: t.updated_at, reverse=True)

        if limit:
            threads = threads[:limit]

        return threads

    async def delete(self, thread_id: str) -> bool:
        """
        Delete a thread.

        Args:
            thread_id: Thread ID

        Returns:
            True if deleted, False if not found

        Example:
            deleted = await thread.delete(thread_id)
        """
        file_path = self.storage_path / f"{thread_id}.json"

        if thread_id in self._cache:
            del self._cache[thread_id]

        if file_path.exists():
            file_path.unlink()
            return True

        return False

    async def update(
        self,
        thread_id: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThreadInfo:
        """
        Update thread properties.

        Args:
            thread_id: Thread ID
            name: New name (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ThreadInfo

        Example:
            await thread.update(thread_id, name="Renamed Chat", tags=["resolved"])
        """
        thread_data = await self._load(thread_id)

        if name is not None:
            thread_data["name"] = name
        if tags is not None:
            thread_data["tags"] = tags
        if metadata is not None:
            thread_data["metadata"] = metadata

        thread_data["updated_at"] = int(time.time())

        self._cache[thread_id] = thread_data
        await self._save(thread_id)

        return await self.get(thread_id)

    async def clear_messages(self, thread_id: str) -> None:
        """
        Clear all messages from a thread.

        Args:
            thread_id: Thread ID

        Example:
            await thread.clear_messages(thread_id)
        """
        thread_data = await self._load(thread_id)
        thread_data["messages"] = []
        thread_data["updated_at"] = int(time.time())

        self._cache[thread_id] = thread_data
        await self._save(thread_id)

    # ========== Internal Methods ==========

    async def _load(self, thread_id: str) -> Dict[str, Any]:
        """Load thread data."""
        if thread_id in self._cache:
            return self._cache[thread_id]

        file_path = self.storage_path / f"{thread_id}.json"

        if not file_path.exists():
            raise ValueError(f"Thread not found: {thread_id}")

        with open(file_path, "r") as f:
            data = json.load(f)

        self._cache[thread_id] = data
        return data

    async def _save(self, thread_id: str) -> None:
        """Save thread data."""
        if thread_id not in self._cache:
            return

        file_path = self.storage_path / f"{thread_id}.json"

        with open(file_path, "w") as f:
            json.dump(self._cache[thread_id], f, indent=2)

    def __repr__(self) -> str:
        return f"Thread(storage_path='{self.storage_path}')"

    # ========================================================================
    # New Composable Architecture Methods
    # ========================================================================

    def as_loader(self, thread_id: str, limit: Optional[int] = None) -> "ThreadLoaderPrimitive":
        """
        Create a primitive that loads conversation history into context.

        Args:
            thread_id: Thread ID to load
            limit: Optional limit on messages

        Returns:
            ThreadLoaderPrimitive that can be used in a pipeline

        Example:
            from langpy.core import Context

            thread = Thread()
            loader = thread.as_loader(thread_id="abc123")
            pipe = Pipe(model="gpt-4o-mini")

            pipeline = loader | pipe
            result = await pipeline.process(Context(query="Hello!"))
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Composable architecture requires langpy.core")
        return ThreadLoaderPrimitive(self, thread_id, limit)

    def as_saver(self, thread_id: str) -> "ThreadSaverPrimitive":
        """
        Create a primitive that saves the conversation to thread.

        Args:
            thread_id: Thread ID to save to

        Returns:
            ThreadSaverPrimitive that can be used in a pipeline

        Example:
            loader = thread.as_loader(thread_id)
            saver = thread.as_saver(thread_id)

            pipeline = loader | pipe | saver
            result = await pipeline.process(Context(query="Hello!"))
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Composable architecture requires langpy.core")
        return ThreadSaverPrimitive(self, thread_id)

    @classmethod
    def as_primitive(
        cls,
        thread_id: str,
        storage_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> "ThreadLoaderPrimitive":
        """
        Class method to create a thread loader primitive directly.

        Args:
            thread_id: Thread ID to load
            storage_path: Optional storage path
            limit: Optional message limit

        Returns:
            ThreadLoaderPrimitive

        Example:
            loader = Thread.as_primitive(thread_id="abc123")
            pipeline = loader | pipe
        """
        thread = cls(storage_path=storage_path)
        return thread.as_loader(thread_id, limit)

    def to_context(self, thread_id: str, query: Optional[str] = None) -> "Context":
        """
        Convert thread messages to a Context object.

        Args:
            thread_id: Thread ID
            query: Optional new query to add

        Returns:
            Context with conversation history

        Example:
            ctx = await thread.to_context(thread_id, query="What did we discuss?")
            result = await pipe.process(ctx)
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Composable architecture requires langpy.core")

        from langpy.core.context import Context, Message as CoreMessage

        ctx = Context(query=query)

        # Load thread data synchronously from cache if available
        if thread_id in self._cache:
            thread_data = self._cache[thread_id]
            for msg_data in thread_data.get("messages", []):
                ctx = ctx.add_message(CoreMessage(
                    role=msg_data["role"],
                    content=msg_data["content"]
                ))

        return ctx


# ============================================================================
# Thread Primitive Classes (for composable architecture)
# ============================================================================

if _NEW_ARCH_AVAILABLE:
    from langpy.core.primitive import BasePrimitive
    from langpy.core.context import Context, Message as CoreMessage
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode

    class ThreadLoaderPrimitive(BasePrimitive):
        """
        Primitive that loads conversation history from a Thread into Context.

        This allows using Thread in composable pipelines.
        """

        def __init__(
            self,
            thread: "Thread",
            thread_id: str,
            limit: Optional[int] = None
        ):
            super().__init__(f"ThreadLoader({thread_id[:8]})")
            self._thread = thread
            self._thread_id = thread_id
            self._limit = limit

        async def _process(self, ctx: Context) -> Result[Context]:
            """Load thread messages into context."""
            try:
                # Get messages from thread
                messages = await self._thread.get_messages(
                    self._thread_id,
                    limit=self._limit,
                    as_dicts=False
                )

                # Add messages to context
                result_ctx = ctx
                for msg in messages:
                    result_ctx = result_ctx.add_message(CoreMessage(
                        role=msg.role,
                        content=msg.content,
                        metadata=msg.metadata
                    ))

                # Store thread info in variables
                result_ctx = result_ctx.set("_thread_id", self._thread_id)

                return Success(result_ctx)

            except Exception as e:
                return Failure(PrimitiveError(
                    code=ErrorCode.UNKNOWN,
                    message=f"Failed to load thread: {e}",
                    primitive=self._name,
                    cause=e
                ))

        def saver(self) -> "ThreadSaverPrimitive":
            """Get a saver primitive for this thread."""
            return ThreadSaverPrimitive(self._thread, self._thread_id)


    class ThreadSaverPrimitive(BasePrimitive):
        """
        Primitive that saves the query and response to a Thread.

        This allows persisting conversations in composable pipelines.
        """

        def __init__(self, thread: "Thread", thread_id: str):
            super().__init__(f"ThreadSaver({thread_id[:8]})")
            self._thread = thread
            self._thread_id = thread_id

        async def _process(self, ctx: Context) -> Result[Context]:
            """Save query and response to thread."""
            try:
                # Save the user query
                if ctx.query:
                    await self._thread.add_message(
                        self._thread_id,
                        "user",
                        ctx.query
                    )

                # Save the assistant response
                if ctx.response:
                    await self._thread.add_message(
                        self._thread_id,
                        "assistant",
                        ctx.response
                    )

                return Success(ctx)

            except Exception as e:
                return Failure(PrimitiveError(
                    code=ErrorCode.UNKNOWN,
                    message=f"Failed to save to thread: {e}",
                    primitive=self._name,
                    cause=e
                ))
else:
    # Stub classes when new architecture is not available
    class ThreadLoaderPrimitive:
        def __init__(self, *args, **kwargs):
            raise ImportError("Composable architecture requires langpy.core")

    class ThreadSaverPrimitive:
        def __init__(self, *args, **kwargs):
            raise ImportError("Composable architecture requires langpy.core")
