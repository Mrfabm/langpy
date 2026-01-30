from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# Type aliases
JsonDict = Dict[str, Any]

class ThreadMessage(BaseModel):
    """A message in a conversation thread."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: Union[str, JsonDict] = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool messages")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")

class Thread(BaseModel):
    """A conversation thread."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = Field(None, description="Optional thread name")
    messages: List[ThreadMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))
    status: str = Field(default="active", description="Thread status: active, archived, deleted")
    tags: List[str] = Field(default_factory=list, description="Thread tags for organization")

class AsyncThread:
    """
    Async thread management system similar to Langbase.
    
    Provides:
    - Thread creation and management
    - Message addition and retrieval
    - Thread persistence to disk
    - Thread listing and deletion
    - Advanced conversation state management
    - Thread archiving and tagging
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize AsyncThread.
        
        Args:
            storage_path: Path to store threads (default: ~/.langpy/threads)
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".langpy" / "threads"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, Thread] = {}

    async def create_thread(
        self, 
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Thread:
        """
        Create a new conversation thread.
        
        Args:
            name: Optional thread name
            metadata: Optional metadata dictionary
            tags: Optional list of tags
            
        Returns:
            Created Thread object
        """
        thread = Thread(
            name=name,
            metadata=metadata or {},
            tags=tags or []
        )
        
        self._threads[thread.id] = thread
        await self._save_thread(thread)
        
        return thread

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """
        Get a thread by ID.
        
        Args:
            thread_id: Thread ID to retrieve
            
        Returns:
            Thread object or None if not found
        """
        if thread_id in self._threads:
            return self._threads[thread_id]
        
        # Try to load from disk
        thread = await self._load_thread(thread_id)
        if thread:
            self._threads[thread_id] = thread
        return thread

    async def list_threads(
        self, 
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Thread]:
        """
        List all available threads with optional filtering.
        
        Args:
            status: Filter by thread status (active, archived, deleted)
            tags: Filter by thread tags
            limit: Maximum number of threads to return
            
        Returns:
            List of Thread objects
        """
        # Load all threads from disk
        thread_files = list(self.storage_path.glob("*.json"))
        threads = []
        
        for thread_file in thread_files:
            thread_id = thread_file.stem
            if thread_id not in self._threads:
                thread = await self._load_thread(thread_id)
                if thread:
                    self._threads[thread_id] = thread
            
            if thread_id in self._threads:
                thread = self._threads[thread_id]
                
                # Apply filters
                if status and thread.status != status:
                    continue
                if tags and not any(tag in thread.tags for tag in tags):
                    continue
                
                threads.append(thread)
        
        # Sort by updated_at (newest first)
        threads.sort(key=lambda t: t.updated_at, reverse=True)
        
        # Apply limit
        if limit:
            threads = threads[:limit]
        
        return threads

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: Union[str, JsonDict],
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThreadMessage:
        """
        Add a message to a thread.
        
        Args:
            thread_id: Thread ID to add message to
            role: Message role (user, assistant, system, tool)
            content: Message content
            name: Optional message name
            tool_call_id: Optional tool call ID
            metadata: Optional message metadata
            
        Returns:
            Created ThreadMessage object
            
        Raises:
            ValueError: If thread not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")
        
        message = ThreadMessage(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            metadata=metadata or {}
        )
        
        thread.messages.append(message)
        thread.updated_at = int(time.time())
        
        await self._save_thread(thread)
        return message

    async def get_messages(
        self, 
        thread_id: str, 
        limit: Optional[int] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[ThreadMessage]:
        """
        Get messages from a thread with advanced filtering.
        
        Args:
            thread_id: Thread ID to get messages from
            limit: Maximum number of messages to return
            before: Return messages before this message ID
            after: Return messages after this message ID
            role: Filter by message role
            
        Returns:
            List of ThreadMessage objects
            
        Raises:
            ValueError: If thread not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")
        
        messages = thread.messages
        
        # Filter by before if specified
        if before:
            try:
                before_index = next(i for i, msg in enumerate(messages) if msg.id == before)
                messages = messages[:before_index]
            except StopIteration:
                messages = []
        
        # Filter by after if specified
        if after:
            try:
                after_index = next(i for i, msg in enumerate(messages) if msg.id == after)
                messages = messages[after_index + 1:]
            except StopIteration:
                messages = []
        
        # Filter by role if specified
        if role:
            messages = [msg for msg in messages if msg.role == role]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]  # Get last N messages
        
        return messages

    async def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread.
        
        Args:
            thread_id: Thread ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return False
        
        # Mark as deleted instead of actually deleting
        thread.status = "deleted"
        thread.updated_at = int(time.time())
        
        await self._save_thread(thread)
        return True

    async def archive_thread(self, thread_id: str) -> bool:
        """
        Archive a thread.
        
        Args:
            thread_id: Thread ID to archive
            
        Returns:
            True if archived, False if not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return False
        
        thread.status = "archived"
        thread.updated_at = int(time.time())
        
        await self._save_thread(thread)
        return True

    async def restore_thread(self, thread_id: str) -> bool:
        """
        Restore a deleted or archived thread.
        
        Args:
            thread_id: Thread ID to restore
            
        Returns:
            True if restored, False if not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return False
        
        thread.status = "active"
        thread.updated_at = int(time.time())
        
        await self._save_thread(thread)
        return True

    async def update_thread_metadata(
        self, 
        thread_id: str, 
        metadata: Dict[str, Any]
    ) -> Optional[Thread]:
        """
        Update thread metadata.
        
        Args:
            thread_id: Thread ID to update
            metadata: New metadata dictionary
            
        Returns:
            Updated Thread object or None if not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return None
        
        thread.metadata.update(metadata)
        thread.updated_at = int(time.time())
        
        await self._save_thread(thread)
        return thread

    async def add_thread_tags(
        self, 
        thread_id: str, 
        tags: List[str]
    ) -> Optional[Thread]:
        """
        Add tags to a thread.
        
        Args:
            thread_id: Thread ID to add tags to
            tags: List of tags to add
            
        Returns:
            Updated Thread object or None if not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return None
        
        for tag in tags:
            if tag not in thread.tags:
                thread.tags.append(tag)
        
        thread.updated_at = int(time.time())
        await self._save_thread(thread)
        return thread

    async def remove_thread_tags(
        self, 
        thread_id: str, 
        tags: List[str]
    ) -> Optional[Thread]:
        """
        Remove tags from a thread.
        
        Args:
            thread_id: Thread ID to remove tags from
            tags: List of tags to remove
            
        Returns:
            Updated Thread object or None if not found
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return None
        
        thread.tags = [tag for tag in thread.tags if tag not in tags]
        thread.updated_at = int(time.time())
        
        await self._save_thread(thread)
        return thread

    async def get_conversation_summary(self, thread_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Args:
            thread_id: Thread ID to summarize
            
        Returns:
            Summary dictionary
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")
        
        user_messages = [msg for msg in thread.messages if msg.role == "user"]
        assistant_messages = [msg for msg in thread.messages if msg.role == "assistant"]
        
        return {
            "thread_id": thread_id,
            "thread_name": thread.name,
            "total_messages": len(thread.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
            "status": thread.status,
            "tags": thread.tags,
            "metadata": thread.metadata
        }

    async def search_messages(
        self, 
        query: str, 
        thread_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for messages containing specific text.
        
        Args:
            query: Text to search for
            thread_ids: Optional list of thread IDs to search in
            limit: Maximum number of results to return
            
        Returns:
            List of matching messages with thread context
        """
        results = []
        
        # Get threads to search
        if thread_ids:
            threads = []
            for thread_id in thread_ids:
                thread = await self.get_thread(thread_id)
                if thread:
                    threads.append(thread)
        else:
            threads = await self.list_threads()
        
        # Search in each thread
        for thread in threads:
            for message in thread.messages:
                if isinstance(message.content, str) and query.lower() in message.content.lower():
                    results.append({
                        "thread_id": thread.id,
                        "thread_name": thread.name,
                        "message_id": message.id,
                        "role": message.role,
                        "content": message.content,
                        "created_at": message.created_at,
                        "metadata": message.metadata
                    })
        
        # Sort by creation time (newest first)
        results.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        return results

    async def _save_thread(self, thread: Thread) -> None:
        """Save thread to disk."""
        thread_file = self.storage_path / f"{thread.id}.json"
        with open(thread_file, 'w') as f:
            json.dump(thread.dict(), f, indent=2)

    async def _load_thread(self, thread_id: str) -> Optional[Thread]:
        """Load thread from disk."""
        thread_file = self.storage_path / f"{thread_id}.json"
        if thread_file.exists():
            with open(thread_file, 'r') as f:
                data = json.load(f)
            return Thread(**data)
        return None 