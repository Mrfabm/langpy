from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, Union
from thread.async_thread import AsyncThread, Thread, ThreadMessage

class SyncThread:
    """
    Sync wrapper around AsyncThread for compatibility.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self._async_thread = AsyncThread(storage_path)

    def create_thread(
        self, 
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Thread:
        """Sync wrapper for create_thread."""
        return asyncio.run(self._async_thread.create_thread(name, metadata))

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Sync wrapper for get_thread."""
        return asyncio.run(self._async_thread.get_thread(thread_id))

    def list_threads(self) -> List[Thread]:
        """Sync wrapper for list_threads."""
        return asyncio.run(self._async_thread.list_threads())

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: Union[str, Dict[str, Any]],
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None
    ) -> ThreadMessage:
        """Sync wrapper for add_message."""
        return asyncio.run(self._async_thread.add_message(thread_id, role, content, name, tool_call_id))

    def get_messages(
        self, 
        thread_id: str, 
        limit: Optional[int] = None,
        before: Optional[str] = None
    ) -> List[ThreadMessage]:
        """Sync wrapper for get_messages."""
        return asyncio.run(self._async_thread.get_messages(thread_id, limit, before))

    def delete_thread(self, thread_id: str) -> bool:
        """Sync wrapper for delete_thread."""
        return asyncio.run(self._async_thread.delete_thread(thread_id))

    def update_thread_metadata(
        self, 
        thread_id: str, 
        metadata: Dict[str, Any]
    ) -> Optional[Thread]:
        """Sync wrapper for update_thread_metadata."""
        return asyncio.run(self._async_thread.update_thread_metadata(thread_id, metadata)) 