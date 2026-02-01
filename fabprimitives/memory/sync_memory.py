"""
Sync Memory Orchestrator - Synchronous wrapper for the memory orchestrator primitive.

Provides synchronous access to memory operations that coordinate parser, chunker, embed, and store operations.
"""

import asyncio
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path

from .models import (
    MemorySettings, DocumentMetadata, MemoryChunk, DocumentProcessingJob,
    DocumentStatus, MemoryQuery, MemorySearchResult, MemoryStats,
    ProcessingProgress
)
from .async_memory import AsyncMemory


class SyncMemory:
    """
    Synchronous memory orchestrator that coordinates parser, chunker, embed, and store operations.
    
    Provides synchronous wrappers around the async memory operations.
    """
    
    def __init__(self, settings: Optional[MemorySettings] = None, register_globally: bool = True):
        """
        Initialize the sync memory orchestrator.
        
        Args:
            settings: Memory configuration settings
        """
        self._async_memory = AsyncMemory(settings, register_globally=register_globally)
    
    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return asyncio.ensure_future(coro)
        else:
            return asyncio.run(coro)

    def upload(
        self,
        content: Union[str, bytes, Path],
        source: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        background: bool = True
    ) -> str:
        """
        Upload a document to memory, automatically running the full pipeline.
        
        This triggers the internal pipeline: Parser → Chunker → Embed → Store
        
        Args:
            content: Document content (file path, bytes, or text)
            source: Optional source identifier
            custom_metadata: Additional custom metadata
            progress_callback: Optional callback for progress updates
            background: Whether to run in background (True) or wait for completion (False)
            
        Returns:
            Job ID for tracking the upload process
        """
        return self._run_async(self._async_memory.upload(
            content=content,
            source=source,
            custom_metadata=custom_metadata,
            progress_callback=progress_callback,
            background=background
        ))
    
    def get_job_status(self, job_id: str) -> Optional[DocumentProcessingJob]:
        """Get the status of a processing job."""
        return self._run_async(self._async_memory.get_job_status(job_id))
    
    def query(
        self,
        query: str,
        k: Optional[int] = None,
        source: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[MemorySearchResult]:
        """
        Query memory for similar content.
        
        Args:
            query: Search query
            k: Number of results to return
            source: Filter by source
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        return self._run_async(self._async_memory.query(
            query=query,
            k=k,
            source=source,
            min_score=min_score
        ))
    
    def get_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        k: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """
        Get memory entries by metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata to match
            k: Number of results to return
            
        Returns:
            List of matching memory entries
        """
        return self._run_async(self._async_memory.get_by_metadata(
            metadata_filter=metadata_filter,
            k=k
        ))
    
    def delete_by_filter(
        self,
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Delete memory entries matching filter criteria.
        
        Args:
            source: Filter by source
            metadata_filter: Additional metadata filters
            
        Returns:
            Number of deleted entries
        """
        return self._run_async(self._async_memory.delete_by_filter(
            source=source,
            metadata_filter=metadata_filter
        ))
    
    def update_metadata(
        self,
        updates: Dict[str, Any],
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Update metadata for entries matching filter criteria.
        
        Args:
            updates: Metadata updates to apply
            source: Filter by source
            metadata_filter: Additional metadata filters
            
        Returns:
            Number of updated entries
        """
        return self._run_async(self._async_memory.update_metadata(
            updates=updates,
            source=source,
            metadata_filter=metadata_filter
        ))
    
    def clear(self):
        """Clear all memory data."""
        self._run_async(self._async_memory.clear())
    
    def get_stats(self) -> MemoryStats:
        """
        Get memory statistics.
        
        Returns:
            Memory statistics
        """
        return self._run_async(self._async_memory.get_stats())
    
    # Standalone primitive access methods
    def parse_document(self, content: Union[str, bytes, Path], **kwargs) -> str:
        """Parse a document using the parser primitive directly."""
        return self._run_async(self._async_memory.parse_document(content, **kwargs))
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text using the chunker primitive directly."""
        return self._run_async(self._async_memory.chunk_text(text, **kwargs))
    
    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using the embed primitive directly."""
        return self._run_async(self._async_memory.embed_texts(texts, **kwargs))
    
    # Langbase-style method aliases
    def retrieve(self, *args, **kwargs):
        """Langbase alias for query."""
        return self.query(*args, **kwargs)

    def stats(self, *args, **kwargs):
        """Langbase alias for get_stats."""
        return self.get_stats(*args, **kwargs)

    def info(self, *args, **kwargs):
        """Langbase alias for get_stats (info)."""
        return self.get_stats(*args, **kwargs)
    
    def delete(self, *args, **kwargs):
        """Langbase alias for delete_by_filter (per-document delete)."""
        return self.delete_by_filter(*args, **kwargs)

    def update(self, *args, **kwargs):
        """Langbase alias for update_metadata (per-document update)."""
        return self.update_metadata(*args, **kwargs)

    def flush(self, *args, **kwargs):
        """Langbase alias for close (flush resources)."""
        return self.close(*args, **kwargs)
    
    def close(self):
        """Close memory connections."""
        self._run_async(self._async_memory.close()) 