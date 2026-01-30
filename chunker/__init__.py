"""
Docling-powered text chunking with overlap support.

This module provides both sync and async chunkers that use Docling's HybridChunker
for robust, structure-aware chunking, matching Langbase's reference implementation.
"""

from .async_chunker import AsyncChunker
from .sync_chunker import SyncChunker
from .settings import ChunkerSettings
from .base import ChunkTooLargeError

__all__ = ["AsyncChunker", "SyncChunker", "ChunkerSettings", "ChunkTooLargeError"]
