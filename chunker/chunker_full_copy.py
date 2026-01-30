"""
chunker_full_copy.py - Full implementation of the Chunker primitive (Docling-powered)

This file is for documentation/reference only. It contains the complete code for the
current chunker primitive, which uses Docling's HybridChunker for initial structural
chunking, then applies character-window slicing with overlap to match Langbase's
reference behavior.

This file is not used in the running codebase, but should be updated whenever the
chunker implementation changes.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional

# -------------------- ChunkTooLargeError --------------------
class ChunkTooLargeError(Exception):
    """Raised when a chunk exceeds the maximum allowed size."""
    def __init__(self, chunk_size: int, max_size: int):
        super().__init__(f"Chunk size {chunk_size} exceeds maximum allowed size {max_size}")
        self.chunk_size = chunk_size
        self.max_size = max_size

# -------------------- ChunkerSettings --------------------
class ChunkerSettings(BaseModel):
    chunk_max_length: int = Field(2000, description="Maximum length of each chunk in characters (1024-30000)")
    chunk_overlap: int = Field(256, description="Character overlap between consecutive chunks (≥256)")
    @validator('chunk_max_length')
    def validate_chunk_max_length(cls, v):
        if v < 1024 or v > 30000:
            raise ValueError('chunk_max_length must be between 1024 and 30000 characters')
        return v
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if v < 256:
            raise ValueError('chunk_overlap must be at least 256 characters')
        if 'chunk_max_length' in values and v >= values['chunk_max_length']:
            raise ValueError('chunk_overlap must be less than chunk_max_length')
        return v

# -------------------- _ChunkerCore --------------------
class _ChunkerCore:
    """
    Stateless text-chunking core that uses Docling's HybridChunker for initial structural chunking,
    then applies character-window slicing with overlap to match Langbase's reference behavior.
    """
    def __init__(self, cfg: ChunkerSettings | None = None) -> None:
        self.cfg = cfg or ChunkerSettings()
        try:
            from docling.chunking import HybridChunker
            self._hybrid_chunker = HybridChunker()
        except ImportError:
            self._hybrid_chunker = None
    async def chunk_text(self, text: str) -> List[str]:
        """
        Split text into character-bounded chunks with sliding-window overlap.
        Uses Docling for initial structural chunking, then applies character-window slicing.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks in order received
            
        Raises:
            ImportError: If Docling is not installed
            ChunkTooLargeError: If any chunk exceeds chunk_max_length
        """
        if not self._hybrid_chunker:
            raise ImportError("Docling is required for chunking. Install with: pip install docling")
        
        if not text:
            return []
        
        # Step 1: Use Docling for initial structural chunking
        from docling_core.types.doc import DoclingDocument
        doc = DoclingDocument(text=text, name="document")
        raw_blocks = [chunk.text for chunk in self._hybrid_chunker.chunk(doc)]
        
        # Step 2: Apply character-window slicing with overlap to respect size constraints
        windowed_chunks = []
        max_length = self.cfg.chunk_max_length
        overlap = self.cfg.chunk_overlap
        stride = max_length - overlap
        
        for block in raw_blocks:
            # If block is shorter than max_length, keep it as-is
            if len(block) <= max_length:
                windowed_chunks.append(block)
                continue
            
            # Apply sliding-window chunking to large blocks
            for start in range(0, len(block), stride):
                end = start + max_length
                chunk = block[start:end]
                windowed_chunks.append(chunk)
                
                # Check size constraint (Langbase parity)
                if len(chunk) > max_length:
                    raise ChunkTooLargeError(len(chunk), max_length)
        
        return windowed_chunks

# -------------------- AsyncChunker --------------------
class AsyncChunker(_ChunkerCore):
    """Stateless async chunker that uses Docling's HybridChunker for chunking."""
    def __init__(self, chunk_max_length: int = 2000, chunk_overlap: int = 256):
        cfg = ChunkerSettings(
            chunk_max_length=chunk_max_length,
            chunk_overlap=chunk_overlap
        )
        super().__init__(cfg)
    async def chunk(self, text: str) -> List[str]:
        """
        Split text into character-bounded chunks with sliding-window overlap (async).
        Args:
            text: Input text to chunk
        Returns:
            List of text chunks in order received
        """
        return await self.chunk_text(text)

# -------------------- SyncChunker --------------------
import asyncio
class SyncChunker(_ChunkerCore):
    """
    Blocking façade that runs the async implementation in the
    event-loop of the current thread, using Docling's HybridChunker for chunking.
    """
    def __init__(self, chunk_max_length: int = 2000, chunk_overlap: int = 256):
        cfg = ChunkerSettings(
            chunk_max_length=chunk_max_length,
            chunk_overlap=chunk_overlap
        )
        super().__init__(cfg)
    def chunk(self, text: str) -> List[str]:
        """
        Split text into character-bounded chunks with sliding-window overlap (sync).
        Args:
            text: Input text to chunk
        Returns:
            List of text chunks in order received
        """
        # Create a new event loop for this thread (Python ≥ 3.11 compatibility)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            coro = super().chunk_text(text)
            return loop.run_until_complete(coro)
        finally:
            loop.close() 