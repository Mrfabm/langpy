import asyncio
from typing import List
from .base import _ChunkerCore
from .settings import ChunkerSettings


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
        Split text into character-bounded chunks with sliding-window overlap.
        
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
