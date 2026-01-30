from __future__ import annotations
from typing import List
from chunker.settings import ChunkerSettings

class ChunkTooLargeError(Exception):
    """Raised when a chunk exceeds the maximum allowed size."""
    def __init__(self, chunk_size: int, max_size: int):
        super().__init__(f"Chunk size {chunk_size} exceeds maximum allowed size {max_size}")
        self.chunk_size = chunk_size
        self.max_size = max_size

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
        
        # Fallback: If no blocks, treat the input as a single block (plain text)
        if not raw_blocks:
            raw_blocks = [text]
        
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
