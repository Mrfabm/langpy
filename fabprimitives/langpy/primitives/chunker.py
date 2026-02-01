"""
Chunker Primitive - Langbase-compatible Text Chunking API.

The Chunker primitive splits text into smaller chunks for embedding.

Usage:
    # Direct API
    result = await lb.chunker.run(content="Long text...", chunk_size=512)
    print(result.chunks)

    # Pipeline composition
    pipeline = parser | chunker | embed
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langpy.core.primitive import BasePrimitive, ChunkerResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


class Chunker(BasePrimitive):
    """
    Chunker primitive - Text segmentation for RAG.

    Splits text into chunks optimized for embedding and retrieval.

    Features:
    - Token-aware chunking
    - Configurable chunk size and overlap
    - Semantic boundary detection
    - Metadata preservation

    Example:
        from langpy import Langpy

        lb = Langpy()

        result = await lb.chunker.run(
            content="Very long document text...",
            chunk_size=512,
            overlap=50
        )

        for chunk in result.chunks:
            print(f"Chunk: {chunk[:50]}...")
    """

    def __init__(
        self,
        client: Any = None,
        name: str = "chunker",
        chunk_size: int = 512,
        overlap: int = 50
    ):
        super().__init__(name=name, client=client)
        self._default_chunk_size = chunk_size
        self._default_overlap = overlap
        self._async_chunker = None

    def _get_async_chunker(self):
        if self._async_chunker is None:
            try:
                from chunker.async_chunker import AsyncChunker
                self._async_chunker = AsyncChunker()
            except ImportError:
                pass  # Use fallback
        return self._async_chunker

    def _simple_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple character-based chunking fallback."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Try to break at sentence/paragraph boundary
            if end < text_len:
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", "! ", "? ", "\n"]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap if overlap < end - start else end

        return chunks

    async def _run(
        self,
        content: str = None,
        chunk_size: int = None,
        overlap: int = None,
        **kwargs
    ) -> ChunkerResponse:
        """
        Chunk text content.

        Args:
            content: Text to chunk
            chunk_size: Maximum chunk size (tokens or chars)
            overlap: Overlap between chunks

        Returns:
            ChunkerResponse with chunks
        """
        try:
            if not content:
                return ChunkerResponse(
                    success=False,
                    error="content required"
                )

            size = chunk_size or self._default_chunk_size
            ovlp = overlap or self._default_overlap

            # Try async chunker
            async_chunker = self._get_async_chunker()
            if async_chunker:
                chunks = await async_chunker.chunk_text(content)
            else:
                chunks = self._simple_chunk(content, size, ovlp)

            return ChunkerResponse(
                success=True,
                chunks=chunks,
                count=len(chunks),
                chunk_size=size,
                overlap=ovlp
            )

        except Exception as e:
            return ChunkerResponse(success=False, error=str(e))

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Process context - chunk text from parsed_text or query."""
        content = ctx.get("parsed_text") or ctx.query

        if not content:
            return Failure(PrimitiveError(
                code=ErrorCode.VALIDATION_ERROR,
                message="No content to chunk",
                primitive=self._name
            ))

        response = await self._run(
            content=content,
            chunk_size=ctx.get("chunk_size"),
            overlap=ctx.get("overlap")
        )

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error,
                primitive=self._name
            ))

        new_ctx = ctx.set("chunks", response.chunks)
        new_ctx = new_ctx.set("chunk_count", response.count)

        return Success(new_ctx)

    # Convenience methods
    async def chunk(self, content: str, **kwargs) -> List[str]:
        """Chunk text and return list of chunks."""
        response = await self._run(content=content, **kwargs)
        return response.chunks if response.success else []
