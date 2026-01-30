"""
Async Memory Implementation with pgvector support and parsing fixes.

This implementation fixes the issue where direct text content was incorrectly
processed through the parser, causing MIME type detection errors.
"""

import asyncio
import logging
import os
import tempfile
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pathlib import Path

from .models import (
    MemorySettings, DocumentMetadata, MemoryChunk, 
    MemoryQuery, MemorySearchResult, MemoryStats
)
from sdk.parser_interface import ParserInterface
from sdk.chunker_interface import ChunkerInterface  
from sdk.embed_interface import EmbedInterface
from stores.base import BaseVectorStore
from sdk.parser_interface import ParseRequest

logger = logging.getLogger(__name__)

class AsyncMemory:
    """
    Async Memory Primitive - orchestrates parser, chunker, embed, and store operations.
    
    Supports both FAISS (local) and pgvector (PostgreSQL) backends.
    """
    
    def __init__(self, settings: MemorySettings):
        self.settings = settings
        self._store: Optional[BaseVectorStore] = None
        self._parser: Optional[ParserInterface] = None
        self._chunker: Optional[ChunkerInterface] = None
        self._embed: Optional[EmbedInterface] = None
        self._initialized = False
        
    async def _ensure_initialized(self):
        """Lazy initialization of all components."""
        if self._initialized:
            return
            
        # Initialize parser
        self._parser = ParserInterface()
        
        # Initialize chunker
        self._chunker = ChunkerInterface()
        
        # Initialize embedder
        self._embed = EmbedInterface()
        
        # Initialize store based on backend
        await self._init_store()
        
        self._initialized = True
        logger.info(f"Memory '{self.settings.name}' initialized with {self.settings.store_backend} backend")
    
    async def _init_store(self):
        """Initialize the vector store based on backend configuration."""
        if self.settings.store_backend == "pgvector":
            from stores.pgvector_store import PgVectorStore
            self._store = PgVectorStore(
                dsn=self.settings.store_uri,
                table_name=f"memory_{self.settings.name}",
                embedding_model=self.settings.embed_model
            )
        else:  # faiss
            from stores.faiss_store import FaissStore
            self._store = FaissStore(
                index_path=f"./memory_{self.settings.name}.faiss",
                embedding_model=self.settings.embed_model
            )
    
    async def _parse(
        self,
        content: Union[str, Path, bytes],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Parse content with fix for direct text content.
        
        The key fix: For direct text strings, return as-is without parsing
        to avoid MIME type detection issues that cause RuntimeError.
        """
        if isinstance(content, str):
            # Direct text content - return as-is to avoid parser MIME type issues
            logger.debug("Processing direct text content, skipping parser")
            result = await self._parser.parse_text(content)
            return result.pages[0] if result.pages else ""
            
        elif isinstance(content, Path):
            # File content - use parser
            logger.debug(f"Processing file content via parser: {content}")
            result = await self._parser.parse_file(content)
            return result.pages[0] if result.pages else ""
            
        elif isinstance(content, bytes):
            # Bytes content - use parser
            logger.debug(f"Processing bytes content via parser")
            result = await self._parser.parse_bytes(content)
            return result.pages[0] if result.pages else ""
            
        else:
            # Fallback to parser for other types
            logger.debug(f"Using parser for unknown content type: {type(content)}")
            result = await self._parser.parse_content(content)
            return result.pages[0] if result.pages else ""
    
    async def upload(
        self,
        content: Union[str, Path, bytes],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentMetadata:
        """
        Upload content to memory and return processing metadata.

        Pipeline: Parser → Chunker → Embed → Store
        """
        await self._ensure_initialized()

        # Copy to avoid mutating the caller's dict
        md: Dict[str, Any] = dict(metadata or {})

        try:
            # Parse content (your _parse should handle str/Path/bytes properly)
            parsed_text: str = await self._parse(content, md)
            if not parsed_text:
                raise ValueError("Parsed text is empty")

            # Chunk
            chunk_texts: List[str] = await self._chunker.chunk_text(parsed_text)
            if not chunk_texts:
                raise ValueError("No chunks produced from parsed text")

            # Embed
            embeddings: List[List[float]] = await self._embed.embed_texts(chunk_texts)
            if len(embeddings) != len(chunk_texts):
                raise RuntimeError(
                    f"Embeddings count ({len(embeddings)}) != chunks count ({len(chunk_texts)})"
                )

            # Prepare per-chunk metadata
            chunk_metas: List[Dict[str, Any]] = []
            total_chunks = len(chunk_texts)
            for i, chunk_text in enumerate(chunk_texts):
                chunk_metas.append(
                    {
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        **md,
                    }
                )

            # Persist
            # Adapt arg names to your vector store API
            await self._store.add(
                texts=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metas,
            )

            # Build document metadata
            doc_id = str(uuid.uuid4())
            doc_metadata = DocumentMetadata(
                id=doc_id,
                name=str(content) if isinstance(content, (str, Path)) else "bytes_content",
                size=len(parsed_text),
                chunks=total_chunks,
                status="completed",
                metadata=md,
            )

            logger.info(
                "Successfully uploaded content: %s chunks, %s chars (doc_id=%s)",
                total_chunks,
                len(parsed_text),
                doc_id,
            )
            return doc_metadata

        except Exception as e:
            logger.exception("Upload failed")
            raise
    
    async def query(
        self, 
        query: str, 
        limit: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[MemorySearchResult]:
        """
        Query memory for similar content.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filter criteria
            
        Returns:
            List of search results
        """
        await self._ensure_initialized()
        
        try:
            # Query the vector store
            results = await self._store.query(query, k=limit, filt=filters)
            
            # Convert to MemorySearchResult objects
            search_results = []
            for result in results:
                search_result = MemorySearchResult(
                    text=result.get("text", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    chunk_index=result.get("chunk_index", 0)
                )
                search_results.append(search_result)
            
            logger.info(f"Query returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    async def clear(self) -> None:
        """Clear all stored content."""
        await self._ensure_initialized()
        await self._store.clear()
        logger.info(f"Cleared memory '{self.settings.name}'")
    
    async def stats(self) -> MemoryStats:
        """Get memory statistics."""
        await self._ensure_initialized()
        
        try:
            token_usage = await self._store.token_usage()
            metadata_stats = await self._store.get_metadata_stats()
            
            return MemoryStats(
                total_chunks=metadata_stats.get("total_documents", 0),
                total_tokens=token_usage,
                backend=self.settings.store_backend,
                embed_model=self.settings.embed_model
            )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return MemoryStats(
                total_chunks=0,
                total_tokens=0,
                backend=self.settings.store_backend,
                embed_model=self.settings.embed_model
            )
    
    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete documents matching filter criteria."""
        await self._ensure_initialized()
        return await self._store.delete_by_filter(filters)
    
    async def update_metadata(self, filters: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """Update metadata for documents matching filter criteria."""
        await self._ensure_initialized()
        return await self._store.update_metadata(filters, updates) 