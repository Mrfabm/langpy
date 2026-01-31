"""
Memory Primitive - Langbase-compatible Memory/RAG API.

The Memory primitive provides vector storage, semantic search,
and RAG (Retrieval-Augmented Generation) capabilities.

Usage:
    # Direct API (Langbase-compatible)
    await memory.add(documents=[{"content": "...", "metadata": {...}}])
    results = await memory.retrieve(query="search term", top_k=5)

    # Pipeline composition (RAG)
    pipeline = memory | pipe
    result = await pipeline.process(ctx)
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

from langpy.core.primitive import BasePrimitive, MemoryResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context, Document
    from langpy.core.result import Result


class Memory(BasePrimitive):
    """
    Memory primitive - Vector storage and RAG.

    Memory is a managed RAG system that handles:
    - Document parsing (PDF, Office, images, etc.)
    - Text chunking
    - Embedding generation
    - Vector storage (FAISS or pgvector)
    - Semantic similarity search

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Add documents
        await lb.memory.add(
            documents=[
                {"content": "Python is a programming language.", "metadata": {"source": "wiki"}},
                {"content": "JavaScript runs in browsers.", "metadata": {"source": "wiki"}}
            ]
        )

        # Or upload files
        await lb.memory.upload(file="document.pdf")

        # Search
        results = await lb.memory.retrieve(
            query="What is Python?",
            top_k=5,
            min_score=0.7
        )

        # Use in RAG pipeline
        pipeline = lb.memory | lb.pipe
        result = await pipeline.process(Context(query="Explain Python"))
    """

    def __init__(
        self,
        client: Any = None,
        name: str = "memory",
        backend: str = "faiss",
        **settings
    ):
        """
        Initialize the Memory primitive.

        Args:
            client: Parent Langpy client
            name: Memory name/namespace
            backend: "faiss" (local) or "pgvector" (PostgreSQL)
            **settings: Additional settings (embed_model, store_uri, etc.)
        """
        super().__init__(name=name, client=client)
        self._backend = backend
        self._settings = settings
        self._async_memory = None

    def _get_async_memory(self):
        """Get or create the underlying AsyncMemory."""
        if self._async_memory is None:
            from memory.async_memory import AsyncMemory
            from memory.models import MemorySettings

            settings = MemorySettings(
                name=self._name,
                store_backend=self._backend,
                store_uri=self._settings.get('store_uri') or os.getenv('LANGPY_PG_DSN'),
                embed_model=self._settings.get('embed_model', 'openai:text-embedding-3-small'),
                **{k: v for k, v in self._settings.items() if k not in ['store_uri', 'embed_model']}
            )
            self._async_memory = AsyncMemory(settings)
        return self._async_memory

    # ========================================================================
    # Langbase-compatible API
    # ========================================================================

    async def _run(
        self,
        action: str = "retrieve",
        query: str = None,
        documents: List[Dict[str, Any]] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        filter: Dict[str, Any] = None,
        **kwargs
    ) -> MemoryResponse:
        """
        Run memory operation.

        Args:
            action: "retrieve", "add", or "delete"
            query: Search query (for retrieve)
            documents: Documents to add (for add)
            top_k: Number of results (for retrieve)
            min_score: Minimum similarity score
            filter: Metadata filter

        Returns:
            MemoryResponse with results
        """
        if action == "retrieve":
            return await self.retrieve(query=query, top_k=top_k, min_score=min_score, filter=filter)
        elif action == "add":
            return await self.add(documents=documents, **kwargs)
        elif action == "delete":
            return await self.delete(filter=filter, **kwargs)
        else:
            return MemoryResponse(success=False, error=f"Unknown action: {action}")

    async def add(
        self,
        documents: List[Dict[str, Any]] = None,
        content: str = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> MemoryResponse:
        """
        Add documents to memory.

        Args:
            documents: List of {"content": str, "metadata": dict}
            content: Single content string (alternative to documents)
            metadata: Metadata for single content

        Returns:
            MemoryResponse with add status
        """
        try:
            mem = self._get_async_memory()
            await mem._ensure_initialized()

            # Normalize input
            if content:
                documents = [{"content": content, "metadata": metadata or {}}]

            if not documents:
                return MemoryResponse(
                    success=False,
                    error="No documents provided"
                )

            added_count = 0
            for doc in documents:
                doc_content = doc.get("content", "")
                doc_metadata = doc.get("metadata", {})

                # Parse, chunk, embed, store
                text = await mem._parse(doc_content, doc_metadata)
                chunks = await mem._chunker.chunk(text)

                for chunk in chunks:
                    embedding = await mem._embed.embed([chunk])
                    await mem._store.add(
                        embeddings=embedding,
                        texts=[chunk],
                        metadatas=[doc_metadata]
                    )
                    added_count += 1

            return MemoryResponse(
                success=True,
                action="add",
                count=added_count,
                documents=None
            )

        except Exception as e:
            return MemoryResponse(success=False, error=str(e), action="add")

    async def upload(
        self,
        file: Union[str, Path],
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> MemoryResponse:
        """
        Upload and process a file.

        Args:
            file: File path
            metadata: Additional metadata

        Returns:
            MemoryResponse with upload status
        """
        try:
            mem = self._get_async_memory()
            await mem._ensure_initialized()

            file_path = Path(file)
            if not file_path.exists():
                return MemoryResponse(
                    success=False,
                    error=f"File not found: {file}"
                )

            # Parse file
            text = await mem._parse(file_path, metadata)

            # Chunk
            chunks = await mem._chunker.chunk(text)

            # Embed and store
            added_count = 0
            file_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                **(metadata or {})
            }

            for chunk in chunks:
                embedding = await mem._embed.embed([chunk])
                await mem._store.add(
                    embeddings=embedding,
                    texts=[chunk],
                    metadatas=[file_metadata]
                )
                added_count += 1

            return MemoryResponse(
                success=True,
                action="upload",
                count=added_count
            )

        except Exception as e:
            return MemoryResponse(success=False, error=str(e), action="upload")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter: Dict[str, Any] = None,
        **kwargs
    ) -> MemoryResponse:
        """
        Search memory for relevant documents.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score (0-1)
            filter: Metadata filter

        Returns:
            MemoryResponse with matching documents
        """
        try:
            mem = self._get_async_memory()
            await mem._ensure_initialized()

            # Embed query
            query_embedding = await mem._embed.embed([query])

            # Search
            results = await mem._store.search(
                query_embedding=query_embedding[0],
                top_k=top_k,
                filter=filter
            )

            # Filter by min_score and format
            documents = []
            for result in results:
                score = result.get("score", 0)
                if score >= min_score:
                    documents.append({
                        "content": result.get("text", ""),
                        "score": score,
                        "metadata": result.get("metadata", {})
                    })

            return MemoryResponse(
                success=True,
                action="retrieve",
                documents=documents,
                count=len(documents)
            )

        except Exception as e:
            return MemoryResponse(success=False, error=str(e), action="retrieve")

    async def delete(
        self,
        ids: List[str] = None,
        filter: Dict[str, Any] = None,
        **kwargs
    ) -> MemoryResponse:
        """
        Delete documents from memory.

        Args:
            ids: Document IDs to delete
            filter: Metadata filter for deletion

        Returns:
            MemoryResponse with deletion status
        """
        try:
            mem = self._get_async_memory()
            await mem._ensure_initialized()

            if ids:
                await mem._store.delete(ids=ids)
            elif filter:
                await mem._store.delete(filter=filter)
            else:
                return MemoryResponse(
                    success=False,
                    error="Provide ids or filter for deletion"
                )

            return MemoryResponse(
                success=True,
                action="delete"
            )

        except Exception as e:
            return MemoryResponse(success=False, error=str(e), action="delete")

    async def stats(self) -> MemoryResponse:
        """
        Get memory statistics.

        Returns:
            MemoryResponse with stats
        """
        try:
            mem = self._get_async_memory()
            await mem._ensure_initialized()

            stats = await mem._store.stats()

            return MemoryResponse(
                success=True,
                action="stats",
                count=stats.get("count", 0),
                documents=[stats]
            )

        except Exception as e:
            return MemoryResponse(success=False, error=str(e), action="stats")

    # ========================================================================
    # Pipeline API: process()
    # ========================================================================

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Process context for pipeline composition (RAG retrieval).

        Reads ctx.query, retrieves relevant documents, adds to ctx.documents.

        Args:
            ctx: Input context with query

        Returns:
            Result[Context] with documents populated
        """
        if not ctx.query:
            return Failure(PrimitiveError(
                code=ErrorCode.VALIDATION_ERROR,
                message="Query required for memory retrieval",
                primitive=self._name
            ))

        # Get retrieval parameters from context
        top_k = ctx.get("top_k", 5)
        min_score = ctx.get("min_score", 0.0)
        filter_dict = ctx.get("filter")

        # Retrieve
        response = await self.retrieve(
            query=ctx.query,
            top_k=top_k,
            min_score=min_score,
            filter=filter_dict
        )

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error or "Memory retrieval failed",
                primitive=self._name
            ))

        # Convert to Context Documents
        from langpy.core.context import Document

        documents = []
        for doc in (response.documents or []):
            documents.append(Document(
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            ))

        # Update context
        new_ctx = ctx.with_documents(documents)
        new_ctx = new_ctx.set("memory_results", response.documents)
        new_ctx = new_ctx.set("memory_count", response.count)

        return Success(new_ctx)

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Simple search interface.

        Args:
            query: Search query
            **kwargs: Additional options

        Returns:
            List of matching documents
        """
        response = await self.retrieve(query=query, **kwargs)
        return response.documents if response.success else []

    async def query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Alias for search()."""
        return await self.search(query, **kwargs)
