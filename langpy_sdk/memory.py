"""
LangPy Memory - Clean SDK wrapper for vector-based storage.

Simple, intuitive interface for storing and searching documents.

Supports both the original API and the new composable architecture:

Original API:
    from langpy_sdk import Memory

    memory = Memory(name="knowledge")
    await memory.add("Paris is the capital of France")
    results = await memory.search("What is France's capital?")
    print(results[0].text)

New Composable API:
    from langpy.core import Context
    from langpy_sdk import Memory, Pipe

    # Memory as a primitive in a RAG pipeline
    memory = Memory(name="knowledge", k=5)
    pipe = Pipe(system_prompt="Answer using context.")

    rag = memory | pipe
    result = await rag.process(Context(query="What is France's capital?"))
"""

from __future__ import annotations
import os
import uuid
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

# Import core types for the new architecture
try:
    from langpy.core.context import Context, Document, TokenUsage
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode
    from langpy.core.primitive import BasePrimitive
    _NEW_ARCH_AVAILABLE = True
except ImportError:
    _NEW_ARCH_AVAILABLE = False
    Context = Any
    Result = Any


@dataclass
class SearchResult:
    """
    A single search result from memory.

    Attributes:
        text: The stored text content
        score: Similarity score (0-1, higher is better)
        metadata: Associated metadata
        id: Document ID
    """
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass
class MemoryStats:
    """
    Memory statistics.

    Attributes:
        total_documents: Number of documents stored
        total_chunks: Number of chunks (same as documents for simple storage)
        backend: Storage backend name
    """
    total_documents: int
    total_chunks: int
    backend: str


class Memory:
    """
    Clean, simple Memory interface for vector storage.

    Provides easy document storage and semantic search.
    Implements IPrimitive for composable RAG pipelines.

    Args:
        name: Name for this memory store (used for persistence)
        backend: Storage backend - "faiss" (local/fast) or "pgvector" (postgres)
        embedding_model: OpenAI embedding model (default: text-embedding-3-small)
        dsn: PostgreSQL connection string (only for pgvector)
        k: Number of documents to retrieve (for process() method, default: 5)
        min_score: Minimum similarity score (for process() method, default: None)
        primitive_name: Name for tracing (default: "MemorySearch")

    Example:
        # Original API - Create memory and manage documents
        memory = Memory(name="docs")
        await memory.add("Python is great for AI")
        results = await memory.search("AI programming language")

        # New Composable API - Use in RAG pipeline
        from langpy.core import Context

        memory = Memory(name="docs", k=5)
        pipe = Pipe(system_prompt="Answer using context.")

        rag = memory | pipe
        result = await rag.process(Context(query="Tell me about Python"))
    """

    def __init__(
        self,
        name: str = "default",
        backend: str = "faiss",
        embedding_model: str = "text-embedding-3-small",
        dsn: Optional[str] = None,
        # New parameters for composable architecture
        k: int = 5,
        min_score: Optional[float] = None,
        primitive_name: str = "MemorySearch",
        filter: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.backend = backend
        self.embedding_model = embedding_model
        self.dsn = dsn or os.getenv("POSTGRES_DSN")

        # New architecture fields
        self._k = k
        self._min_score = min_score
        self._primitive_name = primitive_name
        self._filter = filter

        # Internal state
        self._embedder: Optional[_Embedder] = None
        self._store: Optional[_VectorStore] = None
        self._initialized = False

    @property
    def primitive_name(self) -> str:
        """Return the primitive name for tracing (alias)."""
        return self._primitive_name

    async def _ensure_initialized(self) -> None:
        """Lazy initialization."""
        if self._initialized:
            return

        # Create embedder
        self._embedder = _Embedder(model=self.embedding_model)

        # Create store
        if self.backend == "faiss":
            self._store = _FaissStore(name=self.name)
        elif self.backend == "pgvector":
            if not self.dsn:
                raise ValueError(
                    "pgvector backend requires dsn parameter or POSTGRES_DSN env var"
                )
            self._store = _PgVectorStore(
                name=self.name,
                dsn=self.dsn,
                embedding_model=self.embedding_model
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        await self._store.initialize()
        self._initialized = True

    async def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add text to memory.

        Args:
            text: Text content to store
            metadata: Optional metadata dictionary

        Returns:
            Document ID

        Example:
            doc_id = await memory.add(
                "The sky is blue",
                metadata={"source": "facts", "category": "nature"}
            )
        """
        await self._ensure_initialized()

        embedding = await self._embedder.embed(text)
        doc_id = await self._store.add(text, embedding, metadata or {})
        return doc_id

    async def add_many(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple texts efficiently.

        Args:
            texts: List of text content
            metadata: Optional list of metadata dicts

        Returns:
            List of document IDs

        Example:
            doc_ids = await memory.add_many([
                "First document",
                "Second document",
                "Third document"
            ])
        """
        await self._ensure_initialized()

        if metadata is None:
            metadata = [{} for _ in texts]

        embeddings = await self._embedder.embed_many(texts)
        doc_ids = await self._store.add_many(texts, embeddings, metadata)
        return doc_ids

    async def search(
        self,
        query: str,
        limit: int = 5,
        min_score: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar content.

        Args:
            query: Search query text
            limit: Maximum results to return (default: 5)
            min_score: Minimum similarity score (0-1)
            filter: Metadata filter (exact match)

        Returns:
            List of SearchResult objects, sorted by relevance

        Example:
            results = await memory.search(
                "machine learning",
                limit=10,
                min_score=0.7,
                filter={"category": "tech"}
            )
        """
        await self._ensure_initialized()

        query_embedding = await self._embedder.embed(query)
        raw_results = await self._store.search(query_embedding, limit, filter)

        results = []
        for r in raw_results:
            score = r["score"]
            if min_score is not None and score < min_score:
                continue
            results.append(SearchResult(
                text=r["text"],
                score=score,
                metadata=r.get("metadata", {}),
                id=r.get("id", "")
            ))

        return results

    async def delete(
        self,
        id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Delete documents.

        Args:
            id: Specific document ID to delete
            filter: Delete all matching this metadata filter

        Returns:
            Number of documents deleted

        Example:
            # Delete by ID
            await memory.delete(id="doc_123")

            # Delete by filter
            await memory.delete(filter={"source": "old_data"})
        """
        await self._ensure_initialized()

        if id:
            return await self._store.delete(id)
        elif filter:
            return await self._store.delete_by_filter(filter)
        else:
            raise ValueError("Must provide id or filter")

    async def clear(self) -> None:
        """
        Clear all documents from memory.

        Example:
            await memory.clear()
        """
        await self._ensure_initialized()
        await self._store.clear()

    async def stats(self) -> MemoryStats:
        """
        Get memory statistics.

        Returns:
            MemoryStats object

        Example:
            stats = await memory.stats()
            print(f"Documents: {stats.total_documents}")
        """
        await self._ensure_initialized()
        store_stats = await self._store.stats()
        return MemoryStats(
            total_documents=store_stats["count"],
            total_chunks=store_stats["count"],
            backend=self.backend
        )

    def __repr__(self) -> str:
        return f"Memory(name='{self.name}', backend='{self.backend}')"

    async def quick(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[str]:
        """
        Quick helper for simple searches. Returns just the text content.

        This is the recommended way for simple searches.
        For composition with other primitives, use .process(ctx) instead.

        Args:
            query: Search query text
            limit: Maximum results to return (default: uses instance k)
            min_score: Minimum similarity score (default: uses instance min_score)

        Returns:
            List of text strings from matching documents

        Example:
            memory = Memory(name="docs")
            await memory.add("Python is great for AI")
            results = await memory.quick("AI programming")
            print(results)  # ["Python is great for AI"]
        """
        results = await self.search(
            query=query,
            limit=limit or self._k,
            min_score=min_score or self._min_score
        )
        return [r.text for r in results]

    # ========================================================================
    # New Composable Architecture Methods
    # ========================================================================

    @property
    def name_prop(self) -> str:
        """Return the primitive name for tracing."""
        return self._primitive_name

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process a context by searching for relevant documents (IPrimitive interface).

        This method enables composable RAG pipelines with the | operator.
        It takes the query from the context, searches for relevant documents,
        and adds them to the context for downstream primitives (like Pipe).

        Args:
            ctx: Input context with query

        Returns:
            Result[Context] - Success with documents added or Failure with error

        Example:
            from langpy.core import Context

            memory = Memory(name="knowledge", k=5)
            ctx = Context(query="What is Python?")
            result = await memory.process(ctx)

            if result.is_success():
                for doc in result.unwrap().documents:
                    print(f"{doc.score:.2f}: {doc.content}")
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError(
                "New architecture not available. "
                "Make sure langpy.core is properly installed."
            )

        from langpy.core.context import Context, Document
        from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

        # Start span for tracing
        ctx = ctx.start_span(self._primitive_name, {
            "k": self._k,
            "backend": self.backend,
            "memory_name": self.name
        })

        try:
            # Get the query from context
            query = ctx.query
            if not query:
                ctx = ctx.end_span("error", "No query provided")
                return Failure(PrimitiveError(
                    code=ErrorCode.MISSING_REQUIRED,
                    message="Context must have a query for memory search",
                    primitive=self._primitive_name
                ))

            # Perform the search
            results = await self.search(
                query=query,
                limit=self._k,
                min_score=self._min_score,
                filter=self._filter
            )

            # Convert SearchResults to Documents and add to context
            result_ctx = ctx
            for r in results:
                doc = Document(
                    content=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    id=r.id
                )
                result_ctx = result_ctx.add_document(doc)

            # Store search metadata in variables
            result_ctx = result_ctx.set("_memory_search_count", len(results))
            result_ctx = result_ctx.set("_memory_name", self.name)

            # End span successfully
            result_ctx = result_ctx.end_span("ok")

            return Success(result_ctx)

        except Exception as e:
            # End span with error
            ctx = ctx.end_span("error", str(e))

            # Determine error code
            error_name = type(e).__name__.lower()
            if "connection" in error_name or "connect" in str(e).lower():
                code = ErrorCode.MEMORY_CONNECTION_ERROR
            elif "embed" in error_name or "embedding" in str(e).lower():
                code = ErrorCode.EMBEDDING_ERROR
            else:
                code = ErrorCode.UNKNOWN

            return Failure(PrimitiveError(
                code=code,
                message=str(e),
                primitive=self._primitive_name,
                cause=e
            ))

    def __or__(self, other) -> "Any":
        """
        Sequential composition with the | operator.

        Example:
            pipeline = memory | pipe | validator
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import Pipeline
        return Pipeline([self, other])

    def __and__(self, other) -> "Any":
        """
        Parallel composition with the & operator.

        Example:
            parallel = memory1 & memory2
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import ParallelPrimitives
        return ParallelPrimitives([self, other])

    def __rshift__(self, other) -> "Any":
        """
        Alternative sequential composition with >> operator.

        Example:
            pipeline = memory >> pipe >> validator
        """
        return self.__or__(other)


# ============================================================================
# Internal Components
# ============================================================================

class _Embedder:
    """Simple OpenAI embedder."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Embed single text."""
        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


class _VectorStore:
    """Base vector store interface."""

    async def initialize(self) -> None:
        raise NotImplementedError

    async def add(self, text: str, embedding: List[float], metadata: Dict) -> str:
        raise NotImplementedError

    async def add_many(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> List[str]:
        raise NotImplementedError

    async def search(self, embedding: List[float], limit: int, filter: Optional[Dict]) -> List[Dict]:
        raise NotImplementedError

    async def delete(self, doc_id: str) -> int:
        raise NotImplementedError

    async def delete_by_filter(self, filter: Dict) -> int:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def stats(self) -> Dict:
        raise NotImplementedError


class _FaissStore(_VectorStore):
    """FAISS-based local vector store with persistence."""

    def __init__(self, name: str, persist_dir: Optional[str] = None):
        self.name = name
        self._index = None
        self._docs: List[Dict] = []
        self._use_faiss = True
        # Persistence paths
        self._persist_dir = persist_dir or "./.langpy_memory"
        self._index_path = os.path.join(self._persist_dir, f"{name}.faiss")
        self._docs_path = os.path.join(self._persist_dir, f"{name}_docs.json")

    async def initialize(self) -> None:
        import json

        # Ensure persistence directory exists
        os.makedirs(self._persist_dir, exist_ok=True)

        try:
            import faiss
            self._use_faiss = True

            # Try to load existing index
            if os.path.exists(self._index_path) and os.path.exists(self._docs_path):
                self._index = faiss.read_index(self._index_path)
                with open(self._docs_path, 'r') as f:
                    self._docs = json.load(f)
            else:
                self._index = None
        except ImportError:
            self._use_faiss = False
            self._vectors: List[List[float]] = []

            # Load docs even without FAISS
            if os.path.exists(self._docs_path):
                with open(self._docs_path, 'r') as f:
                    data = json.load(f)
                    self._docs = data.get("docs", [])
                    self._vectors = data.get("vectors", [])

    async def _persist(self) -> None:
        """Save index and docs to disk."""
        import json

        os.makedirs(self._persist_dir, exist_ok=True)

        if self._use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, self._index_path)
            with open(self._docs_path, 'w') as f:
                json.dump(self._docs, f)
        else:
            # Save both vectors and docs for non-FAISS mode
            with open(self._docs_path, 'w') as f:
                json.dump({"docs": self._docs, "vectors": self._vectors}, f)

    async def add(self, text: str, embedding: List[float], metadata: Dict) -> str:
        import numpy as np

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        self._docs.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata
        })

        vec = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self._use_faiss:
            import faiss
            if self._index is None:
                self._index = faiss.IndexFlatIP(len(embedding))
            self._index.add(vec)
        else:
            self._vectors.append(vec[0].tolist())

        # Persist after adding
        await self._persist()

        return doc_id

    async def add_many(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> List[str]:
        doc_ids = []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            doc_id = await self.add(text, emb, meta)
            doc_ids.append(doc_id)
        return doc_ids

    async def search(self, embedding: List[float], limit: int, filter: Optional[Dict]) -> List[Dict]:
        import numpy as np

        if not self._docs:
            return []

        vec = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self._use_faiss and self._index is not None:
            k = min(limit * 2, len(self._docs))
            scores, indices = self._index.search(vec, k)
            scores = scores[0]
            indices = indices[0]
        else:
            vectors = np.array(self._vectors)
            scores = np.dot(vectors, vec[0])
            indices = np.argsort(scores)[::-1][:limit * 2]
            scores = scores[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._docs):
                continue

            doc = self._docs[idx]

            # Apply filter
            if filter:
                match = all(
                    doc["metadata"].get(k) == v
                    for k, v in filter.items()
                )
                if not match:
                    continue

            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "score": float(score),
                "metadata": doc["metadata"]
            })

            if len(results) >= limit:
                break

        return results

    async def delete(self, doc_id: str) -> int:
        original = len(self._docs)
        self._docs = [d for d in self._docs if d["id"] != doc_id]
        deleted = original - len(self._docs)
        if deleted > 0:
            # Note: FAISS index may have stale entries but searches check docs
            await self._persist()
        return deleted

    async def delete_by_filter(self, filter: Dict) -> int:
        original = len(self._docs)
        self._docs = [
            d for d in self._docs
            if not all(d["metadata"].get(k) == v for k, v in filter.items())
        ]
        deleted = original - len(self._docs)
        if deleted > 0:
            await self._persist()
        return deleted

    async def clear(self) -> None:
        self._docs = []
        self._index = None
        if not self._use_faiss:
            self._vectors = []

        # Remove persisted files
        if os.path.exists(self._index_path):
            os.remove(self._index_path)
        if os.path.exists(self._docs_path):
            os.remove(self._docs_path)

    async def stats(self) -> Dict:
        return {"count": len(self._docs)}


class _PgVectorStore(_VectorStore):
    """PostgreSQL + pgvector store."""

    # Embedding dimensions by model
    EMBEDDING_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, name: str, dsn: str, embedding_model: str = "text-embedding-3-small"):
        self.name = name
        self.dsn = dsn
        # Sanitize table name to prevent SQL injection
        self.table = f"memory_{name.replace(chr(34), '').replace(chr(39), '').replace(';', '')}"
        self._pool = None
        self._embedding_dim = self.EMBEDDING_DIMS.get(embedding_model, 1536)

    async def initialize(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Use parameterized dimension
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector({self._embedding_dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def add(self, text: str, embedding: List[float], metadata: Dict) -> str:
        import json

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {self.table} (id, text, embedding, metadata) VALUES ($1, $2, $3, $4)",
                doc_id, text, str(embedding), json.dumps(metadata)
            )

        return doc_id

    async def add_many(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> List[str]:
        doc_ids = []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            doc_id = await self.add(text, emb, meta)
            doc_ids.append(doc_id)
        return doc_ids

    async def search(self, embedding: List[float], limit: int, filter: Optional[Dict]) -> List[Dict]:
        import json

        # Build parameterized query to prevent SQL injection
        params = [str(embedding), limit]
        where_clauses = []
        param_idx = 3  # $1 is embedding, $2 is limit

        if filter:
            for k, v in filter.items():
                # Use parameterized values instead of string interpolation
                where_clauses.append(f"metadata->>${ param_idx} = ${param_idx + 1}")
                params.append(str(k))
                params.append(str(v))
                param_idx += 2

        where = ""
        if where_clauses:
            where = "WHERE " + " AND ".join(where_clauses)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, text, metadata, 1 - (embedding <=> $1::vector) as score
                FROM {self.table}
                {where}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, *params)

        return [{
            "id": r["id"],
            "text": r["text"],
            "score": float(r["score"]),
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {}
        } for r in rows]

    async def delete(self, doc_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(f"DELETE FROM {self.table} WHERE id = $1", doc_id)
            return int(result.split()[-1])

    async def delete_by_filter(self, filter: Dict) -> int:
        # Build parameterized query to prevent SQL injection
        params = []
        where_clauses = []
        param_idx = 1

        for k, v in filter.items():
            where_clauses.append(f"metadata->>${param_idx} = ${param_idx + 1}")
            params.append(str(k))
            params.append(str(v))
            param_idx += 2

        if not where_clauses:
            return 0

        where = " AND ".join(where_clauses)

        async with self._pool.acquire() as conn:
            result = await conn.execute(f"DELETE FROM {self.table} WHERE {where}", *params)
            return int(result.split()[-1])

    async def clear(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {self.table}")

    async def stats(self) -> Dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {self.table}")
            return {"count": row["count"]}
