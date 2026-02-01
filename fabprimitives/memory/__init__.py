"""
Memory Primitive - Orchestrator for parser, chunker, embed, and store operations.

This memory primitive acts as an orchestrator that coordinates the existing primitives:
- Parser: Extracts text from various document formats
- Chunker: Splits text into manageable chunks
- Embed: Generates vector embeddings for chunks
- Store: Stores chunks and embeddings in vector database

When you call upload(), the platform automatically runs:
Parser → Chunker → Embed → Store behind the scenes.

Each stage is also exposed as a standalone primitive that can be invoked directly.
"""

from .models import (
    MemorySettings,
    DocumentMetadata,
    MemoryChunk,
    DocumentProcessingJob,
    DocumentStatus,
    MemoryQuery,
    MemorySearchResult,
    MemoryStats,
    ProcessingProgress,
    MemoryTier,
    FilterExpression,
    CompoundFilter,
    FilterType,
)
from .async_memory import AsyncMemory
from .sync_memory import SyncMemory
from .memory_manager import MemoryManager
from .filter_parser import FilterParser, parse_filter, build_filter
from .upload_service import UploadService, generate_upload_url, get_upload_status
from .embedding_retry import EmbeddingRetryHelper, retry_embedding, with_retry

__all__ = [
    # Main classes
    "AsyncMemory",
    "SyncMemory",
    "MemoryManager",
    
    # Models
    "MemorySettings",
    "DocumentMetadata",
    "MemoryChunk",
    "DocumentProcessingJob",
    "DocumentStatus",
    "MemoryQuery",
    "MemorySearchResult",
    "MemoryStats",
    "ProcessingProgress",
    "MemoryTier",
    "FilterExpression",
    "CompoundFilter",
    "FilterType",
    
    # Filter parser
    "FilterParser",
    "parse_filter",
    "build_filter",
    
    # Upload service
    "UploadService",
    "generate_upload_url",
    "get_upload_status",
    
    # Embedding retry
    "EmbeddingRetryHelper",
    "retry_embedding",
    "with_retry",
]

# Convenience function for creating memory instances
def create_memory(
    name: str = "default",
    backend: str = "faiss",
    dsn: str = None,
    embed_model: str = "openai:text-embedding-3-large",
    chunk_max_length: int = 10000,
    chunk_overlap: int = 256,
    **kwargs
) -> AsyncMemory:
    """
    Create a memory instance with the specified configuration.
    
    Args:
        name: Memory name for identification
        backend: Vector store backend ('faiss', 'pgvector') - defaults to 'faiss'
        dsn: PostgreSQL DSN (only needed for pgvector backend)
        embed_model: Embedding model to use
        chunk_max_length: Maximum length of each chunk
        chunk_overlap: Overlap between consecutive chunks
        **kwargs: Additional settings
        
    Returns:
        Configured AsyncMemory instance
        
    Examples:
        # Simple local FAISS storage
        mem = create_memory("my-notes")
        
        # PostgreSQL with pgvector
        mem = create_memory("my-notes", backend="pgvector", dsn="postgresql://u:p@localhost:5433/db")
        
        # Use environment variable for DSN
        # export LANGPY_PG_DSN="postgresql://u:p@localhost:5433/db"
        mem = create_memory("my-notes", backend="pgvector")
    """
    # Map backend to store configuration
    if backend == "pgvector":
        store_config = {
            "kind": "pgvector",
            "dsn": dsn,
            "table_name": f"memory_{name}",
            "embedding_model": embed_model
        }
        store_uri = dsn
    else:  # faiss
        store_config = {
            "kind": "faiss", 
            "index_path": f"./memory_{name}.faiss",
            "embedding_model": embed_model
        }
        store_uri = None
    
    settings = MemorySettings(
        name=name,
        store_backend=backend,
        store_uri=store_uri,  # PATCH: set store_uri for pgvector
        embed_model=embed_model,
        chunk_max_length=chunk_max_length,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    
    # Store the store config for later use
    settings._store_config = store_config
    
    return AsyncMemory(settings)


def create_sync_memory(
    name: str = "default",
    backend: str = "faiss",
    dsn: str = None,
    embed_model: str = "openai:text-embedding-3-large",
    chunk_max_length: int = 10000,
    chunk_overlap: int = 256,
    **kwargs
) -> SyncMemory:
    """
    Create a sync memory instance with the specified configuration.
    
    Args:
        name: Memory name for identification
        backend: Vector store backend ('faiss', 'pgvector') - defaults to 'faiss'
        dsn: PostgreSQL DSN (only needed for pgvector backend)
        embed_model: Embedding model to use
        chunk_max_length: Maximum length of each chunk
        chunk_overlap: Overlap between consecutive chunks
        **kwargs: Additional settings
        
    Returns:
        Configured SyncMemory instance
    """
    # Map backend to store configuration
    if backend == "pgvector":
        store_config = {
            "kind": "pgvector",
            "dsn": dsn,
            "table_name": f"memory_{name}",
            "embedding_model": embed_model
        }
        store_uri = dsn
    else:  # faiss
        store_config = {
            "kind": "faiss", 
            "index_path": f"./memory_{name}.faiss",
            "embedding_model": embed_model
        }
        store_uri = None
    
    settings = MemorySettings(
        name=name,
        store_backend=backend,
        store_uri=store_uri,  # PATCH: set store_uri for pgvector
        embed_model=embed_model,
        chunk_max_length=chunk_max_length,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    
    # Store the store config for later use
    settings._store_config = store_config
    
    return SyncMemory(settings) 