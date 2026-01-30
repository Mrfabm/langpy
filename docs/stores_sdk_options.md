# Stores SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Stores SDK in LangPy.

## Table of Contents

1. [Store Overview](#store-overview)
2. [Store Creation](#store-creation)
3. [FAISS Store Options](#faiss-store-options)
4. [PGVector Store Options](#pgvector-store-options)
5. [Store Operations](#store-operations)
6. [Advanced Features](#advanced-features)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Store Overview

The Stores SDK provides vector storage backends for the memory primitive, enabling persistent storage and retrieval of vector embeddings with metadata. Two main backends are supported:

- **FAISS Store**: Local file-based vector storage using Facebook's FAISS library
- **PGVector Store**: PostgreSQL-based vector storage using the pgvector extension

## Store Creation

### Factory Method (Recommended)

```python
from stores import get_store

# Create FAISS store
faiss_store = get_store(
    kind="faiss",
    index_path="./my_index.faiss",
    dim=1536,
    embedding_model="openai:text-embedding-3-large"
)

# Create PGVector store
pgvector_store = get_store(
    kind="pgvector",
    dsn="postgresql://user:pass@localhost:5432/db",
    table_name="vectors",
    dim=1536,
    embedding_model="openai:text-embedding-3-large"
)
```

### Direct Creation

```python
# Create FAISS store directly
from stores import FaissStore

faiss_store = FaissStore(
    index_path="./my_index.faiss",
    dim=1536,
    embedding_model="openai:text-embedding-3-large"
)

# Create PGVector store directly
from stores import AsyncPGVectorStore

pgvector_store = AsyncPGVectorStore(
    dsn="postgresql://user:pass@localhost:5432/db",
    table_name="vectors",
    dim=1536,
    embedding_model="openai:text-embedding-3-large"
)
```

### Store Registry

```python
from stores import REGISTRY

# List available store types
available_stores = list(REGISTRY.keys())  # ['faiss', 'pgvector']

# Get store class
store_class = REGISTRY['faiss']  # FaissStore
```

## FAISS Store Options

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_path` | `str` | **Required** | Path to FAISS index file |
| `dim` | `int` | `None` | Embedding dimension (auto-detected if None) |
| `embedding_model` | `str` | `"openai:text-embedding-ada-002"` | Embedding model to use |

### Features

- **Local Storage**: File-based vector storage
- **Fast Similarity Search**: Optimized vector similarity search
- **Metadata Support**: JSON-based metadata storage
- **Auto-Save**: Automatic index persistence
- **Dimension Detection**: Automatic dimension detection from embeddings

### FAISS Store Methods

#### Initialization
```python
from stores import FaissStore

store = FaissStore(
    index_path="./vectors.faiss",
    dim=1536,  # Optional, auto-detected
    embedding_model="openai:text-embedding-3-large"
)
```

#### Adding Data
```python
# Add texts with metadata
await store.add(
    texts=["Document 1", "Document 2"],
    metas=[
        {"source": "doc1.pdf", "category": "technical"},
        {"source": "doc2.pdf", "category": "business"}
    ],
    embeddings=None  # Optional, will be generated if not provided
)
```

#### Querying
```python
# Query with similarity search
results = await store.query(
    query="technical documentation",
    k=5,
    filt={"category": "technical"}  # Optional filter
)
```

#### Management
```python
# Get token usage
tokens = await store.token_usage()

# Clear all data
await store.clear()

# Delete by filter
deleted_count = await store.delete_by_filter({"category": "old"})

# Update metadata
updated_count = await store.update_metadata(
    filt={"category": "technical"},
    updates={"reviewed": True}
)

# Get metadata statistics
stats = await store.get_metadata_stats()

# Save to disk
await store.save()

# Load from disk
await store.load()

# Close store
await store.close()
```

## PGVector Store Options

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dsn` | `str` | **Required** | PostgreSQL connection string |
| `table_name` | `str` | `"memory_vectors"` | Table name for storing vectors |
| `dim` | `int` | `None` | Embedding dimension (auto-detected if None) |
| `embedding_model` | `str` | `"openai:text-embedding-ada-002"` | Embedding model to use |

### Features

- **Persistent Storage**: PostgreSQL-based persistent storage
- **ACID Compliance**: Full ACID transaction support
- **Scalable**: Handles large datasets efficiently
- **Advanced Indexing**: IVFFlat indexing for fast similarity search
- **Metadata Filtering**: Advanced JSONB-based metadata filtering
- **Connection Pooling**: Shared connection pool management

### PGVector Store Methods

#### Initialization
```python
from stores import AsyncPGVectorStore

store = AsyncPGVectorStore(
    dsn="postgresql://user:pass@localhost:5432/db",
    table_name="my_vectors",
    dim=1536,  # Optional, auto-detected
    embedding_model="openai:text-embedding-3-large"
)
```

#### Adding Data
```python
# Add texts with metadata
await store.add(
    texts=["Document 1", "Document 2"],
    metas=[
        {"source": "doc1.pdf", "category": "technical", "tags": ["ai", "ml"]},
        {"source": "doc2.pdf", "category": "business", "tags": ["finance"]}
    ],
    embeds=None  # Optional, will be generated if not provided
)
```

#### Querying
```python
# Query with vector similarity search
results = await store.query(
    query="machine learning concepts",
    k=10,
    filt={"category": "technical"}  # JSONB filter
)

# Advanced filtering
results = await store.query(
    query="AI research",
    k=5,
    filt={
        "category": "technical",
        "tags": ["ai"]  # Array contains filter
    }
)
```

#### Management
```python
# Get token usage
tokens = await store.token_usage()

# Clear all data
await store.clear()

# Delete by filter
deleted_count = await store.delete_by_filter({"category": "outdated"})

# Update metadata
updated_count = await store.update_metadata(
    filt={"category": "technical"},
    updates={"reviewed": True, "updated_at": "2024-01-01"}
)

# Get metadata statistics
stats = await store.get_metadata_stats()

# Get all texts (for hybrid search)
all_texts = await store.get_all_texts()

# Get filtered texts
filtered_texts = await store.get_texts_by_filter({"category": "technical"})
```

### Connection Pool Management

```python
from stores import get_pool, set_pool, close_all_pools

# Get shared connection pool
pool = get_pool("postgresql://user:pass@localhost:5432/db")

# Set shared connection pool
set_pool("postgresql://user:pass@localhost:5432/db", pool)

# Close all pools
close_all_pools()
```

## Store Operations

### Common Interface

All stores implement the `BaseVectorStore` interface:

```python
from stores.base import BaseVectorStore

# Abstract methods that all stores must implement
class BaseVectorStore:
    async def add(self, texts: List[str], metas: List[Dict[str, Any]]) -> None:
        """Add texts with metadata to store."""
        
    async def query(self, query: str, k: int, filt: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query store for similar content."""
        
    async def token_usage(self) -> int:
        """Get total token count."""
    
    async def clear(self) -> None:
        """Clear all stored data."""
    
    async def delete_by_filter(self, filt: Dict[str, Any]) -> int:
        """Delete documents matching filter criteria."""
    
    async def update_metadata(self, filt: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """Update metadata for documents matching filter criteria."""
    
    async def get_metadata_stats(self) -> Dict[str, Any]:
        """Get statistics about stored metadata."""
    
    async def get_all_texts(self) -> List[str]:
        """Get all stored texts (for hybrid search)."""
    
    async def get_texts_by_filter(self, filt: Dict[str, Any]) -> List[str]:
        """Get texts matching filter criteria (for hybrid search)."""
```

### Query Results Format

```python
# Query results structure
result = {
    "text": "Document content",
    "score": 0.85,  # Similarity score (0-1)
    "meta": {
        "source": "document.pdf",
        "category": "technical",
        "tags": ["ai", "ml"]
    },
    "distance": 0.15,  # Distance (lower is better)
    "_id": 123  # Internal ID
}
```

### Metadata Statistics Format

```python
# Metadata statistics structure
stats = {
    "total_documents": 1000,
    "sources": {
        "doc1.pdf": 50,
        "doc2.pdf": 30,
        "web_content": 920
    },
    "tiers": {
        "general": 800,
        "important": 150,
        "critical": 50
    },
    "tags": {
        "ai": 200,
        "ml": 150,
        "nlp": 100
    },
    "total_tokens": 50000
}
```

## Advanced Features

### Embedding Model Integration

```python
# Custom embedding model
store = get_store(
    kind="faiss",
    index_path="./custom_index.faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# OpenAI models
store = get_store(
    kind="pgvector",
    dsn="postgresql://...",
    embedding_model="openai:text-embedding-3-large"
)
```

### Metadata Filtering

#### Simple Filtering
```python
# Simple key-value filter
results = await store.query(
    query="search query",
    k=10,
    filt={"category": "technical"}
)

# Multiple filters
results = await store.query(
    query="search query",
    k=10,
    filt={
        "category": "technical",
        "source": "document.pdf"
    }
)
```

#### Advanced Filtering (PGVector)
```python
# Array contains filter
results = await store.query(
    query="search query",
    k=10,
    filt={"tags": ["ai"]}  # Contains "ai" in tags array
)

# JSONB operations
results = await store.query(
    query="search query",
    k=10,
    filt={
        "metadata->>'priority'": "high",
        "metadata->'tags'": ["important"]
    }
)
```

### Batch Operations

```python
# Batch add
texts = ["Doc 1", "Doc 2", "Doc 3"]
metas = [
    {"source": "doc1.pdf"},
    {"source": "doc2.pdf"},
    {"source": "doc3.pdf"}
]
await store.add(texts, metas)

# Batch delete
deleted = await store.delete_by_filter({"category": "old"})

# Batch update
updated = await store.update_metadata(
    filt={"category": "technical"},
    updates={"reviewed": True}
)
```

### Store Switching

```python
# Create stores with different backends
faiss_store = get_store(kind="faiss", index_path="./faiss_index.faiss")
pgvector_store = get_store(kind="pgvector", dsn="postgresql://...")

# Use the same interface
async def process_with_store(store):
    await store.add(["text"], [{"meta": "data"}])
    results = await store.query("search", k=5)
    return results

# Works with both stores
faiss_results = await process_with_store(faiss_store)
pgvector_results = await process_with_store(pgvector_store)
```

## Error Handling

### Common Errors

```python
# Import errors
try:
    from stores import AsyncPGVectorStore
except ImportError:
    print("asyncpg not installed")

try:
    from stores import FaissStore
except ImportError:
    print("faiss-cpu not installed")

# Store creation errors
try:
    store = get_store("unknown_backend")
except ValueError as e:
    print(f"Unknown backend: {e}")

# Connection errors
try:
    store = AsyncPGVectorStore(dsn="invalid://connection")
    await store.add(["text"], [{"meta": "data"}])
except Exception as e:
    print(f"Connection error: {e}")
```

### Error Recovery

```python
# Safe store operations
async def safe_store_operation(store, texts, metas):
    try:
        await store.add(texts, metas)
        return True
    except Exception as e:
        print(f"Store operation failed: {e}")
        return False

# Connection pool recovery
async def ensure_connection(store):
    try:
        await store.query("test", k=1)
    except Exception as e:
        print(f"Connection issue: {e}")
        # Recreate connection
        store._pool = None
        await store._get_pool()
```

## Examples

### Basic FAISS Store Usage

```python
from stores import get_store
import asyncio

async def faiss_example():
    # Create FAISS store
    store = get_store(
        kind="faiss",
        index_path="./example_index.faiss",
        embedding_model="openai:text-embedding-3-large"
    )
    
    # Add documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text."
    ]
    
    metadata = [
        {"topic": "ml", "difficulty": "beginner"},
        {"topic": "dl", "difficulty": "intermediate"},
        {"topic": "nlp", "difficulty": "beginner"}
    ]
    
    await store.add(documents, metadata)
    
    # Query documents
    results = await store.query(
        query="neural networks",
        k=2,
        filt={"difficulty": "intermediate"}
    )
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['meta']}")
        print("---")
    
    # Get statistics
    stats = await store.get_metadata_stats()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Topics: {stats.get('topics', {})}")
    
    # Close store
    await store.close()

asyncio.run(faiss_example())
```

### Basic PGVector Store Usage

```python
from stores import get_store
import asyncio

async def pgvector_example():
    # Create PGVector store
    store = get_store(
        kind="pgvector",
        dsn="postgresql://postgres:postgres@localhost:5432/langpy",
        table_name="ai_knowledge",
        embedding_model="openai:text-embedding-3-large"
    )
    
    # Add documents with rich metadata
    documents = [
        "Transformer architecture revolutionized NLP.",
        "BERT is a bidirectional encoder representation.",
        "GPT models use autoregressive language modeling."
    ]
    
    metadata = [
        {
            "topic": "transformers",
            "year": 2017,
            "tags": ["attention", "nlp"],
            "importance": "high"
        },
        {
            "topic": "bert",
            "year": 2018,
            "tags": ["bidirectional", "pretraining"],
            "importance": "high"
        },
        {
            "topic": "gpt",
            "year": 2018,
            "tags": ["autoregressive", "generation"],
            "importance": "medium"
        }
    ]
    
    await store.add(documents, metadata)
    
    # Query with advanced filtering
    results = await store.query(
        query="language models",
        k=3,
        filt={
            "importance": "high",
            "tags": ["nlp"]
        }
    )
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Year: {result['meta']['year']}")
        print(f"Tags: {result['meta']['tags']}")
        print("---")
    
    # Update metadata
    updated = await store.update_metadata(
        filt={"year": 2017},
        updates={"reviewed": True}
    )
    print(f"Updated {updated} documents")
    
    # Get statistics
    stats = await store.get_metadata_stats()
    print(f"Total documents: {stats['total_documents']}")

asyncio.run(pgvector_example())
```

### Store Comparison

```python
from stores import get_store
import asyncio
import time

async def compare_stores():
    # Create both stores
    faiss_store = get_store(
        kind="faiss",
        index_path="./comparison_faiss.faiss"
    )
    
    pgvector_store = get_store(
        kind="pgvector",
        dsn="postgresql://postgres:postgres@localhost:5432/langpy",
        table_name="comparison_test"
    )
    
    # Test data
    texts = [f"Document {i}" for i in range(1000)]
    metas = [{"id": i, "category": f"cat_{i % 10}"} for i in range(1000)]
    
    # Benchmark FAISS
    start = time.time()
    await faiss_store.add(texts, metas)
    faiss_add_time = time.time() - start
    
    start = time.time()
    faiss_results = await faiss_store.query("Document 500", k=10)
    faiss_query_time = time.time() - start
    
    # Benchmark PGVector
    start = time.time()
    await pgvector_store.add(texts, metas)
    pgvector_add_time = time.time() - start
    
    start = time.time()
    pgvector_results = await pgvector_store.query("Document 500", k=10)
    pgvector_query_time = time.time() - start
    
    print(f"FAISS - Add: {faiss_add_time:.2f}s, Query: {faiss_query_time:.2f}s")
    print(f"PGVector - Add: {pgvector_add_time:.2f}s, Query: {pgvector_query_time:.2f}s")
    
    # Clean up
    await faiss_store.close()
    await pgvector_store.clear()

asyncio.run(compare_stores())
```

### Advanced Metadata Operations

```python
from stores import get_store
import asyncio

async def metadata_operations():
    store = get_store(
        kind="pgvector",
        dsn="postgresql://postgres:postgres@localhost:5432/langpy",
        table_name="metadata_demo"
    )
    
    # Add documents with complex metadata
    documents = [
        "Research paper on machine learning",
        "Technical documentation for API",
        "User guide for beginners",
        "Advanced tutorial on deep learning"
    ]
    
    metadata = [
        {
            "type": "research",
            "difficulty": "advanced",
            "topics": ["ml", "ai"],
            "created_at": "2024-01-01",
            "author": "Dr. Smith"
        },
        {
            "type": "documentation",
            "difficulty": "intermediate",
            "topics": ["api", "technical"],
            "created_at": "2024-01-02",
            "author": "Engineering Team"
        },
        {
            "type": "guide",
            "difficulty": "beginner",
            "topics": ["tutorial", "basics"],
            "created_at": "2024-01-03",
            "author": "Support Team"
        },
        {
            "type": "tutorial",
            "difficulty": "advanced",
            "topics": ["dl", "neural networks"],
            "created_at": "2024-01-04",
            "author": "Dr. Johnson"
        }
    ]
    
    await store.add(documents, metadata)
    
    # Query by difficulty
    beginner_docs = await store.query(
        query="learning",
        k=5,
        filt={"difficulty": "beginner"}
    )
    print(f"Beginner documents: {len(beginner_docs)}")
    
    # Query by topic array
    ml_docs = await store.query(
        query="machine learning",
        k=5,
        filt={"topics": ["ml"]}
    )
    print(f"ML documents: {len(ml_docs)}")
    
    # Update metadata for advanced documents
    updated = await store.update_metadata(
        filt={"difficulty": "advanced"},
        updates={"reviewed": True, "priority": "high"}
    )
    print(f"Updated {updated} advanced documents")
    
    # Delete beginner documents
    deleted = await store.delete_by_filter({"difficulty": "beginner"})
    print(f"Deleted {deleted} beginner documents")
    
    # Get comprehensive statistics
    stats = await store.get_metadata_stats()
    print(f"Statistics: {stats}")

asyncio.run(metadata_operations())
```

### Custom Embedding Models

```python
from stores import get_store
import asyncio

async def custom_embedding_example():
    # Use different embedding models
    models = [
        "openai:text-embedding-3-large",
        "openai:text-embedding-3-small",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    for model in models:
        print(f"\nTesting with {model}")
        
        store = get_store(
            kind="faiss",
            index_path=f"./embeddings_{model.replace(':', '_').replace('/', '_')}.faiss",
            embedding_model=model
        )
        
        # Add test documents
        documents = [
            "Artificial intelligence is transforming industries.",
            "Machine learning algorithms learn from data.",
            "Deep learning uses neural networks."
        ]
        
        metadata = [
            {"topic": "ai", "id": 1},
            {"topic": "ml", "id": 2},
            {"topic": "dl", "id": 3}
        ]
        
        await store.add(documents, metadata)
        
        # Query and compare results
        results = await store.query("neural networks", k=2)
        
        print(f"Top result: {results[0]['text']}")
        print(f"Score: {results[0]['score']:.4f}")
        
        await store.close()

asyncio.run(custom_embedding_example())
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGPY_PG_DSN` | PostgreSQL DSN for PGVector backend | None |
| `OPENAI_API_KEY` | OpenAI API key for OpenAI embedding models | None |

## Best Practices

1. **Choose the right backend**: Use FAISS for local development and small datasets, PGVector for production and large datasets
2. **Optimize embedding models**: Choose embedding models based on your use case and performance requirements
3. **Use metadata effectively**: Structure metadata for efficient filtering and retrieval
4. **Monitor performance**: Track query times and optimize indexes for large datasets
5. **Handle errors gracefully**: Implement proper error handling for network and storage issues
6. **Manage connections**: Use connection pooling for PGVector to optimize performance
7. **Batch operations**: Use batch operations for better performance with large datasets
8. **Regular maintenance**: Clean up old data and optimize indexes regularly
9. **Backup data**: Implement backup strategies for important vector data
10. **Monitor storage**: Track storage usage and implement cleanup policies

## Troubleshooting

### Common Issues

1. **FAISS not installed**: Install with `pip install faiss-cpu`
2. **asyncpg not installed**: Install with `pip install asyncpg`
3. **PostgreSQL connection errors**: Check DSN format and database availability
4. **pgvector extension missing**: Install pgvector extension in PostgreSQL
5. **Dimension mismatches**: Ensure consistent embedding dimensions
6. **Index corruption**: Recreate indexes if corruption is detected
7. **Memory issues**: Monitor memory usage with large datasets
8. **Connection pool exhaustion**: Adjust pool size or close unused connections
9. **Query performance**: Optimize metadata filters and indexes
10. **Storage space**: Monitor disk usage for FAISS indexes

### Performance Optimization

1. **Index optimization**: Use appropriate indexes for your query patterns
2. **Embedding model selection**: Choose faster models for real-time applications
3. **Batch processing**: Use batch operations for large datasets
4. **Connection pooling**: Reuse connections for better performance
5. **Metadata indexing**: Create indexes on frequently filtered metadata fields
6. **Query optimization**: Optimize filter conditions and result limits
7. **Memory management**: Monitor memory usage and implement cleanup
8. **Storage optimization**: Use compression and efficient storage formats 