# Memory SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Memory SDK in LangPy.

## Table of Contents

1. [Memory Creation](#memory-creation)
2. [MemorySettings Configuration](#memorysettings-configuration)
3. [Memory Operations](#memory-operations)
4. [Advanced RAG Operations](#advanced-rag-operations)
5. [Filter System](#filter-system)
6. [Backend Options](#backend-options)
7. [Embedding Models](#embedding-models)
8. [Examples](#examples)

## Memory Creation

### Factory Method (Recommended)

```python
from sdk import memory

# Create a memory factory
mem_factory = memory()

# Create memory with options
mem_instance = mem_factory.create(
    name="my_memory",                                    # Memory name (required)
    backend="faiss",                                     # Vector store backend
    dsn="postgresql://user:pass@localhost:5432/db",     # PostgreSQL DSN (for pgvector)
    embedding_model="openai:text-embedding-3-large",    # Embedding model
    api_key="sk-...",                                   # API key for embedding model
    **kwargs                                            # Additional settings
)
```

### Direct Creation Methods

```python
# Using create_memory function directly
from memory import create_memory
mem = create_memory(
    name="default",
    backend="faiss", 
    dsn=None,
    embed_model="openai:text-embedding-3-large",
    chunk_max_length=10000,
    chunk_overlap=256,
    **kwargs
)

# Using MemoryInterface directly
from sdk.memory_interface import MemoryInterface
from memory import MemorySettings

settings = MemorySettings(
    name="my_memory",
    store_backend="faiss",
    embed_model="openai:text-embedding-3-large"
)

mem = MemoryInterface(
    async_backend=None,
    sync_backend=None,
    settings=settings
)
```

## MemorySettings Configuration

### Storage Backend Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `store_backend` | `str` | `"faiss"` | Vector store backend: `"faiss"`, `"pgvector"`, or `"docling"` |
| `store_uri` | `str` | `None` | URI for vector store (file path for FAISS, DSN for pgvector) |

### Parser Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `parser_enable_ocr` | `bool` | `True` | Enable OCR for images |
| `parser_ocr_languages` | `List[str]` | `["eng"]` | OCR languages to use |
| `parser_max_file_size` | `int` | `52428800` | Maximum file size for parsing (50MB) |

### Chunker Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_max_length` | `int` | `10000` | Maximum length of each chunk |
| `chunk_overlap` | `int` | `256` | Overlap between consecutive chunks |

### Embedding Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embed_model` | `str` | `"openai:text-embedding-3-large"` | Embedding model to use |

### Reranking Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_reranking` | `bool` | `True` | Enable cross-encoder reranking for better results |
| `reranker_model` | `str` | `"BAAI/bge-reranker-large"` | Cross-encoder model for reranking |
| `rerank_top_k` | `int` | `20` | Number of candidates to rerank |

### Hybrid Search Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_hybrid_search` | `bool` | `True` | Enable hybrid search (ANN + BM25) |
| `hybrid_weight` | `float` | `0.7` | Weight for vector similarity vs keyword matching (0-1) |

### Search Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_k` | `int` | `5` | Number of results to return by default |
| `similarity_threshold` | `float` | `0.7` | Similarity threshold for search |

### Metadata Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_source` | `bool` | `True` | Include source in metadata |
| `include_timestamp` | `bool` | `True` | Include timestamp in metadata |
| `include_tokens` | `bool` | `True` | Include token count in metadata |

### Memory Identification

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | `str` | `"default"` | Memory name for identification |

## Memory Operations

### Upload Operation

```python
job_id = await mem.upload(
    content="text, Path, or bytes",           # Document content (required)
    source="optional_source_id",             # Source identifier (optional)
    custom_metadata={"key": "value"},        # Custom metadata (optional)
    progress_callback=callback_function,     # Progress callback (optional)
    background=True                          # Background processing (optional)
)
```

**Parameters:**
- `content`: Document content as string, bytes, or Path object
- `source`: Optional source identifier for filtering
- `custom_metadata`: Additional metadata to attach to the document
- `progress_callback`: Function to receive progress updates
- `background`: Whether to process in background (True) or wait for completion (False)

### Query Operation

```python
results = await mem.query(
    query="search text",                     # Search query (required)
    k=5,                                    # Number of results (optional)
    source="filter_source",                 # Filter by source (optional)
    min_score=0.7                          # Minimum similarity score (optional)
)
```

**Parameters:**
- `query`: Search query text
- `k`: Number of results to return (overrides default_k)
- `source`: Filter results by source
- `min_score`: Minimum similarity score threshold

### Metadata Operations

#### Get by Metadata
```python
results = await mem.get_by_metadata(
    metadata_filter={"key": "value"},       # Metadata filter (required)
    k=10                                   # Number of results (optional)
)
```

#### Update Metadata
```python
updated_count = await mem.update_metadata(
    updates={"new_key": "new_value"},      # Metadata updates (required)
    source="source_filter",               # Source filter (optional)
    metadata_filter={"key": "value"}      # Additional filters (optional)
)
```

#### Delete by Filter
```python
deleted_count = await mem.delete_by_filter(
    source="source_filter",               # Source filter (optional)
    metadata_filter={"key": "value"}      # Metadata filter (optional)
)
```

### Utility Operations

## Advanced RAG Operations

The Memory SDK provides advanced RAG capabilities with reranking and hybrid search for superior retrieval quality.

### Advanced Query with Reranking

```python
results = await mem.query_advanced(
    query="search text",                     # Search query (required)
    k=5,                                    # Number of results (optional)
    enable_reranking=True,                  # Enable cross-encoder reranking (optional)
    enable_hybrid_search=True,              # Enable hybrid search (optional)
    source="filter_source",                 # Filter by source (optional)
    min_score=0.7                          # Minimum similarity score (optional)
)
```

**Parameters:**
- `query`: Search query text
- `k`: Number of results to return
- `enable_reranking`: Use cross-encoder reranking for better relevance (default: True)
- `enable_hybrid_search`: Combine vector similarity with BM25 keyword search (default: True)
- `source`: Filter results by source
- `min_score`: Minimum similarity score threshold

### RAG Context Extraction

```python
context = await mem.extract_context(
    query="user question",                  # User question (required)
    k=5,                                   # Number of chunks to retrieve (optional)
    enable_reranking=True,                 # Enable reranking (optional)
    enable_hybrid_search=True              # Enable hybrid search (optional)
)
```

**Returns:** Combined context string from retrieved memory chunks

### RAG Response Generation

```python
response = await mem.extract_response(
    query="user question",                  # User question (required)
    context="retrieved context",           # Context from memory (required)
    api_key="your_openai_key",             # OpenAI API key (required)
    model="gpt-4o-mini",                   # LLM model (optional)
    **llm_kwargs                          # Additional LLM parameters
)
```

**Parameters:**
- `query`: User question
- `context`: Retrieved context from memory
- `api_key`: OpenAI API key
- `model`: LLM model to use
- `**llm_kwargs`: Additional parameters (temperature, max_tokens, etc.)

### Complete RAG Workflow

```python
# Complete RAG workflow with advanced features
response = await mem.extract_response(
    query="What is the child policy?",
    context=await mem.extract_context(
        query="What is the child policy?",
        k=5,
        enable_reranking=True,
        enable_hybrid_search=True
    ),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=200
)
```

### Advanced RAG Features

#### Reranking Benefits
- **Cross-encoder reranking** improves relevance by reordering results
- **Better semantic understanding** of query-context relationships
- **Configurable rerank_top_k** for performance vs quality trade-offs

#### Hybrid Search Benefits
- **BM25 keyword matching** catches specific terms and phrases
- **Vector similarity** for semantic understanding
- **Configurable hybrid_weight** to balance keyword vs semantic search

#### Performance Considerations
- **Reranking adds latency** (~10 seconds vs ~1 second for basic search)
- **Higher accuracy** justifies the additional processing time
- **Use for complex queries** where precision matters most

#### Get Job Status
```python
status = await mem.get_job_status("job_id")
```

#### Get Memory Statistics
```python
stats = await mem.get_stats()
```

## Filter System

### Simple Metadata Filter
```python
# Simple key-value filter
metadata_filter = {"category": "important", "type": "document"}
```

### FilterExpression
```python
from memory import FilterExpression

# Create filter expression
filter_expr = FilterExpression(
    field="category",
    operator="eq",
    value="important"
)
```

**Available Operators:**
- `"eq"`: Equal to
- `"neq"`: Not equal to
- `"in"`: In list
- `"nin"`: Not in list
- `"gt"`: Greater than
- `"gte"`: Greater than or equal
- `"lt"`: Less than
- `"lte"`: Less than or equal
- `"contains"`: Contains substring

### CompoundFilter
```python
from memory import CompoundFilter, FilterExpression

# AND filter
and_filter = CompoundFilter(
    and_conditions=[
        FilterExpression(field="category", operator="eq", value="important"),
        FilterExpression(field="type", operator="eq", value="document")
    ]
)

# OR filter
or_filter = CompoundFilter(
    or_conditions=[
        FilterExpression(field="category", operator="eq", value="important"),
        FilterExpression(field="category", operator="eq", value="critical")
    ]
)
```

## Backend Options

### FAISS (Default)
```python
mem = memory().create(
    name="my_memory",
    backend="faiss"
)
```

**Features:**
- Local vector storage
- Fast similarity search
- No external dependencies
- Automatic index management

### PGVector
```python
mem = memory().create(
    name="my_memory",
    backend="pgvector",
    dsn="postgresql://user:pass@localhost:5432/db"
)
```

**Features:**
- PostgreSQL with pgvector extension
- Persistent storage
- ACID compliance
- Scalable for large datasets

### Docling
```python
mem = memory().create(
    name="my_memory",
    backend="docling"
)
```

**Features:**
- Document-based storage
- Specialized for document processing
- Advanced parsing capabilities

## Embedding Models

### OpenAI Models
```python
# Latest and most capable
embed_model="openai:text-embedding-3-large"

# Smaller, faster
embed_model="openai:text-embedding-3-small"

# Legacy model
embed_model="openai:text-embedding-ada-002"
```

### HuggingFace Models
```python
# Any HuggingFace embedding model
embed_model="sentence-transformers/all-MiniLM-L6-v2"
embed_model="sentence-transformers/all-mpnet-base-v2"
```

### Custom Models
```python
# Custom embedding provider
embed_model="custom:my-embedding-model"
```

## Examples

### Basic Usage
```python
from sdk import memory

# Create memory
mem = memory().create("my_notes")

# Upload document
job_id = await mem.upload("path/to/document.pdf")

# Query memory
results = await mem.query("search query", k=10)
```

### Advanced Configuration
```python
from sdk import memory
from memory import MemorySettings

# Create with custom settings
settings = MemorySettings(
    name="advanced_memory",
    store_backend="pgvector",
    store_uri="postgresql://user:pass@localhost:5432/db",
    embed_model="openai:text-embedding-3-large",
    chunk_max_length=8000,
    chunk_overlap=512,
    enable_reranking=True,
    reranker_model="BAAI/bge-reranker-large",
    enable_hybrid_search=True,
    hybrid_weight=0.8,
    similarity_threshold=0.75
)

mem = memory().create(
    name="advanced_memory",
    backend="pgvector",
    dsn="postgresql://user:pass@localhost:5432/db",
    chunk_max_length=8000,
    chunk_overlap=512,
    enable_reranking=True,
    enable_hybrid_search=True,
    hybrid_weight=0.8
)
```

### Complete Workflow
```python
from sdk import memory
import asyncio

async def main():
    # Create memory
    mem = memory().create(
        name="document_storage",
        backend="faiss",
        embedding_model="openai:text-embedding-3-large"
    )
    
    # Upload documents
    job_id1 = await mem.upload("document1.pdf", source="reports")
    job_id2 = await mem.upload("document2.txt", source="notes")
    
    # Wait for processing
    while True:
        status = await mem.get_job_status(job_id1)
        if status and status["status"] == "completed":
            break
        await asyncio.sleep(1)
    
    # Query memory
    results = await mem.query("important findings", k=5)
    
    # Filter by source
    report_results = await mem.query("analysis", source="reports")
    
    # Update metadata
    await mem.update_metadata(
        updates={"reviewed": True},
        source="reports"
    )
    
    # Get statistics
    stats = await mem.get_stats()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")

asyncio.run(main())
```

### Sync Operations
```python
from sdk import memory

# All operations have sync versions
mem = memory().create("sync_memory")

# Sync upload
job_id = mem.upload_sync("document.pdf")

# Sync query
results = mem.query_sync("search query")

# Sync metadata operations
metadata_results = mem.get_by_metadata_sync({"type": "document"})
updated_count = mem.update_metadata_sync({"status": "processed"})
deleted_count = mem.delete_by_filter_sync(source="old_data")

# Sync utilities
status = mem.get_job_status_sync(job_id)
stats = mem.get_stats_sync()
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGPY_PG_DSN` | PostgreSQL DSN for pgvector backend | None |
| `OPENAI_API_KEY` | OpenAI API key for embedding models | None |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Anthropic models) | None |

## Best Practices

1. **Choose the right backend**: Use FAISS for local development and small datasets, PGVector for production and large datasets
2. **Optimize chunk size**: Adjust `chunk_max_length` and `chunk_overlap` based on your document types
3. **Enable reranking**: Keep `enable_reranking=True` for better search quality
4. **Use hybrid search**: Keep `enable_hybrid_search=True` for comprehensive results
5. **Monitor performance**: Use `get_stats()` to monitor memory usage and performance
6. **Filter effectively**: Use metadata filters to narrow down search results
7. **Handle errors**: Always check job status when using background processing

## Troubleshooting

### Common Issues

1. **PostgreSQL connection failed**: Check DSN format and database availability
2. **Embedding API errors**: Verify API keys and model availability
3. **Chunk size too large**: Reduce `chunk_max_length` if hitting token limits
4. **Search quality poor**: Enable reranking and adjust similarity threshold
5. **Memory usage high**: Consider using smaller embedding models or reducing chunk overlap

---

## New in v2.0: Composable Pipeline API

LangPy 2.0 introduces a composable pipeline architecture. Memory now implements the `IPrimitive` interface and can be composed with other primitives using the `|` (sequential) and `&` (parallel) operators.

### The `process()` Method

```python
from langpy.core import Context
from langpy_sdk import Memory

memory = Memory(name="docs", k=5)

# Create context with query
ctx = Context(query="What is Python?")

# Process - searches memory and adds Documents to context
result = await memory.process(ctx)

if result.is_success():
    response_ctx = result.unwrap()
    print(f"Found {len(response_ctx.documents)} documents")
    for doc in response_ctx.documents:
        print(f"  - {doc.content[:50]}... (score: {doc.score:.2f})")
```

### Memory Constructor Options for Composable API

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | `str` | **Required** | Memory store name |
| `k` | `int` | `5` | Number of documents to retrieve |
| `min_score` | `float` | `0.0` | Minimum similarity score threshold |
| `primitive_name` | `str` | `"Memory"` | Name for tracing/logging |
| `filter` | `dict` | `None` | Metadata filter for search |

### Building RAG Pipelines with `|` Operator

The most common use case - chain Memory with Pipe for RAG:

```python
from langpy.core import Context
from langpy_sdk import Memory, Pipe

# Create primitives
memory = Memory(name="knowledge_base", k=5)
answerer = Pipe(
    model="gpt-4o-mini",
    system_prompt="Answer the question using only the provided context."
)

# Compose RAG pipeline
rag = memory | answerer

# Execute
result = await rag.process(Context(query="What is machine learning?"))

if result.is_success():
    ctx = result.unwrap()
    print(f"Answer: {ctx.response}")
    print(f"Based on {len(ctx.documents)} sources")
    print(f"Cost: ${ctx.cost.total_cost:.4f}")
```

### Multi-Memory Retrieval with `&` Operator

Search multiple memory stores in parallel:

```python
from langpy.core import Context, parallel
from langpy_sdk import Memory, Pipe

# Create multiple memory stores
docs_memory = Memory(name="documentation", k=3)
faq_memory = Memory(name="faq", k=3)
examples_memory = Memory(name="examples", k=2)

# Search all in parallel
multi_source = docs_memory & faq_memory & examples_memory

# Then answer
answerer = Pipe(model="gpt-4o-mini", system_prompt="Answer using all context.")

# Complete pipeline
pipeline = multi_source | answerer

result = await pipeline.process(Context(query="How do I use the API?"))
```

### Context Flow Through Pipeline

When Memory processes a Context:

1. **Input**: Context with `query` field
2. **Search**: Performs vector search using `query`
3. **Output**: Context with `documents` field populated

```python
# Before Memory.process()
ctx = Context(query="What is RAG?")
# ctx.documents = []

# After Memory.process()
result = await memory.process(ctx)
ctx = result.unwrap()
# ctx.documents = [Document(content="...", score=0.95), ...]
```

### Using Filters in Pipelines

```python
from langpy_sdk import Memory

# Memory with metadata filter
memory = Memory(
    name="docs",
    k=5,
    filter={"category": "python", "version": "3.x"}
)

# Only searches documents matching the filter
result = await memory.process(Context(query="How to install?"))
```

### Advanced RAG with Reranking

```python
from langpy.core import Context
from langpy_sdk import Memory, Pipe

# Create memory with reranking enabled
memory = Memory(
    name="docs",
    k=5,
    enable_reranking=True,
    rerank_top_k=20
)

pipe = Pipe(model="gpt-4o-mini", system_prompt="Answer accurately.")

rag = memory | pipe
result = await rag.process(Context(query="Complex technical question"))
```

### Cost and Token Tracking

```python
result = await (memory | pipe).process(ctx)
if result.is_success():
    ctx = result.unwrap()

    # Memory doesn't use LLM tokens, but pipeline tracks embedding costs
    print(f"Documents retrieved: {len(ctx.documents)}")

    # Pipe token usage
    print(f"LLM tokens: {ctx.token_usage.total_tokens}")
    print(f"Total cost: ${ctx.cost.total_cost:.4f}")
```

### Backward Compatibility

The original Memory API continues to work:

```python
# Original API (still supported)
results = await memory.query("What is Python?", k=5)
for r in results:
    print(f"{r.text} (score: {r.score})")

# New composable API
result = await memory.process(Context(query="What is Python?"))
for doc in result.unwrap().documents:
    print(f"{doc.content} (score: {doc.score})")
```

### Testing Memory Pipelines

```python
from langpy.testing import mock_memory_results, mock_llm_response, assert_success
from langpy.core import Context

# Create mocks
mock_mem = mock_memory_results([
    "Python is a programming language.",
    "Python was created by Guido van Rossum."
])
mock_llm = mock_llm_response("Python is a popular programming language.")

# Test pipeline
pipeline = mock_mem | mock_llm
result = await pipeline.process(Context(query="What is Python?"))

ctx = assert_success(result)
assert len(ctx.documents) == 2
assert "Python" in ctx.response
```

See [CORE_API.md](CORE_API.md) and [TESTING.md](TESTING.md) for complete documentation.