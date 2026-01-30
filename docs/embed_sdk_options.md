# Embed SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Embed SDK in LangPy.

## Table of Contents

1. [Embed Overview](#embed-overview)
2. [Embed Creation](#embed-creation)
3. [Embed Configuration](#embed-configuration)
4. [Embedding Models](#embedding-models)
5. [Embed Operations](#embed-operations)
6. [Provider Support](#provider-support)
7. [Advanced Features](#advanced-features)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

## Embed Overview

The Embed SDK provides a unified interface for generating text embeddings using multiple providers. It supports OpenAI, HuggingFace, and other embedding services with a consistent API.

### Key Features

- **Multi-provider Support**: OpenAI, HuggingFace, and extensible architecture
- **Async-first Design**: High-performance embedding generation
- **Batch Processing**: Efficient processing of multiple texts
- **Automatic Dimension Detection**: Validates embedding dimensions
- **Progress Tracking**: Monitor embedding generation progress
- **Session Management**: Track embedding operations and results

## Embed Creation

### Using the SDK Factory (Recommended)

```python
from sdk import embed

# Create embed instance with defaults
embed_instance = embed()

# The embed() function returns an EmbedInterface instance
# with default configuration
```

### Direct Interface Creation

```python
from sdk.embed_interface import EmbedInterface

# Create embed interface directly
embed_interface = EmbedInterface(
    default_embedder="openai:text-embedding-3-small"
)

# With custom backends
embed_interface = EmbedInterface(
    async_backend=my_async_backend,
    sync_backend=my_sync_backend,
    default_embedder="openai:text-embedding-3-large"
)
```

## Embed Configuration

### EmbedInterface Parameters

```python
class EmbedInterface:
    def __init__(
        self,
        *,
        async_backend: Optional[Callable[[dict], Any]] = None,
        sync_backend: Optional[Callable[[dict], Any]] = None,
        default_embedder: str = "openai:text-embedding-3-small",
    ) -> None:
        """
        Initialize Embed Interface.
        
        Args:
            async_backend: Optional async backend callable
            sync_backend: Optional sync backend callable
            default_embedder: Default embedder to use
        """
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_backend` | `Optional[Callable]` | `None` | Custom async backend |
| `sync_backend` | `Optional[Callable]` | `None` | Custom sync backend |
| `default_embedder` | `str` | `"openai:text-embedding-3-small"` | Default embedder model |

## Embedding Models

### OpenAI Models

```python
# OpenAI embedding models
openai_models = [
    "openai:text-embedding-3-small",    # 1536 dimensions
    "openai:text-embedding-3-large",    # 3072 dimensions
    "openai:text-embedding-ada-002",    # 1536 dimensions
]

# Create embedder for specific model
embed_instance = embed()
embeddings = await embed_instance.embed_texts(
    ["Hello world"],
    embedder_name="openai:text-embedding-3-large"
)
```

### HuggingFace Models

```python
# HuggingFace embedding models
hf_models = [
    "hf:sentence-transformers/all-MiniLM-L6-v2",    # 384 dimensions
    "hf:sentence-transformers/all-mpnet-base-v2",   # 768 dimensions
    "hf:sentence-transformers/all-distilroberta-v1", # 768 dimensions
]

# Note: HuggingFace integration is currently placeholder
# Full implementation coming soon
```

### Model Information

```python
# Get model information
embed_instance = embed()
info = embed_instance.get_embedder_info()
print(info)
# {
#     "embedder_type": "openai:text-embedding-3-small",
#     "embedding_dimension": 1536,
#     "supported_vendors": ["openai", "hf"],
#     "default_embedder": "openai:text-embedding-3-small",
#     "features": [...]
# }

# Get specific embedder details
details = embed_instance.get_embedder_details("openai:text-embedding-3-large")
print(details)
# {
#     "name": "openai:text-embedding-3-large",
#     "vendor": "openai",
#     "dimension": 3072,
#     "model": "openai:text-embedding-3-large"
# }
```

## Embed Operations

### Async Operations

```python
from sdk import embed

async def async_embed_example():
    embed_instance = embed()
    
    # Embed multiple texts
    texts = ["Hello world", "This is a test", "Another example"]
    embeddings = await embed_instance.embed_texts(texts)
    
    # Embed single text
    single_embedding = await embed_instance.embed_single_text("Hello world")
    
    return embeddings, single_embedding
```

### Sync Operations

```python
from sdk import embed

def sync_embed_example():
    embed_instance = embed()
    
    # Embed multiple texts synchronously
    texts = ["Hello world", "This is a test", "Another example"]
    embeddings = embed_instance.embed_texts_sync(texts)
    
    # Embed single text synchronously
    single_embedding = embed_instance.embed_single_text_sync("Hello world")
    
    return embeddings, single_embedding
```

### Embed Methods

#### `embed_texts(texts, embedder_name=None, progress_callback=None)`

Generate embeddings for multiple texts.

**Parameters:**
- `texts` (List[str]): List of text strings to embed
- `embedder_name` (Optional[str]): Embedder to use (defaults to instance default)
- `progress_callback` (Optional[Callable]): Callback for progress updates

**Returns:**
- `List[List[float]]`: List of embedding vectors aligned with input texts

#### `embed_single_text(text, embedder_name=None)`

Generate embedding for a single text.

**Parameters:**
- `text` (str): Text string to embed
- `embedder_name` (Optional[str]): Embedder to use

**Returns:**
- `List[float]`: Single embedding vector

#### `embed_texts_sync(texts, embedder_name=None)`

Generate embeddings synchronously.

**Parameters:**
- `texts` (List[str]): List of text strings to embed
- `embedder_name` (Optional[str]): Embedder to use

**Returns:**
- `List[List[float]]`: List of embedding vectors

#### `embed_single_text_sync(text, embedder_name=None)`

Generate single embedding synchronously.

**Parameters:**
- `text` (str): Text string to embed
- `embedder_name` (Optional[str]): Embedder to use

**Returns:**
- `List[float]`: Single embedding vector

## Provider Support

### OpenAI Provider

```python
# OpenAI provider configuration
import os
from embed import get_embedder

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-..."

# Create OpenAI embedder
embedder = get_embedder("openai:text-embedding-3-small")

# Embed texts
embeddings = await embedder.embed(["Hello world", "Another text"])
```

### HuggingFace Provider

```python
# HuggingFace provider (placeholder implementation)
from embed import get_embedder

# Create HuggingFace embedder
embedder = get_embedder("hf:sentence-transformers/all-MiniLM-L6-v2")

# Note: Currently returns placeholder embeddings
# Full implementation coming soon
embeddings = await embedder.embed(["Sample text"])
```

### Provider Registry

```python
from embed import REGISTRY

# List all supported providers
print(REGISTRY.keys())  # ['openai', 'hf']

# Get provider class
openai_class = REGISTRY['openai']
hf_class = REGISTRY['hf']

# Create embedder instances
openai_embedder = openai_class("openai:text-embedding-3-small")
hf_embedder = hf_class("hf:sentence-transformers/all-MiniLM-L6-v2")
```

## Advanced Features

### Progress Tracking

```python
from sdk import embed
from sdk.embed_interface import EmbedProgress

async def embed_with_progress():
    embed_instance = embed()
    
    def progress_callback(progress: EmbedProgress):
        print(f"Step: {progress.current_step}")
        print(f"Progress: {progress.progress_percent}%")
        print(f"Elapsed: {progress.get_elapsed_time():.2f}s")
    
    # Embed with progress tracking
    embeddings = await embed_instance.embed_texts(
        ["Text 1", "Text 2", "Text 3"],
        progress_callback=progress_callback
    )
    
    return embeddings
```

### Session Management

```python
from sdk import embed

async def session_example():
    embed_instance = embed()
    
    # Generate embeddings (automatically stored in session)
    await embed_instance.embed_texts(["Hello", "World"])
    await embed_instance.embed_texts(["Another", "Example"])
    
    # Get session results
    results = embed_instance.get_session_results()
    print(f"Total operations: {len(results)}")
    
    # Get latest result
    latest = embed_instance.get_latest_result()
    print(f"Latest operation: {latest}")
    
    # Get statistics
    stats = embed_instance.get_embedding_statistics()
    print(f"Statistics: {stats}")
    
    # Clear session
    embed_instance.clear_session()
```

### Custom Embedders

```python
from embed.base import BaseEmbedder

class CustomEmbedder(BaseEmbedder):
    def __init__(self, model: str):
        self.name = model
        self.dim = 512  # Custom dimension
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Custom embedding logic
        return [[0.1] * self.dim for _ in texts]

# Register custom embedder
from embed import REGISTRY
REGISTRY['custom'] = CustomEmbedder

# Use custom embedder
from embed import get_embedder
embedder = get_embedder("custom:my-model")
```

### Batch Processing

```python
from sdk import embed

async def batch_processing_example():
    embed_instance = embed()
    
    # Large batch of texts
    large_batch = [f"Text {i}" for i in range(1000)]
    
    # Process in chunks
    chunk_size = 100
    all_embeddings = []
    
    for i in range(0, len(large_batch), chunk_size):
        chunk = large_batch[i:i + chunk_size]
        embeddings = await embed_instance.embed_texts(chunk)
        all_embeddings.extend(embeddings)
    
    return all_embeddings
```

## Error Handling

### Basic Error Handling

```python
from sdk import embed
import asyncio

async def error_handling_example():
    embed_instance = embed()
    
    try:
        # This will fail if API key is not set
        embeddings = await embed_instance.embed_texts(["Hello world"])
        return embeddings
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Embedding error: {e}")
    
    return None
```

### Retry Logic

```python
from sdk import embed
import asyncio

async def embed_with_retry(texts, max_retries=3):
    embed_instance = embed()
    
    for attempt in range(max_retries):
        try:
            embeddings = await embed_instance.embed_texts(texts)
            return embeddings
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

### Validation

```python
from sdk import embed

async def validate_inputs():
    embed_instance = embed()
    
    # Validate texts
    texts = ["Hello", "World", ""]
    
    # Filter empty texts
    valid_texts = [text for text in texts if text.strip()]
    
    if not valid_texts:
        raise ValueError("No valid texts to embed")
    
    embeddings = await embed_instance.embed_texts(valid_texts)
    return embeddings
```

## Examples

### Basic Embedding Example

```python
from sdk import embed
import asyncio

async def basic_embedding_example():
    """Basic example of generating embeddings."""
    embed_instance = embed()
    
    # Single text embedding
    text = "Hello, world!"
    embedding = await embed_instance.embed_single_text(text)
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Multiple texts
    texts = [
        "This is the first document.",
        "Here's another document.",
        "And a third one for good measure."
    ]
    
    embeddings = await embed_instance.embed_texts(texts)
    print(f"\nProcessed {len(embeddings)} texts")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    
    return embeddings

# Run example
asyncio.run(basic_embedding_example())
```

### Document Similarity Example

```python
from sdk import embed
import numpy as np
import asyncio

async def document_similarity_example():
    """Example of using embeddings for document similarity."""
    embed_instance = embed()
    
    # Documents to compare
    documents = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The weather is sunny today.",
        "It's a beautiful day outside."
    ]
    
    # Generate embeddings
    embeddings = await embed_instance.embed_texts(documents)
    
    # Convert to numpy arrays for easier computation
    embeddings_array = np.array(embeddings)
    
    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("Document Similarity Matrix:")
    print("=" * 50)
    
    for i, doc1 in enumerate(documents):
        for j, doc2 in enumerate(documents):
            if i <= j:  # Only show upper triangle
                similarity = cosine_similarity(embeddings_array[i], embeddings_array[j])
                print(f"Doc {i+1} vs Doc {j+1}: {similarity:.4f}")
    
    print("\nDocuments:")
    for i, doc in enumerate(documents):
        print(f"Doc {i+1}: {doc}")

asyncio.run(document_similarity_example())
```

### Semantic Search Example

```python
from sdk import embed
import numpy as np
import asyncio

async def semantic_search_example():
    """Example of semantic search using embeddings."""
    embed_instance = embed()
    
    # Knowledge base
    knowledge_base = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text analysis.",
        "Deep learning uses neural networks with multiple layers.",
        "Data science combines statistics and computer science.",
        "Web development involves creating websites and applications.",
        "Database management systems store and organize data.",
        "Cloud computing provides on-demand computing resources."
    ]
    
    # Generate embeddings for knowledge base
    print("Generating embeddings for knowledge base...")
    kb_embeddings = await embed_instance.embed_texts(knowledge_base)
    
    # Search queries
    queries = [
        "What is AI?",
        "How to build websites?",
        "Programming languages",
        "Data storage solutions"
    ]
    
    # Generate query embeddings
    print("Processing search queries...")
    query_embeddings = await embed_instance.embed_texts(queries)
    
    # Perform semantic search
    def find_most_similar(query_embedding, kb_embeddings, top_k=3):
        similarities = []
        for kb_embedding in kb_embeddings:
            similarity = np.dot(query_embedding, kb_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(kb_embedding)
            )
            similarities.append(similarity)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]
    
    # Display results
    print("\nSemantic Search Results:")
    print("=" * 60)
    
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        top_results = find_most_similar(query_embeddings[i], kb_embeddings)
        
        for rank, (idx, score) in enumerate(top_results, 1):
            print(f"{rank}. {knowledge_base[idx]} (Score: {score:.4f})")

asyncio.run(semantic_search_example())
```

### Integration with Parser and Chunker

```python
from sdk import embed, parser, chunker
import asyncio

async def integration_example():
    """Example integrating embed with parser and chunker."""
    # Initialize primitives
    embed_instance = embed()
    parser_instance = parser()
    chunker_instance = chunker()
    
    # Sample text file (replace with actual file path)
    text_content = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to natural intelligence displayed by humans. Leading AI 
    textbooks define the field as the study of "intelligent agents": any 
    device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    Machine learning (ML) is a subset of artificial intelligence that 
    provides systems the ability to automatically learn and improve from 
    experience without being explicitly programmed. ML focuses on the 
    development of computer programs that can access data and use it to 
    learn for themselves.
    """
    
    # Step 1: Parse content (simulated)
    print("Step 1: Parsing content...")
    # In real scenario: parsed_content = await parser_instance.parse_file("document.pdf")
    
    # Step 2: Chunk the content
    print("Step 2: Chunking content...")
    chunks = await chunker_instance.chunk_text(text_content)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    embeddings = await embed_instance.embed_texts(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Step 4: Create searchable index
    print("Step 4: Creating searchable index...")
    chunk_index = list(zip(chunks, embeddings))
    
    # Step 5: Perform semantic search
    print("Step 5: Performing semantic search...")
    query = "What is machine learning?"
    query_embedding = await embed_instance.embed_single_text(query)
    
    # Find most similar chunks
    similarities = []
    for chunk, embedding in chunk_index:
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((chunk, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nQuery: {query}")
    print("Top 3 most similar chunks:")
    for i, (chunk, score) in enumerate(similarities[:3]):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   {chunk[:100]}...")
        print()

asyncio.run(integration_example())
```

### Performance Benchmarking

```python
from sdk import embed
import asyncio
import time

async def performance_benchmark():
    """Benchmark embedding performance."""
    embed_instance = embed()
    
    # Test different batch sizes
    batch_sizes = [1, 10, 50, 100, 500]
    sample_text = "This is a sample text for performance testing."
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Create test data
        texts = [f"{sample_text} {i}" for i in range(batch_size)]
        
        # Measure time
        start_time = time.time()
        embeddings = await embed_instance.embed_texts(texts)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        texts_per_second = batch_size / total_time
        
        results.append({
            'batch_size': batch_size,
            'total_time': total_time,
            'texts_per_second': texts_per_second,
            'embedding_count': len(embeddings)
        })
        
        print(f"  Time: {total_time:.2f}s")
        print(f"  Texts/sec: {texts_per_second:.2f}")
        print()
    
    # Summary
    print("Performance Summary:")
    print("=" * 50)
    for result in results:
        print(f"Batch {result['batch_size']}: "
              f"{result['texts_per_second']:.2f} texts/sec")
    
    return results

asyncio.run(performance_benchmark())
```

### Error Recovery Example

```python
from sdk import embed
import asyncio
import random

async def error_recovery_example():
    """Example of robust error handling and recovery."""
    embed_instance = embed()
    
    # Simulate problematic texts
    texts = [
        "Good text 1",
        "",  # Empty text
        "Good text 2",
        "Good text 3",
        "Very long text " * 1000,  # Potentially problematic
        "Good text 4"
    ]
    
    successful_embeddings = []
    failed_texts = []
    
    # Process texts individually with error recovery
    for i, text in enumerate(texts):
        try:
            if not text.strip():
                print(f"Skipping empty text at index {i}")
                failed_texts.append((i, text, "Empty text"))
                continue
            
            # Add some randomness to simulate network issues
            if random.random() < 0.1:  # 10% chance of simulated failure
                raise Exception("Simulated network error")
            
            embedding = await embed_instance.embed_single_text(text)
            successful_embeddings.append((i, text, embedding))
            print(f"✓ Successfully embedded text {i}")
            
        except Exception as e:
            print(f"✗ Failed to embed text {i}: {e}")
            failed_texts.append((i, text, str(e)))
    
    # Summary
    print("\nProcessing Summary:")
    print(f"Successful: {len(successful_embeddings)}")
    print(f"Failed: {len(failed_texts)}")
    
    if failed_texts:
        print("\nFailed texts:")
        for idx, text, error in failed_texts:
            print(f"  {idx}: {text[:50]}... -> {error}")
    
    return successful_embeddings, failed_texts

asyncio.run(error_recovery_example())
```

## Best Practices

1. **Use async methods**: Prefer async operations for better performance
2. **Batch processing**: Process multiple texts together when possible
3. **Error handling**: Implement robust error handling for API failures
4. **API key management**: Store API keys securely using environment variables
5. **Model selection**: Choose appropriate embedding models for your use case
6. **Validation**: Validate inputs before processing
7. **Progress tracking**: Use progress callbacks for long-running operations
8. **Session management**: Monitor and clear session data when appropriate
9. **Retry logic**: Implement retry mechanisms for transient failures
10. **Performance monitoring**: Track embedding generation performance

## Troubleshooting

### Common Issues

1. **API key not set**: Ensure `OPENAI_API_KEY` environment variable is set
2. **Empty text inputs**: Filter out empty or whitespace-only texts
3. **Rate limiting**: Implement backoff strategies for API rate limits
4. **Memory issues**: Process large batches in chunks
5. **Network timeouts**: Increase timeout values for slow networks
6. **Model not found**: Verify embedder name format (vendor:model)
7. **Dimension mismatches**: Ensure embedding dimensions are consistent
8. **Import errors**: Verify required dependencies are installed
9. **Event loop issues**: Use proper async/sync patterns
10. **Session overflow**: Clear session data periodically

### Performance Optimization

1. **Batch size optimization**: Find optimal batch sizes for your use case
2. **Concurrent processing**: Use asyncio for concurrent operations
3. **Connection pooling**: Reuse HTTP connections when possible
4. **Caching**: Cache embeddings for frequently used texts
5. **Model selection**: Choose faster models for time-sensitive applications
6. **Memory management**: Monitor memory usage with large datasets
7. **Network optimization**: Use faster network connections
8. **Preprocessing**: Clean and preprocess texts before embedding 