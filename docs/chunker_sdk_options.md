# Chunker SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Chunker SDK in LangPy.

## Table of Contents

1. [Chunker Overview](#chunker-overview)
2. [Chunker Creation](#chunker-creation)
3. [Chunker Configuration](#chunker-configuration)
4. [Chunker Operations](#chunker-operations)
5. [Chunker Settings](#chunker-settings)
6. [Advanced Features](#advanced-features)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Chunker Overview

The Chunker SDK provides a unified interface for text chunking using Docling's HybridChunker with fallback to character-based chunking. It maintains Langbase parity with sliding-window overlap and size constraints.

### Key Features

- **Structure-aware Chunking**: Uses Docling's HybridChunker for intelligent text splitting
- **Character-window Fallback**: Reliable fallback when Docling is unavailable
- **Sliding-window Overlap**: Maintains context between chunks
- **Size Constraint Enforcement**: Prevents chunks from exceeding limits
- **Langbase Parity**: Matches Langbase's reference chunking behavior
- **Async-first Design**: High-performance async operations with sync support

### Chunking Process

1. **Initial Structural Chunking**: Docling's HybridChunker analyzes document structure
2. **Character-window Slicing**: Applies sliding-window with overlap for size constraints
3. **Validation**: Ensures all chunks meet size requirements
4. **Fallback Handling**: Falls back to character-based chunking when needed

## Chunker Creation

### Using the SDK Factory (Recommended)

```python
from sdk import chunker

# Create chunker instance with defaults
chunker_instance = chunker()

# The chunker() function returns a ChunkerInterface instance
# with default configuration (chunk_max_length=2000, chunk_overlap=256)
```

### Direct Interface Creation

```python
from sdk.chunker_interface import ChunkerInterface
from chunker import ChunkerSettings

# Create chunker interface directly
chunker_interface = ChunkerInterface()

# With custom settings
custom_settings = ChunkerSettings(
    chunk_max_length=1500,
    chunk_overlap=200
)

chunker_interface = ChunkerInterface(
    default_settings=custom_settings
)

# With custom backends
chunker_interface = ChunkerInterface(
    async_backend=my_async_backend,
    sync_backend=my_sync_backend,
    default_settings=custom_settings
)
```

## Chunker Configuration

### ChunkerInterface Parameters

```python
class ChunkerInterface:
    def __init__(
        self,
        *,
        async_backend: Optional[Callable[[dict], Any]] = None,
        sync_backend: Optional[Callable[[dict], Any]] = None,
        default_settings: Optional[ChunkerSettings] = None,
    ) -> None:
        """
        Initialize Chunker Interface.
        
        Args:
            async_backend: Optional async backend callable
            sync_backend: Optional sync backend callable
            default_settings: Default chunker settings
        """
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_backend` | `Optional[Callable]` | `None` | Custom async backend |
| `sync_backend` | `Optional[Callable]` | `None` | Custom sync backend |
| `default_settings` | `Optional[ChunkerSettings]` | `None` | Default chunker settings |

## Chunker Operations

### Async Operations

```python
from sdk import chunker

async def async_chunk_example():
    chunker_instance = chunker()
    
    # Chunk text with defaults
    text = "This is a long text that needs to be chunked..."
    chunks = await chunker_instance.chunk_text(text)
    
    # Chunk text with custom parameters
    chunks = await chunker_instance.chunk_text(
        text,
        chunk_max_length=1500,
        chunk_overlap=200
    )
    
    # Chunk file
    chunks = await chunker_instance.chunk_file("document.txt")
    
    return chunks
```

### Sync Operations

```python
from sdk import chunker

def sync_chunk_example():
    chunker_instance = chunker()
    
    # Chunk text synchronously
    text = "This is a long text that needs to be chunked..."
    chunks = chunker_instance.chunk_text_sync(text)
    
    # Chunk text with custom parameters
    chunks = chunker_instance.chunk_text_sync(
        text,
        chunk_max_length=1500,
        chunk_overlap=200
    )
    
    # Chunk file synchronously
    chunks = chunker_instance.chunk_file_sync("document.txt")
    
    return chunks
```

### Chunker Methods

#### `chunk_text(text, chunk_max_length=None, chunk_overlap=None, progress_callback=None)`

Chunk text using Docling's HybridChunker with fallback.

**Parameters:**
- `text` (str): Input text to chunk
- `chunk_max_length` (Optional[int]): Maximum length of each chunk (default: 2000)
- `chunk_overlap` (Optional[int]): Overlap between consecutive chunks (default: 256)
- `progress_callback` (Optional[Callable]): Callback for progress updates

**Returns:**
- `List[str]`: List of text chunks

#### `chunk_text_sync(text, chunk_max_length=None, chunk_overlap=None)`

Synchronous version of chunk_text.

**Parameters:**
- `text` (str): Input text to chunk
- `chunk_max_length` (Optional[int]): Maximum length of each chunk
- `chunk_overlap` (Optional[int]): Overlap between consecutive chunks

**Returns:**
- `List[str]`: List of text chunks

#### `chunk_file(file_path, chunk_max_length=None, chunk_overlap=None, encoding="utf-8", progress_callback=None)`

Read a text file and chunk its contents.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the text file
- `chunk_max_length` (Optional[int]): Maximum length of each chunk
- `chunk_overlap` (Optional[int]): Overlap between consecutive chunks
- `encoding` (str): File encoding (default: utf-8)
- `progress_callback` (Optional[Callable]): Callback for progress updates

**Returns:**
- `List[str]`: List of text chunks

#### `chunk_file_sync(file_path, chunk_max_length=None, chunk_overlap=None, encoding="utf-8")`

Synchronous version of chunk_file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the text file
- `chunk_max_length` (Optional[int]): Maximum length of each chunk
- `chunk_overlap` (Optional[int]): Overlap between consecutive chunks
- `encoding` (str): File encoding (default: utf-8)

**Returns:**
- `List[str]`: List of text chunks

## Chunker Settings

### ChunkerSettings Model

```python
class ChunkerSettings(BaseModel):
    chunk_max_length: int = Field(
        2000, 
        description="Maximum length of each chunk in characters (100-30000)"
    )
    chunk_overlap: int = Field(
        256, 
        description="Character overlap between consecutive chunks (≥50)"
    )
```

### Settings Configuration

```python
from chunker import ChunkerSettings

# Create settings with defaults
settings = ChunkerSettings()

# Create settings with custom values
settings = ChunkerSettings(
    chunk_max_length=1500,
    chunk_overlap=200
)

# Validation ranges
# chunk_max_length: 100-30000 characters
# chunk_overlap: 50 characters minimum, must be less than chunk_max_length
```

### Parameter Validation

```python
# Valid settings
valid_settings = ChunkerSettings(
    chunk_max_length=2000,
    chunk_overlap=256
)

# Invalid settings (will raise ValueError)
try:
    invalid_settings = ChunkerSettings(
        chunk_max_length=50,  # Too small (< 100)
        chunk_overlap=256
    )
except ValueError as e:
    print(f"Validation error: {e}")

try:
    invalid_settings = ChunkerSettings(
        chunk_max_length=1000,
        chunk_overlap=1000  # Must be less than chunk_max_length
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Settings Usage

```python
from sdk import chunker
from chunker import ChunkerSettings

# Create custom settings
settings = ChunkerSettings(
    chunk_max_length=1200,
    chunk_overlap=150
)

# Use with chunker interface
chunker_interface = chunker()
chunker_interface._default_settings = settings

# Or create settings using the interface
custom_settings = chunker_interface.create_settings(
    chunk_max_length=1800,
    chunk_overlap=300
)
```

## Advanced Features

### Progress Tracking

```python
from sdk import chunker
from sdk.chunker_interface import ChunkProgress

async def chunk_with_progress():
    chunker_instance = chunker()
    
    def progress_callback(progress: ChunkProgress):
        print(f"Step: {progress.current_step}")
        print(f"Progress: {progress.progress_percent}%")
        print(f"Elapsed: {progress.get_elapsed_time():.2f}s")
    
    # Chunk with progress tracking
    chunks = await chunker_instance.chunk_text(
        "Long text content...",
        progress_callback=progress_callback
    )
    
    return chunks
```

### Session Management

```python
from sdk import chunker

async def session_example():
    chunker_instance = chunker()
    
    # Perform chunking operations (automatically stored in session)
    await chunker_instance.chunk_text("First text")
    await chunker_instance.chunk_text("Second text")
    
    # Get session results
    results = chunker_instance.get_session_results()
    print(f"Total operations: {len(results)}")
    
    # Get latest result
    latest = chunker_instance.get_latest_result()
    print(f"Latest operation: {latest}")
    
    # Get statistics
    stats = chunker_instance.get_chunking_statistics()
    print(f"Statistics: {stats}")
    
    # Clear session
    chunker_instance.clear_session()
```

### Docling Integration

```python
from sdk import chunker

async def docling_integration_example():
    chunker_instance = chunker()
    
    # Check if Docling is available
    info = chunker_instance.get_chunker_info()
    if info["docling_available"]:
        print("Docling is available - using HybridChunker")
    else:
        print("Docling not available - using character-based fallback")
    
    # Chunk structured document
    document_text = """
    # Document Title
    
    ## Section 1
    This is the first section with important content.
    
    ## Section 2
    This is the second section with more content.
    
    ### Subsection 2.1
    Detailed information in subsection.
    """
    
    chunks = await chunker_instance.chunk_text(document_text)
    print(f"Generated {len(chunks)} chunks")
    
    return chunks
```

### Batch File Processing

```python
from sdk import chunker
from pathlib import Path

async def batch_file_processing():
    chunker_instance = chunker()
    
    # Process multiple files
    file_paths = [
        "document1.txt",
        "document2.txt",
        "document3.txt"
    ]
    
    all_chunks = []
    
    for file_path in file_paths:
        if Path(file_path).exists():
            chunks = await chunker_instance.chunk_file(file_path)
            all_chunks.extend(chunks)
            print(f"Processed {file_path}: {len(chunks)} chunks")
    
    return all_chunks
```

### Custom Chunk Sizes

```python
from sdk import chunker

async def custom_chunk_sizes_example():
    chunker_instance = chunker()
    
    text = "A very long document that needs different chunking strategies..."
    
    # Small chunks for detailed analysis
    small_chunks = await chunker_instance.chunk_text(
        text,
        chunk_max_length=500,
        chunk_overlap=50
    )
    
    # Large chunks for context preservation
    large_chunks = await chunker_instance.chunk_text(
        text,
        chunk_max_length=3000,
        chunk_overlap=400
    )
    
    print(f"Small chunks: {len(small_chunks)}")
    print(f"Large chunks: {len(large_chunks)}")
    
    return small_chunks, large_chunks
```

## Error Handling

### Basic Error Handling

```python
from sdk import chunker
from chunker import ChunkTooLargeError

async def error_handling_example():
    chunker_instance = chunker()
    
    try:
        # This might fail with invalid parameters
        chunks = await chunker_instance.chunk_text(
            "Some text",
            chunk_max_length=50,  # Too small
            chunk_overlap=256
        )
    except ValueError as e:
        print(f"Parameter validation error: {e}")
    except ChunkTooLargeError as e:
        print(f"Chunk too large error: {e}")
    except ImportError as e:
        print(f"Docling not available: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None
```

### File Handling Errors

```python
from sdk import chunker
from pathlib import Path

async def file_error_handling():
    chunker_instance = chunker()
    
    try:
        # This will fail if file doesn't exist
        chunks = await chunker_instance.chunk_file("nonexistent.txt")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None
```

### Validation and Recovery

```python
from sdk import chunker
from chunker import ChunkerSettings

async def validation_and_recovery():
    chunker_instance = chunker()
    
    # Validate parameters before chunking
    def validate_parameters(max_length, overlap):
        if max_length < 100:
            return False, "chunk_max_length too small"
        if overlap >= max_length:
            return False, "chunk_overlap must be less than chunk_max_length"
        return True, "Valid"
    
    # Attempt chunking with validation
    text = "Sample text for chunking"
    max_length = 1000
    overlap = 150
    
    valid, message = validate_parameters(max_length, overlap)
    if not valid:
        print(f"Validation failed: {message}")
        # Use safe defaults
        max_length = 2000
        overlap = 256
    
    chunks = await chunker_instance.chunk_text(
        text,
        chunk_max_length=max_length,
        chunk_overlap=overlap
    )
    
    return chunks
```

## Examples

### Basic Chunking Example

```python
from sdk import chunker
import asyncio

async def basic_chunking_example():
    """Basic example of text chunking."""
    chunker_instance = chunker()
    
    # Sample text
    text = """
    This is a sample document that demonstrates text chunking capabilities.
    
    The chunker will split this text into manageable pieces while maintaining
    context through overlapping sections. This ensures that important information
    is not lost at chunk boundaries.
    
    The chunking process uses Docling's HybridChunker when available, which
    provides structure-aware chunking based on document formatting and layout.
    
    When Docling is not available, the chunker falls back to character-based
    chunking with sliding-window overlap to maintain compatibility and
    ensure reliable operation.
    """
    
    # Chunk with defaults
    chunks = await chunker_instance.chunk_text(text)
    
    print(f"Original text length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk length: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f}")
    
    # Display first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    return chunks

# Run example
asyncio.run(basic_chunking_example())
```

### Document Processing Pipeline

```python
from sdk import chunker, parser
import asyncio

async def document_processing_pipeline():
    """Process a document through parsing and chunking."""
    chunker_instance = chunker()
    parser_instance = parser()
    
    # Sample document path (replace with actual file)
    document_path = "sample_document.txt"
    
    # Step 1: Parse document (if needed)
    # In this example, we'll work with plain text
    text = """
    # Research Paper: Text Chunking Strategies
    
    ## Abstract
    This paper explores various text chunking strategies for natural language processing
    applications. We compare character-based, sentence-based, and structure-aware
    chunking methods.
    
    ## Introduction
    Text chunking is a fundamental preprocessing step in many NLP applications.
    The choice of chunking strategy can significantly impact downstream performance.
    
    ## Methodology
    We implemented three chunking approaches:
    1. Fixed character-length chunking
    2. Sentence-boundary chunking
    3. Structure-aware chunking using document formatting
    
    ## Results
    Our experiments show that structure-aware chunking provides the best
    balance between context preservation and computational efficiency.
    
    ## Conclusion
    Structure-aware chunking should be preferred when document structure
    information is available.
    """
    
    # Step 2: Chunk the document
    print("Chunking document...")
    chunks = await chunker_instance.chunk_text(
        text,
        chunk_max_length=1000,
        chunk_overlap=150
    )
    
    # Step 3: Analyze chunks
    print(f"Document processing complete:")
    print(f"- Original length: {len(text)} characters")
    print(f"- Number of chunks: {len(chunks)}")
    print(f"- Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.1f}")
    
    # Step 4: Show chunk boundaries
    print("\nChunk boundaries:")
    for i, chunk in enumerate(chunks):
        first_line = chunk.split('\n')[0][:50]
        print(f"Chunk {i+1}: {first_line}...")
    
    return chunks

asyncio.run(document_processing_pipeline())
```

### Chunk Size Optimization

```python
from sdk import chunker
import asyncio

async def chunk_size_optimization():
    """Optimize chunk size for different use cases."""
    chunker_instance = chunker()
    
    # Test text
    test_text = "This is a test document. " * 200  # Create longer text
    
    # Test different chunk sizes
    chunk_sizes = [500, 1000, 1500, 2000, 2500]
    overlap_ratio = 0.15  # 15% overlap
    
    results = []
    
    for chunk_size in chunk_sizes:
        overlap = int(chunk_size * overlap_ratio)
        
        chunks = await chunker_instance.chunk_text(
            test_text,
            chunk_max_length=chunk_size,
            chunk_overlap=overlap
        )
        
        results.append({
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_chunks': len(chunks),
            'avg_length': sum(len(c) for c in chunks) / len(chunks),
            'coverage': sum(len(c) for c in chunks) / len(test_text)
        })
    
    # Display results
    print("Chunk Size Optimization Results:")
    print("=" * 60)
    print(f"{'Size':<6} {'Overlap':<7} {'Chunks':<7} {'Avg Len':<8} {'Coverage':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['chunk_size']:<6} {result['overlap']:<7} "
              f"{result['num_chunks']:<7} {result['avg_length']:<8.1f} "
              f"{result['coverage']:<8.2f}")
    
    return results

asyncio.run(chunk_size_optimization())
```

### Multi-language Text Chunking

```python
from sdk import chunker
import asyncio

async def multi_language_chunking():
    """Handle multi-language text chunking."""
    chunker_instance = chunker()
    
    # Multi-language text samples
    texts = {
        "English": "This is a sample English text that needs to be chunked properly.",
        "Spanish": "Este es un texto de ejemplo en español que necesita ser dividido correctamente.",
        "French": "Ceci est un exemple de texte français qui doit être divisé correctement.",
        "German": "Dies ist ein Beispieltext auf Deutsch, der ordnungsgemäß aufgeteilt werden muss.",
        "Chinese": "这是一个需要正确分块的中文示例文本。",
        "Japanese": "これは適切にチャンクされる必要がある日本語のサンプルテキストです。"
    }
    
    results = {}
    
    for language, text in texts.items():
        # Repeat text to make it longer
        long_text = text * 10
        
        chunks = await chunker_instance.chunk_text(
            long_text,
            chunk_max_length=300,  # Smaller chunks for multi-language
            chunk_overlap=50
        )
        
        results[language] = {
            'original_length': len(long_text),
            'num_chunks': len(chunks),
            'avg_chunk_length': sum(len(c) for c in chunks) / len(chunks),
            'chunks': chunks
        }
    
    # Display results
    print("Multi-language Chunking Results:")
    print("=" * 50)
    
    for language, result in results.items():
        print(f"\n{language}:")
        print(f"  Original length: {result['original_length']} characters")
        print(f"  Number of chunks: {result['num_chunks']}")
        print(f"  Average chunk length: {result['avg_chunk_length']:.1f}")
        print(f"  First chunk: {result['chunks'][0][:50]}...")
    
    return results

asyncio.run(multi_language_chunking())
```

### Structured Document Chunking

```python
from sdk import chunker
import asyncio

async def structured_document_chunking():
    """Chunk structured documents with proper section handling."""
    chunker_instance = chunker()
    
    # Structured document
    structured_text = """
    # Company Annual Report 2023
    
    ## Executive Summary
    
    This annual report provides a comprehensive overview of our company's
    performance during the fiscal year 2023. We have achieved significant
    growth in multiple areas while maintaining our commitment to sustainability
    and innovation.
    
    Key highlights include:
    - Revenue growth of 25%
    - Expansion into 5 new markets
    - Launch of 3 innovative products
    - Reduction in carbon footprint by 15%
    
    ## Financial Performance
    
    ### Revenue Analysis
    
    Total revenue for 2023 reached $500 million, representing a 25% increase
    from the previous year. This growth was driven by strong performance in
    our core markets and successful expansion into new territories.
    
    ### Profit Margins
    
    Our profit margins improved significantly this year, reaching 18% compared
    to 15% in 2022. This improvement is attributed to operational efficiency
    gains and strategic cost management initiatives.
    
    ## Market Expansion
    
    We successfully entered five new markets during 2023:
    1. Southeast Asia
    2. Eastern Europe
    3. Latin America
    4. Middle East
    5. Africa
    
    Each market entry was carefully planned and executed with local partnerships
    and cultural considerations in mind.
    
    ## Innovation and R&D
    
    ### Product Development
    
    Our R&D team launched three groundbreaking products this year:
    - Product A: Advanced AI-powered analytics platform
    - Product B: Sustainable packaging solution
    - Product C: Mobile application for customer engagement
    
    ### Technology Investments
    
    We invested $50 million in technology upgrades and research initiatives,
    focusing on artificial intelligence, machine learning, and sustainable
    technologies.
    """
    
    # Chunk with different strategies
    strategies = [
        {"name": "Small Chunks", "max_length": 800, "overlap": 100},
        {"name": "Medium Chunks", "max_length": 1500, "overlap": 200},
        {"name": "Large Chunks", "max_length": 2500, "overlap": 300}
    ]
    
    print("Structured Document Chunking Analysis:")
    print("=" * 60)
    
    for strategy in strategies:
        chunks = await chunker_instance.chunk_text(
            structured_text,
            chunk_max_length=strategy["max_length"],
            chunk_overlap=strategy["overlap"]
        )
        
        print(f"\n{strategy['name']}:")
        print(f"  Max length: {strategy['max_length']}")
        print(f"  Overlap: {strategy['overlap']}")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Average length: {sum(len(c) for c in chunks) / len(chunks):.1f}")
        
        # Show section preservation
        sections_in_chunks = 0
        for chunk in chunks:
            if '#' in chunk:  # Contains section headers
                sections_in_chunks += 1
        
        print(f"  Chunks with sections: {sections_in_chunks}")
        print(f"  Section preservation: {sections_in_chunks / len(chunks) * 100:.1f}%")
    
    return chunks

asyncio.run(structured_document_chunking())
```

### Performance Benchmarking

```python
from sdk import chunker
import asyncio
import time

async def performance_benchmarking():
    """Benchmark chunking performance with different text sizes."""
    chunker_instance = chunker()
    
    # Generate test texts of different sizes
    base_text = "This is a sample sentence for performance testing. " * 20
    test_sizes = [1000, 5000, 10000, 25000, 50000]  # Characters
    
    results = []
    
    for size in test_sizes:
        # Create text of target size
        multiplier = size // len(base_text) + 1
        test_text = base_text * multiplier
        test_text = test_text[:size]  # Trim to exact size
        
        print(f"Testing text size: {size} characters")
        
        # Benchmark chunking
        start_time = time.time()
        chunks = await chunker_instance.chunk_text(
            test_text,
            chunk_max_length=2000,
            chunk_overlap=256
        )
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        chars_per_second = size / processing_time
        chunks_per_second = len(chunks) / processing_time
        
        results.append({
            'text_size': size,
            'processing_time': processing_time,
            'num_chunks': len(chunks),
            'chars_per_second': chars_per_second,
            'chunks_per_second': chunks_per_second
        })
    
    # Display results
    print("\nPerformance Benchmarking Results:")
    print("=" * 80)
    print(f"{'Size':<8} {'Time(s)':<8} {'Chunks':<8} {'Chars/s':<10} {'Chunks/s':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['text_size']:<8} {result['processing_time']:<8.3f} "
              f"{result['num_chunks']:<8} {result['chars_per_second']:<10.1f} "
              f"{result['chunks_per_second']:<10.2f}")
    
    return results

asyncio.run(performance_benchmarking())
```

### Error Recovery and Robustness

```python
from sdk import chunker
import asyncio

async def error_recovery_example():
    """Demonstrate error recovery and robustness features."""
    chunker_instance = chunker()
    
    # Test cases with various potential issues
    test_cases = [
        {
            "name": "Normal text",
            "text": "This is normal text that should chunk without issues.",
            "expected_success": True
        },
        {
            "name": "Empty text",
            "text": "",
            "expected_success": True  # Should return empty list
        },
        {
            "name": "Very short text",
            "text": "Hi",
            "expected_success": True
        },
        {
            "name": "Text with special characters",
            "text": "Text with émojis 🚀 and spécial chäractërs ñ",
            "expected_success": True
        },
        {
            "name": "Very long single word",
            "text": "a" * 5000,  # Single word longer than chunk size
            "expected_success": True
        },
        {
            "name": "Mixed content",
            "text": "Regular text\n\nCode:\n```python\nprint('hello')\n```\n\nMore text.",
            "expected_success": True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        
        try:
            # Attempt chunking
            chunks = await chunker_instance.chunk_text(
                test_case["text"],
                chunk_max_length=1000,
                chunk_overlap=100
            )
            
            success = True
            error_msg = None
            
            # Validate results
            if chunks:
                max_chunk_length = max(len(chunk) for chunk in chunks)
                if max_chunk_length > 1000:
                    success = False
                    error_msg = f"Chunk too long: {max_chunk_length}"
            
            results.append({
                "name": test_case["name"],
                "success": success,
                "error": error_msg,
                "num_chunks": len(chunks),
                "text_length": len(test_case["text"])
            })
            
            print(f"  ✓ Success: {len(chunks)} chunks")
            
        except Exception as e:
            results.append({
                "name": test_case["name"],
                "success": False,
                "error": str(e),
                "num_chunks": 0,
                "text_length": len(test_case["text"])
            })
            
            print(f"  ✗ Error: {e}")
    
    # Summary
    print("\nError Recovery Test Results:")
    print("=" * 60)
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(results)}")
    
    if successful < len(results):
        print("\nFailed tests:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['name']}: {result['error']}")
    
    return results

asyncio.run(error_recovery_example())
```

## Best Practices

1. **Choose appropriate chunk sizes**: Balance between context preservation and processing efficiency
2. **Use adequate overlap**: Ensure important information isn't lost at chunk boundaries
3. **Handle edge cases**: Implement error handling for empty text, encoding issues, and large chunks
4. **Monitor performance**: Track chunking performance for large documents
5. **Validate parameters**: Check chunk_max_length and chunk_overlap before processing
6. **Use progress callbacks**: Implement progress tracking for long-running operations
7. **Consider document structure**: Leverage Docling's structure-aware chunking when available
8. **Test with real data**: Test chunking with actual documents from your use case
9. **Manage memory**: Clear session data periodically for long-running applications
10. **Use appropriate encodings**: Handle different text encodings properly

## Troubleshooting

### Common Issues

1. **Docling not installed**: Install with `pip install docling`
2. **Chunks too large**: Reduce chunk_max_length or check for very long words
3. **Poor chunk boundaries**: Adjust chunk_overlap to preserve context
4. **Encoding errors**: Specify correct encoding when reading files
5. **Memory issues**: Process large documents in smaller batches
6. **Performance problems**: Use async operations for better performance
7. **Validation errors**: Check parameter ranges and types
8. **Empty results**: Verify input text is not empty or whitespace-only
9. **Import errors**: Ensure all required dependencies are installed
10. **File access issues**: Check file permissions and paths

### Performance Optimization

1. **Use async methods**: Prefer async operations for better performance
2. **Optimize chunk size**: Find the optimal balance for your use case
3. **Batch processing**: Process multiple documents efficiently
4. **Memory management**: Monitor memory usage with large documents
5. **Session cleanup**: Clear session data when not needed
6. **Progress monitoring**: Use callbacks to track long operations
7. **Efficient file reading**: Use appropriate encodings and buffering
8. **Parallel processing**: Consider parallel processing for multiple documents 