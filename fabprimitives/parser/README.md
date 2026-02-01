# Langpy Parser - Langbase-Compatible Document Parser

A production-ready document parser that perfectly mirrors Langbase's Parser features, interfaces, and operational behavior, powered by Docling for document conversion.

## Features

### ✅ Langbase Parity
- **Job Lifecycle**: `queued` → `processing` → `ready` → `failed`
- **CRUD API**: Full REST API matching Langbase endpoints
- **Async Processing**: Background job processing with status polling
- **Table Extraction**: Structured table reconstruction with metadata
- **OCR Support**: Tesseract integration for scanned documents
- **Multiple Inputs**: File upload, URL download, raw text

### 🚀 Advanced Capabilities
- **Docling Integration**: Powered by Docling for superior document conversion
- **Comprehensive Metadata**: Page-level info, language detection, statistics
- **Configurable Options**: OCR settings, text processing, table strategies
- **Error Handling**: Detailed error codes and messages
- **Webhook Support**: Job completion notifications
- **Docker Ready**: Complete containerized deployment

## Supported Formats

| Category | Formats | Max Size |
|----------|---------|----------|
| **PDF** | Vector & Scanned PDFs | 50MB |
| **Office** | DOCX, PPTX, XLSX | 50MB |
| **Data** | CSV, TSV | 10MB |
| **Text** | HTML, Markdown, TXT | 10MB |
| **Images** | JPEG, PNG, TIFF | 20-50MB |

## Quick Start

### Installation

```bash
# Install dependencies
pip install docling fastapi uvicorn tesserocr pillow pandas

# Or install from requirements
pip install -r parser/requirements.txt
```

### Basic Usage

```python
from parser import parse_sync, ParseRequest, ParserOptions

# Parse a document
request = ParseRequest(
    content=file_content,
    filename="document.pdf",
    options=ParserOptions(enable_ocr=True)
)

result = parse_sync(request)
print(f"Pages: {len(result.pages)}")
print(f"Tables: {len(result.tables)}")
print(f"Characters: {result.metadata.char_count}")
```

### Async Job Processing

```python
import asyncio
from parser import parse_async

async def parse_document():
    # Create job
    job = await parse_async(request)
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    
    # Poll for completion
    while job.status != "ready":
        await asyncio.sleep(1)
        # In real usage, call API to get updated status
    
    # Get result
    result = await get_result(job.id)
    return result
```

## API Reference

### Core Models

#### ParseRequest
```python
class ParseRequest:
    content: Optional[bytes]      # File content
    url: Optional[str]           # URL to download
    filename: Optional[str]      # Original filename
    options: Optional[ParserOptions]  # Parsing options
```

#### ParseResult
```python
class ParseResult:
    pages: List[str]             # Page-ordered text content
    tables: List[TableMeta]      # Extracted tables
    metadata: ParseStats         # Document statistics
```

#### ParseJob
```python
class ParseJob:
    id: str                      # Unique job identifier
    status: JobStatus            # Current status
    created_at: datetime         # Creation timestamp
    updated_at: datetime         # Last update
    input_type: str              # "file", "url", or "text"
    mime_type: Optional[str]     # Detected MIME type
    error_code: Optional[str]    # Error code if failed
    error_message: Optional[str] # Error message if failed
    stats: Optional[ParseStats]  # Parse statistics
    result_url: Optional[str]    # URL to result payload
```

### Parser Options

```python
class ParserOptions:
    # Size limits
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # OCR settings
    enable_ocr: bool = True
    ocr_languages: List[str] = ["eng"]
    ocr_confidence_threshold: float = 0.7
    language_hint: Optional[str] = None
    
    # Text processing
    preserve_whitespace: bool = False
    merge_hyphens: bool = True
    strip_headers_footers: bool = False
    
    # Table extraction
    table_strategy: Literal["docling", "camelot", "none"] = "docling"
    
    # Performance
    parse_timeout: int = 300  # seconds
    
    # Experimental features
    beta_features: bool = False
    experimental_chunking: bool = False
```

## REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/parsers` | Create parse job |
| `POST` | `/parsers/upload` | Upload file and parse |
| `GET` | `/parsers` | List jobs with filters |
| `GET` | `/parsers/{job_id}` | Get job status |
| `GET` | `/parsers/{job_id}/result` | Get parse result |
| `DELETE` | `/parsers/{job_id}` | Delete job |
| `GET` | `/parsers/supported-formats` | Get supported formats |
| `POST` | `/parsers/{job_id}/webhook` | Set webhook URL |

### Authentication

All endpoints require API key authentication via `x-api-key` header:

```bash
curl -H "x-api-key: your-api-key" \
     -X POST "http://localhost:8000/parsers" \
     -F "file=@document.pdf"
```

### Example API Usage

```python
import aiohttp

async def parse_via_api():
    async with aiohttp.ClientSession() as session:
        # Create job
        async with session.post(
            "http://localhost:8000/parsers",
            headers={"x-api-key": "your-api-key"},
            json={"url": "https://example.com/document.pdf"}
        ) as resp:
            job = await resp.json()
        
        # Poll for completion
        while job["status"] != "ready":
            await asyncio.sleep(1)
            async with session.get(
                f"http://localhost:8000/parsers/{job['id']}",
                headers={"x-api-key": "your-api-key"}
            ) as resp:
                job = await resp.json()
        
        # Get result
        async with session.get(
            f"http://localhost:8000/parsers/{job['id']}/result",
            headers={"x-api-key": "your-api-key"}
        ) as resp:
            result = await resp.json()
        
        return result
```

## Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f parser-api
```

### Environment Variables

```bash
# API Configuration
PARSER_API_KEY=your-secret-key
MAX_FILE_SIZE=52428800
STORAGE_PATH=/app/data/parser

# Worker Configuration
MAX_CONCURRENT_JOBS=5
REDIS_URL=redis://redis:6379/0
POSTGRES_URL=postgresql://postgres:postgres@postgres:5432/parser

# Optional: Celery Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

### Standalone Worker

```bash
# Run worker directly
python -m parser.worker

# Or with Celery
celery -A parser.worker worker --loglevel=info
```

## Table Extraction

The parser extracts tables with full structure information:

```python
# Table metadata
table = result.tables[0]
print(f"Dimensions: {table.rows}x{table.columns}")
print(f"Page: {table.page_number}")
print(f"Bounding box: {table.bounding_box}")

# Access table as matrix
matrix = table.matrix
for row in matrix:
    print(row)

# Access individual cells
for cell in table.cells:
    print(f"Cell ({cell.row},{cell.col}): {cell.value}")
    print(f"Spans: {cell.rowspan}x{cell.colspan}")
```

## Error Handling

The parser provides detailed error information:

```python
try:
    result = parse_sync(request)
except FileTooLargeError as e:
    print(f"File too large: {e}")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
except OcrFailureError as e:
    print(f"OCR failed: {e}")
except ParserError as e:
    print(f"Parser error: {e}")
```

## Development

### Running Tests

```bash
# Run parser tests
pytest tests/test_parser.py

# Run integration tests
pytest tests/test_parser_integration.py
```

### Adding New Formats

```python
# Register new parser
from parser.parsers import parser_registry

@parser_registry.register("application/x-custom")
class CustomParser(BaseParser):
    async def parse(self, request: ParseRequest) -> ParseResult:
        # Implementation
        pass
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation**: [Langpy Docs](https://langpy.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/langpy/langpy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/langpy/langpy/discussions) 