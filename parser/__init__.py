"""
Parser Module - Comprehensive document parsing for Langbase compatibility.

Provides both basic and advanced parsing capabilities for various document formats,
including the new Docling-based parser with job lifecycle management.
"""

from .models import (
    # Core models
    ParsedDocument,
    ParseRequest,
    ParseResult,
    ParserOptions,
    
    # New Langbase-compatible models
    ParseJob,
    JobStatus,
    TableMeta,
    TableCell,
    PageMeta,
    ParseStats,
    SupportedFormat,
    
    # Errors
    ParserError,
    UnsupportedFormatError,
    FileTooLargeError,
    CorruptedFileError,
    EncryptedFileError,
    ParseTimeoutError,
    OcrFailureError
)

from .async_parser import AsyncParser
from .sync_parser import SyncParser
from .mime_detection import mime_detector
from .parsers import parser_registry

# New Docling-based parser
try:
    from .docling_parser import (
        DoclingParser,
        parse_async,
        parse_sync,
        get_supported_formats,
        get_parser
    )
    DOCLING_PARSER_AVAILABLE = True
except ImportError:
    DOCLING_PARSER_AVAILABLE = False
    print("Warning: Docling parser not available. Install with: pip install docling")

__all__ = [
    # Core models
    'ParsedDocument',
    'ParseRequest', 
    'ParseResult',
    'ParserOptions',
    
    # New Langbase-compatible models
    'ParseJob',
    'JobStatus',
    'TableMeta',
    'TableCell',
    'PageMeta',
    'ParseStats',
    'SupportedFormat',
    
    # Errors
    'ParserError',
    'UnsupportedFormatError',
    'FileTooLargeError',
    'CorruptedFileError',
    'EncryptedFileError',
    'ParseTimeoutError',
    'OcrFailureError',
    
    # Basic parsers
    'AsyncParser',
    'SyncParser',
    
    # Utilities
    'mime_detector',
    'parser_registry',
    
    # Docling parser (if available)
    'DoclingParser',
    'parse_async',
    'parse_sync',
    'get_supported_formats',
    'get_parser',
    'DOCLING_PARSER_AVAILABLE'
]

# Version info
__version__ = "2.0.0"

# Supported formats summary (legacy)
SUPPORTED_FORMATS = {
    'text': [
        'text/plain', 'text/markdown', 'text/html', 'text/x-rst', 'application/x-tex',
        'application/xml', 'application/json', 'application/x-yaml', 'application/toml',
        'text/csv', 'text/tab-separated-values'
    ],
    'code': [
        'application/x-python', 'application/javascript', 'application/typescript',
        'text/x-java-source', 'text/x-c++src', 'text/x-csrc', 'text/x-csharp',
        'text/x-go', 'text/x-rust', 'application/x-php', 'text/x-ruby',
        'text/x-swift', 'text/x-kotlin', 'text/x-scala', 'text/x-r', 'text/x-matlab',
        'application/sql', 'application/x-sh', 'application/x-powershell',
        'text/css', 'text/x-scss', 'text/x-sass', 'text/x-less'
    ],
    'office': [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # DOCX
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # PPTX
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # XLSX
        'application/vnd.oasis.opendocument.text',  # ODT
        'application/vnd.oasis.opendocument.spreadsheet',  # ODS
        'application/vnd.oasis.opendocument.presentation',  # ODP
        'application/msword',  # DOC
        'application/vnd.ms-powerpoint',  # PPT
        'application/vnd.ms-excel',  # XLS
    ],
    'images': [
        'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff',
        'image/webp', 'image/svg+xml', 'image/x-icon'
    ],
    'email': [
        'message/rfc822', 'application/vnd.ms-outlook', 'application/mbox'
    ],
    'archives': [
        'application/zip', 'application/x-tar', 'application/gzip',
        'application/x-bzip2', 'application/vnd.rar', 'application/x-7z-compressed'
    ],
    'data': [
        'application/x-ndjson', 'application/octet-stream',  # parquet, avro
        'application/x-hdf5'
    ],
    'ebooks': [
        'application/epub+zip', 'application/x-mobipocket-ebook',
        'application/vnd.amazon.ebook', 'application/rtf'
    ]
}

def get_supported_formats() -> dict:
    """Get comprehensive list of supported formats by category (legacy)."""
    return SUPPORTED_FORMATS.copy()

def get_total_supported_formats() -> int:
    """Get total number of supported MIME types (legacy)."""
    total = 0
    for formats in SUPPORTED_FORMATS.values():
        total += len(formats)
    return total

# Quick usage examples
def quick_parse_example():
    """Show quick usage examples."""
    examples = {
        "Basic parsing": """
from parser import parse_file
from pathlib import Path

result = parse_file(Path("document.pdf"))
print(result.document.text)
print(result.document.metadata)
""",
        "Parse with options": """
from parser import parse_document, ParserOptions

options = ParserOptions(
    enable_ocr=True,
    ocr_languages=["eng", "spa"],
    max_file_size=50*1024*1024,  # 50MB
    include_tables=True
)

result = parse_document(content, filename="data.xlsx", options=options)
""",
        "Async parsing": """
import asyncio
from parser import AsyncParser

async def parse_multiple():
    parser = AsyncParser()
    results = await parser.parse_multiple([
        ParseRequest(content=file1, filename="doc1.pdf"),
        ParseRequest(content=file2, filename="doc2.docx"),
    ])
    return results
""",
        "Docling parser (new)": """
from parser import parse_sync, parse_async, ParseRequest, ParserOptions

# Synchronous parsing
request = ParseRequest(
    content=file_content,
    filename="document.pdf",
    options=ParserOptions(enable_ocr=True, table_strategy="docling")
)
result = parse_sync(request)

# Async job-based parsing
job = await parse_async(request)
# Poll for completion and get result
""",
        "Get parser stats": """
from parser import sync_parser

# Parse some documents
sync_parser.parse_file("doc1.pdf")
sync_parser.parse_file("doc2.docx")

# Get statistics
stats = sync_parser.get_stats()
print(f"Total parses: {stats.total_parses}")
print(f"Average time: {stats.average_parse_time:.2f}s")
print(f"Cache hits: {stats.cache_hits}")
"""
    }
    return examples

# Convenience functions for backward compatibility
def parse_file(file_path, options=None):
    """Parse a file using the appropriate parser."""
    if DOCLING_PARSER_AVAILABLE:
        # Use Docling parser for better results
        with open(file_path, 'rb') as f:
            content = f.read()
        
        request = ParseRequest(
            content=content,
            filename=str(file_path),
            options=options
        )
        return parse_sync(request)
    else:
        # Fall back to basic parser
        from .sync_parser import SyncParser
        parser = SyncParser(options)
        return parser.parse_file(file_path)

def parse_document(content, filename=None, options=None):
    """Parse document content using the appropriate parser."""
    if DOCLING_PARSER_AVAILABLE:
        # Use Docling parser for better results
        request = ParseRequest(
            content=content,
            filename=filename,
            options=options
        )
        return parse_sync(request)
    else:
        # Fall back to basic parser
        from .sync_parser import SyncParser
        parser = SyncParser(options)
        return parser.parse(content, filename=filename) 