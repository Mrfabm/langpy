"""
Docling Parser - Production-ready document parser powered by Docling.

Provides Langbase-compatible parsing with job lifecycle, table extraction,
and comprehensive document processing using Docling for conversion.
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import aiohttp
import pandas as pd
from pydantic import BaseModel

# Docling imports
try:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument, TextItem, TableItem, PictureItem
    from docling_core.types.io import DocumentStream
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: Docling not available. Install with: pip install docling")

# OCR imports
try:
    import pytesseract
    from PIL import Image
    import io
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract not available. Install with: pip install pytesseract pillow")

from .models import (
    ParseJob, ParseResult, ParseStats, ParseRequest, ParserOptions,
    TableMeta, TableCell, PageMeta, JobStatus, SupportedFormat,
    ParserError, FileTooLargeError, UnsupportedFormatError,
    CorruptedFileError, OcrFailureError, ParseTimeoutError
)


class DoclingParser:
    """
    Production-ready document parser using Docling for conversion.
    
    Provides Langbase-compatible API with job lifecycle management,
    table extraction, and comprehensive document processing.
    """
    
    def __init__(self, options: Optional[ParserOptions] = None):
        """
        Initialize Docling parser.
        
        Args:
            options: Parser configuration options
        """
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required. Install with: pip install docling")
        
        self.options = options or ParserOptions()
        
        # Initialize converter with proper format detection
        self._converter = DocumentConverter()
        
        # Cache for processed documents
        self._cache = {}
        
        # Supported formats with size limits
        self._supported_formats = {
            "application/pdf": {"max_size": 50 * 1024 * 1024, "description": "PDF documents"},
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {"max_size": 50 * 1024 * 1024, "description": "Word documents"},
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": {"max_size": 50 * 1024 * 1024, "description": "PowerPoint presentations"},
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {"max_size": 50 * 1024 * 1024, "description": "Excel spreadsheets"},
            "text/csv": {"max_size": 10 * 1024 * 1024, "description": "CSV files"},
            "text/tab-separated-values": {"max_size": 10 * 1024 * 1024, "description": "TSV files"},
            "text/html": {"max_size": 10 * 1024 * 1024, "description": "HTML documents"},
            "text/markdown": {"max_size": 10 * 1024 * 1024, "description": "Markdown files"},
            "text/plain": {"max_size": 10 * 1024 * 1024, "description": "Plain text files"},
            "image/jpeg": {"max_size": 20 * 1024 * 1024, "description": "JPEG images"},
            "image/png": {"max_size": 20 * 1024 * 1024, "description": "PNG images"},
            "image/tiff": {"max_size": 50 * 1024 * 1024, "description": "TIFF images"},
        }
    
    def _generate_job_id(self) -> str:
        """Generate unique job identifier."""
        return str(uuid.uuid4())
    
    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _detect_language(self, text: str) -> List[str]:
        """Detect language(s) in text."""
        # Simple language detection - in production, use langdetect or similar
        if not text.strip():
            return ["unknown"]
        
        # Basic heuristics
        languages = []
        if any(ord(c) > 127 for c in text[:1000]):  # Non-ASCII characters
            languages.append("multilingual")
        else:
            languages.append("en")
        
        return languages
    
    def _extract_tables_from_docling(self, doc: Any) -> List[TableMeta]:
        """Extract tables from Docling document."""
        tables = []
        
        for table_item in doc.tables:
            try:
                # Extract table structure
                table_data = table_item.data
                if not table_data or not isinstance(table_data, list):
                    continue
                
                # Convert to pandas DataFrame for easier processing
                df = pd.DataFrame(table_data)
                rows, cols = df.shape
                
                # Create table cells
                cells = []
                for i in range(rows):
                    for j in range(cols):
                        value = str(df.iloc[i, j]) if pd.notna(df.iloc[i, j]) else ""
                        cells.append(TableCell(
                            value=value,
                            row=i,
                            col=j,
                            rowspan=1,
                            colspan=1
                        ))
                
                # Get bounding box if available
                bbox = [0, 0, 100, 100]  # Default
                if hasattr(table_item, 'bbox') and table_item.bbox:
                    bbox = table_item.bbox
                
                # Get page number
                page_num = 1  # Default
                if hasattr(table_item, 'page') and table_item.page:
                    page_num = table_item.page
                
                table_meta = TableMeta(
                    page_number=page_num,
                    bounding_box=bbox,
                    rows=rows,
                    columns=cols,
                    cells=cells
                )
                tables.append(table_meta)
                
            except Exception as e:
                print(f"Warning: Failed to extract table: {e}")
                continue
        
        return tables
    
    def _process_text_content(self, text: str, options: ParserOptions) -> str:
        """Process text content according to options."""
        if not text:
            return text
        
        # Merge hyphens if enabled
        if options.merge_hyphens:
            import re
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Strip headers/footers if enabled
        if options.strip_headers_footers:
            lines = text.split('\n')
            # Simple heuristic: remove first and last few lines if they look like headers/footers
            if len(lines) > 10:
                # Remove potential header (first 2 lines)
                lines = lines[2:]
                # Remove potential footer (last 2 lines)
                lines = lines[:-2]
            text = '\n'.join(lines)
        
        # Preserve whitespace if enabled
        if not options.preserve_whitespace:
            # Normalize whitespace
            text = ' '.join(text.split())
        
        return text
    
    async def _download_url(self, url: str) -> Tuple[bytes, str]:
        """Download content from URL."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ParserError(f"Failed to download URL: {response.status}")
                
                content = await response.read()
                content_type = response.headers.get('content-type', 'application/octet-stream')
                return content, content_type
    
    async def _perform_ocr(self, image_content: bytes, options: ParserOptions) -> str:
        """Perform OCR on image content."""
        if not TESSERACT_AVAILABLE:
            raise OcrFailureError("pytesseract not available", "OCR library not installed")
        
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_content))
            
            # Configure OCR
            ocr_config = f'--oem 3 --psm 6 -l {"+".join(options.ocr_languages)}'
            if options.language_hint:
                ocr_config += f" -l {options.language_hint}"
            
            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(image, config=ocr_config)
            return text.strip()
            
        except Exception as e:
            raise OcrFailureError("OCR processing failed", str(e))
    
    async def _parse_with_docling(self, content: bytes, mime_type: str, options: ParserOptions, filename: str = None) -> ParseResult:
        """Parse content using Docling."""
        start_time = time.time()
        
        try:
            # Use provided filename or fallback to default
            if not filename:
                # Map MIME types to proper file extensions
                mime_to_ext = {
                    "application/pdf": ".pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                    "text/csv": ".csv",
                    "text/tab-separated-values": ".tsv",
                    "text/html": ".html",
                    "text/markdown": ".md",
                    "text/plain": ".txt",
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/tiff": ".tiff",
                }
                
                # Get proper extension
                ext = mime_to_ext.get(mime_type, ".txt")
                filename = f"document{ext}"
            
            # Create DocumentStream from bytes
            import io
            doc_stream = DocumentStream(
                name=filename,
                stream=io.BytesIO(content)
            )
            
            # Convert document using Docling with timeout
            try:
                conversion_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._converter.convert,
                        doc_stream
                    ),
                    timeout=self.options.parse_timeout
                )
            except asyncio.TimeoutError:
                raise ParseTimeoutError(f"Parsing timed out after {self.options.parse_timeout} seconds")
            except Exception as e:
                raise ParserError(f"Docling conversion failed: {str(e)}")
            
            # Get the document from conversion result
            doc = conversion_result.document
            
            # Extract text content
            pages = []
            page_metas = []
            
            # For now, just extract all text as one page
            all_text = ""
            if hasattr(doc, 'texts') and doc.texts:
                for text_item in doc.texts:
                    if hasattr(text_item, 'text'):
                        all_text += text_item.text + "\n"
            
            # If no text found, try alternative extraction
            if not all_text.strip():
                # Try to get text from document content
                if hasattr(doc, 'content'):
                    all_text = str(doc.content)
                elif hasattr(doc, 'text'):
                    all_text = doc.text
                else:
                    all_text = "No text content found"
            
            # Process text content
            processed_text = self._process_text_content(all_text, options)
            pages.append(processed_text)
            
            # Create page metadata
            page_meta = PageMeta(
                page_number=1,
                width=612,  # Default letter size
                height=792,
                rotation=0,
                language=self._detect_language(processed_text)[0],
                text=processed_text
            )
            page_metas.append(page_meta)
            
            # Extract tables (simplified)
            tables = []
            if options.table_strategy != "none" and hasattr(doc, 'tables'):
                tables = self._extract_tables_from_docling(doc)
            
            # Calculate statistics
            char_count = len(processed_text)
            token_estimate = self._estimate_tokens(processed_text)
            languages = list(set([meta.language for meta in page_metas if meta.language]))
            
            parse_time = time.time() - start_time
            
            stats = ParseStats(
                char_count=char_count,
                token_estimate=token_estimate,
                table_count=len(tables),
                page_count=len(pages),
                languages=languages,
                source_name=filename,
                checksum=self._calculate_checksum(content),
                parse_time=parse_time
            )
            
            return ParseResult(
                pages=pages,
                tables=tables,
                metadata=stats
            )
            
        except Exception as e:
            raise ParserError(f"Docling parsing failed: {str(e)}")
    
    async def parse_async(self, request: ParseRequest) -> ParseJob:
        """
        Parse document asynchronously with job lifecycle.
        
        Args:
            request: Parse request with content or URL
            
        Returns:
            Parse job with initial status
        """
        job_id = self._generate_job_id()
        now = datetime.utcnow()
        
        # Create initial job
        job = ParseJob(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            input_type="file" if request.content else "url",
            mime_type=None,
            error_code=None,
            error_message=None,
            stats=None,
            result_url=None
        )
        
        # Start processing in background
        asyncio.create_task(self._process_job(job, request))
        
        return job
    
    async def _process_job(self, job: ParseJob, request: ParseRequest):
        """Process job in background."""
        try:
            # Update status to processing
            job.status = JobStatus.PROCESSING
            job.updated_at = datetime.utcnow()
            
            # Get content
            content = request.content
            mime_type = None
            
            if request.url:
                content, mime_type = await self._download_url(request.url)
                job.input_type = "url"
            
            if not content:
                raise ParserError("No content provided")
            
            # Check file size
            if len(content) > self.options.max_file_size:
                raise FileTooLargeError(len(content), self.options.max_file_size)
            
            # Detect MIME type if not provided
            if not mime_type:
                from .mime_detection import mime_detector
                mime_type = mime_detector.detect_from_bytes(content, request.filename)
            
            # Ensure MIME type is in allowed formats
            if mime_type not in self._supported_formats:
                # Try to map to a supported format
                mime_mapping = {
                    "image/jpg": "image/jpeg",
                    "application/zip": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/octet-stream": "text/plain"
                }
                mime_type = mime_mapping.get(mime_type, "text/plain")
            
            job.mime_type = mime_type
            
            # Check if format is supported
            if mime_type not in self._supported_formats:
                raise UnsupportedFormatError(mime_type)
            
            # Handle OCR for images
            ocr_start_time = None
            if mime_type.startswith('image/') and self.options.enable_ocr:
                ocr_start_time = time.time()
                content = await self._perform_ocr(content, self.options)
                ocr_time = time.time() - ocr_start_time
            else:
                ocr_time = None
            
            # Parse with Docling
            result = await self._parse_with_docling(content, mime_type, self.options, request.filename)
            
            # Update OCR time if used
            if ocr_time:
                result.metadata.ocr_time = ocr_time
            
            # Update job with results
            job.status = JobStatus.READY
            job.updated_at = datetime.utcnow()
            job.stats = result.metadata
            job.result_url = f"/results/{job.id}"  # In production, this would be a real URL
            
        except Exception as e:
            # Update job with error
            job.status = JobStatus.FAILED
            job.updated_at = datetime.utcnow()
            job.error_code = type(e).__name__
            job.error_message = str(e)
    
    def parse_sync(self, request: ParseRequest) -> ParseResult:
        """
        Parse document synchronously.
        
        Args:
            request: Parse request
            
        Returns:
            Parse result
        """
        # Fast-path for images if OCR is disabled (case-insensitive, all common extensions)
        image_exts = (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp")
        if request.filename and request.filename.lower().endswith(image_exts):
            print(f"[DEBUG] Using fast image parser for: {request.filename}")
            if not (request.options and request.options.enable_ocr):
                try:
                    from .fast_image_parser import parse_sync_fast_image
                    return parse_sync_fast_image(request)
                except ImportError:
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(request.content))
                        width, height = img.size
                        stats = ParseStats(
                            char_count=0,
                            token_estimate=0,
                            table_count=0,
                            page_count=1,
                            languages=["unknown"],
                            source_name=request.filename,
                            checksum=self._calculate_checksum(request.content),
                            parse_time=0.01
                        )
                        return ParseResult(
                            pages=[f"Image: {width}x{height}"],
                            tables=[],
                            metadata=stats
                        )
                    except Exception as e:
                        raise ParserError(f"Image fast-path failed: {e}")
            else:
                print(f"[DEBUG] OCR enabled, using Docling for: {request.filename}")
        # For PDFs, try fast parser first if speed is prioritized
        if request.filename and request.filename.lower().endswith('.pdf'):
            try:
                from .fast_pdf_parser import parse_sync_fast_pdf
                return parse_sync_fast_pdf(request)
            except ImportError:
                pass
        # Try Docling for all other formats
        try:
            mime_type = "application/octet-stream"
            if request.filename:
                from .mime_detection import mime_detector
                mime_type = mime_detector.detect_from_bytes(request.content, request.filename)
            if mime_type.startswith("image/") and not (request.options and request.options.enable_ocr):
                print(f"[DEBUG] Using fast image parser for MIME: {mime_type}")
                from .fast_image_parser import parse_sync_fast_image
                return parse_sync_fast_image(request)
            if mime_type not in self._supported_formats:
                mime_mapping = {
                    "image/jpg": "image/jpeg",
                    "application/zip": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/octet-stream": "text/plain"
                }
                mime_type = mime_mapping.get(mime_type, "text/plain")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._parse_with_docling(
                    request.content,
                    mime_type,
                    request.options or self.options,
                    request.filename
                ))
            finally:
                loop.close()
        except Exception as e:
            if request.filename and request.filename.lower().endswith((".txt", ".md", ".csv")):
                try:
                    from .simple_parser import parse_sync_simple
                    return parse_sync_simple(request)
                except ImportError:
                    pass
            raise e
    
    def get_supported_formats(self) -> List[SupportedFormat]:
        """Get list of supported formats with metadata."""
        formats = []
        for mime_type, info in self._supported_formats.items():
            # Get common extensions
            extensions = []
            if mime_type == "application/pdf":
                extensions = [".pdf"]
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                extensions = [".docx"]
            elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                extensions = [".pptx"]
            elif mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                extensions = [".xlsx"]
            elif mime_type == "text/csv":
                extensions = [".csv"]
            elif mime_type == "text/tab-separated-values":
                extensions = [".tsv"]
            elif mime_type == "text/html":
                extensions = [".html", ".htm"]
            elif mime_type == "text/markdown":
                extensions = [".md", ".markdown"]
            elif mime_type == "text/plain":
                extensions = [".txt"]
            elif mime_type == "image/jpeg":
                extensions = [".jpg", ".jpeg"]
            elif mime_type == "image/png":
                extensions = [".png"]
            elif mime_type == "image/tiff":
                extensions = [".tiff", ".tif"]
            
            formats.append(SupportedFormat(
                mime_type=mime_type,
                max_file_size=info["max_size"],
                description=info["description"],
                extensions=extensions
            ))
        
        return formats
    
    def is_supported(self, mime_type: str) -> bool:
        """Check if MIME type is supported."""
        return mime_type in self._supported_formats


# Global parser instance
_parser_instance: Optional[DoclingParser] = None


def get_parser(options: Optional[ParserOptions] = None) -> DoclingParser:
    """Get global parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = DoclingParser(options)
    return _parser_instance


async def parse_async(request: ParseRequest) -> ParseJob:
    """Parse document asynchronously."""
    parser = get_parser()
    return await parser.parse_async(request)


def parse_sync(request: ParseRequest) -> ParseResult:
    """Parse document synchronously."""
    parser = get_parser()
    return parser.parse_sync(request)


def get_supported_formats() -> List[SupportedFormat]:
    """Get supported formats."""
    parser = get_parser()
    return parser.get_supported_formats() 