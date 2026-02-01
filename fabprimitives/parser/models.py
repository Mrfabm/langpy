"""
Parser Models - Core data structures for Langbase-style parser primitive.

Defines the input/output contracts and error types that ensure
compatibility with Langbase's parser API.
"""

from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterator, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class JobStatus(str, Enum):
    """Parser job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ParserError(Exception):
    """Base exception for parser errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class UnsupportedFormatError(ParserError):
    """Raised when file format is not supported."""
    def __init__(self, format_name: str):
        super().__init__(f"Unsupported format: {format_name}", 400)


class FileTooLargeError(ParserError):
    """Raised when file exceeds size limit."""
    def __init__(self, size: int, limit: int):
        super().__init__(f"File size {size} exceeds limit {limit}", 413)


class CorruptedFileError(ParserError):
    """Raised when file is corrupted or unreadable."""
    def __init__(self, filename: str, reason: str):
        super().__init__(f"Corrupted file {filename}: {reason}", 422)


class EncryptedFileError(ParserError):
    """Raised when file is encrypted and password is required."""
    def __init__(self, filename: str):
        super().__init__(f"Encrypted file {filename} requires password", 423)


class ParseTimeoutError(ParserError):
    """Raised when parsing times out."""
    def __init__(self, filename: str, timeout: int):
        super().__init__(f"Parse timeout for {filename} after {timeout}s", 408)


class OcrFailureError(ParserError):
    """Raised when OCR processing fails."""
    def __init__(self, filename: str, reason: str):
        super().__init__(f"OCR failed for {filename}: {reason}", 422)


class TableCell(BaseModel):
    """Represents a single table cell."""
    value: str = Field(..., description="Cell content")
    row: int = Field(..., description="Row index (0-based)")
    col: int = Field(..., description="Column index (0-based)")
    rowspan: int = Field(1, description="Number of rows this cell spans")
    colspan: int = Field(1, description="Number of columns this cell spans")


class TableMeta(BaseModel):
    """Table metadata and structure information."""
    page_number: int = Field(..., description="Page where table appears")
    bounding_box: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates")
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    cells: List[TableCell] = Field(..., description="All table cells")
    
    @property
    def matrix(self) -> List[List[str]]:
        """Convert to 2D matrix format."""
        matrix = [['' for _ in range(self.columns)] for _ in range(self.rows)]
        for cell in self.cells:
            for r in range(cell.row, min(cell.row + cell.rowspan, self.rows)):
                for c in range(cell.col, min(cell.col + cell.colspan, self.columns)):
                    matrix[r][c] = cell.value
        return matrix


class PageMeta(BaseModel):
    """Page-level metadata."""
    page_number: int = Field(..., description="Page number (1-based)")
    width: float = Field(..., description="Page width in points")
    height: float = Field(..., description="Page height in points")
    rotation: int = Field(0, description="Page rotation in degrees")
    language: Optional[str] = Field(None, description="Detected language code")
    text: str = Field(..., description="Page text content")


class ParseStats(BaseModel):
    """Document parsing statistics."""
    char_count: int = Field(..., description="Total character count")
    token_estimate: int = Field(..., description="Estimated token count")
    table_count: int = Field(0, description="Number of tables found")
    page_count: int = Field(..., description="Number of pages")
    languages: List[str] = Field(default_factory=list, description="Detected languages")
    source_name: str = Field(..., description="Original filename or source")
    checksum: str = Field(..., description="SHA256 hash of original content")
    parse_time: float = Field(..., description="Parse time in seconds")
    ocr_time: Optional[float] = Field(None, description="OCR processing time if used")


class ParseResult(BaseModel):
    """Complete parsing result matching Langbase format."""
    pages: List[str] = Field(..., description="Page-ordered text content")
    tables: List[TableMeta] = Field(default_factory=list, description="Extracted tables")
    metadata: ParseStats = Field(..., description="Document statistics and metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pages": self.pages,
            "tables": [table.dict() for table in self.tables],
            "metadata": self.metadata.dict()
        }


class ParseJob(BaseModel):
    """Parser job record matching Langbase API."""
    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    input_type: str = Field(..., description="Input type: file, url, or text")
    mime_type: Optional[str] = Field(None, description="Detected MIME type")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    stats: Optional[ParseStats] = Field(None, description="Parse statistics")
    result_url: Optional[str] = Field(None, description="URL to result payload")
    
    @validator('created_at', 'updated_at', pre=True)
    def parse_datetime(cls, v):
        """Parse datetime strings."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class ParserOptions(BaseModel):
    """Configuration options for parser behavior."""
    # Size limits
    max_file_size: int = Field(50 * 1024 * 1024, description="Maximum file size in bytes (50MB)")
    
    # OCR settings
    enable_ocr: bool = Field(True, description="Enable OCR for images and PDFs")
    ocr_languages: List[str] = Field(["eng"], description="OCR language codes")
    ocr_confidence_threshold: float = Field(0.7, description="Minimum OCR confidence")
    language_hint: Optional[str] = Field(None, description="Language hint for OCR")
    
    # Text processing
    preserve_whitespace: bool = Field(False, description="Preserve original whitespace")
    merge_hyphens: bool = Field(True, description="Merge hyphenated words")
    strip_headers_footers: bool = Field(False, description="Strip headers and footers")
    
    # Table extraction
    table_strategy: Literal["docling", "camelot", "none"] = Field("docling", description="Table extraction strategy")
    
    # Performance
    parse_timeout: int = Field(300, description="Parse timeout in seconds")
    
    # Experimental features
    beta_features: bool = Field(False, description="Enable beta features")
    experimental_chunking: bool = Field(False, description="Enable experimental chunking")


class ParseRequest(BaseModel):
    """Input contract for parser requests."""
    content: Optional[Union[bytes, str]] = Field(None, description="File content")
    url: Optional[str] = Field(None, description="URL to download and parse")
    filename: Optional[str] = Field(None, description="Original filename")
    options: Optional[ParserOptions] = Field(None, description="Parser options")
    
    @validator('content', 'url')
    def validate_input(cls, v, values):
        """Ensure at least one input method is provided."""
        if not v and not values.get('content') and not values.get('url'):
            raise ValueError("Either content or url must be provided")
        return v
    
    def __post_init__(self):
        if self.options is None:
            self.options = ParserOptions()


class SupportedFormat(BaseModel):
    """Supported format information."""
    mime_type: str = Field(..., description="MIME type")
    max_file_size: int = Field(..., description="Maximum file size in bytes")
    description: str = Field(..., description="Format description")
    extensions: List[str] = Field(..., description="Common file extensions")


@dataclass
class ParsedDocument:
    """Legacy parsed document with metadata (for backward compatibility)."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure text is UTF-8 and normalized."""
        if isinstance(self.text, bytes):
            self.text = self.text.decode('utf-8', errors='replace')
        # Normalize newlines
        self.text = self.text.replace('\r\n', '\n').replace('\r', '\n')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) 