"""
Fast Image Parser - Instant image metadata extraction.
"""

import hashlib
import time
from typing import List, Optional
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .models import (
    ParseResult, ParseStats, ParseRequest, ParserOptions,
    TableMeta, TableCell, PageMeta, ParserError
)


class FastImageParser:
    """Fast image parser for instant metadata extraction."""
    
    def __init__(self, options: Optional[ParserOptions] = None):
        self.options = options or ParserOptions()
    
    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content).hexdigest()
    
    def parse_sync(self, request: ParseRequest) -> ParseResult:
        """Parse image instantly."""
        if not PIL_AVAILABLE:
            raise ParserError("PIL/Pillow not available. Install with: pip install Pillow")
        
        start_time = time.time()
        
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(request.content))
            width, height = img.size
            format_name = img.format or "unknown"
            mode = img.mode
            
            # Create basic page metadata
            page_meta = PageMeta(
                page_number=1,
                width=width,
                height=height,
                rotation=0,
                language="unknown",
                text=""  # No text extraction for fast mode
            )
            
            # Calculate statistics
            parse_time = time.time() - start_time
            
            stats = ParseStats(
                char_count=0,  # No text in images
                token_estimate=0,
                table_count=0,
                page_count=1,
                languages=["unknown"],
                source_name=request.filename or "image",
                checksum=self._calculate_checksum(request.content),
                parse_time=parse_time
            )
            
            return ParseResult(
                pages=[f"Image: {width}x{height} {format_name} {mode}"],
                tables=[],
                metadata=stats
            )
            
        except Exception as e:
            raise ParserError(f"Fast image parsing failed: {str(e)}")


def parse_sync_fast_image(request: ParseRequest) -> ParseResult:
    """Fast image parse function."""
    parser = FastImageParser()
    return parser.parse_sync(request) 