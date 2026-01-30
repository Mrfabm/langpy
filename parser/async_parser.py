"""
Async Parser - Basic async document parsing.

Provides async interface for parsing various document formats.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from .models import ParseRequest, ParseResult, ParserOptions, ParserError
from .mime_detection import mime_detector
from .parsers import parser_registry


class AsyncParser:
    """Async parser for various document formats."""
    
    def __init__(self, options: Optional[ParserOptions] = None):
        """
        Initialize async parser.
        
        Args:
            options: Parser configuration options
        """
        self.options = options or ParserOptions()
    
    async def parse(self, content: bytes, content_type: Optional[str] = None, 
                   filename: Optional[str] = None) -> ParseResult:
        """
        Parse document content.
        
        Args:
            content: Document content as bytes
            content_type: MIME type (auto-detected if not provided)
            filename: Original filename
            
        Returns:
            Parse result with extracted text and metadata
        """
        # Detect MIME type if not provided
        if not content_type:
            content_type = mime_detector.detect_from_bytes(content, filename)
        
        # Check if format is supported
        if not parser_registry.is_supported(content_type):
            raise ParserError(f"Unsupported format: {content_type}")
        
        # Get parser class
        parser_class = parser_registry.get_parser(content_type)
        if not parser_class:
            raise ParserError(f"No parser available for: {content_type}")
        
        # Create parse request
        request = ParseRequest(
            content=content,
            content_type=content_type,
            filename=filename,
            options=self.options
        )
        
        # Parse with timeout
        try:
            result = await asyncio.wait_for(
                parser_class().parse(request),
                timeout=self.options.parse_timeout
            )
            return result
        except asyncio.TimeoutError:
            raise ParserError(f"Parse timeout after {self.options.parse_timeout} seconds")
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported MIME types."""
        return parser_registry.get_supported_types()
    
    def is_supported(self, content_type: str) -> bool:
        """Check if MIME type is supported."""
        return parser_registry.is_supported(content_type) 