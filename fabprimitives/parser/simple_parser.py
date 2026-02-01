"""
Simple Fallback Parser - Works without Docling for basic text files.
"""

import hashlib
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from .models import (
    ParseResult, ParseStats, ParseRequest, ParserOptions,
    TableMeta, TableCell, PageMeta, ParserError
)


class SimpleParser:
    """Simple parser for text files that works without Docling."""
    
    def __init__(self, options: Optional[ParserOptions] = None):
        self.options = options or ParserOptions()
    
    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4
    
    def _detect_language(self, text: str) -> List[str]:
        """Detect language(s) in text."""
        if not text.strip():
            return ["unknown"]
        
        # Basic heuristics
        if any(ord(c) > 127 for c in text[:1000]):
            return ["multilingual"]
        else:
            return ["en"]
    
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
            if len(lines) > 10:
                lines = lines[2:-2]
            text = '\n'.join(lines)
        
        # Preserve whitespace if enabled
        if not options.preserve_whitespace:
            text = ' '.join(text.split())
        
        return text
    
    def parse_sync(self, request: ParseRequest) -> ParseResult:
        """Parse document synchronously."""
        start_time = time.time()
        
        try:
            # Decode content if it's bytes
            if isinstance(request.content, bytes):
                try:
                    text = request.content.decode('utf-8')
                except UnicodeDecodeError:
                    text = request.content.decode('latin-1')
            else:
                text = str(request.content)
            
            # Process text content
            processed_text = self._process_text_content(text, request.options or self.options)
            
            # Create pages (split by double newlines for multiple pages)
            pages = [page.strip() for page in processed_text.split('\n\n') if page.strip()]
            if not pages:
                pages = [processed_text]
            
            # Create page metadata
            page_metas = []
            for i, page_text in enumerate(pages, 1):
                page_meta = PageMeta(
                    page_number=i,
                    width=612,
                    height=792,
                    rotation=0,
                    language=self._detect_language(page_text)[0],
                    text=page_text
                )
                page_metas.append(page_meta)
            
            # No tables for simple parser
            tables = []
            
            # Calculate statistics
            total_text = '\n'.join(pages)
            char_count = len(total_text)
            token_estimate = self._estimate_tokens(total_text)
            languages = list(set([meta.language for meta in page_metas if meta.language]))
            
            parse_time = time.time() - start_time
            
            stats = ParseStats(
                char_count=char_count,
                token_estimate=token_estimate,
                table_count=len(tables),
                page_count=len(pages),
                languages=languages,
                source_name=request.filename or "document",
                checksum=self._calculate_checksum(request.content),
                parse_time=parse_time
            )
            
            return ParseResult(
                pages=pages,
                tables=tables,
                metadata=stats
            )
            
        except Exception as e:
            raise ParserError(f"Simple parsing failed: {str(e)}")


def parse_sync_simple(request: ParseRequest) -> ParseResult:
    """Simple parse function."""
    parser = SimpleParser()
    return parser.parse_sync(request) 