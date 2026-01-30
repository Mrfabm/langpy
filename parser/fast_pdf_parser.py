"""
Fast PDF Parser - Quick PDF text extraction using PyPDF2.
"""

import hashlib
import time
from typing import List, Optional
import io

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from .models import (
    ParseResult, ParseStats, ParseRequest, ParserOptions,
    TableMeta, TableCell, PageMeta, ParserError
)


class FastPDFParser:
    """Fast PDF parser using PyPDF2 for quick text extraction."""
    
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
        """Parse PDF document synchronously."""
        if not PYPDF2_AVAILABLE:
            raise ParserError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        start_time = time.time()
        
        try:
            # Create PDF reader
            pdf_file = io.BytesIO(request.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            pages = []
            page_metas = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    # Extract text from page
                    page_text = page.extract_text()
                    
                    # Process text content
                    processed_text = self._process_text_content(page_text, request.options or self.options)
                    pages.append(processed_text)
                    
                    # Create page metadata
                    page_meta = PageMeta(
                        page_number=page_num,
                        width=612,  # Default letter size
                        height=792,
                        rotation=0,
                        language=self._detect_language(processed_text)[0],
                        text=processed_text
                    )
                    page_metas.append(page_meta)
                    
                except Exception as e:
                    # If page extraction fails, add empty page
                    pages.append(f"Page {page_num}: Text extraction failed - {str(e)}")
                    page_meta = PageMeta(
                        page_number=page_num,
                        width=612,
                        height=792,
                        rotation=0,
                        language="unknown",
                        text=""
                    )
                    page_metas.append(page_meta)
            
            # No tables for fast PDF parser
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
                source_name=request.filename or "document.pdf",
                checksum=self._calculate_checksum(request.content),
                parse_time=parse_time
            )
            
            return ParseResult(
                pages=pages,
                tables=tables,
                metadata=stats
            )
            
        except Exception as e:
            raise ParserError(f"Fast PDF parsing failed: {str(e)}")


def parse_sync_fast_pdf(request: ParseRequest) -> ParseResult:
    """Fast PDF parse function."""
    parser = FastPDFParser()
    return parser.parse_sync(request) 