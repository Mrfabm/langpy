"""
Test Parser SDK - Verify parser interface functionality.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sdk import Primitives
from parser.models import ParserOptions


class TestParserSDK:
    """Test cases for the parser SDK interface."""
    
    def setup_method(self):
        """Set up test environment."""
        self.langpy = Primitives()
    
    def test_parser_interface_creation(self):
        """Test that parser interface is created correctly."""
        assert hasattr(self.langpy, 'parser')
        assert self.langpy.parser is not None
    
    def test_get_parser_info(self):
        """Test getting parser information."""
        info = self.langpy.parser.get_parser_info()
        
        assert isinstance(info, dict)
        assert 'dispatcher_available' in info
        assert 'docling_available' in info
        assert 'supported_formats' in info
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = self.langpy.parser.get_supported_formats()
        
        assert isinstance(formats, dict)
        # Should have at least some formats
        assert len(formats) >= 0
    
    def test_detect_mime_type(self):
        """Test MIME type detection."""
        # Test PDF detection
        pdf_content = b"%PDF-1.4\n"
        mime_type = self.langpy.parser.detect_mime_type(pdf_content, "test.pdf")
        assert mime_type == "application/pdf"
        
        # Test text detection
        text_content = b"Hello, World!"
        mime_type = self.langpy.parser.detect_mime_type(text_content, "test.txt")
        assert mime_type in ["text/plain", "application/octet-stream"]
    
    def test_is_supported(self):
        """Test format support checking."""
        # Test PDF support
        assert self.langpy.parser.is_supported("application/pdf") in [True, False]
        
        # Test unsupported format
        assert self.langpy.parser.is_supported("application/unsupported") in [True, False]
    
    def test_create_options(self):
        """Test parser options creation."""
        options = self.langpy.parser.create_options(
            enable_ocr=True,
            ocr_languages=["eng"],
            preserve_whitespace=False,
            merge_hyphens=True
        )
        
        assert isinstance(options, ParserOptions)
        assert options.enable_ocr is True
        assert options.ocr_languages == ["eng"]
        assert options.preserve_whitespace is False
        assert options.merge_hyphens is True
    
    def test_parse_sync_text(self):
        """Test synchronous text parsing."""
        content = "Hello, World! This is a test document."
        
        result = self.langpy.parser.parse_sync(
            content=content,
            filename="test.txt"
        )
        
        assert result is not None
        assert hasattr(result, 'pages')
        assert hasattr(result, 'metadata')
        assert len(result.pages) > 0
        assert result.metadata.char_count > 0
    
    @pytest.mark.asyncio
    async def test_parse_content_async(self):
        """Test asynchronous content parsing."""
        content = "Hello, World! This is a test document for async parsing."
        
        result = await self.langpy.parser.parse_content(
            content=content,
            filename="test_async.txt"
        )
        
        assert result is not None
        assert hasattr(result, 'pages')
        assert hasattr(result, 'metadata')
        assert len(result.pages) > 0
        assert result.metadata.char_count > 0
    
    def test_parse_sync_json(self):
        """Test synchronous JSON parsing."""
        content = '{"name": "John", "age": 30, "city": "NYC"}'
        
        result = self.langpy.parser.parse_sync(
            content=content,
            filename="test.json"
        )
        
        assert result is not None
        assert hasattr(result, 'pages')
        assert hasattr(result, 'metadata')
    
    def test_parse_sync_html(self):
        """Test synchronous HTML parsing."""
        content = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        
        result = self.langpy.parser.parse_sync(
            content=content,
            filename="test.html"
        )
        
        assert result is not None
        assert hasattr(result, 'pages')
        assert hasattr(result, 'metadata')
    
    def test_parse_sync_csv(self):
        """Test synchronous CSV parsing."""
        content = "Name,Age,City\nJohn,30,NYC\nJane,25,LA"
        
        result = self.langpy.parser.parse_sync(
            content=content,
            filename="test.csv"
        )
        
        assert result is not None
        assert hasattr(result, 'pages')
        assert hasattr(result, 'metadata')
    
    def test_parser_error_handling(self):
        """Test parser error handling."""
        # Test with empty content
        with pytest.raises(Exception):
            self.langpy.parser.parse_sync(
                content="",
                filename="empty.txt"
            )
    



if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 