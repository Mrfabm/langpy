"""
Tests for Docling Parser functionality.
"""

import pytest
import asyncio
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.models import (
    ParseRequest, ParseResult, ParserOptions, JobStatus,
    TableMeta, TableCell, ParseStats
)
from parser.docling_parser import DoclingParser, parse_sync, parse_async


class TestDoclingParser:
    """Test cases for Docling parser."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        try:
            parser = DoclingParser()
            assert parser is not None
            assert parser.options is not None
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_supported_formats(self):
        """Test supported formats listing."""
        try:
            parser = DoclingParser()
            formats = parser.get_supported_formats()
            assert len(formats) > 0
            
            # Check for common formats
            mime_types = [fmt.mime_type for fmt in formats]
            assert "application/pdf" in mime_types
            assert "text/plain" in mime_types
            assert "image/jpeg" in mime_types
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_sync_parsing_basic(self):
        """Test basic synchronous parsing."""
        try:
            # Create simple text content
            content = "This is a test document with some content."
            
            request = ParseRequest(
                content=content.encode(),
                filename="test.txt",
                options=ParserOptions(enable_ocr=False)
            )
            
            result = parse_sync(request)
            
            assert isinstance(result, ParseResult)
            assert len(result.pages) > 0
            assert result.metadata.char_count > 0
            assert result.metadata.source_name == "test.txt"
            
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_sync_parsing_with_tables(self):
        """Test parsing with table extraction."""
        try:
            # Create content with tables
            content = """
            Document with Tables
            
            | Name | Age | City |
            |------|-----|------|
            | Alice| 25  | NY   |
            | Bob  | 30  | LA   |
            """
            
            request = ParseRequest(
                content=content.encode(),
                filename="tables.txt",
                options=ParserOptions(
                    table_strategy="docling",
                    preserve_whitespace=True
                )
            )
            
            result = parse_sync(request)
            
            assert isinstance(result, ParseResult)
            assert len(result.pages) > 0
            
            # Note: Table extraction depends on Docling's capabilities
            # We just verify the parser doesn't crash
            
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_parser_options(self):
        """Test different parser options."""
        try:
            content = "Test content with hyphenated words like auto-matic."
            
            # Test with different options
            options_configs = [
                ParserOptions(merge_hyphens=True),
                ParserOptions(merge_hyphens=False),
                ParserOptions(preserve_whitespace=True),
                ParserOptions(strip_headers_footers=True),
            ]
            
            for options in options_configs:
                request = ParseRequest(
                    content=content.encode(),
                    filename="options_test.txt",
                    options=options
                )
                
                result = parse_sync(request)
                assert isinstance(result, ParseResult)
                assert result.metadata.char_count > 0
                
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_error_handling(self):
        """Test error handling."""
        try:
            # Test with unsupported format
            content = b"fake content"
            
            request = ParseRequest(
                content=content,
                filename="test.xyz",  # Unknown extension
                options=ParserOptions()
            )
            
            # This should either work or raise a specific error
            try:
                result = parse_sync(request)
                # If it works, that's fine too
                assert isinstance(result, ParseResult)
            except Exception as e:
                # Should be a specific parser error
                assert "parser" in str(e).lower() or "format" in str(e).lower()
                
        except ImportError:
            pytest.skip("Docling not available")
    
    @pytest.mark.asyncio
    async def test_async_parsing(self):
        """Test async parsing functionality."""
        try:
            content = "Async test document content."
            
            request = ParseRequest(
                content=content.encode(),
                filename="async_test.txt",
                options=ParserOptions(enable_ocr=False)
            )
            
            job = await parse_async(request)
            
            assert isinstance(job, ParseJob)
            assert job.id is not None
            assert job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]
            assert job.input_type == "file"
            
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_checksum_calculation(self):
        """Test checksum calculation."""
        try:
            parser = DoclingParser()
            
            content1 = b"test content"
            content2 = b"test content"
            content3 = b"different content"
            
            checksum1 = parser._calculate_checksum(content1)
            checksum2 = parser._calculate_checksum(content2)
            checksum3 = parser._calculate_checksum(content3)
            
            assert checksum1 == checksum2
            assert checksum1 != checksum3
            assert len(checksum1) == 64  # SHA256 hex length
            
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_token_estimation(self):
        """Test token estimation."""
        try:
            parser = DoclingParser()
            
            text1 = "This is a short text."
            text2 = "This is a much longer text with many more words and characters."
            
            tokens1 = parser._estimate_tokens(text1)
            tokens2 = parser._estimate_tokens(text2)
            
            assert tokens1 > 0
            assert tokens2 > tokens1
            assert tokens2 > len(text2.split())  # Should be more than word count
            
        except ImportError:
            pytest.skip("Docling not available")
    
    def test_language_detection(self):
        """Test language detection."""
        try:
            parser = DoclingParser()
            
            english_text = "This is English text."
            empty_text = ""
            
            languages1 = parser._detect_language(english_text)
            languages2 = parser._detect_language(empty_text)
            
            assert len(languages1) > 0
            assert len(languages2) > 0
            assert "en" in languages1 or "multilingual" in languages1
            
        except ImportError:
            pytest.skip("Docling not available")


class TestParserModels:
    """Test cases for parser models."""
    
    def test_parse_request_validation(self):
        """Test ParseRequest validation."""
        # Valid request with content
        request1 = ParseRequest(
            content=b"test content",
            filename="test.txt"
        )
        assert request1.content == b"test content"
        
        # Valid request with URL
        request2 = ParseRequest(
            url="https://example.com/document.pdf"
        )
        assert request2.url == "https://example.com/document.pdf"
        
        # Invalid request (no content or URL)
        with pytest.raises(ValueError):
            ParseRequest()
    
    def test_parser_options_defaults(self):
        """Test ParserOptions default values."""
        options = ParserOptions()
        
        assert options.max_file_size == 50 * 1024 * 1024
        assert options.enable_ocr is True
        assert options.ocr_languages == ["eng"]
        assert options.table_strategy == "docling"
        assert options.parse_timeout == 300
    
    def test_table_meta_matrix(self):
        """Test TableMeta matrix property."""
        # Create a simple table
        cells = [
            TableCell(value="A", row=0, col=0),
            TableCell(value="B", row=0, col=1),
            TableCell(value="C", row=1, col=0),
            TableCell(value="D", row=1, col=1),
        ]
        
        table = TableMeta(
            page_number=1,
            bounding_box=[0, 0, 100, 100],
            rows=2,
            columns=2,
            cells=cells
        )
        
        matrix = table.matrix
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert matrix[0][0] == "A"
        assert matrix[0][1] == "B"
        assert matrix[1][0] == "C"
        assert matrix[1][1] == "D"


if __name__ == "__main__":
    pytest.main([__file__]) 