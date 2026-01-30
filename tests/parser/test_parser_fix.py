#!/usr/bin/env python3
"""
Simple test script to verify the Docling parser works correctly.
"""

import asyncio
from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

async def test_parser():
    """Test the parser with a simple text document."""
    print("Testing Docling Parser...")
    
    # Create parser
    parser = DoclingParser()
    
    # Create test content
    test_content = b"""This is a test document.

It contains multiple lines of text.

This should be parsed correctly by the Docling parser.

The parser should extract:
- Text content
- Page information
- Metadata
"""
    
    # Create parse request
    request = ParseRequest(
        content=test_content,
        filename="test.txt",
        options=ParserOptions(
            enable_ocr=False,
            preserve_whitespace=True,
            table_strategy="none"
        )
    )
    
    try:
        # Parse synchronously
        result = parser.parse_sync(request)
        
        print("✅ Parse successful!")
        print(f"Pages: {result.metadata.page_count}")
        print(f"Characters: {result.metadata.char_count}")
        print(f"Tables: {result.metadata.table_count}")
        print(f"Parse time: {result.metadata.parse_time:.2f}s")
        
        if result.pages:
            print(f"\nFirst page content (first 100 chars):")
            print(repr(result.pages[0][:100]))
        
        return True
        
    except Exception as e:
        print(f"❌ Parse failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_parser())
        exit(0 if success else 1)
    except RuntimeError as e:
        if "event loop" in str(e):
            # Try running in existing event loop
            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(test_parser())
            exit(0 if success else 1)
        else:
            raise 