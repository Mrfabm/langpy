#!/usr/bin/env python3
"""
Test parser with different file formats.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def test_text_file():
    """Test text file parsing."""
    print("Testing text file parsing...")
    
    content = b"This is a test text file.\nIt has multiple lines.\nAnd some content."
    
    request = ParseRequest(
        content=content,
        filename="test.txt",
        options=ParserOptions(
            enable_ocr=False,
            parse_timeout=30,
            table_strategy="none"
        )
    )
    
    try:
        parser = DoclingParser()
        result = parser.parse_sync(request)
        
        print(f"✅ Text file parsed successfully")
        print(f"   Pages: {result.metadata.page_count}")
        print(f"   Characters: {result.metadata.char_count}")
        print(f"   Parse time: {result.metadata.parse_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Text file parsing failed: {e}")
        return False

def test_csv_file():
    """Test CSV file parsing."""
    print("\nTesting CSV file parsing...")
    
    content = b"""Name,Age,City
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago"""
    
    request = ParseRequest(
        content=content,
        filename="test.csv",
        options=ParserOptions(
            enable_ocr=False,
            parse_timeout=30,
            table_strategy="none"
        )
    )
    
    try:
        parser = DoclingParser()
        result = parser.parse_sync(request)
        
        print(f"✅ CSV file parsed successfully")
        print(f"   Pages: {result.metadata.page_count}")
        print(f"   Characters: {result.metadata.char_count}")
        print(f"   Parse time: {result.metadata.parse_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ CSV file parsing failed: {e}")
        return False

def test_markdown_file():
    """Test Markdown file parsing."""
    print("\nTesting Markdown file parsing...")
    
    content = b"""# Test Document

This is a **markdown** file with some *formatting*.

## Section 1
- Item 1
- Item 2
- Item 3

## Section 2
Some more content here."""
    
    request = ParseRequest(
        content=content,
        filename="test.md",
        options=ParserOptions(
            enable_ocr=False,
            parse_timeout=30,
            table_strategy="none"
        )
    )
    
    try:
        parser = DoclingParser()
        result = parser.parse_sync(request)
        
        print(f"✅ Markdown file parsed successfully")
        print(f"   Pages: {result.metadata.page_count}")
        print(f"   Characters: {result.metadata.char_count}")
        print(f"   Parse time: {result.metadata.parse_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Markdown file parsing failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Parser with Different Formats")
    print("=" * 50)
    
    results = []
    results.append(test_text_file())
    results.append(test_csv_file())
    results.append(test_markdown_file())
    
    print("\n📊 Test Results:")
    print("-" * 20)
    success_count = sum(results)
    total_count = len(results)
    
    print(f"Passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    
    exit(0 if success_count == total_count else 1) 