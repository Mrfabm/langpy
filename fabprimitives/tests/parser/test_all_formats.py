#!/usr/bin/env python3
"""
Comprehensive test for all supported file formats.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def test_format(content: bytes, filename: str, format_name: str):
    """Test a specific format."""
    print(f"Testing {format_name} ({filename})...")
    
    request = ParseRequest(
        content=content,
        filename=filename,
        options=ParserOptions(
            enable_ocr=False,
            parse_timeout=30,
            table_strategy="none"
        )
    )
    
    try:
        parser = DoclingParser()
        result = parser.parse_sync(request)
        
        print(f"   ✅ {format_name} parsed successfully")
        print(f"   📄 Pages: {result.metadata.page_count}")
        print(f"   📝 Characters: {result.metadata.char_count}")
        print(f"   ⏱️  Parse time: {result.metadata.parse_time:.3f}s")
        return True
    except Exception as e:
        print(f"   ❌ {format_name} failed: {e}")
        return False

def run_all_tests():
    """Run tests for all supported formats."""
    print("🧪 Testing All Supported Formats")
    print("=" * 50)
    
    tests = [
        # Text formats
        (b"This is a test text file.\nIt has multiple lines.", "test.txt", "Text File"),
        (b"# Test Document\n\nThis is **markdown** content.", "test.md", "Markdown File"),
        (b"Name,Age,City\nJohn,25,NY\nJane,30,LA", "test.csv", "CSV File"),
        (b"Name\tAge\tCity\nJohn\t25\tNY\nJane\t30\tLA", "test.tsv", "TSV File"),
        (b"<html><body><h1>Test</h1><p>Content</p></body></html>", "test.html", "HTML File"),
        
        # Office formats (minimal valid content)
        (b"PK\x03\x04\x14\x00\x00\x00\x08\x00", "test.docx", "Word Document"),
        (b"PK\x03\x04\x14\x00\x00\x00\x08\x00", "test.pptx", "PowerPoint Document"),
        (b"PK\x03\x04\x14\x00\x00\x00\x08\x00", "test.xlsx", "Excel Document"),
        
        # Image formats (minimal valid headers)
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "test.jpg", "JPEG Image"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "test.png", "PNG Image"),
        (b"II*\x00\x08\x00\x00\x00", "test.tiff", "TIFF Image"),
    ]
    
    results = []
    for content, filename, format_name in tests:
        results.append(test_format(content, filename, format_name))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("-" * 30)
    success_count = sum(results)
    total_count = len(results)
    
    print(f"Passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All formats working!")
    else:
        print("⚠️  Some formats failed")
        failed_indices = [i for i, result in enumerate(results) if not result]
        print(f"Failed formats: {[tests[i][2] for i in failed_indices]}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 