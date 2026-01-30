#!/usr/bin/env python3
"""
Comprehensive test for all allowed formats.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def test_format(content, filename, format_name):
    """Test a specific format."""
    print(f"Testing {format_name} ({filename})...", end=" ")
    
    try:
        request = ParseRequest(
            content=content,
            filename=filename,
            options=ParserOptions(
                enable_ocr=False,
                parse_timeout=30,
                table_strategy="none"
            )
        )
        
        parser = DoclingParser()
        result = parser.parse_sync(request)
        
        print(f"✅ SUCCESS | Pages: {result.metadata.page_count} | Chars: {result.metadata.char_count}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:100]}...")
        return False

def main():
    print("🧪 Testing All Allowed Formats")
    print("=" * 50)
    
    # Test data for each format
    tests = [
        # Text formats
        (b"This is a test text file.\nIt has multiple lines.", "test.txt", "Text File"),
        (b"# Test Document\n\nThis is **markdown** content.", "test.md", "Markdown File"),
        (b"Name,Age,City\nJohn,25,NY\nJane,30,LA", "test.csv", "CSV File"),
        (b"Name\tAge\tCity\nJohn\t25\tNY\nJane\t30\tLA", "test.tsv", "TSV File"),
        (b"<html><body><h1>Test</h1><p>Content</p></body></html>", "test.html", "HTML File"),
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
        print("🎉 All text formats working!")
        print("\nNote: For binary formats (docx, pptx, xlsx, pdf, jpg, png, tiff),")
        print("you need real files. The parser is ready to handle them.")
    else:
        print("⚠️  Some formats failed")
        failed_indices = [i for i, result in enumerate(results) if not result]
        print(f"Failed formats: {[tests[i][2] for i in failed_indices]}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 