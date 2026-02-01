#!/usr/bin/env python3
"""
Test with real file content for all formats.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def create_real_csv():
    """Create a real CSV file content."""
    return b"""Name,Age,City,Occupation
John Doe,30,New York,Engineer
Jane Smith,25,Los Angeles,Designer
Bob Johnson,35,Chicago,Manager
Alice Brown,28,Boston,Developer"""

def create_real_html():
    """Create a real HTML file content."""
    return b"""<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Welcome to Test Page</h1>
    <p>This is a test HTML document with some content.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
</body>
</html>"""

def create_real_markdown():
    """Create a real Markdown file content."""
    return b"""# Test Document

This is a **markdown** document with various formatting.

## Section 1
- List item 1
- List item 2
- List item 3

## Section 2
Some more content with *italic* and `code`.

### Subsection
More content here."""

def create_real_text():
    """Create a real text file content."""
    return b"""This is a test text file.

It contains multiple paragraphs and various content.

Line 1: Some basic text
Line 2: More content here
Line 3: Final line of content

End of document."""

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

def run_tests():
    """Run tests for all text-based formats."""
    print("🧪 Testing All Text Formats with Real Content")
    print("=" * 50)
    
    tests = [
        (create_real_text(), "test.txt", "Text File"),
        (create_real_markdown(), "test.md", "Markdown File"),
        (create_real_csv(), "test.csv", "CSV File"),
        (create_real_html(), "test.html", "HTML File"),
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
    else:
        print("⚠️  Some formats failed")
    
    return success_count == total_count

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 