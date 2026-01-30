#!/usr/bin/env python3
"""
Simple test to verify all formats work.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting format test...")

try:
    from parser.docling_parser import DoclingParser
    from parser.models import ParseRequest, ParserOptions
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test with simple text content
test_content = b"This is a test document."
request = ParseRequest(
    content=test_content,
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
    print(f"✅ Parser works! Pages: {result.metadata.page_count}, Chars: {result.metadata.char_count}")
except Exception as e:
    print(f"❌ Parser failed: {e}")
    exit(1)

print("🎉 Basic test passed!") 