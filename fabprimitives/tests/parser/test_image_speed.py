#!/usr/bin/env python3
"""
Test image parsing speed.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def create_simple_image():
    """Create a simple test image."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()

def test_image_speed():
    """Test image parsing speed."""
    print("Testing image parsing speed...")
    
    # Create test image
    image_content = create_simple_image()
    
    request = ParseRequest(
        content=image_content,
        filename="test.jpg",
        options=ParserOptions(
            enable_ocr=False,
            parse_timeout=5,
            table_strategy="none"
        )
    )
    
    try:
        parser = DoclingParser()
        start_time = time.time()
        result = parser.parse_sync(request)
        parse_time = time.time() - start_time
        
        print(f"✅ Image parsed in {parse_time:.3f}s")
        print(f"   Pages: {result.metadata.page_count}")
        print(f"   Characters: {result.metadata.char_count}")
        print(f"   Content: {result.pages[0]}")
        
        if parse_time < 0.1:
            print("🚀 Image parsing is now instant!")
        elif parse_time < 1.0:
            print("⚡ Image parsing is fast!")
        else:
            print("⚠️  Image parsing is still slow")
        
        return True
    except Exception as e:
        print(f"❌ Image parsing failed: {e}")
        return False

if __name__ == "__main__":
    success = test_image_speed()
    exit(0 if success else 1) 