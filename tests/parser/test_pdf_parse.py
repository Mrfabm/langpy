#!/usr/bin/env python3
"""
Test PDF parsing specifically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def test_pdf_parse():
    """Test PDF parsing."""
    print("Testing PDF parsing...")
    
    # Create parser with shorter timeout
    parser = DoclingParser(ParserOptions(
        enable_ocr=False,
        parse_timeout=60,  # 60 seconds timeout
        table_strategy="none"
    ))
    
    # Create a simple PDF-like content (this won't be a real PDF, just for testing)
    test_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF
"""
    
    # Create parse request
    request = ParseRequest(
        content=test_content,
        filename="test.pdf",
        options=ParserOptions(
            enable_ocr=False,
            parse_timeout=60,
            table_strategy="none"
        )
    )
    
    try:
        print("Starting PDF parse...")
        result = parser.parse_sync(request)
        
        print("✅ PDF Parse successful!")
        print(f"Pages: {result.metadata.page_count}")
        print(f"Characters: {result.metadata.char_count}")
        print(f"Parse time: {result.metadata.parse_time:.2f}s")
        
        if result.pages:
            print(f"First page content: {result.pages[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF Parse failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_parse()
    exit(0 if success else 1) 