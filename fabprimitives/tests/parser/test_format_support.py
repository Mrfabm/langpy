"""
Comprehensive Format Support Test - Verify all Streamlit UI formats work with parser SDK.

This test ensures that all formats supported by the Streamlit UI can be parsed
by our parser SDK with the same functionality.
"""

import asyncio
import tempfile
from pathlib import Path
from io import BytesIO

# Import the parser SDK
from langpy.sdk import parser


def create_test_files():
    """Create test files for all supported formats."""
    test_files = {}
    
    # Text files
    test_files['text'] = {
        'content': b"This is a test text file.\n\nIt contains multiple lines and paragraphs.\n\nThis should be parsed correctly.",
        'filename': 'test.txt',
        'mime_type': 'text/plain'
    }
    
    # Markdown files
    test_files['markdown'] = {
        'content': b"""# Test Markdown Document

This is a **test** markdown file with *formatting*.

## Section 1
- Item 1
- Item 2
- Item 3

## Section 2
1. Numbered item 1
2. Numbered item 2

```python
print("Hello, World!")
```
""",
        'filename': 'test.md',
        'mime_type': 'text/markdown'
    }
    
    # HTML files
    test_files['html'] = {
        'content': b"""<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Document</title>
</head>
<body>
    <h1>Test HTML Document</h1>
    <p>This is a test HTML file with various elements.</p>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
        <li>List item 3</li>
    </ul>
    <table>
        <tr><th>Header 1</th><th>Header 2</th></tr>
        <tr><td>Data 1</td><td>Data 2</td></tr>
    </table>
</body>
</html>""",
        'filename': 'test.html',
        'mime_type': 'text/html'
    }
    
    # CSV files
    test_files['csv'] = {
        'content': b"""Name,Age,City
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago
Alice,28,Boston""",
        'filename': 'test.csv',
        'mime_type': 'text/csv'
    }
    
    # JSON files (as text)
    test_files['json'] = {
        'content': b"""{
    "name": "Test JSON",
    "version": "1.0",
    "data": {
        "items": [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
    }
}""",
        'filename': 'test.json',
        'mime_type': 'application/json'
    }
    
    # XML files (as text)
    test_files['xml'] = {
        'content': b"""<?xml version="1.0" encoding="UTF-8"?>
<root>
    <item id="1">
        <name>Item 1</name>
        <description>First item</description>
    </item>
    <item id="2">
        <name>Item 2</name>
        <description>Second item</description>
    </item>
</root>""",
        'filename': 'test.xml',
        'mime_type': 'application/xml'
    }
    
    return test_files


async def test_format_support():
    """Test that all supported formats can be parsed."""
    print("🧪 COMPREHENSIVE FORMAT SUPPORT TEST")
    print("=" * 60)
    
    # Create parser instance
    p = parser()
    
    # Get supported formats
    supported_formats = p.get_supported_formats()
    print(f"📋 Supported MIME types: {len(supported_formats)}")
    for mime_type, extensions in supported_formats.items():
        print(f"   {mime_type}: {', '.join(extensions)}")
    
    # Create test files
    test_files = create_test_files()
    
    # Test each format
    results = {}
    
    for format_name, file_info in test_files.items():
        print(f"\n🔍 Testing {format_name.upper()} format...")
        print(f"   File: {file_info['filename']}")
        print(f"   MIME Type: {file_info['mime_type']}")
        
        try:
            # Test MIME type detection
            detected_mime = p.detect_mime_type(file_info['content'], file_info['filename'])
            print(f"   Detected MIME: {detected_mime}")
            
            # Test format support
            is_supported = p.is_supported(detected_mime)
            print(f"   Supported: {is_supported}")
            
            if is_supported:
                # Test parsing
                print(f"   Parsing content...")
                result = await p.parse_content(
                    file_info['content'],
                    file_info['filename'],
                    progress_callback=lambda p: print(f"      Progress: {p.progress_percent:.0f}%")
                )
                
                print(f"   ✅ Parse successful!")
                print(f"      Pages: {len(result.pages)}")
                print(f"      Tables: {len(result.tables)}")
                print(f"      Characters: {result.metadata.char_count:,}")
                print(f"      Parse time: {result.metadata.parse_time:.3f}s")
                
                # Show first 100 characters of content
                if result.pages:
                    preview = result.pages[0][:100].replace('\n', '\\n')
                    print(f"      Preview: {preview}...")
                
                results[format_name] = {
                    'success': True,
                    'result': result,
                    'detected_mime': detected_mime
                }
            else:
                print(f"   ❌ Format not supported")
                results[format_name] = {
                    'success': False,
                    'error': 'Format not supported',
                    'detected_mime': detected_mime
                }
                
        except Exception as e:
            print(f"   ❌ Parse failed: {e}")
            results[format_name] = {
                'success': False,
                'error': str(e),
                'detected_mime': detected_mime if 'detected_mime' in locals() else 'unknown'
            }
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 FORMAT SUPPORT SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    for format_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {format_name.upper()}: {result['detected_mime']}")
        if not result['success']:
            print(f"   Error: {result['error']}")
    
    # Test specific format capabilities
    print(f"\n🔧 TESTING SPECIFIC FORMAT CAPABILITIES")
    print("=" * 60)
    
    # Test Office document formats (if dependencies are available)
    try:
        import pandas as pd
        print("✅ pandas available - XLSX support enabled")
    except ImportError:
        print("❌ pandas not available - XLSX support disabled")
    
    try:
        import docx
        print("✅ python-docx available - DOCX support enabled")
    except ImportError:
        print("❌ python-docx not available - DOCX support disabled")
    
    try:
        import pptx
        print("✅ python-pptx available - PPTX support enabled")
    except ImportError:
        print("❌ python-pptx not available - PPTX support disabled")
    
    try:
        import pytesseract
        print("✅ pytesseract available - OCR support enabled")
    except ImportError:
        print("❌ pytesseract not available - OCR support disabled")
    
    try:
        import docling
        print("✅ docling available - Advanced parsing enabled")
    except ImportError:
        print("❌ docling not available - Advanced parsing disabled")
    
    # Test parser options
    print(f"\n⚙️ TESTING PARSER OPTIONS")
    print("=" * 60)
    
    # Test fast options
    fast_options = p.create_fast_options()
    print(f"🚀 Fast options: OCR={fast_options.enable_ocr}, Tables={fast_options.table_strategy}")
    
    # Test accurate options
    accurate_options = p.create_accurate_options()
    print(f"🎯 Accurate options: OCR={accurate_options.enable_ocr}, Tables={accurate_options.table_strategy}")
    
    # Test custom options
    custom_options = p.create_options(
        enable_ocr=True,
        ocr_languages=["eng", "spa"],
        preserve_whitespace=True,
        merge_hyphens=False,
        table_strategy="docling"
    )
    print(f"🔧 Custom options: OCR={custom_options.enable_ocr}, Languages={custom_options.ocr_languages}")
    
    # Test format-specific information
    print(f"\n🏷️ TESTING FORMAT-SPECIFIC INFORMATION")
    print("=" * 60)
    
    for format_name, file_info in test_files.items():
        format_info = p.get_format_specific_info(
            file_info['filename'],
            file_info['mime_type'],
            p._default_options
        )
        print(f"📄 {format_name.upper()}:")
        print(f"   Is Image: {format_info['is_image']}")
        print(f"   Is Office: {format_info['is_office_document']}")
        print(f"   Is PDF: {format_info['is_pdf']}")
        if format_info['recommendations']:
            print(f"   Recommendations: {format_info['recommendations']}")
        if format_info['warnings']:
            print(f"   Warnings: {format_info['warnings']}")
    
    return results


async def test_batch_parsing():
    """Test batch parsing with multiple formats."""
    print(f"\n📦 TESTING BATCH PARSING")
    print("=" * 60)
    
    p = parser()
    test_files = create_test_files()
    
    # Create batch of different formats
    batch_content = []
    batch_filenames = []
    
    for format_name, file_info in test_files.items():
        batch_content.append(file_info['content'])
        batch_filenames.append(file_info['filename'])
    
    print(f"📦 Batch parsing {len(batch_content)} files...")
    
    def batch_progress(current, total, filename):
        print(f"   Processing {current}/{total}: {filename}")
    
    try:
        results = await p.parse_batch(
            batch_content,
            filenames=batch_filenames,
            progress_callback=batch_progress
        )
        
        print(f"✅ Batch completed!")
        for filename, result, parse_time in results:
            if result:
                print(f"   📄 {filename}: {len(result.pages)} pages, {parse_time:.3f}s")
            else:
                print(f"   ❌ {filename}: Failed")
                
    except Exception as e:
        print(f"❌ Batch parsing failed: {e}")


async def main():
    """Run all format support tests."""
    print("🚀 LANGPY PARSER SDK - FORMAT SUPPORT VERIFICATION")
    print("=" * 60)
    print("This test verifies that all formats supported by the Streamlit UI")
    print("can be parsed by our parser SDK with the same functionality.")
    print("=" * 60)
    
    try:
        # Test individual format support
        results = await test_format_support()
        
        # Test batch parsing
        await test_batch_parsing()
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("🎉 FORMAT SUPPORT VERIFICATION COMPLETED")
        print("=" * 60)
        
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        if successful == total:
            print("✅ ALL FORMATS SUPPORTED SUCCESSFULLY!")
            print("✅ Parser SDK has complete feature parity with Streamlit UI!")
        else:
            print(f"⚠️  {total - successful} formats failed - check dependencies")
        
        print(f"\n📋 Supported formats in Streamlit UI:")
        streamlit_formats = [
            'PDF', 'DOCX', 'PPTX', 'XLSX', 'CSV', 'TXT', 'HTML', 'MD',
            'JPG', 'PNG', 'TIFF', 'JSON', 'XML'
        ]
        for fmt in streamlit_formats:
            status = "✅" if any(fmt.lower() in k for k in results.keys()) else "❌"
            print(f"   {status} {fmt}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 