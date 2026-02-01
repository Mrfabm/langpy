"""
Test Parser - Simple test script for the parser.

Tests the parser with sample files to ensure it works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from parser import AsyncParser, ParserOptions


async def test_parser():
    """Test the parser with sample files."""
    print("Testing Langpy Parser...")
    print("=" * 50)
    
    # Create parser
    parser = AsyncParser()
    
    # Test files
    test_files = [
        ("test_files/sample.txt", "text/plain"),
        ("test_files/sample.json", "application/json"),
        ("test_files/sample.html", "text/html"),
        ("test_files/sample.py", "application/x-python"),
        ("test_files/sample_content_types.xml", "application/xml"),
        ("test_files/sample.pdf", "application/pdf"),
    ]
    
    for file_path, expected_mime in test_files:
        full_path = Path(__file__).parent / file_path
        
        if full_path.exists():
            print(f"\nTesting: {file_path}")
            print("-" * 30)
            
            try:
                # Read file
                content = full_path.read_bytes()
                
                # Parse file
                result = await parser.parse(content, expected_mime, full_path.name)
                
                # Display results
                print(f"✅ Success!")
                print(f"   Parse time: {result.parse_time:.3f}s")
                print(f"   Text length: {len(result.document.text)} chars")
                print(f"   Metadata: {result.document.metadata}")
                
                # Show first 150 chars of text
                preview = result.document.text[:150]
                if len(result.document.text) > 150:
                    preview += "..."
                print(f"   Preview: {preview}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        else:
            print(f"\n⚠️  File not found: {file_path}")
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_parser()) 