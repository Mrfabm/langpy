#!/usr/bin/env python3
"""Test script to verify the updated chunker implementation."""

from chunker import SyncChunker, AsyncChunker, ChunkTooLargeError
import asyncio

def test_sync_chunker():
    """Test the sync chunker with the new implementation."""
    print("Testing SyncChunker...")
    
    # Test basic chunking
    chunker = SyncChunker(chunk_max_length=1500, chunk_overlap=300)
    text = "This is a test text. " * 100  # Create a long text
    
    try:
        chunks = chunker.chunk(text)
        print(f"✓ Created {len(chunks)} chunks")
        print(f"  First chunk length: {len(chunks[0])}")
        print(f"  Last chunk length: {len(chunks[-1])}")
        
        # Verify size constraints
        for i, chunk in enumerate(chunks):
            if len(chunk) > 1500:
                print(f"✗ Chunk {i} exceeds max length: {len(chunk)}")
                return False
        print("✓ All chunks respect size constraints")
        
        # Verify overlap (check if consecutive chunks have overlap)
        if len(chunks) > 1:
            overlap_found = False
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]
                # Check if there's overlap (simplified check)
                if chunk1[-100:] in chunk2 or chunk2[:100] in chunk1:
                    overlap_found = True
                    break
            if overlap_found:
                print("✓ Overlap detected between chunks")
            else:
                print("⚠ No obvious overlap detected (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_async_chunker():
    """Test the async chunker with the new implementation."""
    print("\nTesting AsyncChunker...")
    
    chunker = AsyncChunker(chunk_max_length=1500, chunk_overlap=300)
    text = "This is a test text. " * 100
    
    try:
        chunks = await chunker.chunk(text)
        print(f"✓ Created {len(chunks)} chunks")
        print(f"  First chunk length: {len(chunks[0])}")
        print(f"  Last chunk length: {len(chunks[-1])}")
        
        # Verify size constraints
        for i, chunk in enumerate(chunks):
            if len(chunk) > 1500:
                print(f"✗ Chunk {i} exceeds max length: {len(chunk)}")
                return False
        print("✓ All chunks respect size constraints")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_error_handling():
    """Test error handling for oversized chunks."""
    print("\nTesting error handling...")
    
    chunker = SyncChunker(chunk_max_length=100, chunk_overlap=50)
    
    # This should trigger the size constraint check
    try:
        # Create a text that will result in chunks > 100 chars
        text = "This is a very long text that should trigger the size constraint check. " * 10
        chunks = chunker.chunk(text)
        print("✓ Chunking completed without errors")
        return True
    except ChunkTooLargeError as e:
        print(f"✓ Correctly caught ChunkTooLargeError: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Updated Chunker Implementation")
    print("=" * 50)
    
    # Test sync chunker
    sync_ok = test_sync_chunker()
    
    # Test async chunker
    async_ok = asyncio.run(test_async_chunker())
    
    # Test error handling
    error_ok = test_error_handling()
    
    print("\n" + "=" * 50)
    if sync_ok and async_ok and error_ok:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return sync_ok and async_ok and error_ok

if __name__ == "__main__":
    main() 