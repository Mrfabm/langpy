#!/usr/bin/env python3

# Test with direct imports
from chunker.sync_chunker import SyncChunker
from chunker.async_chunker import AsyncChunker
from chunker.settings import ChunkerSettings

print("Testing with direct imports...")

try:
    # Test sync chunker
    chunker1 = SyncChunker(chunk_max_length=1500, chunk_overlap=300)
    print("✓ SyncChunker created successfully")
    
    text = "This is a sample text that will be chunked into smaller pieces. " * 10
    chunks = chunker1.chunk(text)
    print(f"✓ Sync chunking successful: {len(chunks)} chunks created")
    
    # Test async chunker
    import asyncio
    async def test_async():
        chunker2 = AsyncChunker(chunk_max_length=1500, chunk_overlap=300)
        chunks = await chunker2.chunk(text)
        print(f"✓ Async chunking successful: {len(chunks)} chunks created")
    
    asyncio.run(test_async())
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Direct import test complete.") 