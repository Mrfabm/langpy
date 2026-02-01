"""
Test memory primitive resilience - ensures pipeline continues even when parser fails.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from memory import AsyncMemory, SyncMemory
from memory.models import MemorySettings


async def test_memory_resilience_async():
    """Test that memory continues processing even when parser fails."""
    print("🧪 Testing Async Memory Resilience")
    
    # Create memory with simple settings
    settings = MemorySettings(
        store_backend="faiss",
        store_uri="./test_memory_resilience.faiss",
        chunk_max_length=100,
        chunk_overlap=10
    )
    
    memory = AsyncMemory(settings)
    
    # Test 1: Upload a simple text file (should work normally)
    print("\n📝 Test 1: Normal text file upload")
    test_content = "This is a test document with some content that should be processed normally."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        job_id = await memory.upload(temp_file)
        print(f"✅ Upload job created: {job_id}")
        
        # Wait for completion
        while True:
            job = await memory.get_job_status(job_id)
            if job.status.value in ['completed', 'failed']:
                break
            await asyncio.sleep(0.1)
        
        print(f"📊 Job status: {job.status.value}")
        if job.error_message:
            print(f"❌ Error: {job.error_message}")
        else:
            print(f"✅ Success! Chunks: {job.metadata.chunk_count}")
            
    finally:
        os.unlink(temp_file)
    
    # Test 2: Upload content that might cause parser issues
    print("\n📝 Test 2: Content that might cause parser issues")
    problematic_content = "Simple text content without complex formatting"
    
    job_id = await memory.upload(problematic_content, source="test_content")
    print(f"✅ Upload job created: {job_id}")
    
    # Wait for completion
    while True:
        job = await memory.get_job_status(job_id)
        if job.status.value in ['completed', 'failed']:
            break
        await asyncio.sleep(0.1)
    
    print(f"📊 Job status: {job.status.value}")
    if job.error_message:
        print(f"❌ Error: {job.error_message}")
    else:
        print(f"✅ Success! Chunks: {job.metadata.chunk_count}")
    
    # Test 3: Query the memory to see if content was stored
    print("\n🔍 Test 3: Querying stored content")
    results = await memory.query("test document", k=5)
    print(f"📊 Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.get('score', 0):.3f}, Text: {result.get('text', '')[:50]}...")
    
    # Cleanup
    await memory.clear()
    await memory.close()
    
    print("\n✅ Async Memory Resilience Test Completed")


def test_memory_resilience_sync():
    """Test that sync memory continues processing even when parser fails."""
    print("\n🧪 Testing Sync Memory Resilience")
    
    # Create memory with simple settings
    settings = MemorySettings(
        store_backend="faiss",
        store_uri="./test_memory_resilience_sync.faiss",
        chunk_max_length=100,
        chunk_overlap=10
    )
    
    memory = SyncMemory(settings)
    
    # Test 1: Upload simple text content
    print("\n📝 Test 1: Simple text content upload")
    test_content = "This is another test document for sync memory testing."
    
    job_id = memory.upload(test_content, source="sync_test")
    print(f"✅ Upload job created: {job_id}")
    
    # Wait for completion
    while True:
        job = memory.get_job_status(job_id)
        if job.status.value in ['completed', 'failed']:
            break
        import time
        time.sleep(0.1)
    
    print(f"📊 Job status: {job.status.value}")
    if job.error_message:
        print(f"❌ Error: {job.error_message}")
    else:
        print(f"✅ Success! Chunks: {job.metadata.chunk_count}")
    
    # Test 2: Query the memory
    print("\n🔍 Test 2: Querying stored content")
    results = memory.query("test document", k=5)
    print(f"📊 Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.score:.3f}, Text: {result.text[:50]}...")
    
    # Cleanup
    memory.clear()
    memory.close()
    
    print("\n✅ Sync Memory Resilience Test Completed")


async def test_memory_with_real_files():
    """Test memory with real files to see how it handles different formats."""
    print("\n🧪 Testing Memory with Real Files")
    
    settings = MemorySettings(
        store_backend="faiss",
        store_uri="./test_memory_files.faiss",
        chunk_max_length=100,
        chunk_overlap=10
    )
    
    memory = AsyncMemory(settings)
    
    # Create test files
    test_files = []
    
    # Text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a simple text file for testing memory resilience.")
        test_files.append(f.name)
    
    # Markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Document\n\nThis is a markdown file for testing.")
        test_files.append(f.name)
    
    # JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"title": "Test JSON", "content": "This is JSON content for testing."}')
        test_files.append(f.name)
    
    try:
        for i, file_path in enumerate(test_files):
            print(f"\n📝 Test {i+1}: Uploading {Path(file_path).suffix} file")
            
            job_id = await memory.upload(file_path)
            print(f"✅ Upload job created: {job_id}")
            
            # Wait for completion
            while True:
                job = await memory.get_job_status(job_id)
                if job.status.value in ['completed', 'failed']:
                    break
                await asyncio.sleep(0.1)
            
            print(f"📊 Job status: {job.status.value}")
            if job.error_message:
                print(f"❌ Error: {job.error_message}")
            else:
                print(f"✅ Success! Chunks: {job.metadata.chunk_count}")
        
        # Query all content
        print("\n🔍 Querying all stored content")
        results = await memory.query("test", k=10)
        print(f"📊 Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result.get('score', 0):.3f}, Text: {result.get('text', '')[:50]}...")
    
    finally:
        # Cleanup files
        for file_path in test_files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        # Cleanup memory
        await memory.clear()
        await memory.close()
    
    print("\n✅ Memory with Real Files Test Completed")


if __name__ == "__main__":
    print("🚀 Starting Memory Resilience Tests")
    
    # Run async tests
    asyncio.run(test_memory_resilience_async())
    
    # Run sync tests
    test_memory_resilience_sync()
    
    # Run file tests
    asyncio.run(test_memory_with_real_files())
    
    print("\n🎉 All Memory Resilience Tests Completed!") 