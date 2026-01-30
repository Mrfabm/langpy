#!/usr/bin/env python3
"""
Comprehensive test for the Memory Primitive.

This test covers:
- Async and sync memory operations
- Document upload and processing
- Vector search and retrieval
- Metadata operations
- Error handling
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import pytest

# Test imports
from memory import AsyncMemory, SyncMemory, create_memory, create_sync_memory
from memory.models import MemorySettings, DocumentMetadata
from sdk.memory_interface import MemoryInterface


@pytest.mark.asyncio
async def test_async_memory_basic():
    """Test basic async memory operations."""
    print("🧪 Testing Async Memory Basic Operations...")
    
    # Create memory instance
    settings = MemorySettings(
        name="test_async",
        store_backend="faiss",
        embed_model="openai:text-embedding-3-small",  # Use smaller model for testing
        chunk_max_length=1024,
        chunk_overlap=100
    )
    
    memory = AsyncMemory(settings)
    
    # Test 1: Upload simple text
    print("  📝 Testing text upload...")
    test_text = """
    This is a test document about artificial intelligence and machine learning.
    AI has become increasingly important in modern technology.
    Machine learning algorithms can process large amounts of data efficiently.
    """
    
    job_id = await memory.upload(
        content=test_text,
        source="test_doc_1",
        custom_metadata={"category": "technology", "author": "test_user"}
    )
    print(f"    ✅ Upload job created: {job_id}")
    
    # Test 2: Check job status
    print("  🔍 Checking job status...")
    job = await memory.get_job_status(job_id)
    if job:
        print(f"    ✅ Job status: {job.status}")
    else:
        print("    ❌ Job not found")
    
    # Test 3: Query memory
    print("  🔎 Testing memory query...")
    results = await memory.query("artificial intelligence", k=3)
    print(f"    ✅ Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"      Result {i+1}: Score={result.score:.3f}, Source={result.source}")
    
    # Test 4: Get stats
    print("  📊 Getting memory stats...")
    stats = await memory.get_stats()
    print(f"    ✅ Stats: {stats}")
    
    print("✅ Async Memory Basic Test Completed\n")


def test_sync_memory_basic():
    """Test basic sync memory operations."""
    print("🧪 Testing Sync Memory Basic Operations...")
    
    # Create memory instance
    settings = MemorySettings(
        name="test_sync",
        store_backend="faiss",
        embed_model="openai:text-embedding-3-small",
        chunk_max_length=1024,
        chunk_overlap=100
    )
    
    memory = SyncMemory(settings)
    
    # Test 1: Upload simple text
    print("  📝 Testing text upload...")
    test_text = """
    Python is a versatile programming language used for web development,
    data science, and automation. It has a simple syntax and large ecosystem.
    Many developers prefer Python for its readability and extensive libraries.
    """
    
    job_id = memory.upload(
        content=test_text,
        source="test_doc_2",
        custom_metadata={"category": "programming", "language": "python"}
    )
    print(f"    ✅ Upload job created: {job_id}")
    
    # Test 2: Check job status
    print("  🔍 Checking job status...")
    job = memory.get_job_status(job_id)
    if job:
        print(f"    ✅ Job status: {job.status}")
    else:
        print("    ❌ Job not found")
    
    # Test 3: Query memory
    print("  🔎 Testing memory query...")
    results = memory.query("Python programming", k=3)
    print(f"    ✅ Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"      Result {i+1}: Score={result.score:.3f}, Source={result.source}")
    
    # Test 4: Get stats
    print("  📊 Getting memory stats...")
    stats = memory.get_stats()
    print(f"    ✅ Stats: {stats}")
    
    print("✅ Sync Memory Basic Test Completed\n")


@pytest.mark.asyncio
async def test_memory_interface():
    """Test the SDK memory interface."""
    print("🧪 Testing Memory Interface...")
    
    # Create interface
    interface = MemoryInterface()
    
    # Test 1: Add text via interface
    print("  📝 Testing interface add_text...")
    test_text = """
    Natural language processing (NLP) is a field of artificial intelligence
    that focuses on the interaction between computers and human language.
    NLP techniques are used in chatbots, translation services, and text analysis.
    """
    
    result = await interface.add_text(
        text=test_text,
        source="test_doc_3"
    )
    print(f"    ✅ Interface add_text result: {result}")
    
    # Test 2: Query via interface
    print("  🔎 Testing interface query...")
    results = await interface.query("natural language processing", k=2)
    print(f"    ✅ Interface query found {len(results)} results")
    
    # Test 3: Get stats via interface
    print("  📊 Getting interface stats...")
    stats = await interface.get_stats()
    print(f"    ✅ Interface stats: {stats}")
    
    print("✅ Memory Interface Test Completed\n")


@pytest.mark.asyncio
async def test_file_upload():
    """Test uploading files to memory."""
    print("🧪 Testing File Upload...")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        This is a test file for memory upload.
        It contains information about data structures and algorithms.
        Arrays, linked lists, and trees are fundamental data structures.
        Sorting and searching are essential algorithms in computer science.
        """)
        temp_file_path = f.name
    
    try:
        # Create memory instance
        memory = create_memory("test_file_upload", embed_model="openai:text-embedding-3-small")
        
        # Upload file
        print("  📁 Uploading file...")
        job_id = await memory.upload(
            content=Path(temp_file_path),
            source="test_file.txt",
            custom_metadata={"file_type": "text", "test": True}
        )
        print(f"    ✅ File upload job: {job_id}")
        
        # Query the uploaded content
        print("  🔎 Querying uploaded file content...")
        results = await memory.query("data structures", k=2)
        print(f"    ✅ Found {len(results)} results from file")
        
    finally:
        # Clean up
        os.unlink(temp_file_path)
    
    print("✅ File Upload Test Completed\n")


@pytest.mark.asyncio
async def test_metadata_operations():
    """Test metadata operations."""
    print("🧪 Testing Metadata Operations...")
    
    memory = create_memory("test_metadata", embed_model="openai:text-embedding-3-small")
    
    # Upload documents with metadata
    print("  📝 Uploading documents with metadata...")
    
    docs = [
        ("Document about cats", "cat_doc", {"animal": "cat", "category": "pets"}),
        ("Document about dogs", "dog_doc", {"animal": "dog", "category": "pets"}),
        ("Document about cars", "car_doc", {"vehicle": "car", "category": "transport"})
    ]
    
    job_ids = []
    for content, source, metadata in docs:
        job_id = await memory.upload(content, source, metadata)
        job_ids.append(job_id)
        print(f"    ✅ Uploaded {source}")
    
    # Test metadata filtering
    print("  🔍 Testing metadata filtering...")
    
    # Get all pet documents
    pet_results = await memory.get_by_metadata({"category": "pets"})
    print(f"    ✅ Found {len(pet_results)} pet documents")
    
    # Get cat documents
    cat_results = await memory.get_by_metadata({"animal": "cat"})
    print(f"    ✅ Found {len(cat_results)} cat documents")
    
    # Test metadata updates
    print("  🔄 Testing metadata updates...")
    updated = await memory.update_metadata(
        updates={"category": "domestic_animals"},
        metadata_filter={"category": "pets"}
    )
    print(f"    ✅ Updated {updated} documents")
    
    # Test deletion
    print("  🗑️ Testing deletion...")
    deleted = await memory.delete_by_filter(metadata_filter={"vehicle": "car"})
    print(f"    ✅ Deleted {deleted} car documents")
    
    print("✅ Metadata Operations Test Completed\n")


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling scenarios."""
    print("🧪 Testing Error Handling...")
    
    memory = create_memory("test_errors", embed_model="openai:text-embedding-3-small")
    
    # Test 1: Invalid content
    print("  ❌ Testing invalid content...")
    try:
        await memory.upload(content=None, source="invalid")
        print("    ❌ Should have raised error")
    except Exception as e:
        print(f"    ✅ Caught expected error: {type(e).__name__}")
    
    # Test 2: Invalid job ID
    print("  ❌ Testing invalid job ID...")
    job = await memory.get_job_status("invalid_job_id")
    if job is None:
        print("    ✅ Correctly returned None for invalid job")
    else:
        print("    ❌ Should have returned None")
    
    # Test 3: Query with no results
    print("  ❌ Testing query with no results...")
    results = await memory.query("completely unrelated query that should not match anything", k=5)
    print(f"    ✅ Query returned {len(results)} results (expected 0)")
    
    print("✅ Error Handling Test Completed\n")


@pytest.mark.asyncio
async def test_convenience_functions():
    """Test convenience functions."""
    print("🧪 Testing Convenience Functions...")
    
    # Test create_memory
    print("  🏭 Testing create_memory...")
    async_memory = create_memory(
        name="convenience_test",
        backend="faiss",
        embed_model="openai:text-embedding-3-small",
        chunk_max_length=1024
    )
    print("    ✅ Async memory created successfully")
    
    # Test create_sync_memory
    print("  🏭 Testing create_sync_memory...")
    sync_memory = create_sync_memory(
        name="convenience_sync_test",
        backend="faiss",
        embed_model="openai:text-embedding-3-small",
        chunk_max_length=1024
    )
    print("    ✅ Sync memory created successfully")
    
    # Test basic operations with convenience-created instances
    print("  📝 Testing operations with convenience instances...")
    
    # Async
    job_id = await async_memory.upload("Test content for convenience function", "convenience_test")
    print(f"    ✅ Async upload job: {job_id}")
    
    # Sync
    job_id_sync = sync_memory.upload("Test content for sync convenience function", "convenience_sync_test")
    print(f"    ✅ Sync upload job: {job_id_sync}")
    
    print("✅ Convenience Functions Test Completed\n")


async def main():
    """Run all memory primitive tests."""
    print("🚀 Starting Memory Primitive Tests\n")
    print("=" * 50)
    
    try:
        # Run all tests
        await test_async_memory_basic()
        await test_memory_interface()
        await test_file_upload()
        await test_metadata_operations()
        await test_error_handling()
        await test_convenience_functions()
        
        print("=" * 50)
        print("🎉 All Memory Primitive Tests Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if we're in the virtual environment
    import sys
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider activating .venv\\Scripts\\activate")
    
    # Run sync test first
    test_sync_memory_basic()
    # Run async tests
    asyncio.run(main()) 