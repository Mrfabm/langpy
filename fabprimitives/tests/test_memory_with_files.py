#!/usr/bin/env python3
"""
Test Memory Primitive with Actual Files.

This test creates actual files and tests the memory primitive's
upload -> parser -> chunker -> embed -> store pipeline.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Test imports
from memory import AsyncMemory, SyncMemory, create_memory, create_sync_memory
from memory.models import MemorySettings


async def test_memory_with_files():
    """Test memory primitive with actual files."""
    print("🧪 Testing Memory Primitive with Files...")
    
    # Create memory instance
    settings = MemorySettings(
        name="test_files",
        store_backend="faiss",
        embed_model="openai:text-embedding-3-small",
        chunk_max_length=1024,
        chunk_overlap=100
    )
    
    memory = AsyncMemory(settings)
    
    # Create test files
    test_files = []
    
    # Test 1: Create a text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        This is a test document about artificial intelligence and machine learning.
        AI has become increasingly important in modern technology.
        Machine learning algorithms can process large amounts of data efficiently.
        Neural networks are a key component of modern AI systems.
        Deep learning has revolutionized computer vision and natural language processing.
        """)
        test_files.append(f.name)
    
    # Test 2: Create a markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""
        # Python Programming Guide
        
        Python is a versatile programming language used for:
        - Web development
        - Data science
        - Automation
        - Machine learning
        
        ## Key Features
        - Simple syntax
        - Large ecosystem
        - Extensive libraries
        - Cross-platform compatibility
        """)
        test_files.append(f.name)
    
    # Test 3: Create a JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("""
        {
            "title": "Data Science Concepts",
            "topics": [
                "Statistical Analysis",
                "Data Visualization",
                "Machine Learning",
                "Big Data Processing"
            ],
            "description": "Comprehensive guide to data science methodologies and tools."
        }
        """)
        test_files.append(f.name)
    
    try:
        # Upload each file
        job_ids = []
        for i, file_path in enumerate(test_files):
            print(f"  📁 Uploading file {i+1}: {Path(file_path).name}")
            job_id = await memory.upload(
                content=file_path,
                source=f"test_file_{i+1}",
                custom_metadata={"file_type": Path(file_path).suffix, "test": True}
            )
            job_ids.append(job_id)
            print(f"    ✅ Upload job created: {job_id}")
        
        # Wait a bit for processing
        print("  ⏳ Waiting for processing to complete...")
        await asyncio.sleep(5)
        
        # Check job statuses
        print("  🔍 Checking job statuses...")
        for i, job_id in enumerate(job_ids):
            job = await memory.get_job_status(job_id)
            if job:
                print(f"    File {i+1} status: {job.status}")
                if job.status.value == "completed":
                    print(f"      ✅ Successfully processed {job.metadata.chunk_count} chunks")
                elif job.status.value == "failed":
                    print(f"      ❌ Failed: {job.error_message}")
            else:
                print(f"    ❌ Job {job_id} not found")
        
        # Test queries
        print("  🔎 Testing memory queries...")
        
        # Query 1: AI/ML related
        results = await memory.query("artificial intelligence machine learning", k=3)
        print(f"    ✅ AI/ML query found {len(results)} results")
        for i, result in enumerate(results):
            print(f"      Result {i+1}: Score={result['score']:.3f}, Source={result['metadata']['source']}")
        
        # Query 2: Python related
        results = await memory.query("Python programming language", k=3)
        print(f"    ✅ Python query found {len(results)} results")
        for i, result in enumerate(results):
            print(f"      Result {i+1}: Score={result['score']:.3f}, Source={result['metadata']['source']}")
        
        # Query 3: Data science related
        results = await memory.query("data science statistical analysis", k=3)
        print(f"    ✅ Data science query found {len(results)} results")
        for i, result in enumerate(results):
            print(f"      Result {i+1}: Score={result['score']:.3f}, Source={result['metadata']['source']}")
        
        # Get stats
        print("  📊 Getting memory stats...")
        stats = await memory.get_stats()
        print(f"    ✅ Stats: {stats}")
        
    finally:
        # Clean up test files
        for file_path in test_files:
            try:
                os.unlink(file_path)
            except:
                pass
    
    print("✅ Memory with Files Test Completed\n")


async def test_memory_with_pdf():
    """Test memory primitive with a PDF file if available."""
    print("🧪 Testing Memory Primitive with PDF...")
    
    # Check if we have a test PDF
    test_pdf = r"C:\Users\USER\Desktop\AGENTS\Important\Dump Here\Amakoro Lodge Fact Sheet.pdf"
    
    if not os.path.exists(test_pdf):
        print("  ⚠️ Test PDF not found, skipping PDF test")
        return
    
    # Create memory instance
    settings = MemorySettings(
        name="test_pdf",
        store_backend="faiss",
        embed_model="openai:text-embedding-3-small",
        chunk_max_length=1024,
        chunk_overlap=100
    )
    
    memory = AsyncMemory(settings)
    
    try:
        print(f"  📁 Uploading PDF: {Path(test_pdf).name}")
        job_id = await memory.upload(
            content=test_pdf,
            source="test_pdf",
            custom_metadata={"file_type": "pdf", "test": True}
        )
        print(f"    ✅ Upload job created: {job_id}")
        
        # Wait for processing
        print("  ⏳ Waiting for PDF processing to complete...")
        await asyncio.sleep(10)
        
        # Check job status
        job = await memory.get_job_status(job_id)
        if job:
            print(f"    PDF status: {job.status}")
            if job.status.value == "completed":
                print(f"      ✅ Successfully processed {job.metadata.chunk_count} chunks")
                print(f"      📄 Parsed text length: {len(job.parsed_text)} characters")
                
                # Test query on PDF content
                results = await memory.query("lodge accommodation", k=3)
                print(f"    ✅ Lodge query found {len(results)} results")
                for i, result in enumerate(results):
                    print(f"      Result {i+1}: Score={result['score']:.3f}")
            elif job.status.value == "failed":
                print(f"      ❌ Failed: {job.error_message}")
        
    except Exception as e:
        print(f"    ❌ PDF test failed: {e}")
    
    print("✅ Memory with PDF Test Completed\n")


async def main():
    """Run all memory tests with files."""
    print("🚀 Starting Memory Primitive File Tests\n")
    print("=" * 50)
    
    try:
        # Run tests
        await test_memory_with_files()
        await test_memory_with_pdf()
        
        print("=" * 50)
        print("🎉 All Memory Primitive File Tests Completed Successfully!")
        
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
    
    # Run tests
    asyncio.run(main()) 