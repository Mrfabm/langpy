#!/usr/bin/env python3
"""
Simple test script to check basic memory functionality with FAISS backend
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_simple_memory():
    """Test basic memory functionality with FAISS backend."""
    try:
        print("Testing simple memory with FAISS backend...")
        
        # Import memory components directly
        from memory.async_memory import AsyncMemory
        from memory.models import MemorySettings
        
        # Create settings with FAISS backend
        settings = MemorySettings(
            store_backend="faiss",
            store_uri="./test_memory.faiss",
            embed_model="openai:text-embedding-3-small",
            chunk_max_length=1024,
            chunk_overlap=256,
            enable_reranking=False,
            enable_hybrid_search=False
        )
        
        print("✅ Settings created")
        
        # Create memory instance
        memory = AsyncMemory(settings)
        print("✅ Memory instance created")
        
        # Test upload
        print("Testing upload...")
        test_text = "This is a test document about artificial intelligence and machine learning."
        
        job_id = await memory.upload(
            content=test_text,
            source="test_doc.txt",
            background=False
        )
        print(f"✅ Upload completed, job_id: {job_id}")
        
        # Check job status
        job = await memory.get_job_status(job_id)
        print(f"✅ Job status: {job.status.value}")
        print(f"✅ Chunks created: {len(job.chunks) if job.chunks else 0}")
        
        # Test query
        print("Testing query...")
        results = await memory.query("artificial intelligence", k=3)
        print(f"✅ Query successful, found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Text: {result['text'][:100]}...")
        
        # Test stats
        print("Testing stats...")
        stats = await memory.get_stats()
        print(f"✅ Stats: {stats.total_documents} docs, {stats.total_chunks} chunks")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_simple_memory()) 