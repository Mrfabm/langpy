#!/usr/bin/env python3
"""
Test script to verify threaded memory operations work correctly.
"""

import asyncio
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from memory import AsyncMemory, MemorySettings


def run_async_in_thread(coro_func, *args, **kwargs):
    """Run async operation in dedicated thread."""
    executor = ThreadPoolExecutor(max_workers=1)
    
    def worker():
        """Worker function that runs in dedicated thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create the coroutine
            coro = coro_func(*args, **kwargs)
            # Run it
            result = loop.run_until_complete(coro)
            return result
        except Exception as e:
            return f"ERROR: {e}"
        finally:
            loop.close()
    
    # Submit to thread pool
    future = executor.submit(worker)
    result = future.result(timeout=60)  # 60 second timeout
    
    if isinstance(result, str) and result.startswith("ERROR:"):
        print(f"Operation failed: {result[7:]}")
        return None
    
    return result


def test_threaded_memory():
    """Test memory operations using threaded approach."""
    print("🧪 Testing Threaded Memory Operations")
    print("=" * 50)
    
    try:
        # Test memory initialization in thread
        print("1. Testing memory initialization...")
        
        def init_memory():
            """Initialize memory in a dedicated thread with its own event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                settings = MemorySettings(
                    store_backend="pgvector",
                    store_uri="postgresql://postgres:postgres@localhost:5433/postgres",
                    embed_model="openai:text-embedding-3-small",
                    chunk_max_length=1024,
                    chunk_overlap=256,
                    enable_reranking=True,
                    enable_hybrid_search=True,
                    rerank_top_k=20,
                    hybrid_weight=0.7
                )
                return AsyncMemory(settings)
            except Exception as e:
                return f"ERROR: {e}"
        
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(init_memory)
        result = future.result(timeout=30)
        
        if isinstance(result, str) and result.startswith("ERROR:"):
            print(f"❌ Failed to initialize memory: {result[7:]}")
            return
        else:
            memory = result
            print("✅ Memory initialized successfully")
        
        # Test upload in thread
        print("\n2. Testing upload in thread...")
        test_text = """
        Artificial Intelligence and Machine Learning
        
        Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines 
        that work and react like humans. Machine learning is a subset of AI that enables computers to learn 
        and improve from experience without being explicitly programmed.
        """
        
        job_id = run_async_in_thread(memory.upload, content=test_text, meta={"category": "test"}, background=False)
        
        if job_id:
            print(f"✅ Upload returned job_id: {job_id}")
            
            # Test get job status in thread
            print("\n3. Testing get job status in thread...")
            job = run_async_in_thread(memory.get_job_status, job_id)
            if job:
                print(f"✅ Job status: {job.status.value}")
                print(f"✅ Job completed_at: {job.completed_at}")
                print(f"✅ Chunks created: {len(job.chunks) if job.chunks else 0}")
            else:
                print("❌ Failed to get job status")
        else:
            print("❌ Upload failed - no job ID returned")
        
        # Test get stats in thread
        print("\n4. Testing get stats in thread...")
        stats = run_async_in_thread(memory.get_stats)
        if stats:
            print(f"✅ Stats retrieved: {stats.total_documents} documents, {stats.total_chunks} chunks")
        else:
            print("❌ Failed to get stats")

        # Test query in thread
        print("\n5. Testing query in thread...")
        results = run_async_in_thread(
            memory.query_advanced,
            query="machine learning",
            k=3,
            min_score=0.0
        )
        if results:
            print(f"✅ Query returned {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Text: {result['text'][:100]}...")
        else:
            print("❌ Query returned no results or failed.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_threaded_memory() 