#!/usr/bin/env python3
"""
Test script to verify memory_full_copy.py works correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_full_copy():
    """Test the memory_full_copy.py functionality."""
    try:
        print("Testing memory_full_copy.py...")
        
        # Import from the full copy
        from memory.memory_full_copy import AsyncMemory
        from memory.models import MemorySettings
        
        print("✅ Memory full copy import successful")
        
        # Create settings
        settings = MemorySettings(
            store_backend="faiss",
            store_uri="./test_full_copy.faiss",
            embed_model="openai:text-embedding-3-small",
            chunk_max_length=1024,
            chunk_overlap=256
        )
        
        # Create memory instance
        memory = AsyncMemory(settings)
        print("✅ Memory instance created from full copy")
        
        # Test upload
        test_text = "This is a test document about artificial intelligence and machine learning."
        
        job_id = await memory.upload(
            content=test_text,
            source="test_doc.txt"
        )
        print(f"✅ Upload completed, job_id: {job_id}")
        
        # Check job status
        job = await memory.get_job_status(job_id)
        print(f"✅ Job status: {job.status.value}")
        print(f"✅ Chunks created: {len(job.chunks) if job.chunks else 0}")
        
        # Test query
        results = await memory.query("artificial intelligence", k=3)
        print(f"✅ Query successful, found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Text: {result['text'][:100]}...")
        
        print("✅ All tests passed for memory_full_copy.py!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_full_copy()) 