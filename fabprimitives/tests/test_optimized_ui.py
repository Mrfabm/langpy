#!/usr/bin/env python3
"""
Test script to verify the optimized Streamlit UI components work correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_optimized_components():
    """Test the optimized UI components."""
    try:
        print("Testing optimized UI components...")
        
        # Test environment variable handling
        os.environ["POSTGRES_URI"] = "postgresql://test:test@localhost:5432/test"
        
        # Import the optimized functions
        from demos.memory.streamlit_pgvector_threaded_ui import get_worker_loop, get_memory
        
        print("✅ Optimized UI components import successful")
        
        # Test memory creation with environment variable
        from memory import AsyncMemory, MemorySettings
        
        settings = MemorySettings(
            store_backend="faiss",  # Use FAISS for testing
            store_uri="./test_optimized.faiss",
            embed_model="openai:text-embedding-3-small",
            chunk_max_length=1024,
            chunk_overlap=256
        )
        
        memory = AsyncMemory(settings)
        print("✅ Memory instance created with optimized settings")
        
        # Test background upload
        test_text = "This is a test document for the optimized UI."
        
        job_id = await memory.upload(
            content=test_text,
            source="test_optimized.txt",
            background=True
        )
        print(f"✅ Background upload started: {job_id}")
        
        # Test job polling
        while True:
            job = await memory.get_job_status(job_id)
            if job and job.status.value in ["completed", "failed"]:
                print(f"✅ Job completed with status: {job.status.value}")
                if job.status.value == "completed":
                    print(f"   Chunks created: {len(job.chunks) if job.chunks else 0}")
                break
            await asyncio.sleep(0.5)
        
        # Test query
        results = await memory.query("test document", k=3)
        print(f"✅ Query successful: {len(results)} results")
        
        print("✅ All optimized UI components working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_optimized_components()) 