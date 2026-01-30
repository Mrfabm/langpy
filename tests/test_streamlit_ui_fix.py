#!/usr/bin/env python3
"""
Test script to verify Streamlit UI memory initialization works.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from memory import AsyncMemory, MemorySettings


async def test_memory_initialization():
    """Test memory initialization with fixed settings."""
    print("🧪 Testing Memory Initialization")
    print("=" * 40)
    
    try:
        # Test with fixed settings that should work
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
        print("✅ MemorySettings created successfully")
        
        # Try to create memory instance
        memory = AsyncMemory(settings)
        print("✅ AsyncMemory instance created successfully")
        
        # Try to get stats (this is where the event loop error was happening)
        print("Testing get_stats()...")
        stats = await memory.get_stats()
        if stats:
            print(f"✅ Stats retrieved: {stats.total_documents} documents, {stats.total_chunks} chunks")
        else:
            print("⚠️ Stats returned None (backend might not be ready)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_memory_initialization()) 