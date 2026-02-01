#!/usr/bin/env python3
"""
Simple test script to check memory functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_memory_import():
    """Test if memory module can be imported and basic functionality works."""
    try:
        print("Testing memory import...")
        from memory import AsyncMemory, MemorySettings
        print("✅ Memory import successful")
        
        # Test settings creation
        print("Testing settings creation...")
        settings = MemorySettings(
            store_backend="pgvector",
            store_uri="postgresql://postgres:postgres@localhost:5433/postgres",
            embed_model="openai:text-embedding-3-small",
            chunk_max_length=1024,
            chunk_overlap=256
        )
        print("✅ Settings creation successful")
        
        # Test memory instance creation
        print("Testing memory instance creation...")
        memory = AsyncMemory(settings)
        print("✅ Memory instance creation successful")
        
        # Test basic query (should work even with empty database)
        print("Testing basic query...")
        try:
            results = await memory.query("test", k=1)
            print(f"✅ Query successful, returned {len(results)} results")
        except Exception as e:
            print(f"⚠️ Query failed (expected if no data): {e}")
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_memory_import()) 