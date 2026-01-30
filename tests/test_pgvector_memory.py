#!/usr/bin/env python3
"""
Test script to verify memory works correctly with pgvector backend.
This tests the fixed event loop handling with background=False.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from memory import AsyncMemory, MemorySettings


async def test_pgvector_memory():
    """Test memory with pgvector backend."""
    print("🧪 Testing Memory with pgvector Backend")
    print("=" * 50)
    
    # Create memory instance with pgvector
    settings = MemorySettings(
        store_backend="pgvector",
        store_uri="postgresql://postgres:postgres@localhost:5433/postgres",
        embed_model="openai:text-embedding-3-small",
        chunk_max_length=1024,
        chunk_overlap=256
    )
    memory = AsyncMemory(settings)
    
    # Test with simple text
    print("\n1. Testing upload with pgvector...")
    test_text = """
    Artificial Intelligence and Machine Learning
    
    Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that work and react like humans. Machine learning is a subset of AI that enables computers to learn 
    and improve from experience without being explicitly programmed.
    
    Key Applications:
    - Virtual assistants like Siri and Alexa
    - Recommendation systems on streaming platforms
    - Autonomous vehicles and robotics
    - Medical diagnosis and drug discovery
    """
    
    try:
        print("   Calling memory.upload(background=False)...")
        job_id = await memory.upload(
            content=test_text,
            source="ai_article.txt",
            meta={"category": "technology", "author": "AI Expert"},
            background=False  # Run synchronously
        )
        print(f"   ✅ Upload returned job_id: {job_id}")
        
        # Check job status immediately
        print("   Checking job status...")
        job = await memory.get_job_status(job_id)
        print(f"   ✅ Job status: {job.status.value}")
        print(f"   ✅ Job completed_at: {job.completed_at}")
        print(f"   ✅ Parsed text length: {len(job.parsed_text) if job.parsed_text else 0}")
        print(f"   ✅ Chunks created: {len(job.chunks) if job.chunks else 0}")
        
        if job.status.value == "completed":
            print("   🎉 SUCCESS: Memory upload completed successfully!")
            
            # Test querying
            print("\n2. Testing query with pgvector...")
            results = await memory.query(
                query="machine learning applications",
                k=3,
                min_score=0.5
            )
            print(f"   ✅ Query returned {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result.score:.3f}")
                print(f"      Text: {result.text[:100]}...")
                print(f"      Source: {result.metadata.source}")
        else:
            print(f"   ❌ FAILED: Job status is {job.status.value}")
            if job.error_message:
                print(f"   Error: {job.error_message}")
                
    except Exception as e:
        print(f"   ❌ Exception during upload: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with a second document
    print("\n3. Testing second document upload...")
    test_text2 = """
    Neural Networks and Deep Learning
    
    Neural networks are computational models inspired by biological neural networks. 
    They consist of interconnected nodes (neurons) that process information and can 
    learn patterns from data. Deep learning uses multiple layers of neural networks 
    to solve complex problems.
    
    Deep learning has revolutionized:
    - Computer vision and image recognition
    - Natural language processing
    - Speech recognition and synthesis
    - Game playing and strategy
    """
    
    try:
        job_id2 = await memory.upload(
            content=test_text2,
            source="neural_networks.txt",
            meta={"category": "technology", "topic": "neural_networks"},
            background=False
        )
        print(f"   ✅ Second upload completed: {job_id2}")
        
        # Test advanced query
        print("\n4. Testing advanced query...")
        results = await memory.query_advanced(
            query="deep learning applications",
            k=5,
            min_score=0.3,
            enable_reranking=False,
            enable_hybrid_search=False
        )
        print(f"   ✅ Advanced query returned {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.3f}")
            print(f"      Text: {result['text'][:80]}...")
            print(f"      Source: {result['metadata']['source']}")
            
    except Exception as e:
        print(f"   ❌ Exception during second upload: {e}")
        import traceback
        traceback.print_exc()
    
    # Get memory stats
    print("\n5. Getting memory statistics...")
    try:
        stats = await memory.get_stats()
        print(f"   ✅ Total documents: {stats.total_documents}")
        print(f"   ✅ Total chunks: {stats.total_chunks}")
        print(f"   ✅ Storage backend: {stats.storage_backend}")
        print(f"   ✅ Success rate: {stats.success_rate:.2%}")
    except Exception as e:
        print(f"   ❌ Exception getting stats: {e}")
    
    print("\n" + "=" * 50)
    print("pgvector Memory Test completed!")


if __name__ == "__main__":
    asyncio.run(test_pgvector_memory()) 