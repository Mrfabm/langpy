import asyncio
from stores.pgvector_store import AsyncPGVectorStore

async def test_dimension_fix():
    """Test the dimension detection fix."""
    print("🧪 Testing Dimension Detection Fix")
    print("=" * 40)
    
    # Create pgvector store instance
    store = AsyncPGVectorStore(
        dsn="postgresql://postgres:password@localhost:5432/memory_db",
        embedding_model="openai:text-embedding-3-large"
    )
    
    print(f"Initial dim: {store.dim}")
    
    # Test dimension determination
    await store._determine_dim()
    print(f"After _determine_dim: {store.dim}")
    
    # Test with actual content
    texts = ["This is a test document for dimension detection."]
    metas = [{"source": "test", "tier": "general"}]
    
    print("\n📝 Testing add method...")
    await store.add(texts, metas)
    print("✅ Add method completed successfully!")
    
    print("\n🔍 Testing query method...")
    results = await store.query("test document", k=1)
    print(f"✅ Query completed! Found {len(results)} results")
    
    if results:
        print(f"Score: {results[0].get('score', 0):.3f}")
        print(f"Text: {results[0].get('text', '')[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_dimension_fix()) 