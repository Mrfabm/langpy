import asyncio
from embed import get_embedder

async def test_dimension_detection():
    """Test embedding dimension detection."""
    print("🧪 Testing Embedding Dimension Detection")
    print("=" * 40)
    
    # Test with text-embedding-3-large
    embedder = get_embedder("openai:text-embedding-3-large")
    
    try:
        test_embedding = await embedder.embed(["test"])
        dimension = len(test_embedding[0])
        print(f"✅ Detected dimension: {dimension}")
        print(f"✅ Embedding model: openai:text-embedding-3-large")
        print(f"✅ First few values: {test_embedding[0][:5]}")
    except Exception as e:
        print(f"❌ Error detecting dimension: {e}")

if __name__ == "__main__":
    asyncio.run(test_dimension_detection()) 