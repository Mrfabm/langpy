import asyncio
from sdk import embed
from dotenv import load_dotenv
import os

load_dotenv(override=True)

async def test_embed():
    print("Testing embed functionality...")
    
    embed_instance = embed()
    
    # Test with simple text
    texts = ["Hello world", "This is a test"]
    
    try:
        embeddings = await embed_instance.embed_texts(texts)
        print(f"Successfully generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First embedding preview: {embeddings[0][:5]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_embed()) 