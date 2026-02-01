import asyncio
from sdk import chunker, parser, embed 
from dotenv import load_dotenv
import os 

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

async def main():
    print("Testing SDK integration with chunker, parser, and embed...")
    
    # Test parser
    print("\n1. Testing parser...")
    parser_instance = parser()
    text = await parser_instance.parse_file(r"C:\Users\USER\Desktop\AGENTS\Important\Dump Here\Amakoro Lodge Fact Sheet.pdf")
    
    # FIX: ParseResult has 'pages' attribute, not 'text'
    # Join all pages into one text for chunking
    all_text = "\n".join(text.pages)
    print(f"Parsed text length: {len(all_text)} characters")
    print(f"Number of pages: {len(text.pages)}")
    
    # Test chunker
    print("\n2. Testing chunker...")
    chunker_instance = chunker()
    chunks = await chunker_instance.chunk_text(all_text)  # Use chunk_text(), not chunk()
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk preview: {chunks[0][:100]}...")
    
    # Test embed
    print("\n3. Testing embed...")
    embed_instance = embed()
    embeddings = await embed_instance.embed_texts(chunks)
    print(embeddings)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First embedding preview: {embeddings[0][:5]}...")

if __name__ == "__main__":
    asyncio.run(main()) 