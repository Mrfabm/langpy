import asyncio
from langpy.sdk import parser
from chunker import AsyncChunker

PDF_PATH = r"C:\Users\USER\Desktop\AGENTS\Important\Dump Here\Amakoro Lodge Fact Sheet.pdf"

async def main():
    parser_instance = parser()
    result = await parser_instance.parse_file(PDF_PATH)
    # Assume result.pages is a list of page texts
    if hasattr(result, 'pages'):
        text = "\n".join(result.pages)
    elif hasattr(result, 'content'):
        text = result.content
    else:
        print("Parser result does not contain pages or content.")
        return
    print(f"Parsed document length: {len(text)} characters")
    chunker = AsyncChunker(chunk_max_length=1500, chunk_overlap=300)
    chunks = await chunker.chunk(text)
    print(f"Chunked into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1} (length {len(chunk)}):\n{chunk[:200]}\n---")
    if len(chunks) > 3:
        print(f"... and {len(chunks) - 3} more chunks.")

if __name__ == "__main__":
    asyncio.run(main()) 