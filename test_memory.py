"""Test Memory primitive after fixes."""
import asyncio
import os
from dotenv import load_dotenv
from langpy import Langpy

load_dotenv()

async def test_memory():
    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    print("Testing Memory primitive...")
    print("-" * 50)

    # Test 1: Add documents
    print("\n[TEST 1] Adding documents...")
    documents = [
        {"content": "Python is a programming language."},
        {"content": "JavaScript runs in browsers."},
        {"content": "TypeScript adds types to JavaScript."}
    ]

    response = await lb.memory.add(documents=documents)
    print(f"  Success: {response.success}")
    print(f"  Count: {response.count}")
    if not response.success:
        print(f"  Error: {response.error}")

    # Test 2: Retrieve documents
    print("\n[TEST 2] Retrieving documents...")
    response = await lb.memory.retrieve(
        query="What is Python?",
        top_k=2
    )
    print(f"  Success: {response.success}")
    if response.success and response.documents:
        print(f"  Found {len(response.documents)} documents:")
        for i, doc in enumerate(response.documents):
            print(f"    [{i+1}] Score: {doc.get('score', 0):.3f}")
            print(f"        Content: {doc.get('content', 'N/A')[:60]}...")
    else:
        print(f"  Error: {response.error}")

    # Test 3: Stats
    print("\n[TEST 3] Memory stats...")
    response = await lb.memory.stats()
    print(f"  Success: {response.success}")
    if response.success:
        print(f"  Count: {response.count}")
        if response.documents:
            print(f"  Stats: {response.documents[0]}")

if __name__ == "__main__":
    asyncio.run(test_memory())
