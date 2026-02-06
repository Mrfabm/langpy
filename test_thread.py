"""Test Thread primitive directly."""
import asyncio
from langpy import Langpy
import os
from dotenv import load_dotenv

load_dotenv()

async def test_thread():
    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    print("Testing Thread primitive...")
    print("-" * 50)

    # Test 1: Create thread
    print("\n[TEST 1] Creating thread...")
    response = await lb.thread.create(metadata={"role": "test"})
    print(f"  Success: {response.success}")
    print(f"  Thread ID: {response.thread_id}")
    print(f"  Type of thread_id: {type(response.thread_id)}")
    print(f"  Response: {response}")

    if response.success and response.thread_id:
        thread_id = response.thread_id

        # Test 2: Append messages
        print(f"\n[TEST 2] Appending messages to thread {thread_id}...")
        append_response = await lb.thread.append(
            thread_id=thread_id,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        print(f"  Success: {append_response.success}")
        print(f"  Error: {append_response.error}")

        # Test 3: List messages
        print(f"\n[TEST 3] Listing messages from thread {thread_id}...")
        list_response = await lb.thread.list(thread_id=thread_id)
        print(f"  Success: {list_response.success}")
        print(f"  Messages count: {len(list_response.messages) if list_response.messages else 0}")
        if list_response.messages:
            for i, msg in enumerate(list_response.messages):
                print(f"    [{i+1}] {msg.get('role')}: {msg.get('content')}")
    else:
        print("\n[ERROR] Thread creation failed!")
        print(f"  Response object: {response}")
        print(f"  Success: {response.success}")
        print(f"  Thread ID: {response.thread_id}")
        print(f"  Error: {response.error if hasattr(response, 'error') else 'No error'}")

if __name__ == "__main__":
    asyncio.run(test_thread())
