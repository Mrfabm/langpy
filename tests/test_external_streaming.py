"""
Test script to verify streaming helper functionality works in external projects.
This simulates how the package would work when installed in another project.
"""

import asyncio
import os
from dotenv import load_dotenv

# Simulate importing from the installed package
try:
    from sdk import agent, OpenAI
    print("✅ Successfully imported from installed package")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

load_dotenv(override=True)

async def test_streaming():
    """Test the streaming functionality with stream=True during agent creation."""
    print("\n" + "="*60)
    print("TESTING STREAMING WITH stream=True DURING AGENT CREATION")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    try:
        # Create backend and agent interface
        openai_backend = OpenAI(api_key=api_key)
        agent_interface = agent(async_backend=openai_backend.call_async)
        
        # Create agent with stream=True
        agent_instance = agent_interface.create_agent(
            name="test_agent",
            instructions="You are a helpful assistant. Answer briefly.",
            input="What is 2+2?",
            model="gpt-4o-mini",
            api_key=api_key,
            stream=True  # This should automatically use the helper function
        )
        
        print("Streaming response:")
        print("-" * 50)
        
        # Just call run() - it should automatically handle streaming
        full_response = await agent_instance.run()
        
        print("-" * 50)
        print("Streaming completed!")
        print(f"Full response length: {len(full_response)} characters")
        print(f"Response content: {full_response}")
        
        print("\n✅ Streaming test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Streaming test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming()) 