#!/usr/bin/env python3
"""
Test async agent functionality.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_async_agent():
    print("Testing async agent functionality...")
    
    try:
        from sdk import agent, OpenAI
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No OPENAI_API_KEY found in environment")
            return
        
        print(f"✅ API key found: {api_key[:20]}...")
        
        # Create OpenAI backend
        openai_backend = OpenAI(api_key=api_key)
        print("✅ OpenAI backend created")
        
        # Create agent interface with async backend
        agent_interface = agent(
            async_backend=openai_backend.call_async,
            sync_backend=openai_backend.call_sync
        )
        print("✅ Agent interface created with async backend")
        
        # Create agent instance
        agent_instance = agent_interface.create_agent(
            name="async_test_agent",
            instructions="You are a helpful assistant. Respond briefly.",
            input="What is 2+2?",
            model="gpt-4o-mini",
            api_key=api_key
        )
        print("✅ Agent instance created")
        
        # Test async run
        print("🔄 Running agent asynchronously...")
        response = await agent_instance.run()
        print("✅ Async run completed")
        
        # Print response
        choice = response.choices[0]
        msg = choice.message
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', 'No content')
        else:
            content = str(msg)
        
        print(f"📝 Response: {content}")
        
        # Test multiple concurrent runs
        print("\n🔄 Testing concurrent async runs...")
        tasks = []
        for i in range(3):
            task = agent_instance.run(input=f"Count to {i+1}")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        print(f"✅ Completed {len(responses)} concurrent runs")
        
        for i, response in enumerate(responses):
            choice = response.choices[0]
            msg = choice.message
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', 'No content')
            else:
                content = str(msg)
            print(f"  Response {i+1}: {content[:50]}...")
        
        print("\n✅ All async tests passed!")
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_async_agent()) 