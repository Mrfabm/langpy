#!/usr/bin/env python3
"""
Debug script to test agent functionality step by step.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_agent():
    print("1. Testing imports...")
    try:
        from sdk import agent, OpenAI
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    print("\n2. Testing OpenAI backend creation...")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"API key found: {api_key[:20]}..." if api_key else "No API key found")
        
        openai_backend = OpenAI(api_key=api_key)
        print("✅ OpenAI backend created")
        print(f"Available methods: {[m for m in dir(openai_backend) if not m.startswith('_')]}")
    except Exception as e:
        print(f"❌ OpenAI backend error: {e}")
        return
    
    print("\n3. Testing agent interface creation...")
    try:
        agent_interface = agent(
            async_backend=openai_backend.call_async,
            sync_backend=openai_backend.call_sync
        )
        print("✅ Agent interface created")
    except Exception as e:
        print(f"❌ Agent interface error: {e}")
        return
    
    print("\n4. Testing agent instance creation...")
    try:
        agent_instance = agent_interface.create_agent(
            name="test_agent",
            instructions="You are a helpful assistant.",
            input="Hello, how are you?",
            model="gpt-4o-mini",
            api_key=api_key
        )
        print("✅ Agent instance created")
    except Exception as e:
        print(f"❌ Agent instance error: {e}")
        return
    
    print("\n5. Testing agent run...")
    try:
        response = await agent_instance.run()
        print("✅ Agent run successful")
        print(f"Response type: {type(response)}")
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                if hasattr(choice.message, 'content'):
                    print(f"Content: {choice.message.content}")
                elif isinstance(choice.message, dict):
                    print(f"Content: {choice.message.get('content', 'No content')}")
                else:
                    print(f"Message: {choice.message}")
            else:
                print(f"Choice: {choice}")
        else:
            print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Agent run error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent()) 