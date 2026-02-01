#!/usr/bin/env python3
"""
Test streaming agent functionality.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_streaming_agent():
    print("Testing streaming agent functionality...")
    
    try:
        from sdk import OpenAI
        from agent import AsyncAgent
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No OPENAI_API_KEY found in environment")
            return
        
        print(f"✅ API key found: {api_key[:20]}...")
        
        # Create OpenAI backend
        openai_backend = OpenAI(api_key=api_key)
        print("✅ OpenAI backend created")
        
        # Create AsyncAgent directly (bypassing SDK wrapper)
        agent = AsyncAgent(async_llm=openai_backend.call_async)
        print("✅ AsyncAgent created")
        
        # Test streaming run
        print("🔄 Running agent with streaming...")
        print("📝 Streaming response:")
        print("-" * 50)
        
        async for chunk in agent.run(
            model="gpt-4o-mini",
            input="Explain the benefits of streaming responses in AI applications.",
            instructions="You are a helpful assistant. Respond in a detailed way.",
            stream=True,
            apiKey=api_key
        ):
            if hasattr(chunk, 'choices') and chunk.choices:
                for choice in chunk.choices:
                    if hasattr(choice, 'delta') and choice.delta:
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            print(choice.delta.content, end='', flush=True)
                        elif isinstance(choice.delta, dict) and 'content' in choice.delta:
                            print(choice.delta['content'], end='', flush=True)
        
        print("\n" + "-" * 50)
        print("✅ Streaming completed!")
        
        # Test non-streaming for comparison
        print("\n🔄 Running agent without streaming...")
        response = await agent.run(
            model="gpt-4o-mini",
            input="Explain the benefits of streaming responses in AI applications.",
            instructions="You are a helpful assistant. Respond in a detailed way.",
            stream=False,
            apiKey=api_key
        )
        
        choice = response.choices[0]
        msg = choice.message
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', 'No content')
        else:
            content = str(msg)
        
        print(f"📝 Non-streaming response length: {len(content)} characters")
        
        print("\n✅ All streaming tests passed!")
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming_agent()) 