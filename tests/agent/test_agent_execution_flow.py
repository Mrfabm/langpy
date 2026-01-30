#!/usr/bin/env python3
"""
Test agent execution flow with detailed logging.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_agent_execution_flow():
    print("🔍 Testing Agent Execution Flow")
    print("=" * 50)
    
    try:
        from sdk import agent, OpenAI
        
        # Step 1: Initialize
        print("1️⃣ Initializing Agent...")
        api_key = os.getenv("OPENAI_API_KEY")
        openai_backend = OpenAI(api_key=api_key)
        
        agent_interface = agent(
            async_backend=openai_backend.call_async,
            sync_backend=openai_backend.call_sync
        )
        print("✅ Agent interface created")
        
        # Step 2: Create Agent Instance
        print("\n2️⃣ Creating Agent Instance...")
        agent_instance = agent_interface.create_agent(
            name="flow_test_agent",
            instructions="You are a helpful assistant. Explain your thinking process.",
            input="What is 2 + 2? Please explain your reasoning.",
            model="gpt-4o-mini",
            api_key=api_key
        )
        print("✅ Agent instance created")
        print(f"   - Name: {agent_instance.name}")
        print(f"   - Model: {agent_instance.model}")
        print(f"   - Instructions: {agent_instance.instructions[:50]}...")
        
        # Step 3: Execute Agent
        print("\n3️⃣ Executing Agent...")
        print("   - Sending request to LLM...")
        print("   - Processing response...")
        print("   - Formatting output...")
        
        response = await agent_instance.run()
        print("✅ Agent execution completed")
        
        # Step 4: Process Response
        print("\n4️⃣ Processing Response...")
        choice = response.choices[0]
        msg = choice.message
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', 'No content')
        else:
            content = str(msg)
        
        print("📝 Response:")
        print(f"   {content}")
        
        # Step 5: Test Streaming
        print("\n5️⃣ Testing Streaming Flow...")
        print("   - Creating streaming generator...")
        print("   - Processing chunks...")
        
        chunk_count = 0
        async for chunk in agent_instance.stream():
            chunk_count += 1
            if chunk_count == 1:
                print("   - First chunk received")
            elif chunk_count % 5 == 0:
                print(f"   - Chunk {chunk_count} received")
        
        print(f"✅ Streaming completed ({chunk_count} chunks)")
        
        print("\n🎉 Agent Execution Flow Test PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Agent Execution Flow Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_execution_flow()) 