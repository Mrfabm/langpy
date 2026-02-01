"""
Test script to verify streaming helper functionality works with all LLM providers.
This tests OpenAI, Anthropic, and Mistral backends.
"""

import asyncio
import os
from dotenv import load_dotenv

# Import from the installed package
try:
    from sdk import agent, OpenAI, Anthropic, Mistral
    print("✅ Successfully imported all LLM backends from installed package")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

load_dotenv(override=True)

async def test_openai_streaming():
    """Test streaming with OpenAI backend."""
    print("\n" + "="*60)
    print("TESTING STREAMING WITH OPENAI BACKEND")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found - skipping OpenAI test")
        return False
    
    try:
        openai_backend = OpenAI(api_key=api_key)
        agent_interface = agent(async_backend=openai_backend.call_async)
        
        agent_instance = agent_interface.create_agent(
            name="openai_test_agent",
            instructions="You are a helpful assistant. Answer briefly.",
            input="What is the capital of France?",
            model="gpt-4o-mini",
            api_key=api_key,
            stream=True
        )
        
        print("OpenAI streaming response:")
        print("-" * 50)
        
        full_response = await agent_instance.run()
        
        print("-" * 50)
        print(f"OpenAI response: {full_response}")
        print("✅ OpenAI streaming test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI streaming test FAILED: {e}")
        return False

async def test_anthropic_streaming():
    """Test streaming with Anthropic backend."""
    print("\n" + "="*60)
    print("TESTING STREAMING WITH ANTHROPIC BACKEND")
    print("="*60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found - skipping Anthropic test")
        return False
    
    try:
        anthropic_backend = Anthropic(api_key=api_key)
        agent_interface = agent(async_backend=anthropic_backend.call_async)
        
        agent_instance = agent_interface.create_agent(
            name="anthropic_test_agent",
            instructions="You are a helpful assistant. Answer briefly.",
            input="What is the capital of Japan?",
            model="claude-3-haiku-20240307",
            api_key=api_key,
            stream=True
        )
        
        print("Anthropic streaming response:")
        print("-" * 50)
        
        full_response = await agent_instance.run()
        
        print("-" * 50)
        print(f"Anthropic response: {full_response}")
        print("✅ Anthropic streaming test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic streaming test FAILED: {e}")
        return False

async def test_mistral_streaming():
    """Test streaming with Mistral backend."""
    print("\n" + "="*60)
    print("TESTING STREAMING WITH MISTRAL BACKEND")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found - skipping Mistral test")
        return False
    
    try:
        mistral_backend = Mistral(api_key=api_key)
        agent_interface = agent(async_backend=mistral_backend.call_async)
        
        agent_instance = agent_interface.create_agent(
            name="mistral_test_agent",
            instructions="You are a helpful assistant. Answer briefly.",
            input="What is the capital of Italy?",
            model="mistral-small-latest",
            api_key=api_key,
            stream=True
        )
        
        print("Mistral streaming response:")
        print("-" * 50)
        
        full_response = await agent_instance.run()
        
        print("-" * 50)
        print(f"Mistral response: {full_response}")
        print("✅ Mistral streaming test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Mistral streaming test FAILED: {e}")
        return False

async def main():
    """Run all LLM streaming tests."""
    print("🚀 TESTING STREAMING FUNCTIONALITY WITH ALL LLM PROVIDERS")
    print("="*80)
    
    results = []
    
    # Test each LLM provider
    results.append(await test_openai_streaming())
    results.append(await test_anthropic_streaming())
    results.append(await test_mistral_streaming())
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL LLM PROVIDERS SUPPORT STREAMING WITH stream=True!")
        print("   The streaming helper functionality works universally!")
    elif passed > 0:
        print("⚠️  Some LLM providers support streaming, others may need API keys")
    else:
        print("❌ No LLM providers tested successfully - check API keys")

if __name__ == "__main__":
    asyncio.run(main()) 