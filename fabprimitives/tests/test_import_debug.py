#!/usr/bin/env python3
"""
Debug import issue.
"""

import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTesting imports...")

try:
    import sdk
    print(f"✅ sdk imported from: {sdk.__file__}")
except Exception as e:
    print(f"❌ sdk import error: {e}")

try:
    from sdk import agent
    print(f"✅ agent imported from: {agent.__module__}")
    print(f"✅ agent function signature: {agent.__code__.co_varnames}")
except Exception as e:
    print(f"❌ agent import error: {e}")

try:
    from sdk import OpenAI
    print(f"✅ OpenAI imported from: {OpenAI.__module__}")
except Exception as e:
    print(f"❌ OpenAI import error: {e}")

# Test the exact same code as the demo
print("\nTesting demo code...")
try:
    import os
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    openai_backend = OpenAI(api_key=api_key)
    print("✅ OpenAI backend created")
    
    agent_interface = agent(
        async_backend=openai_backend.call_async,
        sync_backend=openai_backend.call_sync
    )
    print("✅ Agent interface created")
except Exception as e:
    print(f"❌ Demo code error: {e}")
    import traceback
    traceback.print_exc() 