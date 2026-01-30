import asyncio
from dotenv import load_dotenv 
from sdk import agent, OpenAI
import os 

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

async def main():
    openai_backend = OpenAI(api_key=api_key)
    agent_interface = agent(async_backend=openai_backend.call_async)
    message = "What makes a great ai agent?"
    
    agent_instance = agent_interface.create_agent(
        name="my_agent",
        instructions="you are the agent answers question in a well formatted table",
        input=message,
        model="gpt-4o-mini",
        api_key=api_key,
        stream=True  # This will automatically use the helper function
    )
    
    print("Streaming response:")
    print("-" * 50)
    
    # Just call run() - it automatically handles streaming and printing!
    full_response = await agent_instance.run()
    
    print("\n" + "-" * 50)
    print("Streaming completed!")
    print(f"Full response length: {len(full_response)} characters")

if __name__ == "__main__":
    asyncio.run(main()) 