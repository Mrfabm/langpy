import asyncio
from dotenv import load_dotenv 
from langpy.sdk import agent, OpenAI
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
        api_key=api_key
    )
    
    response = await agent_instance.run()
    choice = response.choices[0]
    msg = choice.message
    if hasattr(msg, 'content'):
        print(msg.content)
    elif isinstance(msg, dict):
        print(msg.get('content', 'No content'))
    else:
        print(msg)

asyncio.run(main()) 