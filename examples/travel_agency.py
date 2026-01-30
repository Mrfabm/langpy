"""
Travel Planning Agency - Clean SDK Demo
========================================

This example demonstrates the new clean LangPy SDK:
- Agent: For intelligent conversation with tools
- Memory: For storing travel knowledge
- Thread: For conversation management
- Pipe: For simple LLM calls

Run with: python -m langpy_sdk.examples.travel_agency
"""

import asyncio
import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from langpy_sdk import Agent, Memory, Thread, Pipe, tool


# ============================================================================
# TOOLS - Define capabilities for the agent
# ============================================================================

@tool(
    "get_destination",
    "Get information about a travel destination",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
)
def get_destination(city: str) -> str:
    """Get destination information."""
    destinations = {
        "paris": "Paris, France - The City of Light. Best time: Apr-Jun, Sep-Oct. Must see: Eiffel Tower, Louvre, Notre-Dame. Daily budget: $150",
        "tokyo": "Tokyo, Japan - A blend of traditional and ultramodern. Best time: Mar-May, Sep-Nov. Must see: Shibuya, Senso-ji, Mt Fuji. Daily budget: $120",
        "rome": "Rome, Italy - The Eternal City. Best time: Apr-Jun, Sep-Oct. Must see: Colosseum, Vatican, Trevi Fountain. Daily budget: $130",
        "bali": "Bali, Indonesia - Island of the Gods. Best time: Apr-Oct (dry season). Must see: Ubud, Tanah Lot, beaches. Daily budget: $60",
        "new york": "New York, USA - The Big Apple. Best time: Apr-Jun, Sep-Nov. Must see: Central Park, Statue of Liberty, Times Square. Daily budget: $200",
    }
    return destinations.get(city.lower(), f"No info for {city}. Try: Paris, Tokyo, Rome, Bali, New York")


@tool(
    "calculate_budget",
    "Calculate trip budget",
    {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "days": {"type": "integer"},
            "travelers": {"type": "integer", "default": 1}
        },
        "required": ["destination", "days"]
    }
)
def calculate_budget(destination: str, days: int, travelers: int = 1) -> str:
    """Calculate trip budget."""
    daily_costs = {"paris": 150, "tokyo": 120, "rome": 130, "bali": 60, "new york": 200}
    cost = daily_costs.get(destination.lower(), 100)
    total = cost * days * travelers
    return f"Budget for {days} days in {destination} ({travelers} travelers): ${total}"


@tool(
    "check_weather",
    "Check weather for a destination",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "month": {"type": "string"}
        },
        "required": ["city", "month"]
    }
)
def check_weather(city: str, month: str) -> str:
    """Check weather conditions."""
    weather = {
        ("tokyo", "april"): "Mild 10-19C, cherry blossoms!",
        ("tokyo", "july"): "Hot & humid 24-31C, rainy season",
        ("paris", "april"): "Mild 10-16C, partly cloudy",
        ("paris", "july"): "Warm 18-25C, sunny",
        ("bali", "july"): "Warm 23-29C, dry season - perfect!",
    }
    key = (city.lower(), month.lower())
    return weather.get(key, f"Weather data not available for {city} in {month}")


# ============================================================================
# TRAVEL AGENCY CLASS
# ============================================================================

class TravelAgency:
    """Travel planning agency using the clean SDK."""

    def __init__(self):
        # Create agent with tools
        self.agent = Agent(
            model="gpt-4o-mini",
            tools=[get_destination, calculate_budget, check_weather],
            temperature=0.7
        )

        # Create memory for knowledge
        self.memory = Memory(name="travel_tips")

        # Create thread manager
        self.thread = Thread()

        # Create pipe for quick responses
        self.pipe = Pipe(model="gpt-4o-mini")

        self.current_thread = None

    async def initialize(self):
        """Initialize with travel knowledge."""
        print("Initializing Travel Agency...")

        # Add travel tips to memory
        tips = [
            "Book flights 6-8 weeks in advance for best prices",
            "Always get travel insurance for international trips",
            "Learn basic phrases in the local language",
            "Pack light - you can buy essentials at your destination",
            "Keep digital copies of important documents",
        ]
        await self.memory.add_many(tips)
        print(f"  Added {len(tips)} travel tips to memory")

        # Create conversation thread
        self.current_thread = await self.thread.create(
            "Travel Planning Session",
            tags=["travel"]
        )
        print(f"  Created thread: {self.current_thread}")
        print("Ready!\n")

    async def chat(self, message: str) -> str:
        """Process a user message."""
        # Add to thread
        await self.thread.add_message(self.current_thread, "user", message)

        # Get relevant tips
        tips = await self.memory.search(message, limit=2)
        tips_context = ""
        if tips:
            tips_context = "\n\nTravel tips:\n" + "\n".join(f"- {t.text}" for t in tips)

        # Get conversation history
        history = await self.thread.get_messages(self.current_thread, limit=6)

        # Build system prompt
        system = f"""You are a helpful travel planning assistant.
Use your tools to provide accurate destination info, budgets, and weather.
Be friendly and enthusiastic about travel!{tips_context}"""

        # Run agent
        response = await self.agent.run(history, system=system)

        # Get content (handle empty responses from tool calls)
        content = response.content or "I processed your request."

        # Save response
        await self.thread.add_message(self.current_thread, "assistant", content)

        return content

    async def quick_summary(self, text: str) -> str:
        """Get a quick summary using pipe."""
        return await self.pipe.summarize(text, style="concise")


# ============================================================================
# DEMO
# ============================================================================

async def run_demo():
    """Run the travel agency demo."""
    print("=" * 60)
    print("   TRAVEL PLANNING AGENCY")
    print("   Powered by LangPy Clean SDK")
    print("=" * 60)
    print()

    agency = TravelAgency()
    await agency.initialize()

    # Demo conversations
    queries = [
        "Tell me about Tokyo as a travel destination",
        "What's the weather like in April?",
        "Calculate a budget for 5 days for 2 people",
    ]

    for query in queries:
        print(f"You: {query}")
        print("-" * 40)
        response = await agency.chat(query)
        print(f"Agent: {response}")
        print()
        print("=" * 60)
        print()

    # Show thread summary
    info = await agency.thread.get(agency.current_thread)
    print(f"\nSession: {info.name}")
    print(f"Messages: {info.message_count}")


async def interactive():
    """Run in interactive mode."""
    print("=" * 60)
    print("   TRAVEL AGENCY - Interactive Mode")
    print("   Type 'quit' to exit")
    print("=" * 60)
    print()

    agency = TravelAgency()
    await agency.initialize()

    print("Agent: Welcome! I can help you plan your next trip.")
    print("       Ask about destinations, weather, or budgets.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nSafe travels!")
                break
            if not user_input:
                continue

            response = await agency.chat(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive())
    else:
        asyncio.run(run_demo())
