"""
Pattern 1: Augmented LLM
========================

The foundational pattern - an LLM enhanced with retrieval, tools, and memory.

This example shows how to BUILD an augmented LLM by composing primitives:
    - Agent: LLM with tool execution
    - Memory: Long-term knowledge storage
    - Thread: Conversation history
    - tool decorator: External capabilities

Architecture:
    Query → Memory.search() + Thread.get_messages() + Agent.run(tools) → Response

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Agent, Memory, Thread, Pipe, tool


# =============================================================================
# STEP 1: Define tools using the @tool decorator primitive
# =============================================================================

@tool(
    "search_web",
    "Search the web for information",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
)
def search_web(query: str) -> str:
    """Simulated web search."""
    return f"Search results for '{query}': [Simulated results about {query}]"


@tool(
    "get_current_time",
    "Get the current date and time",
    {"type": "object", "properties": {}}
)
def get_current_time() -> str:
    """Get current time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool(
    "calculate",
    "Perform mathematical calculations",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
)
def calculate(expression: str) -> str:
    """Safe calculator."""
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        try:
            return f"Result: {eval(expression)}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Error: Invalid expression"


# =============================================================================
# PATTERN 1: AUGMENTED LLM WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def augmented_llm_pattern():
    """
    Build an augmented LLM by composing Agent + Memory + Thread + tools.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 1: AUGMENTED LLM")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 2: Create the primitives (our building blocks)
    # =========================================================================

    # Agent primitive with tools
    agent = Agent(
        model="gpt-4o-mini",
        tools=[search_web, get_current_time, calculate],
        temperature=0.7
    )

    # Memory primitive for knowledge
    memory = Memory(name="augmented_llm_kb")

    # Thread primitive for conversation history
    thread = Thread()

    print("Primitives created:")
    print("  - Agent: LLM with tools (search, time, calculate)")
    print("  - Memory: for storing and retrieving knowledge")
    print("  - Thread: for conversation history")
    print()

    # =========================================================================
    # STEP 3: Add knowledge to Memory primitive
    # =========================================================================

    knowledge = [
        "LangPy is a Python framework for building AI agents.",
        "LangPy has 9 primitives: Pipe, Memory, Agent, Workflow, Thread, Tools, Parser, Chunker, Embed.",
        "LangPy is designed to be composable like Lego blocks.",
    ]

    print("Adding knowledge to Memory...")
    await memory.add_many(knowledge)
    print(f"Added {len(knowledge)} facts to memory")
    print()

    # =========================================================================
    # STEP 4: Create a conversation thread
    # =========================================================================

    thread_id = await thread.create("Augmented LLM Session", tags=["augmented-llm"])
    print(f"Created thread: {thread_id}")
    print()

    # =========================================================================
    # STEP 5: Chat function - compose all primitives together
    # =========================================================================

    async def chat(message: str) -> str:
        """
        Chat with augmented capabilities - composing all primitives:
        1. Memory.search() → retrieve relevant knowledge
        2. Thread.get_messages() → get conversation history
        3. Agent.run() → process with tools available
        4. Thread.add_message() → save to history
        """

        # COMPOSE: Retrieve from Memory
        context = ""
        try:
            results = await memory.search(message, limit=3)
            if results:
                context = "\n\nRelevant knowledge:\n" + "\n".join(f"- {r.text}" for r in results)
        except Exception:
            pass

        # COMPOSE: Get history from Thread
        history = []
        try:
            messages = await thread.get_messages(thread_id, limit=10)
            history = [{"role": m.role, "content": m.content} for m in messages]
        except Exception:
            pass

        # Build system prompt with context
        system = """You are a helpful AI assistant with access to:
- Tools for web search, calculations, and time
- A knowledge memory for context
- Conversation history for continuity

Use your tools when needed. Be helpful and accurate."""
        if context:
            system += context

        # COMPOSE: Save user message to Thread
        await thread.add_message(thread_id, "user", message)

        # COMPOSE: Run Agent with tools
        response = await agent.run(
            history + [{"role": "user", "content": message}],
            system=system
        )

        content = response.content or "I processed your request."

        # COMPOSE: Save response to Thread
        await thread.add_message(thread_id, "assistant", content)

        return content

    # =========================================================================
    # STEP 6: Test the augmented LLM
    # =========================================================================

    queries = [
        "What is LangPy?",           # Uses Memory
        "What time is it?",          # Uses tool
        "Calculate 15 * 7 + 23",     # Uses tool
    ]

    for query in queries:
        print(f"User: {query}")
        print("-" * 40)
        response = await chat(query)
        print(f"Assistant: {response}")
        print()

    # =========================================================================
    # STEP 7: Show conversation history from Thread
    # =========================================================================

    print("=" * 60)
    print("Conversation History (from Thread primitive):")
    messages = await thread.get_messages(thread_id)
    for msg in messages:
        print(f"  [{msg.role}]: {msg.content[:50]}...")

    print()
    print("=" * 60)


# =============================================================================
# SIMPLE AUGMENTED LLM - Minimal composition example
# =============================================================================

async def simple_augmented_llm(query: str, knowledge: list[str]) -> str:
    """
    Minimal augmented LLM - pure primitive composition.

    Args:
        query: User's question
        knowledge: List of knowledge facts

    Returns:
        Augmented response
    """
    # Create primitives
    agent = Agent(model="gpt-4o-mini", tools=[calculate, get_current_time])
    memory = Memory(name="simple_augmented")

    # Add knowledge
    await memory.add_many(knowledge)

    # Retrieve context
    results = await memory.search(query, limit=3)
    context = "\n".join([r.text for r in results])

    # Run agent with context
    response = await agent.run(
        f"Context:\n{context}\n\nQuestion: {query}",
        system="Answer using the context and your tools."
    )

    return response.content


# =============================================================================
# CONVERSATIONAL AUGMENTED LLM
# =============================================================================

async def conversational_demo():
    """
    Conversational augmented LLM with memory.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   CONVERSATIONAL AUGMENTED LLM")
    print("   Agent + Memory + Thread composition")
    print("=" * 60 + "\n")

    # Primitives
    pipe = Pipe(model="gpt-4o-mini")
    memory = Memory(name="conv_augmented")
    thread = Thread()

    # Setup
    await memory.add_many([
        "The Eiffel Tower is 330 meters tall.",
        "The Eiffel Tower was built in 1889.",
        "The Eiffel Tower is located in Paris, France.",
    ])
    thread_id = await thread.create("Eiffel Tower Chat")

    # Multi-turn conversation
    questions = [
        "How tall is the Eiffel Tower?",
        "When was it built?",  # Uses conversation context
        "Where is it?",        # Uses conversation context
    ]

    for question in questions:
        # Get history
        history = await thread.get_messages(thread_id)
        history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-4:]])

        # Retrieve from memory
        results = await memory.search(question, limit=2)
        knowledge = "\n".join([r.text for r in results])

        # Generate
        prompt = f"""Conversation:
{history_text}

Knowledge:
{knowledge}

User: {question}"""

        response = await pipe.run(prompt, system="Answer based on knowledge and conversation context.")

        # Save to thread
        await thread.add_message(thread_id, "user", question)
        await thread.add_message(thread_id, "assistant", response.content)

        print(f"User: {question}")
        print(f"Assistant: {response.content}")
        print()


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all augmented LLM demonstrations."""
    await augmented_llm_pattern()
    await conversational_demo()


if __name__ == "__main__":
    asyncio.run(demo())
