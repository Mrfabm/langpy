"""
Pattern 8: Memory Agent (RAG)
=============================

Retrieval-Augmented Generation using direct primitive composition.

This example shows how to BUILD a RAG system by composing primitives:
    - Memory: stores and retrieves knowledge
    - Pipe: generates responses
    - Thread: maintains conversation history

Architecture:
    Query → Memory.search() → Pipe.run(context + query) → Response

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Memory, Pipe, Thread


# =============================================================================
# PATTERN 8: RAG WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def memory_agent_pattern():
    """
    Build a RAG agent by composing Memory + Pipe + Thread primitives.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 8: MEMORY AGENT (RAG)")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Create the primitives (our building blocks)
    # =========================================================================

    memory = Memory(name="langpy_knowledge")  # Vector storage
    pipe = Pipe(model="gpt-4o-mini")          # LLM for generation
    thread = Thread()                          # Conversation history

    print("Primitives created:")
    print("  - Memory: for storing and retrieving knowledge")
    print("  - Pipe: for generating responses")
    print("  - Thread: for conversation history")
    print()

    # =========================================================================
    # STEP 2: Add knowledge to Memory primitive
    # =========================================================================

    knowledge = [
        "LangPy is a Python framework for building AI agents.",
        "LangPy has 9 primitives: Pipe, Memory, Agent, Workflow, Thread, Tools, Parser, Chunker, Embed.",
        "The Pipe primitive makes simple LLM calls without tool execution.",
        "The Memory primitive provides vector storage for RAG.",
        "The Agent primitive enables tool execution with function calling.",
        "LangPy primitives are composable like Lego blocks.",
    ]

    print("Adding knowledge to Memory...")
    await memory.add_many(knowledge)

    stats = await memory.stats()
    print(f"Memory now contains {stats.document_count} documents")
    print()

    # =========================================================================
    # STEP 3: Create a conversation thread
    # =========================================================================

    thread_id = await thread.create("RAG Demo Session", tags=["rag", "demo"])
    print(f"Created thread: {thread_id}")
    print()

    # =========================================================================
    # STEP 4: RAG Query - Compose primitives together
    # =========================================================================

    async def rag_query(question: str) -> str:
        """
        RAG query using primitive composition:
        1. Memory.search() → retrieve relevant context
        2. Pipe.run() → generate response with context
        3. Thread.add_message() → save to history
        """

        # COMPOSE: Memory retrieval
        results = await memory.search(question, limit=3)
        context = "\n".join([f"- {r.text}" for r in results])

        # COMPOSE: Build augmented prompt
        augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer based on the context. If the context doesn't contain the answer, say so."""

        # COMPOSE: Generate with Pipe
        response = await pipe.run(
            augmented_prompt,
            system="You are a helpful assistant that answers based on provided context."
        )

        # COMPOSE: Save to Thread history
        await thread.add_message(thread_id, "user", question)
        await thread.add_message(thread_id, "assistant", response.content)

        return response.content

    # =========================================================================
    # STEP 5: Run RAG queries
    # =========================================================================

    questions = [
        "What is LangPy?",
        "How many primitives does LangPy have?",
        "What is the Memory primitive used for?",
    ]

    for question in questions:
        print(f"Q: {question}")
        print("-" * 40)

        answer = await rag_query(question)
        print(f"A: {answer}")
        print()

    # =========================================================================
    # STEP 6: Show conversation history from Thread
    # =========================================================================

    print("=" * 60)
    print("Conversation history (from Thread primitive):")
    print("-" * 40)

    messages = await thread.get_messages(thread_id)
    for msg in messages[-6:]:  # Last 3 Q&A pairs
        print(f"[{msg.role}]: {msg.content[:80]}...")

    print()
    print("=" * 60)


# =============================================================================
# SIMPLE RAG FUNCTION - Minimal composition example
# =============================================================================

async def simple_rag(question: str, knowledge_base: list[str]) -> str:
    """
    Minimal RAG in just a few lines - pure primitive composition.

    Args:
        question: User's question
        knowledge_base: List of knowledge strings

    Returns:
        Generated answer
    """
    # Create primitives
    memory = Memory(name="simple_rag")
    pipe = Pipe(model="gpt-4o-mini")

    # Add knowledge
    await memory.add_many(knowledge_base)

    # Retrieve relevant context
    results = await memory.search(question, limit=3)
    context = "\n".join([r.text for r in results])

    # Generate with context
    response = await pipe.run(
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        system="Answer based on the provided context."
    )

    return response.content


# =============================================================================
# CONVERSATIONAL RAG - With history
# =============================================================================

async def conversational_rag_demo():
    """
    RAG with conversation memory - composing Memory + Pipe + Thread.
    """
    print("\n" + "=" * 60)
    print("   CONVERSATIONAL RAG")
    print("   Memory + Pipe + Thread composition")
    print("=" * 60 + "\n")

    # Primitives
    memory = Memory(name="conv_rag")
    pipe = Pipe(model="gpt-4o-mini")
    thread = Thread()

    # Setup
    await memory.add_many([
        "The weather in Tokyo is usually mild in spring.",
        "Cherry blossoms bloom in Tokyo from late March to early April.",
        "Tokyo Tower is 333 meters tall.",
    ])
    thread_id = await thread.create("Tokyo Chat")

    # Conversation loop
    questions = [
        "When do cherry blossoms bloom in Tokyo?",
        "What's a famous landmark there?",  # Uses conversation context
    ]

    for question in questions:
        # Get conversation history for context
        history = await thread.get_messages(thread_id)
        history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-4:]])

        # Retrieve from memory
        results = await memory.search(question, limit=2)
        knowledge = "\n".join([r.text for r in results])

        # Generate with both history and knowledge
        prompt = f"""Conversation history:
{history_text}

Relevant knowledge:
{knowledge}

User: {question}

Respond naturally, using both conversation context and knowledge."""

        response = await pipe.run(prompt)

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
    """Run all RAG pattern demonstrations."""
    await memory_agent_pattern()
    await conversational_rag_demo()


if __name__ == "__main__":
    asyncio.run(demo())
