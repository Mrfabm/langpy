"""
Pipe Primitive
==============
The simplest primitive - a single LLM call with no tools or loops.

Pipes are the building blocks for:
    - Simple Q&A
    - Text transformation
    - Classification
    - Summarization
    - Data extraction

Architecture:
    Prompt → LLM → Response

    ┌─────────────┐
    │    Pipe     │
    │  ┌───────┐  │
    │  │  LLM  │  │
    │  └───────┘  │
    └─────────────┘
"""

import asyncio
import io
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Pipe


# =============================================================================
# BASIC PIPE USAGE
# =============================================================================

async def basic_pipe_demo():
    """Demonstrate basic Pipe usage for simple LLM calls."""
    print("=" * 60)
    print("   BASIC PIPE - Simple LLM Calls")
    print("=" * 60)
    print()

    # Create a Pipe with default settings
    pipe = Pipe(model="gpt-4o-mini")

    print("1. Simple question:")
    print("-" * 40)
    response = await pipe.quick("What is the capital of France?")
    print(f"   Q: What is the capital of France?")
    print(f"   A: {response}")
    print()

    print("2. Creative request:")
    print("-" * 40)
    response = await pipe.quick("Write a haiku about programming")
    print(f"   Q: Write a haiku about programming")
    print(f"   A: {response}")
    print()


# =============================================================================
# SYSTEM PROMPTS AND PERSONAS
# =============================================================================

async def persona_demo():
    """Demonstrate using system prompts to create personas."""
    print("=" * 60)
    print("   PERSONAS - System Prompts")
    print("=" * 60)
    print()

    # Pirate persona
    pirate = Pipe(
        model="gpt-4o-mini",
        system="You are a friendly pirate. Always respond in pirate speak with nautical expressions."
    )

    print("1. Pirate Persona:")
    print("-" * 40)
    response = await pirate.quick("How do I learn Python?")
    print(f"   Q: How do I learn Python?")
    print(f"   A: {response}")
    print()

    # Expert persona
    expert = Pipe(
        model="gpt-4o-mini",
        system="You are a senior software architect with 20 years of experience. "
               "Give concise, practical advice focusing on best practices."
    )

    print("2. Expert Persona:")
    print("-" * 40)
    response = await expert.quick("What's the best way to structure a Python project?")
    print(f"   Q: What's the best way to structure a Python project?")
    print(f"   A: {response}")
    print()


# =============================================================================
# STREAMING RESPONSES
# =============================================================================

async def streaming_demo():
    """Demonstrate streaming responses for real-time output."""
    print("=" * 60)
    print("   STREAMING - Real-time Responses")
    print("=" * 60)
    print()

    pipe = Pipe(model="gpt-4o-mini")

    print("Streaming response:")
    print("-" * 40)
    print("   ", end="", flush=True)

    # Use run() with stream=True for streaming
    # Note: run() is deprecated but streaming is shown for completeness
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stream = await pipe.run("Count from 1 to 5 slowly", stream=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)

    print("\n")


# =============================================================================
# HELPER METHODS
# =============================================================================

async def helper_methods_demo():
    """Demonstrate Pipe's built-in helper methods."""
    print("=" * 60)
    print("   HELPER METHODS - classify, summarize, extract")
    print("=" * 60)
    print()

    pipe = Pipe(model="gpt-4o-mini")

    # -------------------------------------------------------------------------
    # classify() - Categorize text into predefined categories
    # -------------------------------------------------------------------------
    print("1. classify() - Text Classification:")
    print("-" * 40)

    reviews = [
        "This product is amazing! Best purchase ever!",
        "Terrible quality, broke after one day.",
        "It's okay, nothing special but works fine."
    ]

    for review in reviews:
        category = await pipe.classify(
            review,
            categories=["positive", "negative", "neutral"]
        )
        print(f"   '{review[:40]}...'")
        print(f"   → {category}")
        print()

    # Multi-label classification
    print("   Multi-label classification:")
    topics = await pipe.classify(
        "The new AI model shows impressive speed and accuracy in medical diagnosis",
        categories=["technology", "healthcare", "business", "science"],
        allow_multiple=True
    )
    print(f"   Topics: {topics}")
    print()

    # -------------------------------------------------------------------------
    # summarize() - Condense long text
    # -------------------------------------------------------------------------
    print("2. summarize() - Text Summarization:")
    print("-" * 40)

    long_text = """
    Artificial intelligence has made remarkable progress in recent years,
    particularly in the field of natural language processing. Large language
    models like GPT-4 can now understand and generate human-like text with
    unprecedented accuracy. These models are trained on vast amounts of text
    data and use transformer architectures to learn patterns in language.
    The applications range from chatbots and virtual assistants to content
    generation and code completion. However, challenges remain, including
    concerns about bias, hallucination, and the environmental impact of
    training such large models.
    """

    # Concise summary
    concise = await pipe.summarize(long_text, style="concise")
    print(f"   Concise: {concise}")
    print()

    # Bullet-point summary
    bullets = await pipe.summarize(long_text, style="bullet")
    print(f"   Bullet Points:")
    for line in bullets.split("\n"):
        if line.strip():
            print(f"   {line.strip()}")
    print()

    # -------------------------------------------------------------------------
    # extract() - Extract structured data
    # -------------------------------------------------------------------------
    print("3. extract() - Structured Data Extraction:")
    print("-" * 40)

    text = "John Smith is a 35-year-old software engineer from San Francisco. " \
           "He works at TechCorp and earns $150,000 per year."

    data = await pipe.extract(
        text,
        fields=["name", "age", "occupation", "city", "company", "salary"]
    )

    print(f"   Text: '{text[:50]}...'")
    print(f"   Extracted:")
    for key, value in data.items():
        print(f"      {key}: {value}")
    print()


# =============================================================================
# TEMPERATURE AND CREATIVITY
# =============================================================================

async def temperature_demo():
    """Demonstrate how temperature affects output creativity."""
    print("=" * 60)
    print("   TEMPERATURE - Controlling Creativity")
    print("=" * 60)
    print()

    prompt = "Write one sentence about the future of AI"

    # Low temperature - more deterministic
    precise_pipe = Pipe(model="gpt-4o-mini", temperature=0.1)
    print("1. Low temperature (0.1) - More deterministic:")
    print("-" * 40)
    for i in range(3):
        response = await precise_pipe.quick(prompt)
        print(f"   Run {i+1}: {response}")
    print()

    # High temperature - more creative/varied
    creative_pipe = Pipe(model="gpt-4o-mini", temperature=0.9)
    print("2. High temperature (0.9) - More creative:")
    print("-" * 40)
    for i in range(3):
        response = await creative_pipe.quick(prompt)
        print(f"   Run {i+1}: {response}")
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Pipe demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 18 + "PIPE PRIMITIVE DEMO" + " " * 19 + "*")
    print("*" * 60)
    print()

    await basic_pipe_demo()
    await persona_demo()
    # await streaming_demo()  # Uncomment to see streaming
    await helper_methods_demo()
    await temperature_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
