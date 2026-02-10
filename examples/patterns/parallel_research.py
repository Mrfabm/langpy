"""
Parallel Research Pattern
==========================

Research multiple topics simultaneously using map_over operator.

Pattern: map_over() with parallel execution
"""

import asyncio
import os
from dotenv import load_dotenv
from langpy import Langpy, map_over, Context

load_dotenv()


async def main():
    """Parallel research with map_over operator."""
    print("\n" + "="*70)
    print("PATTERN: Parallel Research")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Define topics to research
    topics = [
        "Python asyncio",
        "Machine learning basics",
        "API design patterns"
    ]

    print(f"\n[STARTING] Researching {len(topics)} topics in parallel...")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")

    # Create parallel research mapper
    researcher = map_over(
        items=lambda ctx: ctx.get("topics", []),
        apply=lb.agent,
        parallel=True,  # Run in parallel
        name="parallel-researcher"
    )

    # Execute
    result = await researcher.process(Context(
        topics=topics,
        input="Provide a brief 2-sentence explanation of this topic",
        model="openai:gpt-4o-mini",
        instructions="You are a technical educator. Be concise and clear."
    ))

    if result.is_success():
        ctx = result.unwrap()
        map_results = ctx.get("map_results", [])

        print(f"\n[COMPLETED] Researched {len(map_results)} topics")
        print("\n[RESULTS]")
        for i, result_ctx in enumerate(map_results, 1):
            if hasattr(result_ctx, 'response'):
                print(f"\n  Topic {i}: {topics[i-1]}")
                print(f"  {result_ctx.response[:150]}...")
    else:
        print(f"\n[FAILED] {result.error()}")

    print("\n" + "="*70)
    print("Pattern demonstrates:")
    print("  ✓ map_over() for parallel processing")
    print("  ✓ Process multiple independent tasks")
    print("  ✓ Efficient use of async/await")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
