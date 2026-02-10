"""
Iterative Refinement Pattern
=============================

Keep improving content until quality threshold is met.

Pattern: loop_while() with quality evaluation
"""

import asyncio
import os
from dotenv import load_dotenv
from langpy import Langpy, loop_while, Context

load_dotenv()


async def main():
    """Iterative refinement with loop_while operator."""
    print("\n" + "="*70)
    print("PATTERN: Iterative Refinement")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Create refinement loop
    refiner = loop_while(
        condition=lambda ctx: ctx.get("quality", 0.0) < 0.8 and ctx.get("iterations", 0) < 3,
        body=lb.agent,
        max_iterations=3,
        name="quality-refiner"
    )

    # Initial context
    ctx = Context(
        content="Python is a programming language.",
        quality=0.3,
        iterations=0,
        input="Improve this text to be more detailed and engaging",
        model="openai:gpt-4o-mini",
        instructions="You are a writing expert. Make improvements while keeping it concise."
    )

    print("\n[STARTING] Initial content:", ctx.get("content"))
    print(f"[STARTING] Initial quality: {ctx.get('quality')}")

    # Run refinement loop
    result = await refiner.process(ctx)

    if result.is_success():
        final_ctx = result.unwrap()
        print("\n[COMPLETED] Final content:", final_ctx.response[:200] + "...")
        print(f"[COMPLETED] Iterations: {final_ctx.get('iterations', 'N/A')}")
    else:
        print(f"\n[FAILED] {result.error()}")

    print("\n" + "="*70)
    print("Pattern demonstrates:")
    print("  ✓ loop_while() for iteration")
    print("  ✓ Quality threshold condition")
    print("  ✓ Max iterations safety limit")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
