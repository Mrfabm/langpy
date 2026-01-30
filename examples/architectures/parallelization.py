"""
Pattern 4: Parallelization
==========================

Run multiple LLM calls simultaneously for speed or diverse perspectives.

This example shows how to BUILD parallel processing by composing primitives:
    - Pipe: Multiple instances run concurrently
    - asyncio.gather: Python's native parallelism

Architecture:
    Input → [Pipe.run(), Pipe.run(), Pipe.run()] → Aggregate → Output

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Pipe


# =============================================================================
# PATTERN 4: PARALLELIZATION WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def parallelization_pattern():
    """
    Build parallel processing by composing Pipe primitives with asyncio.gather.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 4: PARALLELIZATION")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Create Pipe primitives (our building blocks)
    # =========================================================================

    pipe = Pipe(model="gpt-4o-mini")

    print("Pipe primitive created for parallel execution")
    print()

    # =========================================================================
    # STEP 2: Define parallel tasks as prompts
    # =========================================================================

    topic = "Adopting AI in small businesses"

    tasks = {
        "optimist": f"Analyze the POSITIVE aspects and opportunities of: {topic}",
        "critic": f"Analyze the RISKS and potential problems with: {topic}",
        "pragmatist": f"Give a BALANCED, practical analysis of: {topic}",
    }

    print(f"Topic: {topic}")
    print(f"Running {len(tasks)} perspectives in PARALLEL...")
    print()

    # =========================================================================
    # STEP 3: Run all tasks in parallel using asyncio.gather
    # =========================================================================

    # COMPOSE: Create coroutines for parallel execution
    async def run_perspective(name: str, prompt: str) -> tuple[str, str]:
        response = await pipe.run(prompt, system=f"You are a {name}. Be thorough.")
        return name, response.content

    # Run all in parallel
    results = await asyncio.gather(*[
        run_perspective(name, prompt)
        for name, prompt in tasks.items()
    ])

    # Convert to dict
    results_dict = {name: content for name, content in results}

    # =========================================================================
    # STEP 4: Show individual results
    # =========================================================================

    print("Individual perspectives:")
    for name, content in results_dict.items():
        print(f"\n  {name.upper()}:")
        print(f"    {content[:150]}...")

    # =========================================================================
    # STEP 5: Aggregate results with another Pipe call
    # =========================================================================

    print("\n" + "-" * 40)
    print("Aggregating perspectives...")

    all_perspectives = "\n\n".join(
        f"**{name}:**\n{content}"
        for name, content in results_dict.items()
    )

    merged = await pipe.run(
        f"""Merge these different perspectives into one comprehensive analysis:

Topic: {topic}

Perspectives:
{all_perspectives}

Create a balanced synthesis:""",
        system="You synthesize multiple viewpoints into balanced analysis."
    )

    print("\nMerged analysis:")
    print(merged.content[:400] + "...")
    print()

    print("=" * 60)


# =============================================================================
# SIMPLE PARALLEL FUNCTION - Minimal composition example
# =============================================================================

async def simple_parallel(prompts: list[str]) -> list[str]:
    """
    Minimal parallel execution - pure primitive composition.

    Args:
        prompts: List of prompts to run in parallel

    Returns:
        List of responses
    """
    pipe = Pipe(model="gpt-4o-mini")

    # Run all prompts in parallel
    results = await asyncio.gather(*[pipe.run(p) for p in prompts])

    return [r.content for r in results]


# =============================================================================
# MULTI-PERSPECTIVE ANALYSIS
# =============================================================================

async def multi_perspective_demo():
    """
    Multi-perspective analysis with parallel execution.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   MULTI-PERSPECTIVE ANALYSIS")
    print("   Parallel Pipe execution")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    topic = "Remote work policies"

    # Define perspectives
    perspectives = {
        "employee": "You represent employee interests. Focus on work-life balance.",
        "manager": "You represent management. Focus on productivity and coordination.",
        "HR": "You represent HR. Focus on policy and compliance.",
    }

    print(f"Topic: {topic}")
    print(f"Running {len(perspectives)} perspectives in parallel...\n")

    # COMPOSE: Parallel Pipe calls
    async def get_perspective(role: str, system: str) -> tuple[str, str]:
        response = await pipe.run(
            f"Give your perspective on: {topic}",
            system=system
        )
        return role, response.content

    results = await asyncio.gather(*[
        get_perspective(role, system)
        for role, system in perspectives.items()
    ])

    for role, content in results:
        print(f"{role.upper()}: {content[:100]}...")
        print()


# =============================================================================
# ENSEMBLE CLASSIFICATION
# =============================================================================

async def ensemble_classification_demo():
    """
    Ensemble classification with voting.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   ENSEMBLE CLASSIFICATION")
    print("   Parallel classifiers with voting")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    text = "The product arrived damaged and customer service was unhelpful."
    categories = ["positive", "negative", "neutral"]

    print(f"Text: {text}")
    print(f"Categories: {categories}\n")

    # COMPOSE: Multiple classification strategies in parallel
    strategies = {
        "direct": f"Classify as {categories}: {text}\n\nCategory:",
        "reasoning": f"Think step by step. Classify as {categories}: {text}\n\nReasoning and category:",
        "keywords": f"What sentiment keywords appear? Classify as {categories}: {text}\n\nCategory:",
    }

    async def classify(name: str, prompt: str) -> tuple[str, str]:
        response = await pipe.run(prompt, system="Respond with the category.")
        return name, response.content

    results = await asyncio.gather(*[
        classify(name, prompt)
        for name, prompt in strategies.items()
    ])

    # Show individual results
    print("Individual classifiers:")
    votes = []
    for name, content in results:
        # Extract category from response
        category = "neutral"
        for cat in categories:
            if cat in content.lower():
                category = cat
                break
        votes.append(category)
        print(f"  {name}: {category}")

    # Majority vote
    from collections import Counter
    winner = Counter(votes).most_common(1)[0][0]
    print(f"\nEnsemble result (majority vote): {winner}")


# =============================================================================
# SPEED OPTIMIZATION - Independent tasks
# =============================================================================

async def speed_optimization_demo():
    """
    Speed optimization through parallel independent tasks.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   SPEED OPTIMIZATION")
    print("   Parallel independent tasks")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    text = "LangPy is a Python framework for building AI agents using composable primitives."

    # Independent analysis tasks
    tasks = [
        ("summary", f"Summarize in one sentence: {text}"),
        ("keywords", f"Extract 5 keywords: {text}"),
        ("questions", f"Generate 2 questions about: {text}"),
        ("translation", f"Translate to Spanish: {text}"),
    ]

    print(f"Input: {text[:50]}...")
    print(f"Running {len(tasks)} independent tasks in parallel...\n")

    import time
    start = time.time()

    # COMPOSE: All Pipe calls run in parallel
    async def run_task(name: str, prompt: str) -> tuple[str, str]:
        response = await pipe.run(prompt)
        return name, response.content

    results = await asyncio.gather(*[run_task(n, p) for n, p in tasks])

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s (parallel)\n")

    for name, content in results:
        print(f"{name}: {content[:80]}...")
    print()


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all parallelization demonstrations."""
    await parallelization_pattern()
    await multi_perspective_demo()
    await ensemble_classification_demo()
    await speed_optimization_demo()


if __name__ == "__main__":
    asyncio.run(demo())
