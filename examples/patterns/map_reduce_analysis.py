"""
Map-Reduce Analysis Pattern
============================

Process multiple items in parallel (map), then aggregate results (reduce).

Pattern: map_over() → reduce()
"""

import asyncio
import os
from dotenv import load_dotenv
from langpy import Langpy, map_over, reduce, pipeline, Context

load_dotenv()


async def main():
    """Map-reduce pattern for comprehensive analysis."""
    print("\n" + "="*70)
    print("PATTERN: Map-Reduce Analysis")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Topics to analyze
    topics = ["Python", "JavaScript", "Rust"]

    print(f"\n[PHASE 1: MAP] Analyzing {len(topics)} languages in parallel...")

    # Map: Analyze each language
    mapper = map_over(
        items=lambda ctx: ctx.get("topics", []),
        apply=lb.agent,
        parallel=True,
        name="language-analyzer"
    )

    # Reduce: Synthesize findings
    reducer = reduce(
        inputs=lambda ctx: [
            r.response if hasattr(r, 'response') else str(r)
            for r in ctx.get("map_results", [])
        ],
        combine=lambda results: {
            "summary": "\n\n".join(f"Language {i+1}:\n{r[:200]}..."
                                   for i, r in enumerate(results)),
            "count": len(results),
            "total_length": sum(len(r) for r in results)
        },
        name="synthesizer"
    )

    # Compose into map-reduce pipeline
    map_reduce = pipeline(mapper, reducer, name="map-reduce-analysis")

    # Execute
    result = await map_reduce.process(Context(
        topics=topics,
        input="Analyze this programming language: strengths, use cases, and ecosystem",
        model="openai:gpt-4o-mini",
        instructions="Be concise but comprehensive."
    ))

    if result.is_success():
        ctx = result.unwrap()
        reduced_result = ctx.get("reduce_result", {})

        print(f"\n[PHASE 2: REDUCE] Combined {reduced_result.get('count', 0)} analyses")
        print(f"[STATS] Total content: {reduced_result.get('total_length', 0)} chars")

        print("\n[FINAL RESULT]")
        print(reduced_result.get("summary", "No summary available"))
    else:
        print(f"\n[FAILED] {result.error()}")

    print("\n" + "="*70)
    print("Pattern demonstrates:")
    print("  ✓ map_over() for parallel processing")
    print("  ✓ reduce() for aggregation")
    print("  ✓ pipeline() for composition")
    print("  ✓ Classic map-reduce pattern")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
