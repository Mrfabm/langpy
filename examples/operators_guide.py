"""
LangPy Operators Guide - Working Examples
==========================================

Comprehensive examples demonstrating all LangPy operators and composition patterns.
"""

import asyncio
import os
from dotenv import load_dotenv
from langpy import (
    Langpy,
    Context,
    pipeline,
    parallel,
    when,
    branch,
    loop_while,
    map_over,
    reduce,
    retry,
    recover
)

load_dotenv()


# ============================================================================
# EXAMPLE 1: Sequential Composition (| operator)
# ============================================================================

async def example_1_sequential():
    """Sequential composition with | operator."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Sequential Composition (|)")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Simple chain: agent1 | agent2
    agent1 = lb.agent
    agent2 = lb.agent

    chain = agent1 | agent2

    result = await chain.process(Context(
        input="What is Python?",
        model="openai:gpt-4o-mini"
    ))

    if result.is_success():
        ctx = result.unwrap()
        print(f"[RESULT] {ctx.response[:200]}...")


# ============================================================================
# EXAMPLE 2: Parallel Composition (& operator)
# ============================================================================

async def example_2_parallel():
    """Parallel composition with & operator."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Parallel Composition (&)")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Three agents in parallel
    agent1 = lb.agent
    agent2 = lb.agent
    agent3 = lb.agent

    multi = agent1 & agent2 & agent3

    result = await multi.process(Context(
        input="What is AI?",
        model="openai:gpt-4o-mini"
    ))

    if result.is_success():
        ctx = result.unwrap()
        print(f"[RESULT] Got {len(ctx.response)} chars from parallel execution")


# ============================================================================
# EXAMPLE 3: Conditional (when)
# ============================================================================

async def example_3_conditional():
    """Conditional branching with when()."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Conditional (when)")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Create conditional: route based on query length
    short_answer = lb.agent
    long_answer = lb.agent

    conditional = when(
        condition=lambda ctx: len(ctx.get("input", "")) > 50,
        then_do=long_answer,
        else_do=short_answer,
        name="length-router"
    )

    # Test with short query
    result1 = await conditional.process(Context(
        input="What is AI?",
        model="openai:gpt-4o-mini"
    ))

    # Test with long query
    result2 = await conditional.process(Context(
        input="Can you explain artificial intelligence in detail with examples?" * 3,
        model="openai:gpt-4o-mini"
    ))

    print("[SHORT QUERY] Routed to short_answer")
    print("[LONG QUERY] Routed to long_answer")


# ============================================================================
# EXAMPLE 4: Multi-way Branch
# ============================================================================

async def example_4_branch():
    """Multi-way branching with branch()."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Multi-way Branch")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Create different agents for different tasks
    research_agent = lb.agent
    summary_agent = lb.agent
    translate_agent = lb.agent
    default_agent = lb.agent

    # Create router
    router = branch(
        router=lambda ctx: ctx.get("task_type", "general"),
        routes={
            "research": research_agent,
            "summarize": summary_agent,
            "translate": translate_agent,
        },
        default=default_agent,
        name="task-router"
    )

    # Test different routes
    result = await router.process(Context(
        task_type="research",
        input="Research Python async",
        model="openai:gpt-4o-mini"
    ))

    print("[ROUTED] task_type='research' → research_agent")


# ============================================================================
# EXAMPLE 5: Loop (loop_while)
# ============================================================================

async def example_5_loop():
    """Iteration with loop_while()."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Loop (loop_while)")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Create loop that refines until quality threshold
    refiner = lb.agent

    loop = loop_while(
        condition=lambda ctx: ctx.get("iterations", 0) < 2,  # Simple counter
        body=refiner,
        max_iterations=3,
        name="refine-loop"
    )

    result = await loop.process(Context(
        input="Improve: Python is good",
        model="openai:gpt-4o-mini",
        iterations=0
    ))

    if result.is_success():
        ctx = result.unwrap()
        print(f"[COMPLETED] Loop finished")


# ============================================================================
# EXAMPLE 6: Map (map_over)
# ============================================================================

async def example_6_map():
    """For-each pattern with map_over()."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Map (map_over)")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Process multiple topics in parallel
    mapper = map_over(
        items=lambda ctx: ctx.get("topics", []),
        apply=lb.agent,
        parallel=True,
        name="topic-mapper"
    )

    result = await mapper.process(Context(
        topics=["Python", "JavaScript", "Rust"],
        input="Explain this language in one sentence",
        model="openai:gpt-4o-mini"
    ))

    if result.is_success():
        ctx = result.unwrap()
        map_results = ctx.get("map_results", [])
        print(f"[COMPLETED] Processed {len(map_results)} topics in parallel")

        for i, result_ctx in enumerate(map_results):
            if hasattr(result_ctx, 'response'):
                print(f"  [{i+1}] {result_ctx.response[:80]}...")


# ============================================================================
# EXAMPLE 7: Reduce (reduce)
# ============================================================================

async def example_7_reduce():
    """Aggregation with reduce()."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Reduce")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # First, gather some data
    ctx = Context(
        research1="Finding 1: Python is versatile",
        research2="Finding 2: Python has great libraries",
        research3="Finding 3: Python is beginner-friendly"
    )

    # Reduce: combine findings
    reducer = reduce(
        inputs=["research1", "research2", "research3"],
        combine=lambda results: "\n".join(f"- {r}" for r in results),
        name="synthesizer"
    )

    result = await reducer.process(ctx)

    if result.is_success():
        combined = result.unwrap().get("reduce_result")
        print("[COMBINED RESULTS]")
        print(combined)


# ============================================================================
# EXAMPLE 8: Retry (retry)
# ============================================================================

async def example_8_retry():
    """Resilience with retry()."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Retry")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Wrap agent with retry
    resilient_agent = retry(
        primitive=lb.agent,
        max_attempts=3,
        delay=0.5,
        backoff_multiplier=2.0,
        name="resilient-agent"
    )

    result = await resilient_agent.process(Context(
        input="What is AI?",
        model="openai:gpt-4o-mini"
    ))

    if result.is_success():
        print("[SUCCESS] Agent completed (with retry protection)")


# ============================================================================
# EXAMPLE 9: Recover (recover)
# ============================================================================

async def example_9_recover():
    """Error handling with recover()."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Recover")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Wrap agent with error recovery
    safe_agent = recover(
        primitive=lb.agent,
        handler=lambda err, ctx: ctx.set("fallback", "Used fallback response"),
        name="safe-agent"
    )

    result = await safe_agent.process(Context(
        input="What is AI?",
        model="openai:gpt-4o-mini"
    ))

    if result.is_success():
        print("[SUCCESS] Agent completed (with error recovery)")


# ============================================================================
# EXAMPLE 10: Map-Reduce Pattern
# ============================================================================

async def example_10_map_reduce():
    """Map-reduce pattern: process in parallel, then aggregate."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Map-Reduce Pattern")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Map: Research topics in parallel
    mapper = map_over(
        items=lambda ctx: ctx.get("topics", []),
        apply=lb.agent,
        parallel=True
    )

    # Reduce: Combine results
    reducer = reduce(
        inputs=lambda ctx: [
            r.response if hasattr(r, 'response') else str(r)
            for r in ctx.get("map_results", [])
        ],
        combine=lambda results: "\n\n".join(f"Topic {i+1}: {r[:100]}..."
                                            for i, r in enumerate(results))
    )

    # Compose
    map_reduce = pipeline(mapper, reducer, name="map-reduce")

    result = await map_reduce.process(Context(
        topics=["Python", "JavaScript"],
        input="Explain this language briefly",
        model="openai:gpt-4o-mini"
    ))

    if result.is_success():
        ctx = result.unwrap()
        print("[RESULT] Map-Reduce Complete")
        print(ctx.get("reduce_result", "")[:200] + "...")


# ============================================================================
# EXAMPLE 11: Complex Composition
# ============================================================================

async def example_11_complex():
    """Complex operator composition."""
    print("\n" + "="*70)
    print("EXAMPLE 11: Complex Composition")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # Build complex pipeline:
    # 1. Conditional routing
    # 2. Retry wrapper
    # 3. Sequential composition

    quick_agent = lb.agent
    thorough_agent = lb.agent

    # Conditional: route by urgency
    conditional = when(
        condition=lambda ctx: ctx.get("urgent", False),
        then_do=quick_agent,
        else_do=thorough_agent
    )

    # Wrap with retry
    resilient = retry(conditional, max_attempts=2)

    # Test
    result = await resilient.process(Context(
        input="What is Python?",
        model="openai:gpt-4o-mini",
        urgent=True
    ))

    if result.is_success():
        print("[SUCCESS] Complex pipeline completed")
        print("  → Routed by urgency")
        print("  → Protected by retry")


# ============================================================================
# EXAMPLE 12: Workflow + Operators
# ============================================================================

async def example_12_workflow():
    """Operators in workflow steps."""
    print("\n" + "="*70)
    print("EXAMPLE 12: Workflow + Operators")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    wf = lb.workflow(name="operator-workflow")

    # Step 1: Map over topics
    wf.step(
        id="research",
        primitive=map_over(
            items=lambda ctx: ctx.get("topics", []),
            apply=lb.agent,
            parallel=True
        )
    )

    # Step 2: Reduce results
    wf.step(
        id="synthesize",
        primitive=reduce(
            inputs=lambda ctx: [
                r.response if hasattr(r, 'response') else str(r)
                for r in ctx.get("map_results", [])
            ],
            combine=lambda results: f"Synthesized {len(results)} findings"
        ),
        after=["research"]
    )

    result = await wf.run(
        topics=["AI", "ML"],
        input="Brief explanation",
        model="openai:gpt-4o-mini"
    )

    if result.success:
        print("[SUCCESS] Workflow with operators completed")
        synthesized = result.outputs.get("synthesize", {}).get("reduce_result", "")
        print(f"  Result: {synthesized}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run all operator examples."""
    print("\n" + "="*70)
    print("LANGPY OPERATORS GUIDE")
    print("Comprehensive examples of all operators and patterns")
    print("="*70)

    examples = [
        ("Sequential (|)", example_1_sequential),
        ("Parallel (&)", example_2_parallel),
        ("Conditional (when)", example_3_conditional),
        ("Branch", example_4_branch),
        ("Loop", example_5_loop),
        ("Map", example_6_map),
        ("Reduce", example_7_reduce),
        ("Retry", example_8_retry),
        ("Recover", example_9_recover),
        ("Map-Reduce", example_10_map_reduce),
        ("Complex Composition", example_11_complex),
        ("Workflow + Operators", example_12_workflow),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i:2}. {name}")

    print("\n" + "="*70)
    print("Running Quick Demo (Examples 5, 6, 7)...")
    print("="*70)

    # Run quick demo
    try:
        await example_5_loop()
        await example_6_map()
        await example_7_reduce()
    except Exception as e:
        print(f"\n[ERROR] {e}")

    print("\n" + "="*70)
    print("Quick demo complete!")
    print("To run all examples, uncomment them in main()")
    print("="*70)

    # Uncomment to run all examples:
    # for name, func in examples:
    #     try:
    #         await func()
    #     except Exception as e:
    #         print(f"\n[ERROR in {name}] {e}")


if __name__ == "__main__":
    asyncio.run(main())
