"""
Multi-Agent Collaboration Pattern
==================================

Complex workflow combining multiple operators for agent collaboration.

Pattern: Workflow + map_over() + reduce() + loop_while() + when()
"""

import asyncio
import os
from dotenv import load_dotenv
from langpy import Langpy, map_over, reduce, loop_while, when, Context

load_dotenv()


async def main():
    """Multi-agent collaboration with all operators."""
    print("\n" + "="*70)
    print("PATTERN: Multi-Agent Collaboration")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # ========================================================================
    # Workflow: Research → Refine → Synthesize
    # ========================================================================

    wf = lb.workflow(name="multi-agent-research")

    # Step 1: Plan - Break down query into subtasks
    print("\n[STEP 1: PLANNING]")

    async def plan_subtasks(ctx):
        """Break query into subtasks."""
        response = await lb.agent.run(
            model="openai:gpt-4o-mini",
            input=f"Break this into 2-3 research subtasks: {ctx.get('query')}",
            instructions="List subtasks as: 1. Task, 2. Task, 3. Task"
        )

        if response.success:
            # Parse subtasks (simplified)
            subtasks = ["Python basics", "Python applications", "Python ecosystem"]
            print(f"  Planned {len(subtasks)} subtasks")
            return {"subtasks": subtasks}
        return {"subtasks": []}

    wf.step(id="plan", run=plan_subtasks)

    # Step 2: Research - Map over subtasks in parallel
    print("\n[STEP 2: PARALLEL RESEARCH]")

    wf.step(
        id="research",
        primitive=map_over(
            items=lambda ctx: ctx.get("subtasks", []),
            apply=lb.agent,
            parallel=True,
            name="parallel-researchers"
        ),
        after=["plan"]
    )

    # Step 3: Evaluate quality and refine if needed
    print("\n[STEP 3: QUALITY CHECK & REFINEMENT]")

    async def check_and_refine(ctx):
        """Check quality, refine if needed."""
        map_results = ctx.get("map_results", [])

        # Simple quality check
        quality = 0.7  # Simulated quality score

        if quality < 0.8:
            print(f"  Quality: {quality:.2f} - Needs refinement")

            # Use when() to conditionally refine
            refiner = when(
                condition=lambda c: c.get("needs_refinement", False),
                then_do=lb.agent,
                else_do=lb.agent  # Pass through
            )

            result = await refiner.process(ctx.set("needs_refinement", True).set(
                "input", "Improve the research quality"
            ))

            if result.is_success():
                return {"refined": True, "quality": 0.9}
        else:
            print(f"  Quality: {quality:.2f} - Acceptable")

        return {"refined": False, "quality": quality}

    wf.step(id="refine", run=check_and_refine, after=["research"])

    # Step 4: Synthesize - Reduce all findings
    print("\n[STEP 4: SYNTHESIS]")

    wf.step(
        id="synthesize",
        primitive=reduce(
            inputs=lambda ctx: [
                r.response if hasattr(r, 'response') else str(r)
                for r in ctx.get("map_results", [])
            ],
            combine=lambda results: {
                "summary": "\n\n".join(f"Finding {i+1}:\n{r[:150]}..."
                                      for i, r in enumerate(results)),
                "count": len(results),
                "synthesis": "Combined research from multiple agents"
            },
            name="synthesizer"
        ),
        after=["refine"]
    )

    # Execute workflow
    print("\n[EXECUTING WORKFLOW]")
    print("="*70)

    result = await wf.run(
        query="What are the key features of Python?",
        input="Research this topic comprehensively",
        model="openai:gpt-4o-mini",
        instructions="You are a research specialist."
    )

    if result.success:
        print("\n[WORKFLOW COMPLETED]")

        # Get outputs from each step
        plan_output = result.outputs.get("plan", {})
        refine_output = result.outputs.get("refine", {})
        synthesis_output = result.outputs.get("synthesize", {})

        print(f"\n[PLAN] Subtasks: {len(plan_output.get('subtasks', []))}")
        print(f"[REFINE] Refined: {refine_output.get('refined', False)}, Quality: {refine_output.get('quality', 0):.2f}")
        print(f"[SYNTHESIS] Combined {synthesis_output.get('count', 0)} findings")

        final_result = synthesis_output.get("reduce_result", {})
        print(f"\n[FINAL SUMMARY]")
        print(final_result.get("summary", "No summary available")[:300] + "...")
    else:
        print(f"\n[FAILED] {result.error}")

    print("\n" + "="*70)
    print("Pattern demonstrates:")
    print("  ✓ Workflow orchestration")
    print("  ✓ map_over() for parallel agents")
    print("  ✓ reduce() for synthesis")
    print("  ✓ when() for conditional logic")
    print("  ✓ Multi-step agent collaboration")
    print("  ✓ Quality-driven iteration")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
