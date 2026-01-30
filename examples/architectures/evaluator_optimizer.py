"""
Pattern 6: Evaluator-Optimizer
==============================

Iterative refinement using direct primitive composition.

This example shows how to BUILD a self-improving loop by composing primitives:
    - Pipe: Generator (creates output)
    - Pipe: Evaluator (scores and provides feedback)
    - Loop: Iterate until quality threshold met

Architecture:
    Task → Generate → Evaluate → (if below threshold) → Improve → Evaluate → ...

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Pipe


# =============================================================================
# PATTERN 6: EVALUATOR-OPTIMIZER WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def evaluator_optimizer_pattern():
    """
    Build a self-improving loop by composing two Pipe primitives.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 6: EVALUATOR-OPTIMIZER")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Create the primitives (our building blocks)
    # =========================================================================

    generator = Pipe(model="gpt-4o-mini")  # Generates output
    evaluator = Pipe(model="gpt-4o-mini")  # Evaluates and provides feedback

    print("Primitives created:")
    print("  - Generator Pipe: creates and improves output")
    print("  - Evaluator Pipe: scores and provides feedback")
    print()

    # =========================================================================
    # STEP 2: Define the task and criteria
    # =========================================================================

    task = "Write a Python function that checks if a number is prime"
    criteria = [
        "Code is correct and handles edge cases",
        "Code is readable and well-structured",
        "Code is efficient",
    ]

    print(f"Task: {task}")
    print(f"Criteria: {criteria}")
    print(f"Threshold: 0.85")
    print(f"Max iterations: 3")
    print()

    # =========================================================================
    # STEP 3: Implement the generate-evaluate-improve loop
    # =========================================================================

    threshold = 0.85
    max_iterations = 3
    current_output = None
    feedback = None

    for iteration in range(1, max_iterations + 1):
        print(f"--- Iteration {iteration} ---")

        # COMPOSE: Generator creates/improves output
        if current_output and feedback:
            # Improvement prompt
            gen_prompt = f"""Improve this code based on the feedback:

Task: {task}

Current code:
{current_output}

Feedback:
{feedback}

Provide improved code:"""
        else:
            # Initial generation
            gen_prompt = f"""Complete this task:

{task}

Provide high-quality code:"""

        gen_response = await generator.run(
            gen_prompt,
            system="You are an expert programmer who writes clean, efficient code."
        )
        current_output = gen_response.content
        print(f"Generated output (first 200 chars):")
        print(current_output[:200] + "...")
        print()

        # COMPOSE: Evaluator scores the output
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        eval_prompt = f"""Evaluate this code against the criteria:

Task: {task}

Code:
{current_output}

Criteria:
{criteria_text}

Respond in this EXACT format:
SCORE: [0.0 to 1.0]
FEEDBACK: [specific feedback]
SUGGESTIONS: [improvements needed]"""

        eval_response = await evaluator.run(
            eval_prompt,
            system="You are a critical code reviewer. Be specific about issues."
        )

        # Parse score from evaluation
        score = 0.5
        eval_text = eval_response.content
        for line in eval_text.split("\n"):
            if line.strip().startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                except ValueError:
                    pass

        print(f"Evaluation score: {score:.2f}")
        print(f"Evaluation: {eval_text[:200]}...")
        print()

        # Check if threshold met
        if score >= threshold:
            print(f"Threshold met! Score {score:.2f} >= {threshold}")
            break

        # Extract feedback for next iteration
        feedback = eval_text

    print()
    print("=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(current_output)


# =============================================================================
# SIMPLE OPTIMIZER FUNCTION - Minimal composition example
# =============================================================================

async def simple_optimizer(task: str, criteria: list[str], threshold: float = 0.8, max_iter: int = 3) -> str:
    """
    Minimal evaluator-optimizer in just a few lines - pure primitive composition.

    Args:
        task: Task to complete
        criteria: List of evaluation criteria
        threshold: Quality threshold (0.0 to 1.0)
        max_iter: Maximum iterations

    Returns:
        Optimized output
    """
    generator = Pipe(model="gpt-4o-mini")
    evaluator = Pipe(model="gpt-4o-mini")

    output = None
    feedback = None

    for _ in range(max_iter):
        # Generate/improve
        if output and feedback:
            prompt = f"Improve based on feedback:\n\nTask: {task}\nCurrent: {output}\nFeedback: {feedback}"
        else:
            prompt = f"Complete: {task}"

        response = await generator.run(prompt)
        output = response.content

        # Evaluate
        eval_response = await evaluator.run(
            f"Score 0-1 for these criteria: {criteria}\n\nOutput: {output}\n\nRespond: SCORE: [number]"
        )

        # Parse score
        try:
            score = float(eval_response.content.split("SCORE:")[1].split()[0])
        except:
            score = 0.5

        if score >= threshold:
            break

        feedback = eval_response.content

    return output


# =============================================================================
# WRITING OPTIMIZER
# =============================================================================

async def writing_optimizer_demo():
    """
    Optimize writing quality through iterative refinement.
    """
    print("\n" + "=" * 60)
    print("   WRITING OPTIMIZER")
    print("   Generator + Evaluator composition")
    print("=" * 60 + "\n")

    generator = Pipe(model="gpt-4o-mini")
    evaluator = Pipe(model="gpt-4o-mini")

    task = "Write a compelling opening paragraph for an article about AI agents"
    criteria = ["Engaging hook", "Clear thesis", "Professional tone"]

    print(f"Task: {task}")
    print()

    output = None
    for iteration in range(1, 4):
        # Generate
        if output:
            prompt = f"Improve this based on feedback. Task: {task}\n\nCurrent:\n{output}\n\nFeedback:\n{feedback}"
        else:
            prompt = f"Task: {task}"

        response = await generator.run(prompt, system="You write engaging content.")
        output = response.content

        print(f"Iteration {iteration}:")
        print(output[:300] + "..." if len(output) > 300 else output)
        print()

        # Evaluate
        eval_response = await evaluator.run(
            f"Rate 0-1 for: {criteria}\n\nText: {output}\n\nFormat: SCORE: [number]\nFEEDBACK: [details]",
            system="You are a critical editor."
        )

        # Parse
        try:
            score = float(eval_response.content.split("SCORE:")[1].split()[0])
        except:
            score = 0.5

        print(f"Score: {score:.2f}")

        if score >= 0.85:
            print("Quality threshold met!")
            break

        feedback = eval_response.content
        print(f"Feedback: {feedback[:100]}...")
        print()


# =============================================================================
# CODE OPTIMIZER
# =============================================================================

async def code_optimizer_demo():
    """
    Optimize code quality through iterative refinement.
    """
    print("\n" + "=" * 60)
    print("   CODE OPTIMIZER")
    print("   Generator + Evaluator composition")
    print("=" * 60 + "\n")

    generator = Pipe(model="gpt-4o-mini")
    evaluator = Pipe(model="gpt-4o-mini")

    task = "Write a function to find the nth Fibonacci number efficiently"

    output = await generator.run(
        f"Write code: {task}",
        system="You write efficient, clean code."
    )

    print("Initial code:")
    print(output.content)
    print()

    # Evaluate and improve
    eval_response = await evaluator.run(
        f"Review this code for: correctness, efficiency, readability.\n\n{output.content}\n\nFormat: SCORE: [0-1]\nISSUES: [list]",
        system="You are a senior code reviewer."
    )
    print(f"Evaluation: {eval_response.content}")
    print()

    # Improve based on feedback
    improved = await generator.run(
        f"Improve this code based on the review:\n\nCode:\n{output.content}\n\nReview:\n{eval_response.content}",
        system="You write optimized, clean code."
    )

    print("Improved code:")
    print(improved.content)


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all evaluator-optimizer demonstrations."""
    await evaluator_optimizer_pattern()
    await writing_optimizer_demo()
    await code_optimizer_demo()


if __name__ == "__main__":
    asyncio.run(demo())
