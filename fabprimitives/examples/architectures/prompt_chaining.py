"""
Pattern 2: Prompt Chaining
==========================

Sequential task decomposition using direct primitive composition.

This example shows how to BUILD a prompt chain by composing primitives:
    - Pipe: Each step in the chain
    - Multiple Pipe.run() calls: Output of one feeds into next

Architecture:
    Input → Pipe.run() → Pipe.run() → Pipe.run() → Output

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Pipe


# =============================================================================
# PATTERN 2: PROMPT CHAINING WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def prompt_chain_pattern():
    """
    Build a prompt chain by composing multiple Pipe.run() calls.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 2: PROMPT CHAINING")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Create the Pipe primitive (our building block)
    # =========================================================================

    pipe = Pipe(model="gpt-4o-mini")

    print("Pipe primitive created for chained processing")
    print()

    # =========================================================================
    # STEP 2: Define input and chain the Pipe calls
    # =========================================================================

    input_text = """
    LangPy is a Python framework for building AI agents. It provides 9 core
    primitives: Pipe, Memory, Agent, Workflow, Thread, Tools, Parser, Chunker,
    and Embed. These primitives are composable like Lego blocks.
    """

    print("Input:", input_text.strip()[:80] + "...")
    print()

    # =========================================================================
    # STEP 3: Chain Pipe.run() calls - output flows to next input
    # =========================================================================

    # CHAIN STEP 1: Extract key facts
    print("Step 1: Extract key facts")
    print("-" * 40)
    step1 = await pipe.run(
        f"Extract the key facts from this text as bullet points:\n\n{input_text}",
        system="You extract key facts concisely."
    )
    print(step1.content)
    print()

    # CHAIN STEP 2: Analyze (using step1 output)
    print("Step 2: Analyze the extracted facts")
    print("-" * 40)
    step2 = await pipe.run(
        f"Analyze these facts and identify main themes:\n\n{step1.content}",
        system="You analyze information and identify patterns."
    )
    print(step2.content)
    print()

    # CHAIN STEP 3: Summarize (using step2 output)
    print("Step 3: Generate final summary")
    print("-" * 40)
    step3 = await pipe.run(
        f"Write a one-paragraph summary:\n\n{step2.content}",
        system="You write clear, concise summaries."
    )
    print(step3.content)
    print()

    print("=" * 60)


# =============================================================================
# SIMPLE CHAIN FUNCTION - Minimal composition example
# =============================================================================

async def simple_chain(text: str, steps: list[str]) -> str:
    """
    Minimal prompt chain - pure primitive composition.

    Args:
        text: Input text
        steps: List of instruction prompts

    Returns:
        Final output after all steps
    """
    pipe = Pipe(model="gpt-4o-mini")
    result = text

    # Chain: each step's output becomes next step's input
    for instruction in steps:
        response = await pipe.run(f"{instruction}\n\nInput:\n{result}")
        result = response.content

    return result


# =============================================================================
# DOCUMENT PROCESSING CHAIN
# =============================================================================

async def document_processing_demo():
    """
    Document processing: Extract → Analyze → Summarize
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   DOCUMENT PROCESSING CHAIN")
    print("   Extract → Analyze → Summarize")
    print("=" * 60 + "\n")

    # Create the Pipe primitive
    pipe = Pipe(model="gpt-4o-mini")

    document = """
    The quarterly report shows a 15% increase in revenue compared to last year.
    Customer satisfaction scores improved by 8 points. However, operating costs
    rose by 12% due to supply chain challenges. The new product line launched
    in Q3 exceeded sales projections by 20%.
    """

    print("Document:", document.strip()[:100] + "...")
    print()

    # COMPOSE: Chain of Pipe.run() calls
    # Step 1
    extract = await pipe.run(
        f"Extract all numerical metrics from this report:\n\n{document}",
        system="Extract metrics with their context."
    )
    print("1. Extracted metrics:")
    print(extract.content)
    print()

    # Step 2 (uses step 1 output)
    analyze = await pipe.run(
        f"Analyze these metrics - identify positive and negative trends:\n\n{extract.content}",
        system="Analyze business metrics objectively."
    )
    print("2. Analysis:")
    print(analyze.content)
    print()

    # Step 3 (uses step 2 output)
    summary = await pipe.run(
        f"Write a 2-sentence executive summary:\n\n{analyze.content}",
        system="Write concise executive summaries."
    )
    print("3. Executive Summary:")
    print(summary.content)
    print()


# =============================================================================
# CONTENT GENERATION CHAIN
# =============================================================================

async def content_generation_demo():
    """
    Content generation: Outline → Draft → Polish
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   CONTENT GENERATION CHAIN")
    print("   Outline → Draft → Polish")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    topic = "Benefits of composable AI architectures"

    print(f"Topic: {topic}")
    print()

    # COMPOSE: Chain Pipe calls
    # Step 1: Outline
    outline = await pipe.run(
        f"Create a brief outline (3 main points) for: {topic}",
        system="Create clear, logical outlines."
    )
    print("1. Outline:")
    print(outline.content)
    print()

    # Step 2: Draft (uses outline)
    draft = await pipe.run(
        f"Write a short blog post based on this outline:\n\n{outline.content}",
        system="Write engaging, informative content."
    )
    print("2. Draft (preview):")
    print(draft.content[:400] + "...")
    print()

    # Step 3: Polish (uses draft)
    polished = await pipe.run(
        f"Polish this for clarity. Keep it concise:\n\n{draft.content}",
        system="Edit for clarity and engagement."
    )
    print("3. Polished (preview):")
    print(polished.content[:400] + "...")
    print()


# =============================================================================
# TRANSLATION CHAIN WITH REVIEW
# =============================================================================

async def translation_chain_demo():
    """
    Translation with review: Translate → Review → Refine
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   TRANSLATION CHAIN")
    print("   Translate → Review → Refine")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    text = "The early bird catches the worm, but the second mouse gets the cheese."

    print(f"Original: {text}")
    print()

    # COMPOSE: Chain Pipe calls for translation pipeline
    # Step 1: Translate
    translation = await pipe.run(
        f"Translate to Spanish:\n\n{text}",
        system="Translate naturally, preserving idioms where possible."
    )
    print(f"1. Translation: {translation.content}")

    # Step 2: Review (uses translation)
    review = await pipe.run(
        f"Review this translation for accuracy:\n\nOriginal: {text}\nTranslation: {translation.content}",
        system="Review translations critically."
    )
    print(f"2. Review: {review.content}")

    # Step 3: Refine (uses review)
    refined = await pipe.run(
        f"Based on this review, provide the best final translation:\n\n{review.content}",
        system="Provide polished, natural translations."
    )
    print(f"3. Final: {refined.content}")
    print()


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all prompt chaining demonstrations."""
    await prompt_chain_pattern()
    await document_processing_demo()
    await content_generation_demo()
    await translation_chain_demo()


if __name__ == "__main__":
    asyncio.run(demo())
