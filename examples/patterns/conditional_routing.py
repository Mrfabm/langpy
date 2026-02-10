"""
Conditional Routing Pattern
============================

Route requests to different agents based on conditions.

Pattern: when() and branch() for dynamic routing
"""

import asyncio
import os
from dotenv import load_dotenv
from langpy import Langpy, when, branch, Context

load_dotenv()


async def main():
    """Conditional routing with when() and branch()."""
    print("\n" + "="*70)
    print("PATTERN: Conditional Routing")
    print("="*70)

    lb = Langpy(api_key=os.getenv("OPENAI_API_KEY"))

    # ========================================================================
    # Part 1: Binary Routing with when()
    # ========================================================================

    print("\n[PART 1: Binary Routing]")

    # Simple conditional: route based on query length
    simple_router = when(
        condition=lambda ctx: len(ctx.get("input", "")) > 50,
        then_do=lb.agent,  # Long query → detailed response
        else_do=lb.agent,  # Short query → quick response
        name="length-router"
    )

    # Test short query
    result1 = await simple_router.process(Context(
        input="What is AI?",
        model="openai:gpt-4o-mini"
    ))

    if result1.is_success():
        print(f"[SHORT QUERY] Routed to quick response")
        print(f"  Response: {result1.unwrap().response[:100]}...")

    # Test long query
    result2 = await simple_router.process(Context(
        input="Can you provide a comprehensive explanation of artificial intelligence, including its history, current applications, and future potential?" * 2,
        model="openai:gpt-4o-mini"
    ))

    if result2.is_success():
        print(f"\n[LONG QUERY] Routed to detailed response")
        print(f"  Response: {result2.unwrap().response[:100]}...")

    # ========================================================================
    # Part 2: Multi-way Routing with branch()
    # ========================================================================

    print("\n\n[PART 2: Multi-way Routing]")

    # Create router for different task types
    task_router = branch(
        router=lambda ctx: ctx.get("task_type", "general"),
        routes={
            "research": lb.agent,
            "summarize": lb.agent,
            "translate": lb.agent,
        },
        default=lb.agent,
        name="task-router"
    )

    # Test different routes
    tasks = [
        ("research", "Research the benefits of Python"),
        ("summarize", "Summarize: Python is a versatile language used in many fields."),
        ("translate", "Translate to Spanish: Hello, how are you?"),
        ("unknown", "Just a general question"),
    ]

    for task_type, input_text in tasks:
        result = await task_router.process(Context(
            task_type=task_type,
            input=input_text,
            model="openai:gpt-4o-mini",
            instructions=f"You are a {task_type} specialist." if task_type != "unknown" else "You are a general assistant."
        ))

        if result.is_success():
            route_used = task_type if task_type in ["research", "summarize", "translate"] else "default"
            print(f"\n[ROUTE: {task_type}] → {route_used}")
            print(f"  Input: {input_text[:60]}...")
            print(f"  Output: {result.unwrap().response[:80]}...")

    print("\n" + "="*70)
    print("Pattern demonstrates:")
    print("  ✓ when() for binary routing")
    print("  ✓ branch() for multi-way routing")
    print("  ✓ Dynamic path selection")
    print("  ✓ Default fallback handling")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
