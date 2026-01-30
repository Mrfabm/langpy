"""
Pattern 5: Orchestration-Workers
================================

A supervisor decomposes tasks and delegates to specialized workers.

This example shows how to BUILD an orchestration system by composing primitives:
    - Pipe: Orchestrator (decomposes tasks, synthesizes results)
    - Pipe: Workers (specialized handlers for subtasks)

Architecture:
    Task → Pipe.run(decompose) → [Worker Pipes in parallel] → Pipe.run(synthesize)

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Pipe


# =============================================================================
# PATTERN 5: ORCHESTRATOR-WORKERS WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def orchestrator_pattern():
    """
    Build an orchestration system by composing Pipe primitives.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 5: ORCHESTRATION-WORKERS")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Create the primitives (our building blocks)
    # =========================================================================

    # Orchestrator Pipe - decomposes and synthesizes
    orchestrator = Pipe(model="gpt-4o-mini")

    # Worker Pipes - specialized handlers
    workers = {
        "researcher": Pipe(model="gpt-4o-mini"),
        "writer": Pipe(model="gpt-4o-mini"),
        "editor": Pipe(model="gpt-4o-mini"),
    }

    worker_roles = {
        "researcher": "You are a research specialist. Find facts and information.",
        "writer": "You are a content writer. Create engaging content.",
        "editor": "You are an editor. Review and improve content.",
    }

    print("Primitives created:")
    print("  - Orchestrator Pipe: decomposes tasks, synthesizes results")
    print("  - Researcher Pipe: finds information")
    print("  - Writer Pipe: creates content")
    print("  - Editor Pipe: reviews and improves")
    print()

    # =========================================================================
    # STEP 2: Define the task
    # =========================================================================

    task = "Create a short guide about getting started with LangPy"

    print(f"Task: {task}")
    print()

    # =========================================================================
    # STEP 3: Orchestrator decomposes the task
    # =========================================================================

    print("Step 1: Orchestrator decomposes task...")
    print("-" * 40)

    workers_desc = "\n".join(f"- {name}: {role}" for name, role in worker_roles.items())

    decomposition = await orchestrator.run(
        f"""Break down this task into subtasks for the team.

Task: {task}

Team members:
{workers_desc}

Respond in this format (one per line):
SUBTASK: [description] -> ASSIGN: [worker_name]""",
        system="You are a project manager. Assign tasks efficiently."
    )

    print(decomposition.content)
    print()

    # Parse subtasks
    subtasks = []
    for line in decomposition.content.split("\n"):
        if "SUBTASK:" in line and "ASSIGN:" in line:
            try:
                parts = line.split("->")
                desc = parts[0].replace("SUBTASK:", "").strip()
                worker_name = parts[1].replace("ASSIGN:", "").strip().lower()

                # Match to actual worker
                for w in workers.keys():
                    if w in worker_name:
                        subtasks.append({"description": desc, "worker": w})
                        break
            except:
                continue

    if not subtasks:
        # Fallback
        subtasks = [{"description": task, "worker": "writer"}]

    # =========================================================================
    # STEP 4: Execute workers in parallel
    # =========================================================================

    print("Step 2: Workers execute subtasks in parallel...")
    print("-" * 40)

    async def execute_subtask(subtask: dict) -> dict:
        worker_name = subtask["worker"]
        worker_pipe = workers[worker_name]
        system = worker_roles[worker_name]

        response = await worker_pipe.run(
            f"Your task: {subtask['description']}\n\nComplete it thoroughly.",
            system=system
        )

        return {
            "worker": worker_name,
            "task": subtask["description"],
            "result": response.content
        }

    # COMPOSE: Run all workers in parallel
    results = await asyncio.gather(*[execute_subtask(st) for st in subtasks])

    for r in results:
        print(f"[{r['worker']}]: {r['result'][:80]}...")
    print()

    # =========================================================================
    # STEP 5: Orchestrator synthesizes final result
    # =========================================================================

    print("Step 3: Orchestrator synthesizes results...")
    print("-" * 40)

    all_results = "\n\n".join(
        f"### {r['worker'].upper()} - {r['task']}\n{r['result']}"
        for r in results
    )

    synthesis = await orchestrator.run(
        f"""Synthesize this team's work into a cohesive final deliverable.

Original task: {task}

Team work:
{all_results}

Create a polished final output:""",
        system="You synthesize team work into polished deliverables."
    )

    print("FINAL RESULT:")
    print(synthesis.content[:600] + "..." if len(synthesis.content) > 600 else synthesis.content)
    print()

    print("=" * 60)


# =============================================================================
# SIMPLE ORCHESTRATOR FUNCTION - Minimal composition example
# =============================================================================

async def simple_orchestrator(task: str, workers: dict[str, str]) -> str:
    """
    Minimal orchestrator - pure primitive composition.

    Args:
        task: Task to complete
        workers: Dict of {worker_name: role_description}

    Returns:
        Final synthesized result
    """
    pipe = Pipe(model="gpt-4o-mini")

    # Decompose
    workers_text = "\n".join(f"- {name}: {role}" for name, role in workers.items())
    decomposition = await pipe.run(
        f"Break into subtasks for: {workers_text}\n\nTask: {task}\n\nFormat: worker_name: subtask",
        system="Assign subtasks to workers."
    )

    # Execute workers (simplified - assign to all)
    results = []
    for name, role in workers.items():
        response = await pipe.run(f"As a {role}, help with: {task}", system=f"You are a {role}.")
        results.append(f"{name}: {response.content}")

    # Synthesize
    synthesis = await pipe.run(
        f"Synthesize into final output:\n\n" + "\n\n".join(results),
        system="Create cohesive final deliverable."
    )

    return synthesis.content


# =============================================================================
# CONTENT TEAM ORCHESTRATION
# =============================================================================

async def content_team_demo():
    """
    Content creation with researcher, writer, editor.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   CONTENT TEAM ORCHESTRATION")
    print("   Researcher → Writer → Editor")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    task = "Write a brief intro to AI agents"

    print(f"Task: {task}\n")

    # COMPOSE: Sequential worker pipeline
    # Step 1: Researcher gathers info
    research = await pipe.run(
        f"Research and gather key facts about: {task}",
        system="You are a researcher. Find important facts."
    )
    print(f"RESEARCHER: {research.content[:150]}...")
    print()

    # Step 2: Writer creates draft (using research)
    draft = await pipe.run(
        f"Write content based on this research:\n\n{research.content}",
        system="You are a writer. Create engaging content."
    )
    print(f"WRITER: {draft.content[:150]}...")
    print()

    # Step 3: Editor polishes (using draft)
    final = await pipe.run(
        f"Edit and improve this draft:\n\n{draft.content}",
        system="You are an editor. Polish the content."
    )
    print(f"EDITOR (Final): {final.content[:200]}...")
    print()


# =============================================================================
# ANALYSIS TEAM ORCHESTRATION
# =============================================================================

async def analysis_team_demo():
    """
    Analysis with data analyst, strategist, reporter.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   ANALYSIS TEAM ORCHESTRATION")
    print("   Parallel workers + synthesis")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")
    topic = "The impact of AI on job markets"

    print(f"Topic: {topic}\n")

    # COMPOSE: Parallel analysis from different perspectives
    async def analyze(role: str, system: str) -> tuple[str, str]:
        response = await pipe.run(f"Analyze: {topic}", system=system)
        return role, response.content

    workers = [
        ("data_analyst", "You analyze trends and statistics."),
        ("strategist", "You develop strategic recommendations."),
        ("reporter", "You summarize findings clearly."),
    ]

    results = await asyncio.gather(*[analyze(role, sys) for role, sys in workers])

    for role, content in results:
        print(f"{role.upper()}: {content[:100]}...")
    print()

    # Synthesize
    all_work = "\n\n".join(f"{role}: {content}" for role, content in results)
    final = await pipe.run(
        f"Synthesize this team's analysis into a cohesive report:\n\n{all_work}",
        system="Create a unified analysis report."
    )

    print("SYNTHESIZED REPORT:")
    print(final.content[:300] + "...")
    print()


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all orchestrator demonstrations."""
    await orchestrator_pattern()
    await content_team_demo()
    await analysis_team_demo()


if __name__ == "__main__":
    asyncio.run(demo())
