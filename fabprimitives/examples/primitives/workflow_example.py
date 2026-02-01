"""
Workflow Primitive
==================
Orchestrate multi-step processes with dependencies and parallel execution.

Workflows enable:
    - Sequential step execution
    - Parallel step execution
    - Dependency management (step A must complete before step B)
    - Retries and error handling
    - Complex AI pipelines

Architecture:
    Steps → Dependency Resolution → Execute (parallel where possible) → Results

    ┌─────────────────────────────────────────────────────────┐
    │                     Workflow                            │
    │  ┌──────┐   ┌──────┐   ┌──────┐                         │
    │  │Step A│ → │Step B│ → │Step D│                         │
    │  └──────┘   └──────┘   └──────┘                         │
    │       ↘                ↗                                │
    │         ┌──────┐                                        │
    │         │Step C│  (parallel with B)                     │
    │         └──────┘                                        │
    └─────────────────────────────────────────────────────────┘
"""

import asyncio
import io
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Workflow, Step


# =============================================================================
# BASIC WORKFLOW
# =============================================================================

async def basic_workflow_demo():
    """Demonstrate basic sequential workflow."""
    print("=" * 60)
    print("   BASIC WORKFLOW - Sequential Steps")
    print("=" * 60)
    print()

    # Define handlers
    def step_a(inputs):
        print("      Executing Step A: Fetching data...")
        return {"data": [1, 2, 3, 4, 5]}

    def step_b(inputs):
        data = inputs.get("step_a", {}).get("data", [])
        print(f"      Executing Step B: Processing {len(data)} items...")
        return {"processed": [x * 2 for x in data]}

    def step_c(inputs):
        processed = inputs.get("step_b", {}).get("processed", [])
        total = sum(processed)
        print(f"      Executing Step C: Calculating total = {total}")
        return {"total": total}

    # Create workflow
    workflow = Workflow("data-pipeline")

    workflow.add_step(Step(name="step_a", handler=step_a))
    workflow.add_step(Step(name="step_b", handler=step_b, depends_on=["step_a"]))
    workflow.add_step(Step(name="step_c", handler=step_c, depends_on=["step_b"]))

    print("Workflow structure:")
    print(workflow.visualize())
    print()

    print("Executing workflow:")
    print("-" * 40)
    result = await workflow.run()

    print()
    print(f"Status: {result.status}")
    print(f"Duration: {result.duration:.3f}s")
    print(f"Final output: {result.outputs}")
    print()


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

async def parallel_workflow_demo():
    """Demonstrate parallel step execution."""
    print("=" * 60)
    print("   PARALLEL WORKFLOW - Concurrent Steps")
    print("=" * 60)
    print()

    async def fetch_users(inputs):
        print("      Fetching users... (0.5s)")
        await asyncio.sleep(0.5)
        return {"users": ["Alice", "Bob", "Carol"]}

    async def fetch_products(inputs):
        print("      Fetching products... (0.5s)")
        await asyncio.sleep(0.5)
        return {"products": ["Widget", "Gadget", "Gizmo"]}

    async def fetch_orders(inputs):
        print("      Fetching orders... (0.5s)")
        await asyncio.sleep(0.5)
        return {"orders": [101, 102, 103]}

    def combine_data(inputs):
        users = inputs.get("fetch_users", {}).get("users", [])
        products = inputs.get("fetch_products", {}).get("products", [])
        orders = inputs.get("fetch_orders", {}).get("orders", [])
        print(f"      Combining: {len(users)} users, {len(products)} products, {len(orders)} orders")
        return {
            "summary": f"{len(users)} users, {len(products)} products, {len(orders)} orders"
        }

    workflow = Workflow("parallel-fetch")

    # These three can run in parallel (no dependencies)
    workflow.add_step(Step(name="fetch_users", handler=fetch_users))
    workflow.add_step(Step(name="fetch_products", handler=fetch_products))
    workflow.add_step(Step(name="fetch_orders", handler=fetch_orders))

    # This depends on all three above
    workflow.add_step(Step(
        name="combine",
        handler=combine_data,
        depends_on=["fetch_users", "fetch_products", "fetch_orders"]
    ))

    print("Workflow structure:")
    print(workflow.visualize())
    print()

    print("Executing workflow (parallel=True):")
    print("-" * 40)
    result = await workflow.run(parallel=True)

    print()
    print(f"Status: {result.status}")
    print(f"Duration: {result.duration:.3f}s (faster than sequential!)")
    print(f"Output: {result.outputs.get('combine')}")
    print()


# =============================================================================
# DECORATOR STYLE
# =============================================================================

async def decorator_workflow_demo():
    """Demonstrate @workflow.step decorator style."""
    print("=" * 60)
    print("   DECORATOR STYLE - @workflow.step")
    print("=" * 60)
    print()

    workflow = Workflow("decorated-pipeline")

    @workflow.step("fetch", retry=2)
    async def fetch(inputs):
        print("      Fetching data...")
        await asyncio.sleep(0.1)
        return {"items": ["a", "b", "c"]}

    @workflow.step("transform", depends_on=["fetch"])
    def transform(inputs):
        items = inputs.get("fetch", {}).get("items", [])
        print(f"      Transforming {len(items)} items...")
        return {"transformed": [x.upper() for x in items]}

    @workflow.step("save", depends_on=["transform"])
    def save(inputs):
        data = inputs.get("transform", {}).get("transformed", [])
        print(f"      Saving {data}...")
        return {"saved": True, "count": len(data)}

    print("Workflow structure:")
    print(workflow.visualize())
    print()

    print("Executing workflow:")
    print("-" * 40)
    result = await workflow.run()

    print()
    print(f"Status: {result.status}")
    print(f"All outputs: {result.outputs}")
    print()


# =============================================================================
# RETRY AND ERROR HANDLING
# =============================================================================

async def retry_workflow_demo():
    """Demonstrate retries and error handling."""
    print("=" * 60)
    print("   RETRIES - Error Handling and Recovery")
    print("=" * 60)
    print()

    attempt_count = {"value": 0}

    async def flaky_operation(inputs):
        attempt_count["value"] += 1
        print(f"      Attempt {attempt_count['value']}...")
        if attempt_count["value"] < 3:
            raise Exception("Temporary failure")
        return {"success": True}

    workflow = Workflow("retry-demo")
    workflow.add_step(Step(
        name="flaky_step",
        handler=flaky_operation,
        retry=3  # Retry up to 3 times
    ))

    print("Step with retry=3:")
    print("-" * 40)

    result = await workflow.run()

    print()
    print(f"Status: {result.status}")
    print(f"Attempts: {attempt_count['value']}")
    print(f"Step result: {result.get('flaky_step')}")
    print()


# =============================================================================
# TIMEOUT HANDLING
# =============================================================================

async def timeout_workflow_demo():
    """Demonstrate step timeouts."""
    print("=" * 60)
    print("   TIMEOUTS - Time-Limited Steps")
    print("=" * 60)
    print()

    async def slow_operation(inputs):
        print("      Starting slow operation...")
        await asyncio.sleep(5)  # Will timeout
        return {"completed": True}

    async def fast_operation(inputs):
        print("      Starting fast operation...")
        await asyncio.sleep(0.1)
        return {"completed": True}

    workflow = Workflow("timeout-demo")
    workflow.add_step(Step(
        name="slow_step",
        handler=slow_operation,
        timeout=1.0  # 1 second timeout
    ))
    workflow.add_step(Step(
        name="fast_step",
        handler=fast_operation,
        timeout=2.0
    ))

    print("Steps with timeouts:")
    print("-" * 40)

    result = await workflow.run()

    print()
    print(f"Overall status: {result.status}")
    for name, step_result in result.steps.items():
        print(f"   {name}: {step_result.status.value}")
        if step_result.error:
            print(f"      Error: {step_result.error}")
    print()


# =============================================================================
# CONDITIONAL STEPS
# =============================================================================

async def conditional_workflow_demo():
    """Demonstrate conditional step execution."""
    print("=" * 60)
    print("   CONDITIONAL STEPS - Dynamic Execution")
    print("=" * 60)
    print()

    def check_data(inputs):
        print("      Checking data validity...")
        # Return flag to control downstream steps
        return {"is_valid": True, "data": [1, 2, 3]}

    def process_valid(inputs):
        data = inputs.get("check", {}).get("data", [])
        print(f"      Processing valid data: {data}")
        return {"result": sum(data)}

    def handle_invalid(inputs):
        print("      Handling invalid data...")
        return {"error": "Data was invalid"}

    # Create workflow
    workflow = Workflow("conditional-demo")

    workflow.add_step(Step(name="check", handler=check_data))

    # This step only runs if data is valid
    workflow.add_step(Step(
        name="process_valid",
        handler=process_valid,
        depends_on=["check"],
        condition=lambda inputs: inputs.get("check", {}).get("is_valid", False)
    ))

    # This step only runs if data is invalid
    workflow.add_step(Step(
        name="handle_invalid",
        handler=handle_invalid,
        depends_on=["check"],
        condition=lambda inputs: not inputs.get("check", {}).get("is_valid", True)
    ))

    print("Workflow structure:")
    print(workflow.visualize())
    print()

    print("Executing workflow (with valid data):")
    print("-" * 40)

    result = await workflow.run()

    print()
    for name, step_result in result.steps.items():
        status = step_result.status.value
        if status == "skipped":
            print(f"   {name}: SKIPPED (condition not met)")
        else:
            print(f"   {name}: {status}")
    print()


# =============================================================================
# PRIMITIVES IN WORKFLOW - Using Pipe, Agent, Memory in Steps
# =============================================================================

async def primitives_workflow_demo():
    """Demonstrate using LangPy primitives within workflow steps."""
    print("=" * 60)
    print("   PRIMITIVES IN WORKFLOW - Pipe, Memory, Agent")
    print("=" * 60)
    print()

    from langpy_sdk import Pipe, Memory, Agent, tool

    # -------------------------------------------------------------------------
    # Create primitives that will be used in workflow steps
    # -------------------------------------------------------------------------

    # Memory for knowledge storage
    memory = Memory(name="workflow_demo")
    await memory.clear()

    # Pipe for LLM calls
    pipe = Pipe(model="gpt-4o-mini")

    # Tool for agent
    @tool(
        "search_knowledge",
        "Search the knowledge base",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )
    async def search_knowledge(query: str) -> str:
        results = await memory.search(query, limit=3)
        if results:
            return "\n".join([f"- {r.text}" for r in results])
        return "No results found"

    # Agent with memory search tool
    agent = Agent(model="gpt-4o-mini", tools=[search_knowledge])

    # -------------------------------------------------------------------------
    # Define workflow steps that USE the primitives
    # -------------------------------------------------------------------------

    async def ingest_knowledge(inputs):
        """Step 1: Add knowledge to Memory primitive."""
        knowledge = inputs.get("knowledge", [])
        print(f"      Ingesting {len(knowledge)} documents into Memory...")
        await memory.add_many(knowledge)
        return {"ingested": len(knowledge)}

    async def retrieve_context(inputs):
        """Step 2: Use Memory primitive to search for context."""
        query = inputs.get("query", "")
        print(f"      Searching Memory for: '{query}'...")
        results = await memory.search(query, limit=3)
        context = "\n".join([r.text for r in results])
        return {"context": context, "num_results": len(results)}

    async def generate_answer(inputs):
        """Step 3: Use Pipe primitive to generate answer with context."""
        query = inputs.get("query", "")
        context = inputs.get("retrieve", {}).get("context", "")

        print(f"      Generating answer with Pipe...")
        prompt = f"""Based on this context:
{context}

Answer this question: {query}"""

        response = await pipe.quick(prompt)
        return {"answer": response}

    async def agent_followup(inputs):
        """Step 4: Use Agent primitive for follow-up with tools."""
        answer = inputs.get("generate", {}).get("answer", "")

        print(f"      Agent checking answer with tools...")
        response = await agent.quick(
            f"Verify this answer is correct: {answer}",
            system="You are a fact-checker. Use the search_knowledge tool to verify claims."
        )
        return {"verified": response}

    # -------------------------------------------------------------------------
    # Create the workflow
    # -------------------------------------------------------------------------

    workflow = Workflow("rag-with-primitives")

    workflow.add_step(Step(name="ingest", handler=ingest_knowledge))
    workflow.add_step(Step(name="retrieve", handler=retrieve_context, depends_on=["ingest"]))
    workflow.add_step(Step(name="generate", handler=generate_answer, depends_on=["retrieve"]))
    workflow.add_step(Step(name="verify", handler=agent_followup, depends_on=["generate"]))

    print("RAG Workflow with Primitives:")
    print(workflow.visualize())
    print()

    print("This workflow demonstrates:")
    print("   - Step 1 (ingest): Uses Memory.add_many()")
    print("   - Step 2 (retrieve): Uses Memory.search()")
    print("   - Step 3 (generate): Uses Pipe.quick()")
    print("   - Step 4 (verify): Uses Agent.quick() with tools")
    print()

    # Note: Actual execution requires OPENAI_API_KEY
    print("To run this workflow:")
    print("-" * 40)
    print("""
    result = await workflow.run({
        "knowledge": [
            "Python was created by Guido van Rossum in 1991.",
            "Python is known for its readable syntax.",
            "Python supports multiple programming paradigms."
        ],
        "query": "When was Python created?"
    })
    print(result.outputs["generate"]["answer"])
    """)
    print()


# =============================================================================
# PARALLEL PRIMITIVES - Multiple LLM Calls
# =============================================================================

async def parallel_primitives_demo():
    """Demonstrate parallel primitive execution in workflow."""
    print("=" * 60)
    print("   PARALLEL PRIMITIVES - Multiple LLM Perspectives")
    print("=" * 60)
    print()

    from langpy_sdk import Pipe

    # Create multiple pipes with different personas
    optimist = Pipe(
        model="gpt-4o-mini",
        system="You are an optimist. Find the positive angle in everything."
    )

    pessimist = Pipe(
        model="gpt-4o-mini",
        system="You are a pessimist. Point out potential problems and risks."
    )

    analyst = Pipe(
        model="gpt-4o-mini",
        system="You are a neutral analyst. Provide balanced, factual analysis."
    )

    # Workflow steps using different pipes
    async def get_optimist_view(inputs):
        topic = inputs.get("topic", "")
        print(f"      Optimist analyzing...")
        response = await optimist.quick(f"Share your view on: {topic}")
        return {"view": response}

    async def get_pessimist_view(inputs):
        topic = inputs.get("topic", "")
        print(f"      Pessimist analyzing...")
        response = await pessimist.quick(f"Share your view on: {topic}")
        return {"view": response}

    async def get_analyst_view(inputs):
        topic = inputs.get("topic", "")
        print(f"      Analyst analyzing...")
        response = await analyst.quick(f"Share your view on: {topic}")
        return {"view": response}

    async def synthesize(inputs):
        """Combine all perspectives."""
        opt = inputs.get("optimist", {}).get("view", "")
        pess = inputs.get("pessimist", {}).get("view", "")
        anal = inputs.get("analyst", {}).get("view", "")

        print(f"      Synthesizing perspectives...")
        synthesizer = Pipe(model="gpt-4o-mini")
        response = await synthesizer.quick(f"""
Synthesize these three perspectives into a balanced summary:

OPTIMIST: {opt}

PESSIMIST: {pess}

ANALYST: {anal}

Provide a balanced conclusion.""")
        return {"summary": response}

    # Create workflow with parallel steps
    workflow = Workflow("multi-perspective")

    # These three run in PARALLEL (no dependencies between them)
    workflow.add_step(Step(name="optimist", handler=get_optimist_view))
    workflow.add_step(Step(name="pessimist", handler=get_pessimist_view))
    workflow.add_step(Step(name="analyst", handler=get_analyst_view))

    # This waits for all three
    workflow.add_step(Step(
        name="synthesize",
        handler=synthesize,
        depends_on=["optimist", "pessimist", "analyst"]
    ))

    print("Multi-Perspective Analysis Workflow:")
    print(workflow.visualize())
    print()

    print("Execution pattern:")
    print("-" * 40)
    print("""
    +----------+  +-----------+  +---------+
    | Optimist |  | Pessimist |  | Analyst |  <- Run in PARALLEL
    |   Pipe   |  |   Pipe    |  |  Pipe   |
    +----+-----+  +-----+-----+  +----+----+
         |              |             |
         +--------------+-------------+
                        |
                        v
                 +--------------+
                 |  Synthesize  |  <- Waits for all three
                 |     Pipe     |
                 +--------------+
    """)
    print()


# =============================================================================
# AI PIPELINE WORKFLOW (Simulated)
# =============================================================================

async def ai_pipeline_demo():
    """Demonstrate an AI-focused workflow pipeline."""
    print("=" * 60)
    print("   AI PIPELINE - RAG Ingestion Example")
    print("=" * 60)
    print()

    # Simulated AI operations (to run without API key)
    def parse_document(inputs):
        print("      Parsing document...")
        return {"text": "LangPy is a framework for building AI applications with 9 primitives."}

    def chunk_text(inputs):
        text = inputs.get("parse", {}).get("text", "")
        print(f"      Chunking {len(text)} chars...")
        return {"chunks": [text]}

    def generate_embeddings(inputs):
        chunks = inputs.get("chunk", {}).get("chunks", [])
        print(f"      Generating embeddings for {len(chunks)} chunks...")
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in chunks]}

    def store_in_memory(inputs):
        embeddings = inputs.get("embed", {}).get("embeddings", [])
        print(f"      Storing {len(embeddings)} embeddings...")
        return {"stored": True, "count": len(embeddings)}

    # Create RAG ingestion workflow
    workflow = Workflow("rag-ingestion")

    workflow.add_step(Step(name="parse", handler=parse_document))
    workflow.add_step(Step(name="chunk", handler=chunk_text, depends_on=["parse"]))
    workflow.add_step(Step(name="embed", handler=generate_embeddings, depends_on=["chunk"]))
    workflow.add_step(Step(name="store", handler=store_in_memory, depends_on=["embed"]))

    print("RAG Ingestion Pipeline:")
    print(workflow.visualize())
    print()

    print("Executing pipeline:")
    print("-" * 40)

    result = await workflow.run({"document_path": "sample.pdf"})

    print()
    print(f"Pipeline status: {result.status}")
    print(f"Documents stored: {result.outputs.get('store', {}).get('count', 0)}")
    print()


# =============================================================================
# WORKFLOW VISUALIZATION
# =============================================================================

async def visualization_demo():
    """Demonstrate workflow visualization."""
    print("=" * 60)
    print("   VISUALIZATION - Understanding Workflow Structure")
    print("=" * 60)
    print()

    # Complex workflow
    workflow = Workflow("complex-pipeline")

    workflow.add_step(Step(name="start", handler=lambda x: {}))
    workflow.add_step(Step(name="validate", handler=lambda x: {}, depends_on=["start"]))
    workflow.add_step(Step(name="fetch_a", handler=lambda x: {}, depends_on=["validate"]))
    workflow.add_step(Step(name="fetch_b", handler=lambda x: {}, depends_on=["validate"]))
    workflow.add_step(Step(name="fetch_c", handler=lambda x: {}, depends_on=["validate"]))
    workflow.add_step(Step(name="merge", handler=lambda x: {}, depends_on=["fetch_a", "fetch_b", "fetch_c"]))
    workflow.add_step(Step(name="transform", handler=lambda x: {}, depends_on=["merge"], retry=2))
    workflow.add_step(Step(name="save", handler=lambda x: {}, depends_on=["transform"]))

    print("Complex workflow visualization:")
    print("-" * 40)
    print(workflow.visualize())
    print()

    print("Execution order:")
    print("-" * 40)
    print("""
    1. start
    2. validate (after start)
    3. fetch_a, fetch_b, fetch_c (parallel, after validate)
    4. merge (after all fetches)
    5. transform (after merge, with retry)
    6. save (after transform)
    """)
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Workflow demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 15 + "WORKFLOW PRIMITIVE DEMO" + " " * 16 + "*")
    print("*" * 60)
    print()

    await basic_workflow_demo()
    await parallel_workflow_demo()
    await decorator_workflow_demo()
    await retry_workflow_demo()
    await timeout_workflow_demo()
    await conditional_workflow_demo()
    await primitives_workflow_demo()    # Shows Pipe, Memory, Agent in steps
    await parallel_primitives_demo()    # Shows parallel Pipe calls
    await ai_pipeline_demo()
    await visualization_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
