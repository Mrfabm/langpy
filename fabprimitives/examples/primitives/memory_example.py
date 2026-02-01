"""
Memory Primitive
================
Vector-based storage for semantic search and retrieval.

Memory enables:
    - Store documents with embeddings
    - Semantic search (meaning-based, not just keywords)
    - Metadata filtering
    - RAG (Retrieval-Augmented Generation) pipelines

Architecture:
    Documents → Embed → Vector Store → Semantic Search

    ┌──────────────────────────────┐
    │           Memory             │
    │  ┌────────┐    ┌──────────┐  │
    │  │ Embed  │ →  │  Vector  │  │
    │  │        │    │   Store  │  │
    │  └────────┘    └──────────┘  │
    │                     ↓        │
    │            Search Results    │
    └──────────────────────────────┘
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

from langpy_sdk import Memory, Pipe


# =============================================================================
# BASIC MEMORY OPERATIONS
# =============================================================================

async def basic_memory_demo():
    """Demonstrate basic Memory add and search operations."""
    print("=" * 60)
    print("   BASIC MEMORY - Add and Search")
    print("=" * 60)
    print()

    # Create memory store (uses FAISS locally by default)
    memory = Memory(name="demo_basic")

    # Clear previous data
    await memory.clear()

    print("1. Adding documents:")
    print("-" * 40)

    documents = [
        "Python is a versatile programming language known for its readability.",
        "JavaScript is the language of the web browser.",
        "Machine learning models learn patterns from data.",
        "Databases store and retrieve structured information.",
        "APIs allow different software systems to communicate.",
    ]

    for doc in documents:
        doc_id = await memory.add(doc)
        print(f"   Added: '{doc[:40]}...' (id: {doc_id})")

    print()

    # Search
    print("2. Semantic search:")
    print("-" * 40)

    query = "How do I learn coding?"
    results = await memory.search(query, limit=3)

    print(f"   Query: '{query}'")
    print()
    for i, result in enumerate(results):
        print(f"   Result {i+1} (score: {result.score:.3f}):")
        print(f"      '{result.text}'")
        print()


# =============================================================================
# BATCH ADDING
# =============================================================================

async def batch_add_demo():
    """Demonstrate efficient batch document adding."""
    print("=" * 60)
    print("   BATCH ADD - Efficient Bulk Operations")
    print("=" * 60)
    print()

    memory = Memory(name="demo_batch")
    await memory.clear()

    # Knowledge base about programming languages
    knowledge = [
        "Python was created by Guido van Rossum and released in 1991.",
        "Python uses indentation for code blocks instead of curly braces.",
        "Python supports multiple programming paradigms: procedural, OOP, and functional.",
        "JavaScript was created by Brendan Eich in just 10 days.",
        "JavaScript is single-threaded but supports asynchronous programming.",
        "TypeScript is a superset of JavaScript that adds static typing.",
        "Rust guarantees memory safety without a garbage collector.",
        "Go was designed at Google for simplicity and concurrency.",
        "Java follows the 'write once, run anywhere' principle.",
    ]

    print("Adding multiple documents efficiently:")
    print("-" * 40)

    doc_ids = await memory.add_many(knowledge)
    print(f"   Added {len(doc_ids)} documents in a single call")
    print()

    # Test search
    queries = [
        "Who created Python?",
        "memory safe programming language",
        "async programming"
    ]

    print("Searching the knowledge base:")
    print("-" * 40)

    for query in queries:
        results = await memory.search(query, limit=2)
        print(f"\n   Query: '{query}'")
        for result in results:
            print(f"      {result.score:.3f}: {result.text[:60]}...")

    print()


# =============================================================================
# METADATA FILTERING
# =============================================================================

async def metadata_demo():
    """Demonstrate metadata storage and filtering."""
    print("=" * 60)
    print("   METADATA - Filtering Search Results")
    print("=" * 60)
    print()

    memory = Memory(name="demo_metadata")
    await memory.clear()

    print("Adding documents with metadata:")
    print("-" * 40)

    # Add documents with category metadata
    docs_with_meta = [
        ("Python is great for data science", {"category": "programming", "language": "python"}),
        ("Use pip to install Python packages", {"category": "programming", "language": "python"}),
        ("CSS styles web page elements", {"category": "programming", "language": "css"}),
        ("Machine learning requires labeled training data", {"category": "ml", "level": "beginner"}),
        ("Neural networks have layers of neurons", {"category": "ml", "level": "intermediate"}),
        ("Transformers use attention mechanisms", {"category": "ml", "level": "advanced"}),
    ]

    for text, metadata in docs_with_meta:
        await memory.add(text, metadata=metadata)
        print(f"   '{text[:35]}...' [{metadata}]")

    print()

    # Search with metadata filter
    print("Filtered search examples:")
    print("-" * 40)

    # Filter by category
    print("\n1. Filter: category=ml")
    results = await memory.search(
        "learning techniques",
        limit=3,
        filter={"category": "ml"}
    )
    for r in results:
        print(f"      {r.score:.3f}: {r.text[:50]}...")

    # Filter by language
    print("\n2. Filter: language=python")
    results = await memory.search(
        "how to install",
        limit=3,
        filter={"language": "python"}
    )
    for r in results:
        print(f"      {r.score:.3f}: {r.text[:50]}...")

    print()


# =============================================================================
# SIMPLE RAG PATTERN
# =============================================================================

async def simple_rag_demo():
    """Demonstrate a simple RAG pattern with Memory + Pipe."""
    print("=" * 60)
    print("   SIMPLE RAG - Memory + Pipe Composition")
    print("=" * 60)
    print()

    # Create memory with knowledge base
    memory = Memory(name="demo_rag")
    await memory.clear()

    # Add knowledge about LangPy
    knowledge = [
        "LangPy is a Python framework for building AI applications.",
        "LangPy has 9 primitives: Pipe, Agent, Memory, Thread, Workflow, Parser, Chunker, Embed, and Tools.",
        "The Pipe primitive makes simple LLM calls without tool execution.",
        "The Agent primitive can use tools to interact with external systems.",
        "Memory stores documents using vector embeddings for semantic search.",
        "Thread manages conversation history for multi-turn interactions.",
        "Workflow orchestrates multiple steps with dependencies and parallel execution.",
        "Parser extracts text from various document formats like PDF and images.",
        "Chunker splits long documents into smaller pieces for embedding.",
        "Embed converts text into vector representations for similarity search.",
        "Tools are defined using the @tool decorator with JSON Schema parameters.",
    ]

    await memory.add_many(knowledge)
    print(f"Knowledge base: {len(knowledge)} documents")
    print()

    # Create pipe for generation
    pipe = Pipe(
        model="gpt-4o-mini",
        system="You are a helpful assistant. Answer questions based on the provided context. "
               "If the context doesn't contain the answer, say so."
    )

    # RAG function
    async def ask(question: str) -> str:
        """Simple RAG: retrieve context, then generate answer."""

        # Step 1: Retrieve relevant documents
        results = await memory.search(question, limit=3)
        context = "\n".join([f"- {r.text}" for r in results])

        # Step 2: Generate answer with context
        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        response = await pipe.quick(prompt)
        return response

    # Test RAG
    questions = [
        "What is LangPy?",
        "How many primitives does LangPy have?",
        "What is the difference between Pipe and Agent?",
        "How do I manage conversations in LangPy?",
    ]

    print("RAG Q&A:")
    print("-" * 40)

    for question in questions:
        print(f"\nQ: {question}")
        answer = await ask(question)
        print(f"A: {answer}")

    print()


# =============================================================================
# MEMORY STATISTICS
# =============================================================================

async def stats_demo():
    """Demonstrate memory statistics and management."""
    print("=" * 60)
    print("   MEMORY STATS - Monitoring Your Store")
    print("=" * 60)
    print()

    memory = Memory(name="demo_stats")
    await memory.clear()

    # Add some documents
    for i in range(10):
        await memory.add(f"Document number {i} with some content about topic {i % 3}")

    # Get statistics
    stats = await memory.stats()

    print("Memory Statistics:")
    print("-" * 40)
    print(f"   Total documents: {stats.total_documents}")
    print(f"   Total chunks: {stats.total_chunks}")
    print(f"   Backend: {stats.backend}")
    print()

    # Delete some documents
    print("Deleting documents by filter:")
    print("-" * 40)

    # Add documents with metadata for deletion demo
    await memory.clear()
    await memory.add("Keep this document", metadata={"keep": True})
    await memory.add("Delete this document", metadata={"keep": False})
    await memory.add("Also delete this", metadata={"keep": False})

    deleted = await memory.delete(filter={"keep": False})
    print(f"   Deleted {deleted} documents")

    stats = await memory.stats()
    print(f"   Remaining documents: {stats.total_documents}")
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Memory demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 17 + "MEMORY PRIMITIVE DEMO" + " " * 17 + "*")
    print("*" * 60)
    print()

    await basic_memory_demo()
    await batch_add_demo()
    await metadata_demo()
    await simple_rag_demo()
    await stats_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
