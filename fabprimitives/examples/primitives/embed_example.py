"""
Embed Primitive
===============
Convert text into vector embeddings for semantic search and similarity.

Embeddings enable:
    - Semantic search (find similar meaning, not just keywords)
    - Document clustering
    - Recommendation systems
    - RAG pipelines

Architecture:
    Text → Embedding Model → Vector [float, float, ...]

    ┌─────────────────┐
    │      Embed      │
    │  ┌───────────┐  │
    │  │ Embedding │  │
    │  │   Model   │  │
    │  └───────────┘  │
    │        ↓        │
    │   [0.1, 0.3,    │
    │    0.5, ...]    │
    └─────────────────┘
"""

import asyncio
import io
import os
import sys
import math

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Simple Embed class for standalone example
# (The langpy.primitives.Embed expects a client parameter)
class Embed:
        """Simple embedder wrapper."""

        def __init__(self, model: str = "text-embedding-3-small"):
            self.model = model
            self._client = None

        @property
        def dimensions(self) -> int:
            """Return embedding dimensions for the model."""
            dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            return dims.get(self.model, 1536)

        def _get_client(self):
            if self._client is None:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            return self._client

        async def embed(self, texts: list[str]) -> list[list[float]]:
            """Embed multiple texts."""
            client = self._get_client()
            response = await client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]

        async def embed_one(self, text: str) -> list[float]:
            """Embed a single text."""
            embeddings = await self.embed([text])
            return embeddings[0] if embeddings else []


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# =============================================================================
# BASIC EMBEDDING
# =============================================================================

async def basic_embed_demo():
    """Demonstrate basic text embedding."""
    print("=" * 60)
    print("   BASIC EMBEDDING - Text to Vectors")
    print("=" * 60)
    print()

    embedder = Embed(model="text-embedding-3-small")

    print(f"Model: {embedder.model}")
    print(f"Dimensions: {embedder.dimensions}")
    print()

    # Embed a single text
    print("1. Single text embedding:")
    print("-" * 40)
    text = "The quick brown fox jumps over the lazy dog"
    embedding = await embedder.embed_one(text)

    print(f"   Text: '{text}'")
    print(f"   Vector length: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   Last 5 values: {embedding[-5:]}")
    print()


# =============================================================================
# BATCH EMBEDDING
# =============================================================================

async def batch_embed_demo():
    """Demonstrate batch embedding for efficiency."""
    print("=" * 60)
    print("   BATCH EMBEDDING - Multiple Texts")
    print("=" * 60)
    print()

    embedder = Embed(model="text-embedding-3-small")

    texts = [
        "Python is a programming language",
        "JavaScript runs in the browser",
        "Machine learning uses data",
        "Cats are popular pets",
        "Dogs are loyal companions"
    ]

    print("Embedding multiple texts in one call:")
    print("-" * 40)

    embeddings = await embedder.embed(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"   {i+1}. '{text[:35]}...'")
        print(f"      Vector: [{emb[0]:.4f}, {emb[1]:.4f}, ... {emb[-1]:.4f}]")
    print()

    print(f"Total embeddings: {len(embeddings)}")
    print(f"Each with {len(embeddings[0])} dimensions")
    print()


# =============================================================================
# SEMANTIC SIMILARITY
# =============================================================================

async def similarity_demo():
    """Demonstrate semantic similarity using embeddings."""
    print("=" * 60)
    print("   SEMANTIC SIMILARITY - Finding Related Text")
    print("=" * 60)
    print()

    embedder = Embed(model="text-embedding-3-small")

    # Query and candidate texts
    query = "How do I train a neural network?"

    candidates = [
        "Deep learning models require training data and optimization",
        "The weather today is sunny and warm",
        "Python is great for machine learning projects",
        "I enjoy cooking Italian food",
        "Backpropagation is used to train neural networks",
        "Cats and dogs make wonderful pets"
    ]

    print(f"Query: '{query}'")
    print()
    print("Candidates ranked by similarity:")
    print("-" * 40)

    # Embed everything
    all_texts = [query] + candidates
    embeddings = await embedder.embed(all_texts)

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    # Calculate similarities
    similarities = []
    for text, emb in zip(candidates, candidate_embeddings):
        score = cosine_similarity(query_embedding, emb)
        similarities.append((score, text))

    # Sort by similarity (descending)
    similarities.sort(reverse=True)

    for score, text in similarities:
        bar = "█" * int(score * 20)
        print(f"   {score:.3f} {bar}")
        print(f"         '{text}'")
        print()


# =============================================================================
# SEMANTIC CLUSTERING
# =============================================================================

async def clustering_demo():
    """Demonstrate semantic clustering using embeddings."""
    print("=" * 60)
    print("   SEMANTIC CLUSTERING - Grouping Similar Items")
    print("=" * 60)
    print()

    embedder = Embed(model="text-embedding-3-small")

    # Texts from different categories
    texts = [
        # Programming
        "Python is a great programming language",
        "JavaScript is used for web development",
        "Rust focuses on memory safety",
        # Food
        "Pizza is an Italian dish",
        "Sushi comes from Japan",
        "Tacos are popular in Mexico",
        # Animals
        "Lions are the kings of the jungle",
        "Eagles soar through the sky",
        "Dolphins are intelligent marine mammals"
    ]

    print("Finding similarity matrix:")
    print("-" * 40)

    embeddings = await embedder.embed(texts)

    # Show which texts are most similar
    print("\nTop similar pairs:")
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            pairs.append((score, texts[i][:25], texts[j][:25]))

    pairs.sort(reverse=True)
    for score, t1, t2 in pairs[:6]:
        print(f"   {score:.3f}: '{t1}...' <-> '{t2}...'")
    print()


# =============================================================================
# EMBEDDING FOR RAG
# =============================================================================

async def rag_embedding_demo():
    """Demonstrate embeddings in a RAG context."""
    print("=" * 60)
    print("   RAG CONTEXT - Embeddings for Retrieval")
    print("=" * 60)
    print()

    embedder = Embed(model="text-embedding-3-small")

    # Simulate a knowledge base
    knowledge_base = [
        "LangPy is a Python framework for building AI applications.",
        "LangPy has 9 primitives: Pipe, Agent, Memory, Thread, Workflow, Parser, Chunker, Embed, and Tools.",
        "The Memory primitive uses vector embeddings for semantic search.",
        "Agents can use tools to interact with external systems.",
        "Workflows orchestrate multiple steps with dependencies.",
        "The Pipe primitive makes simple LLM calls without tools.",
        "Embeddings convert text to vectors for similarity matching.",
    ]

    # User query
    query = "What primitives does LangPy have?"

    print(f"Knowledge Base: {len(knowledge_base)} documents")
    print(f"Query: '{query}'")
    print()

    # Embed everything
    all_texts = [query] + knowledge_base
    embeddings = await embedder.embed(all_texts)

    query_emb = embeddings[0]
    doc_embeddings = embeddings[1:]

    # Find most relevant documents
    scores = []
    for doc, emb in zip(knowledge_base, doc_embeddings):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, doc))

    scores.sort(reverse=True)

    print("Retrieved documents (top 3):")
    print("-" * 40)
    for i, (score, doc) in enumerate(scores[:3]):
        print(f"   {i+1}. Score: {score:.3f}")
        print(f"      '{doc}'")
        print()

    print("These documents would be passed to the LLM as context.")
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Embed demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 17 + "EMBED PRIMITIVE DEMO" + " " * 18 + "*")
    print("*" * 60)
    print()

    await basic_embed_demo()
    await batch_embed_demo()
    await similarity_demo()
    await clustering_demo()
    await rag_embedding_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
