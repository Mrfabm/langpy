"""
Chunker Primitive
=================
Split text into smaller chunks for embedding and retrieval.

Chunking is essential for:
    - Processing long documents that exceed context limits
    - Creating granular search results
    - Optimizing embedding quality (smaller chunks = better focus)
    - Building RAG pipelines

Architecture:
    Long Text → Chunker → [Chunk1, Chunk2, Chunk3, ...]

    ┌────────────────────────┐
    │        Chunker         │
    │  ┌──────────────────┐  │
    │  │    Long Text     │  │
    │  └────────┬─────────┘  │
    │           ↓            │
    │  ┌────┐ ┌────┐ ┌────┐  │
    │  │ C1 │ │ C2 │ │ C3 │  │
    │  └────┘ └────┘ └────┘  │
    └────────────────────────┘
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

# Simple Chunker implementation for standalone example
# (The langpy.primitives.Chunker expects a client parameter)
class Chunker:
        """Simple text chunker."""

        def __init__(
            self,
            chunk_size: int = 500,
            overlap: int = 50,
            separator: str = "\n\n"
        ):
            self.chunk_size = chunk_size
            self.overlap = overlap
            self.separator = separator

        async def chunk(self, text: str) -> list[str]:
            """Split text into chunks."""
            if not text:
                return []

            chunks = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)

                # Try to break at natural boundaries
                if end < text_len:
                    # Look for paragraph break
                    para_break = text.rfind("\n\n", start, end)
                    if para_break > start + self.chunk_size // 2:
                        end = para_break + 2
                    else:
                        # Look for sentence break
                        for sep in [". ", "! ", "? ", "\n"]:
                            sent_break = text.rfind(sep, start, end)
                            if sent_break > start + self.chunk_size // 2:
                                end = sent_break + len(sep)
                                break

                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                start = end - self.overlap if self.overlap < end - start else end

            return chunks


# =============================================================================
# SAMPLE DOCUMENTS
# =============================================================================

SAMPLE_ARTICLE = """
Artificial Intelligence: A Comprehensive Overview

Introduction to AI

Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind, such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. A subset of artificial intelligence is machine learning, which refers to the concept that computer programs can automatically learn from and adapt to new data without being assisted by humans.

History of AI Development

The field of AI research was founded at a workshop at Dartmouth College in 1956. The attendees became the founders and leaders of AI research. They and their students produced programs that the press described as astonishing: computers were learning checkers strategies, solving word problems in algebra, proving logical theorems, and speaking English.

By the middle of the 1960s, research in the U.S. was heavily funded by the Department of Defense and laboratories had been established around the world. AI's founders were optimistic about the future: Herbert Simon predicted, "machines will be capable, within twenty years, of doing any work a man can do."

Types of Artificial Intelligence

AI can be categorized in several ways. The most common classification divides AI into three categories based on capabilities:

1. Narrow AI (Weak AI): This type of AI is designed to perform a narrow task, such as facial recognition, internet searches, or driving a car. Most current AI implementations fall into this category.

2. General AI (Strong AI): This type of AI would have generalized human cognitive abilities. When presented with an unfamiliar task, a strong AI system could find a solution without human intervention.

3. Superintelligent AI: This refers to an intellect that is much smarter than the best human brains in practically every field, including scientific creativity, general wisdom, and social skills.

Applications of AI

AI is being applied across various industries with significant impact:

Healthcare: AI algorithms can analyze complex medical data to help diagnose diseases, recommend treatments, and predict patient outcomes.

Finance: AI powers fraud detection systems, algorithmic trading, and personalized banking services.

Transportation: Self-driving cars and traffic management systems rely heavily on AI technologies.

Entertainment: Streaming services use AI to recommend content, while video games use AI to create responsive, adaptive gaming experiences.

Future Considerations

As AI continues to evolve, important ethical considerations must be addressed. These include concerns about privacy, job displacement, algorithmic bias, and the potential for AI systems to be used maliciously. Responsible AI development requires careful consideration of these issues alongside technical advancement.
"""


# =============================================================================
# BASIC CHUNKING
# =============================================================================

async def basic_chunking_demo():
    """Demonstrate basic text chunking."""
    print("=" * 60)
    print("   BASIC CHUNKING - Splitting Long Text")
    print("=" * 60)
    print()

    # Default chunker
    chunk_size = 500
    overlap = 50
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)

    print(f"Original text length: {len(SAMPLE_ARTICLE)} characters")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print()

    chunks = await chunker.chunk(SAMPLE_ARTICLE)

    print(f"Number of chunks: {len(chunks)}")
    print("-" * 40)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        preview = chunk[:100].replace("\n", " ")
        print(f"   '{preview}...'")

    print()


# =============================================================================
# CONFIGURABLE CHUNK SIZES
# =============================================================================

async def chunk_size_demo():
    """Demonstrate different chunk sizes."""
    print("=" * 60)
    print("   CHUNK SIZES - Comparing Different Settings")
    print("=" * 60)
    print()

    text = SAMPLE_ARTICLE[:2000]  # Use first part of article
    print(f"Text length: {len(text)} characters")
    print()

    # Small chunks (more granular)
    small_chunker = Chunker(chunk_size=200, overlap=20)
    small_chunks = await small_chunker.chunk(text)

    print("1. Small chunks (200 chars):")
    print(f"   Count: {len(small_chunks)} chunks")
    print(f"   Use for: Fine-grained search, detailed retrieval")
    print()

    # Medium chunks (balanced)
    medium_chunker = Chunker(chunk_size=500, overlap=50)
    medium_chunks = await medium_chunker.chunk(text)

    print("2. Medium chunks (500 chars):")
    print(f"   Count: {len(medium_chunks)} chunks")
    print(f"   Use for: General RAG, balanced context")
    print()

    # Large chunks (more context)
    large_chunker = Chunker(chunk_size=1000, overlap=100)
    large_chunks = await large_chunker.chunk(text)

    print("3. Large chunks (1000 chars):")
    print(f"   Count: {len(large_chunks)} chunks")
    print(f"   Use for: Summarization, document-level context")
    print()


# =============================================================================
# OVERLAP DEMONSTRATION
# =============================================================================

async def overlap_demo():
    """Demonstrate the effect of overlap on chunk boundaries."""
    print("=" * 60)
    print("   OVERLAP - Maintaining Context Across Chunks")
    print("=" * 60)
    print()

    text = """First paragraph contains important context about the topic.

Second paragraph builds on the first with more details.

Third paragraph concludes the thought started earlier.

Fourth paragraph introduces a new concept entirely."""

    print("Original text sections:")
    print("-" * 40)
    print(text)
    print()

    # Without overlap
    no_overlap = Chunker(chunk_size=100, overlap=0)
    chunks_no_overlap = await no_overlap.chunk(text)

    print("1. WITHOUT overlap (0):")
    print("-" * 40)
    for i, chunk in enumerate(chunks_no_overlap):
        print(f"   Chunk {i+1}: '{chunk[:60]}...'")
    print()

    # With overlap
    with_overlap = Chunker(chunk_size=100, overlap=30)
    chunks_with_overlap = await with_overlap.chunk(text)

    print("2. WITH overlap (30 chars):")
    print("-" * 40)
    for i, chunk in enumerate(chunks_with_overlap):
        print(f"   Chunk {i+1}: '{chunk[:60]}...'")
    print()

    print("Notice how overlap helps maintain context between chunks.")
    print()


# =============================================================================
# CHUNKING FOR RAG
# =============================================================================

async def rag_chunking_demo():
    """Demonstrate chunking in a RAG pipeline context."""
    print("=" * 60)
    print("   RAG CHUNKING - Preparing Documents for Retrieval")
    print("=" * 60)
    print()

    # Simulate processing multiple documents
    documents = [
        ("AI Overview", SAMPLE_ARTICLE[:1500]),
        ("ML Basics", """
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

Supervised Learning involves training on labeled data, where the algorithm learns to map inputs to known outputs. Common algorithms include linear regression, decision trees, and neural networks.

Unsupervised Learning works with unlabeled data, finding hidden patterns or intrinsic structures. Clustering and dimensionality reduction are typical applications.

Reinforcement Learning trains agents to make decisions by rewarding desired behaviors. This approach powers game-playing AI and robotics applications.
        """),
    ]

    chunker = Chunker(chunk_size=400, overlap=40)

    print("Processing documents for RAG:")
    print("-" * 40)

    all_chunks = []
    for title, content in documents:
        chunks = await chunker.chunk(content)
        for i, chunk in enumerate(chunks):
            # In a real system, you'd store metadata with each chunk
            all_chunks.append({
                "source": title,
                "chunk_id": i,
                "text": chunk
            })
        print(f"   '{title}': {len(chunks)} chunks")

    print()
    print(f"Total chunks for embedding: {len(all_chunks)}")
    print()

    print("Sample chunks with metadata:")
    print("-" * 40)
    for chunk_data in all_chunks[:3]:
        print(f"   Source: {chunk_data['source']}, Chunk: {chunk_data['chunk_id']}")
        preview = chunk_data['text'][:80].replace("\n", " ")
        print(f"   Text: '{preview}...'")
        print()


# =============================================================================
# SEMANTIC CHUNKING PREVIEW
# =============================================================================

async def semantic_chunking_preview():
    """Preview of semantic chunking concepts."""
    print("=" * 60)
    print("   SEMANTIC CHUNKING - Intelligent Boundaries")
    print("=" * 60)
    print()

    print("Standard chunking splits by character count.")
    print("Semantic chunking considers meaning:")
    print()
    print("   - Split at paragraph boundaries")
    print("   - Keep related sentences together")
    print("   - Maintain heading with its content")
    print("   - Avoid splitting code blocks or lists")
    print()

    # Demonstrate paragraph-aware chunking
    chunker = Chunker(chunk_size=500, overlap=50)

    text = """# Introduction

This section introduces the topic with background information that sets the stage for what follows.

# Main Concepts

Here we discuss the core ideas:
1. First concept with explanation
2. Second concept with details
3. Third concept with examples

# Conclusion

Summary and final thoughts bring everything together."""

    chunks = await chunker.chunk(text)

    print("Paragraph-aware chunking result:")
    print("-" * 40)
    for i, chunk in enumerate(chunks):
        lines = chunk.count("\n")
        print(f"   Chunk {i+1}: {len(chunk)} chars, {lines} line breaks")
        first_line = chunk.split("\n")[0]
        print(f"      Starts with: '{first_line}'")
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Chunker demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 16 + "CHUNKER PRIMITIVE DEMO" + " " * 17 + "*")
    print("*" * 60)
    print()

    await basic_chunking_demo()
    await chunk_size_demo()
    await overlap_demo()
    await rag_chunking_demo()
    await semantic_chunking_preview()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
