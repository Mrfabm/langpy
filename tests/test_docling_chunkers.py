#!/usr/bin/env python3
"""Test script to check available Docling chunking capabilities."""

try:
    from docling.chunking import HybridChunker
    print("✓ HybridChunker available from docling.chunking")
except ImportError as e:
    print(f"✗ HybridChunker not available: {e}")

try:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    print("✓ HybridChunker available from docling_core")
except ImportError as e:
    print(f"✗ HybridChunker not available from docling_core: {e}")

try:
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    print("✓ HierarchicalChunker available")
except ImportError as e:
    print(f"✗ HierarchicalChunker not available: {e}")

try:
    from docling_core.transforms.chunker.base_chunker import BaseChunker
    print("✓ BaseChunker available")
except ImportError as e:
    print(f"✗ BaseChunker not available: {e}")

try:
    from docling_core.types.doc import DoclingDocument
    print("✓ DoclingDocument available")
except ImportError as e:
    print(f"✗ DoclingDocument not available: {e}")

print("\nTesting basic chunking functionality:")
try:
    from docling.chunking import HybridChunker
    chunker = HybridChunker()
    print("✓ HybridChunker instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate HybridChunker: {e}") 