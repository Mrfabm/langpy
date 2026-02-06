"""
Simple in-memory vector store for testing and development.

Uses cosine similarity for semantic search.
"""

from __future__ import annotations
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseVectorStore


class MemoryStore(BaseVectorStore):
    """Simple in-memory vector store using numpy for similarity search."""

    def __init__(self, embedding_model: str = "openai:text-embedding-3-small"):
        """
        Initialize memory store.

        Args:
            embedding_model: Embedding model name (for compatibility)
        """
        self.embedding_model = embedding_model
        self._texts: List[str] = []
        self._embeddings: List[List[float]] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._embedder = None

    def _get_embedder(self):
        """Get or create embedder."""
        if self._embedder is None:
            import os
            from embed.openai_async import OpenAIAsyncEmbedder
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LANGPY_API_KEY')
            self._embedder = OpenAIAsyncEmbedder(
                model=self.embedding_model,
                api_key=api_key
            )
        return self._embedder

    async def add(self, texts: List[str], metas: List[Dict[str, Any]]) -> None:
        """
        Add texts with metadata to store.

        Args:
            texts: List of text chunks to add
            metas: List of metadata dicts (one per text)
        """
        if not texts:
            return

        # Get embeddings
        embedder = self._get_embedder()
        embeddings = await embedder.embed(texts)

        # Store
        self._texts.extend(texts)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(metas)

    async def query(
        self,
        query: str,
        k: int,
        filt: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query store for similar content.

        Args:
            query: Query string
            k: Number of results
            filt: Optional metadata filter

        Returns:
            List of results with text, score, and metadata
        """
        if not self._texts:
            return []

        # Get query embedding
        embedder = self._get_embedder()
        query_embeddings = await embedder.embed([query])
        query_embedding = query_embeddings[0]

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self._embeddings):
            # Check filter
            if filt:
                metadata = self._metadatas[i]
                if not self._matches_filter(metadata, filt):
                    continue

            # Cosine similarity
            score = self._cosine_similarity(query_embedding, emb)
            similarities.append((i, score))

        # Sort by score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for idx, score in similarities[:k]:
            results.append({
                "text": self._texts[idx],
                "score": float(score),
                "metadata": self._metadatas[idx].copy()
            })

        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _matches_filter(self, metadata: Dict[str, Any], filt: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filt.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    async def token_usage(self) -> int:
        """Get total token count (approximate)."""
        # Rough estimate: ~0.75 tokens per word
        total_words = sum(len(text.split()) for text in self._texts)
        return int(total_words * 0.75)

    async def clear(self) -> None:
        """Clear all stored data."""
        self._texts = []
        self._embeddings = []
        self._metadatas = []

    async def delete_by_filter(self, filt: Dict[str, Any]) -> int:
        """Delete documents matching filter criteria."""
        indices_to_keep = []
        deleted_count = 0

        for i, metadata in enumerate(self._metadatas):
            if self._matches_filter(metadata, filt):
                deleted_count += 1
            else:
                indices_to_keep.append(i)

        # Keep only non-matching items
        self._texts = [self._texts[i] for i in indices_to_keep]
        self._embeddings = [self._embeddings[i] for i in indices_to_keep]
        self._metadatas = [self._metadatas[i] for i in indices_to_keep]

        return deleted_count

    async def update_metadata(
        self,
        filt: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> int:
        """Update metadata for documents matching filter criteria."""
        updated_count = 0

        for metadata in self._metadatas:
            if self._matches_filter(metadata, filt):
                metadata.update(updates)
                updated_count += 1

        return updated_count

    async def get_metadata_stats(self) -> Dict[str, Any]:
        """Get statistics about stored metadata."""
        if not self._metadatas:
            return {"count": 0, "keys": []}

        # Count unique keys
        keys = set()
        for metadata in self._metadatas:
            keys.update(metadata.keys())

        return {
            "count": len(self._metadatas),
            "keys": sorted(list(keys)),
            "total_texts": len(self._texts)
        }

    async def get_all_texts(self) -> List[str]:
        """Get all stored texts."""
        return self._texts.copy()

    async def get_texts_by_filter(self, filt: Dict[str, Any]) -> List[str]:
        """Get texts matching filter criteria."""
        matching_texts = []
        for i, metadata in enumerate(self._metadatas):
            if self._matches_filter(metadata, filt):
                matching_texts.append(self._texts[i])
        return matching_texts
