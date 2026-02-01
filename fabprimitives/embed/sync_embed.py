"""
SyncEmbedder - Synchronous wrapper around async embedders.

Uses asyncio.run() to delegate all operations to async embedders,
eliminating code duplication while maintaining the same API.
"""

import asyncio
from typing import List, Optional

from .openai_async import OpenAIAsyncEmbedder


class SyncEmbedder:
    """
    SyncEmbedder provides a synchronous interface to embedding models.

    This class wraps async embedders using asyncio.run() to provide blocking
    functionality for generating embeddings.
    """

    def __init__(
        self,
        model: str = "openai:text-embedding-3-small",
        *,
        api_key: Optional[str] = None
    ):
        """
        Initialize SyncEmbedder.

        Args:
            model: Model name in 'vendor:model' format (e.g., 'openai:text-embedding-3-small')
            api_key: API key (defaults to environment variable)
        """
        self._async_embedder = OpenAIAsyncEmbedder(model, api_key=api_key)
        self.name = self._async_embedder.name
        self.model = self._async_embedder.model
        self.dim = self._async_embedder.dim

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "SyncEmbedder methods cannot be called from within an async context. "
                "Use async embedders instead."
            )

        return asyncio.run(coro)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts (blocking).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        return self._run_async(self._async_embedder.embed(texts))

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (blocking).

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (list of floats)
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []


def get_sync_embedder(name: str) -> SyncEmbedder:
    """
    Get a sync embedder instance by name.

    Args:
        name: Model name in 'vendor:model' format (e.g., 'openai:text-embedding-3-small')

    Returns:
        SyncEmbedder instance
    """
    return SyncEmbedder(name)
