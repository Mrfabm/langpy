from __future__ import annotations
import typing as t
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """Pluggable async embedder interface (vector dimension is backend-specific)."""

    name: str                       # vendor:model id e.g. 'openai:text-embedding-3-small'
    dim: int                        # embedding dimension

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list[embedding] aligned with *texts*.""" 