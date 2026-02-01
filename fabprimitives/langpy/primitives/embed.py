"""
Embed Primitive - Langbase-compatible Embedding API.

The Embed primitive converts text to vector embeddings.

Usage:
    # Direct API
    result = await lb.embed.run(texts=["Hello", "World"])
    print(result.embeddings)

    # Pipeline composition
    pipeline = chunker | embed | store
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langpy.core.primitive import BasePrimitive, EmbedResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


class Embed(BasePrimitive):
    """
    Embed primitive - Text to vector embeddings.

    Converts text into vector representations for semantic search.

    Supported models:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - HuggingFace: sentence-transformers models
    - Custom models via adapters

    Example:
        from langpy import Langpy

        lb = Langpy()

        result = await lb.embed.run(
            texts=["Hello world", "Goodbye world"],
            model="openai:text-embedding-3-small"
        )

        for i, embedding in enumerate(result.embeddings):
            print(f"Text {i}: {len(embedding)} dimensions")
    """

    def __init__(
        self,
        client: Any = None,
        name: str = "embed",
        model: str = "openai:text-embedding-3-small"
    ):
        super().__init__(name=name, client=client)
        self._default_model = model
        self._embedder = None

    def _get_embedder(self, model: str):
        """Get or create embedder for model."""
        provider = model.split(":")[0] if ":" in model else "openai"
        model_name = model.split(":")[-1] if ":" in model else model

        try:
            if provider == "openai":
                from embed.openai_async import OpenAIAsyncEmbedder
                return OpenAIAsyncEmbedder(model=model_name)
            elif provider in ["huggingface", "hf"]:
                from embed.huggingface_async import HFAsyncEmbedder
                return HFAsyncEmbedder(model=model_name)
            else:
                # Try OpenAI as default
                from embed.openai_async import OpenAIAsyncEmbedder
                return OpenAIAsyncEmbedder(model=model_name)
        except ImportError:
            return None

    async def _run(
        self,
        texts: List[str] = None,
        text: str = None,
        model: str = None,
        api_key: str = None,
        **kwargs
    ) -> EmbedResponse:
        """
        Generate embeddings.

        Args:
            texts: List of texts to embed
            text: Single text (alternative)
            model: Embedding model
            api_key: API key (or use env var)

        Returns:
            EmbedResponse with embeddings
        """
        try:
            # Normalize input
            if text and not texts:
                texts = [text]

            if not texts:
                return EmbedResponse(
                    success=False,
                    error="texts required"
                )

            # Get model
            embed_model = model or self._default_model

            # Get API key
            resolved_key = api_key
            if not resolved_key and self._client:
                resolved_key = getattr(self._client, '_api_key', None)
            if not resolved_key:
                resolved_key = os.getenv("OPENAI_API_KEY")

            # Get embedder
            embedder = self._get_embedder(embed_model)

            if embedder:
                embeddings = await embedder.embed(texts)
            else:
                # Fallback: use OpenAI API directly
                import httpx

                provider = embed_model.split(":")[0] if ":" in embed_model else "openai"
                model_name = embed_model.split(":")[-1] if ":" in embed_model else embed_model

                if provider == "openai":
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.openai.com/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {resolved_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": model_name,
                                "input": texts
                            },
                            timeout=60.0
                        )
                        response.raise_for_status()
                        data = response.json()
                        embeddings = [item["embedding"] for item in data["data"]]
                else:
                    return EmbedResponse(
                        success=False,
                        error=f"No embedder available for {embed_model}"
                    )

            dimensions = len(embeddings[0]) if embeddings else 0

            return EmbedResponse(
                success=True,
                embeddings=embeddings,
                model=embed_model,
                dimensions=dimensions,
                count=len(embeddings)
            )

        except Exception as e:
            return EmbedResponse(success=False, error=str(e))

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Process context - embed chunks from context."""
        chunks = ctx.get("chunks")
        text = ctx.query

        if not chunks and text:
            chunks = [text]

        if not chunks:
            return Failure(PrimitiveError(
                code=ErrorCode.VALIDATION_ERROR,
                message="No text to embed (provide chunks or query)",
                primitive=self._name
            ))

        response = await self._run(
            texts=chunks,
            model=ctx.get("embed_model") or self._default_model,
            api_key=ctx.get("api_key")
        )

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error,
                primitive=self._name
            ))

        new_ctx = ctx.set("embeddings", response.embeddings)
        new_ctx = new_ctx.set("embedding_dimensions", response.dimensions)

        return Success(new_ctx)

    # Convenience methods
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed texts and return vectors."""
        response = await self._run(texts=texts, **kwargs)
        return response.embeddings if response.success else []

    async def embed_one(self, text: str, **kwargs) -> List[float]:
        """Embed single text."""
        embeddings = await self.embed([text], **kwargs)
        return embeddings[0] if embeddings else []
