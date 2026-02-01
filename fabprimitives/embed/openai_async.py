import os
import httpx
from .base import BaseEmbedder

class OpenAIAsyncEmbedder(BaseEmbedder):
    def __init__(self, model: str = "openai:text-embedding-3-small", *, api_key: str | None = None):
        vendor, self.model = model.split(":", 1)
        self.name = model
        self.dim = 1536
        self.key = api_key or os.getenv("OPENAI_API_KEY")

    async def embed(self, texts):
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model, "input": texts},
            )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]] 