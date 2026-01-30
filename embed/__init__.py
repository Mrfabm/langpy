from .openai_async import OpenAIAsyncEmbedder
from .huggingface_async import HFAsyncEmbedder
from .sync_embed import SyncEmbedder, get_sync_embedder

REGISTRY = {
    "openai": OpenAIAsyncEmbedder,
    "hf": HFAsyncEmbedder,
}

def get_embedder(name: str):
    """Get async embedder instance by name (e.g., 'openai:text-embedding-3-small')."""
    vendor = name.split(":", 1)[0]
    cls = REGISTRY.get(vendor)
    if not cls:
        raise ValueError(f"Unknown embedder vendor {vendor}")
    return cls(name)

__all__ = [
    "OpenAIAsyncEmbedder",
    "HFAsyncEmbedder",
    "SyncEmbedder",
    "get_embedder",
    "get_sync_embedder",
    "REGISTRY",
] 