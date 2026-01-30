from .base import BaseEmbedder

class HFAsyncEmbedder(BaseEmbedder):
    def __init__(self, model: str):
        self.name = model
        self.dim = 768

    async def embed(self, texts):
        # TODO: integrate Inference-API / local model
        return [[0.0] * self.dim for _ in texts] 