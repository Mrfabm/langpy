from __future__ import annotations
import httpx, json
from typing import Any, Dict, AsyncGenerator
from agent.async_agent import AgentRunResponse, AgentStreamChunk

Json = Dict[str, Any]

async def run(payload: Json) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
    url     = "https://api.mistral.ai/v1/chat/completions"
    key     = payload.pop("apiKey")
    body    = {**payload, "messages": payload.pop("input")}
    body    = {k: v for k, v in body.items() if v is not None}
    stream  = body.get("stream", False)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=90) as c:
        if stream:
            r = await c.post(url, headers=headers, json=body, timeout=None)
            r.raise_for_status()
            async def gen() -> AsyncGenerator[AgentStreamChunk, None]:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "): continue
                    if line.strip() == "data: [DONE]": break
                    yield AgentStreamChunk(**json.loads(line[6:]))
            return gen()
        r = await c.post(url, headers=headers, json=body)
        r.raise_for_status()
        return AgentRunResponse(**r.json())
