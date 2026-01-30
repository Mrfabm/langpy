from __future__ import annotations
import httpx, json
from typing import Any, Dict, AsyncGenerator
from agent.async_agent import AgentRunResponse, AgentStreamChunk

Json = Dict[str, Any]

async def run(payload: Json) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
    url  = "https://api.anthropic.com/v1/messages"
    key  = payload.pop("apiKey")
    body = {
        "model": payload["model"].split(":",1)[-1],
        "messages": payload.pop("input"),
        "max_tokens": payload.get("max_tokens", 1024),
        "stream": payload.get("stream", False),
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90) as c:
        if body["stream"]:
            r = await c.post(url, headers=headers, json=body, timeout=None)
            r.raise_for_status()
            async def gen() -> AsyncGenerator[AgentStreamChunk, None]:
                async for chunk in r.aiter_bytes():
                    if not chunk.strip(): continue
                    for part in chunk.split(b"\n\n"):
                        if part: yield AgentStreamChunk(**json.loads(part))
            return gen()
        r = await c.post(url, headers=headers, json=body)
        r.raise_for_status()
        return AgentRunResponse(**r.json())
