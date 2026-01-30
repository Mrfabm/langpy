"""
Google Gemini (Pro & Flash) adapter – REST JSON
Docs: https://ai.google.dev/api/rest/v1beta/models/generateContent
"""

from __future__ import annotations
import os, json, httpx, asyncio
from typing import Any, Dict, AsyncGenerator
from agent.async_agent import AgentRunResponse, AgentStreamChunk

Json = Dict[str, Any]

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent?key={key}"
)

async def run(payload: Json) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
    api_key = payload.pop("apiKey") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    model_name = payload["model"].split(":", 1)[-1]         # gemini:gemini-pro -> gemini-pro
    stream     = payload.get("stream", False)

    # Langbase -> Gemini message mapping
    parts = [{"text": m["content"]} for m in payload.pop("input")]
    body  = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": payload.get("temperature", 0.7),
            "topP":        payload.get("top_p", 1.0),
            "maxOutputTokens": payload.get("max_tokens", 1024),
        },
        "stream": stream,
    }

    url = _GEMINI_ENDPOINT.format(model=model_name, key=api_key)
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=90) as client:
        if stream:
            r = await client.post(url, headers=headers, json=body, timeout=None)
            r.raise_for_status()

            async def gen() -> AsyncGenerator[AgentStreamChunk, None]:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    data = json.loads(line[6:])
                    delta = data["candidates"][0]["content"]["parts"][0]["text"]
                    yield AgentStreamChunk(
                        id="gemini",
                        object="agent.chunk",
                        created=0,
                        model=model_name,
                        choices=[{"delta": {"content": delta}}],
                    )
            return gen()

        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return AgentRunResponse(
            id="gemini",
            object="agent.run",
            created=0,
            model=model_name,
            choices=[{"message": {"content": content}}],
        )
