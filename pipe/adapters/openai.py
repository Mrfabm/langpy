from openai import AsyncOpenAI
from agent.async_agent import AgentRunResponse, AgentStreamChunk, StreamDelta, RunChoice
import os, time
from typing import Any, Dict, AsyncGenerator

Json = Dict[str, Any]

async def run(payload: Json) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = payload.get("model", "gpt-4")
    messages = payload.get("input", [])
    stream = payload.get("stream", False)
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice", "auto")
    temperature = payload.get("temperature", 0.7)
    max_tokens = payload.get("max_tokens", 1000)
    
    # Wrap model string mapping once inside the adapter
    openai_model = model.replace("openai:", "") if model.startswith("openai:") else model
    
    openai_params = {
        "model": openai_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    if tools:
        openai_params["tools"] = [
            {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters
                }
            }
            for tool in tools
        ]
        openai_params["tool_choice"] = tool_choice
    
    if stream:
        async def stream_generator():
            async for chunk in client.chat.completions.create(**openai_params):
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta.model_dump() if choice.delta else {}
                    yield AgentStreamChunk(
                        id=chunk.id,
                        object="agent.chunk",
                        created=chunk.created,
                        model=chunk.model,
                        choices=[StreamDelta(
                            delta=delta,
                            index=choice.index,
                            finish_reason=choice.finish_reason
                        )],
                        usage=chunk.usage.model_dump() if chunk.usage else None
                    )
        return stream_generator()
    else:
        response = await client.chat.completions.create(**openai_params)
        
        # Surface usage cost
        if response.usage:
            print(f"[{response.usage.completion_tokens} tokens]")
        
        return AgentRunResponse(
            id=response.id,
            object="agent.run",
            created=response.created,
            model=response.model,
            choices=[
                RunChoice(
                    message=choice.message.model_dump(),
                    finish_reason=choice.finish_reason,
                    index=choice.index
                )
                for choice in response.choices
            ],
            usage=response.usage.model_dump() if response.usage else None
        )


async def openai_async_llm(payload: Json) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
    """
    OpenAI async LLM backend function for use with LangPy primitives.
    
    Args:
        payload: Dictionary containing model, input, stream, and other parameters
        
    Returns:
        AgentRunResponse for non-streaming, AsyncGenerator for streaming
    """
    return await run(payload)
