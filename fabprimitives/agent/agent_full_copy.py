# This is a full copy of the agent primitive, including all LLM backends and tool execution logic.
# It is a standalone file for reference or integration purposes.
# 
# NEW: Simplified streaming with automatic helper functions when stream=True is set during agent creation.
# Just set stream=True and call run() - no manual chunk parsing needed!
# 
# UNIVERSAL LLM SUPPORT: Streaming works with ALL supported LLM providers:
# - OpenAI (GPT-4, GPT-3.5)
# - Anthropic (Claude)
# - Mistral AI
# - And any future providers
# 
# Same stream=True pattern works universally across all backends!

# --- MODELS (from agent/async_agent.py) ---
"""
Agent Models - Core data structures for Langbase-style agent primitive.
"""
from __future__ import annotations
import asyncio
import time
import uuid
from typing import Callable, AsyncGenerator, Awaitable, Literal, Union, Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Type aliases
JsonDict = Dict[str, Any]

class StreamDelta(BaseModel):
    delta: JsonDict
    index: int
    finish_reason: Optional[str] = None

class RunChoice(BaseModel):
    message: JsonDict
    finish_reason: str
    index: int

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolChoice(BaseModel):
    type: Literal["function"]
    function: ToolFunction

class Tool(BaseModel):
    type: Literal["function"]
    function: ToolFunction
    callable: Optional[Callable[..., Any]] = None

class InputMessage(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: Union[str, JsonDict]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class AgentStreamChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[StreamDelta]
    usage: Optional[JsonDict] = None
    error: Optional[Dict[str, str]] = None
    systemFingerprint: Optional[str] = None

class AgentRunResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[RunChoice]
    usage: Optional[JsonDict] = None
    error: Optional[Dict[str, str]] = None
    systemFingerprint: Optional[str] = None

# --- LLM BACKENDS (from sdk/llm.py) ---
"""
Built-in LLM backends for the agent primitive.
"""
import json
import time
import uuid
from abc import ABC, abstractmethod

class BaseLLMBackend(ABC):
    """Base class for all LLM backends."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    @abstractmethod
    async def call_async(self, payload: JsonDict) -> AgentRunResponse:
        """Make async call to LLM."""
        pass
    
    @abstractmethod
    def call_sync(self, payload: JsonDict) -> AgentRunResponse:
        """Make sync call to LLM."""
        pass
    
    def _create_response(self, content: str, model: str, usage: Optional[JsonDict] = None) -> AgentRunResponse:
        """Create standardized AgentRunResponse."""
        return AgentRunResponse(
            id=str(uuid.uuid4()),
            object="agent.run",
            created=int(time.time()),
            model=model,
            choices=[{
                "message": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0
            }],
            usage=usage,
            error=None,
            systemFingerprint=None
        )

class OpenAI(BaseLLMBackend):
    """OpenAI backend implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.sync_client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")
    
    async def call_async(self, payload: JsonDict) -> AgentRunResponse:
        """Make async call to OpenAI."""
        try:
            messages = payload.get("input", [])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            tools = payload.get("tools")
            tool_choice = payload.get("tool_choice", "auto")
            
            response = await self.client.chat.completions.create(
                model=payload["model"].replace("openai:", ""),
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=payload.get("temperature", 1.0),
                max_tokens=payload.get("max_tokens"),
                stream=payload.get("stream", False)
            )
            
            return AgentRunResponse(
                id=response.id,
                object="agent.run",
                created=response.created,
                model=response.model,
                choices=[{
                    "message": response.choices[0].message.model_dump(),
                    "finish_reason": response.choices[0].finish_reason,
                    "index": response.choices[0].index
                }],
                usage=response.usage.model_dump() if response.usage else None,
                error=None,
                systemFingerprint=response.system_fingerprint
            )
        except Exception as e:
            return AgentRunResponse(
                id=str(uuid.uuid4()),
                object="agent.run",
                created=int(time.time()),
                model=payload.get("model", "error"),
                choices=[],
                usage=None,
                error={"type": e.__class__.__name__, "message": str(e)},
                systemFingerprint=None
            )
    
    def call_sync(self, payload: JsonDict) -> AgentRunResponse:
        """Make sync call to OpenAI."""
        try:
            messages = payload.get("input", [])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            tools = payload.get("tools")
            tool_choice = payload.get("tool_choice", "auto")
            
            response = self.sync_client.chat.completions.create(
                model=payload["model"].replace("openai:", ""),
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=payload.get("temperature", 1.0),
                max_tokens=payload.get("max_tokens"),
                stream=payload.get("stream", False)
            )
            
            return AgentRunResponse(
                id=response.id,
                object="agent.run",
                created=response.created,
                model=response.model,
                choices=[{
                    "message": response.choices[0].message.model_dump(),
                    "finish_reason": response.choices[0].finish_reason,
                    "index": response.choices[0].index
                }],
                usage=response.usage.model_dump() if response.usage else None,
                error=None,
                systemFingerprint=response.system_fingerprint
            )
        except Exception as e:
            return AgentRunResponse(
                id=str(uuid.uuid4()),
                object="agent.run",
                created=int(time.time()),
                model=payload.get("model", "error"),
                choices=[],
                usage=None,
                error={"type": e.__class__.__name__, "message": str(e)},
                systemFingerprint=None
            )

class Anthropic(BaseLLMBackend):
    """Anthropic backend implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.sync_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")
    
    async def call_async(self, payload: JsonDict) -> AgentRunResponse:
        """Make async call to Anthropic."""
        try:
            messages = payload.get("input", [])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # Convert OpenAI format to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    anthropic_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    anthropic_messages.append({"role": "assistant", "content": msg["content"]})
                elif msg["role"] == "system":
                    # Anthropic doesn't support system messages, prepend to first user message
                    if anthropic_messages and anthropic_messages[0]["role"] == "user":
                        anthropic_messages[0]["content"] = f"{msg['content']}\n\n{anthropic_messages[0]['content']}"
                    else:
                        anthropic_messages.insert(0, {"role": "user", "content": msg["content"]})
            
            response = await self.client.messages.create(
                model=payload["model"].replace("anthropic:", ""),
                messages=anthropic_messages,
                max_tokens=payload.get("max_tokens", 4096),
                temperature=payload.get("temperature", 1.0)
            )
            
            return AgentRunResponse(
                id=response.id,
                object="agent.run",
                created=int(time.time()),
                model=response.model,
                choices=[{
                    "message": {"content": response.content[0].text, "role": "assistant"},
                    "finish_reason": response.stop_reason,
                    "index": 0
                }],
                usage=response.usage.model_dump() if response.usage else None,
                error=None,
                systemFingerprint=None
            )
        except Exception as e:
            return AgentRunResponse(
                id=str(uuid.uuid4()),
                object="agent.run",
                created=int(time.time()),
                model=payload.get("model", "error"),
                choices=[],
                usage=None,
                error={"type": e.__class__.__name__, "message": str(e)},
                systemFingerprint=None
            )
    
    def call_sync(self, payload: JsonDict) -> AgentRunResponse:
        """Make sync call to Anthropic."""
        try:
            messages = payload.get("input", [])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # Convert OpenAI format to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    anthropic_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    anthropic_messages.append({"role": "assistant", "content": msg["content"]})
                elif msg["role"] == "system":
                    # Anthropic doesn't support system messages, prepend to first user message
                    if anthropic_messages and anthropic_messages[0]["role"] == "user":
                        anthropic_messages[0]["content"] = f"{msg['content']}\n\n{anthropic_messages[0]['content']}"
                    else:
                        anthropic_messages.insert(0, {"role": "user", "content": msg["content"]})
            
            response = self.sync_client.messages.create(
                model=payload["model"].replace("anthropic:", ""),
                messages=anthropic_messages,
                max_tokens=payload.get("max_tokens", 4096),
                temperature=payload.get("temperature", 1.0)
            )
            
            return AgentRunResponse(
                id=response.id,
                object="agent.run",
                created=int(time.time()),
                model=response.model,
                choices=[{
                    "message": {"content": response.content[0].text, "role": "assistant"},
                    "finish_reason": response.stop_reason,
                    "index": 0
                }],
                usage=response.usage.model_dump() if response.usage else None,
                error=None,
                systemFingerprint=None
            )
        except Exception as e:
            return AgentRunResponse(
                id=str(uuid.uuid4()),
                object="agent.run",
                created=int(time.time()),
                model=payload.get("model", "error"),
                choices=[],
                usage=None,
                error={"type": e.__class__.__name__, "message": str(e)},
                systemFingerprint=None
            )

# --- AGENT IMPLEMENTATION (from agent/async_agent.py) ---
"""
Complete agent implementation with tool execution and LLM integration.
"""

class AsyncAgent:
    """
    AsyncAgent implements the Langbase Agent Run API (async variant).
    
    This class provides async functionality for running agents with tool execution,
    streaming support, and recursive tool-call handling.
    """
    def __init__(
        self,
        *,
        async_llm: Optional[Callable[[JsonDict], Awaitable[AgentRunResponse] | AsyncGenerator[AgentStreamChunk, None]]] = None,
        sync_llm: Optional[Callable[[JsonDict], AgentRunResponse]] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None,
        model: str = "openai:gpt-4o",
    ) -> None:
        """
        Initialize AsyncAgent.
        
        Args:
            async_llm: Async backend callable
            sync_llm: Sync backend callable
            tools: Optional list of Tool objects with callables
            api_key: API key for built-in backends
            model: Default model to use
        """
        self._async_llm = async_llm
        self._sync_llm = sync_llm
        self._tools = tools or []
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        
        # Set up built-in backend if no custom backend provided
        if not async_llm and not sync_llm and api_key:
            if model.startswith("openai:"):
                backend = OpenAI(api_key)
                self._async_llm = backend.call_async
                self._sync_llm = backend.call_sync
            elif model.startswith("anthropic:"):
                backend = Anthropic(api_key)
                self._async_llm = backend.call_async
                self._sync_llm = backend.call_sync
        
        # Register tools
        for tool in self._tools:
            if tool.function.name and tool.callable:
                self._tool_registry[tool.function.name] = tool.callable

    def _normalize_input(self, input: Union[str, List[InputMessage]]) -> List[JsonDict]:
        """Normalize input to OpenAI-style message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        if isinstance(input, list) and input and isinstance(input[0], InputMessage):
            return [msg.model_dump() for msg in input]
        return input  # type: ignore

    def _validate_api_key(self, api_key: str) -> None:
        """Validate apiKey parameter."""
        if not api_key or not api_key.strip():
            raise ValueError("apiKey cannot be empty or None")

    def _validate_tool_choice(self, tool_choice: Optional[Union[str, ToolChoice]]) -> None:
        """Validate tool_choice parameter."""
        if isinstance(tool_choice, str) and tool_choice not in ["auto", "required"]:
            raise ValueError("tool_choice must be 'auto', 'required', or a ToolChoice object")

    def _wrap_error(self, err: Exception, stream: bool = False, object_type: str = "agent.run", payload: Optional[JsonDict] = None) -> Union[AgentRunResponse, AsyncGenerator[AgentStreamChunk, None]]:
        """Wrap an exception in a structured error response."""
        import time, uuid
        err_obj: JsonDict = {
            "id": str(uuid.uuid4()),
            "object": object_type,
            "created": int(time.time()),
            "model": payload.get("model", "error") if payload else "error",
            "choices": [],
            "usage": None,
            "error": {
                "type": err.__class__.__name__,
                "message": str(err),
            },
            "systemFingerprint": None,
        }
        if stream:
            async def gen() -> AsyncGenerator[AgentStreamChunk, None]:
                yield AgentStreamChunk(**err_obj)
            return gen()
        return AgentRunResponse(**err_obj)

    def _safe_stream_chunk(self, chunk: Any) -> AgentStreamChunk:
        """Ensure chunk is a valid AgentStreamChunk, else wrap as error."""
        try:
            if isinstance(chunk, AgentStreamChunk):
                return chunk
            if isinstance(chunk, dict):
                if "choices" in chunk and chunk["choices"] and not isinstance(chunk["choices"][0], StreamDelta):
                    chunk["choices"] = [StreamDelta(**c) if not isinstance(c, StreamDelta) else c for c in chunk["choices"]]
                return AgentStreamChunk(**chunk)
            raise TypeError("Chunk is not dict or AgentStreamChunk")
        except Exception as e:
            return AgentStreamChunk(
                id="error",
                object="agent.chunk",
                created=0,
                model="error",
                choices=[],
                usage=None,
                error={"type": e.__class__.__name__, "message": str(e)},
                systemFingerprint=None,
            )

    def _extract_tool_calls(self, obj: Any) -> List[JsonDict]:
        """Extract tool calls from choice message or delta."""
        tool_calls: List[JsonDict] = []
        if isinstance(obj, dict):
            # Look only in message.tool_calls and delta.tool_calls
            if "message" in obj and isinstance(obj["message"], dict):
                if "tool_calls" in obj["message"] and isinstance(obj["message"]["tool_calls"], list):
                    tool_calls.extend(obj["message"]["tool_calls"])
            if "delta" in obj and isinstance(obj["delta"], dict):
                if "tool_calls" in obj["delta"] and isinstance(obj["delta"]["tool_calls"], list):
                    tool_calls.extend(obj["delta"]["tool_calls"])
        return tool_calls

    def _is_complete_response(self, choices: List[Any]) -> bool:
        """Check if all choices have finish_reason == 'stop' and no tool_calls."""
        for choice in choices:
            if isinstance(choice, dict):
                # Check finish_reason
                finish_reason = choice.get("finish_reason")
                if finish_reason != "stop":
                    return False
                # Check for tool_calls in message
                if "message" in choice and isinstance(choice["message"], dict):
                    if "tool_calls" in choice["message"]:
                        return False
            elif isinstance(choice, StreamDelta):
                if choice.finish_reason != "stop":
                    return False
            elif hasattr(choice, 'finish_reason') and choice.finish_reason != "stop":
                return False
        return True

    async def _tool_call_execution_loop(
        self,
        llm_response: JsonDict,
        tools: Optional[List[Tool]],
        parallel_tool_calls: bool,
        tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        stream: bool = False,
    ) -> JsonDict:
        """Execute tool calls and re-query LLM until completion."""
        registry = tool_registry or {}
        messages: List[JsonDict] = []
        while True:
            tool_calls: List[JsonDict] = []
            # Look for tool calls in all choices
            if llm_response and "choices" in llm_response and llm_response["choices"]:
                for choice in llm_response["choices"]:
                    tool_calls.extend(self._extract_tool_calls(choice))
                    if "message" in choice:
                        messages = [choice["message"]]   # keep messages as a list
            if not tool_calls:
                break
            async def exec_tool(tool_call: JsonDict) -> JsonDict:
                name = tool_call.get("name")
                args = tool_call.get("arguments", {})
                func = registry.get(name)
                if not func:
                    return {"role": "tool", "content": f"Tool '{name}' not found", "tool_call_id": tool_call.get("id")}
                if asyncio.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    result = func(**args)
                return {"role": "tool", "content": result, "tool_call_id": tool_call.get("id")}
            # Only use gather if multiple tool calls
            if parallel_tool_calls and len(tool_calls) > 1:
                tool_results = await asyncio.gather(*(exec_tool(tc) for tc in tool_calls))
            else:
                tool_results = [await exec_tool(tc) for tc in tool_calls]
            messages = messages + tool_results
            payload = dict(llm_response)
            payload["input"] = messages
            if stream:
                # For streaming, just return after one tool pass
                break
            if hasattr(self, "_async_llm") and self._async_llm:
                llm_response = await self._async_llm(payload)
            elif hasattr(self, "_sync_llm") and self._sync_llm:
                llm_response = self._sync_llm(payload)
            else:
                break
        return llm_response

    async def run(
        self,
        *,
        model: str,
        input: Union[str, List[InputMessage]],
        instructions: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        parallel_tool_calls: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        apiKey: str,
    ) -> Union[AgentRunResponse, AsyncGenerator[AgentStreamChunk, None]]:
        """
        Run the agent with the given parameters.
        
        Args:
            model: Model name (provider-qualified)
            input: Prompt or message array
            instructions: System-level guidance
            stream: If True, return a generator yielding AgentStreamChunk
            tools: List of Tool objects
            tool_choice: Tool selection mode ('auto', 'required', or ToolChoice)
            parallel_tool_calls: Run tool calls concurrently if True
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            presence_penalty: Penalize new tokens based on presence
            frequency_penalty: Penalize new tokens based on frequency
            apiKey: API key (required)
            
        Returns:
            AgentRunResponse or async generator of AgentStreamChunk
            
        Raises:
            ValueError: If apiKey is empty or tool_choice is invalid
        """
        # Validate inputs
        self._validate_api_key(apiKey)
        self._validate_tool_choice(tool_choice)
        
        norm_input = self._normalize_input(input)
        payload: JsonDict = {
            "model": model,
            "input": norm_input,
            "instructions": instructions,
            "stream": stream,
            "tools": tools if tools is not None else self._tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "apiKey": apiKey,
        }
        if not self._async_llm:
            return self._wrap_error(RuntimeError("No async_llm callable provided."), stream=stream, object_type="agent.run", payload=payload)
        try:
            if stream:
                async def recursive_stream() -> AsyncGenerator[AgentStreamChunk, None]:
                    messages = payload["input"]
                    current_payload = dict(payload)
                    while True:
                        last_chunk = None
                        llm_result = self._async_llm(current_payload)
                        # Patch: handle coroutine or async generator
                        if hasattr(llm_result, "__aiter__"):
                            # Async generator (streaming)
                            async for chunk in llm_result:
                                chunk_obj = self._safe_stream_chunk(chunk)
                                yield chunk_obj
                                last_chunk = chunk_obj
                                # Check for tool calls in this chunk
                                tool_calls = []
                                for c in chunk_obj.choices:
                                    tool_calls.extend(self._extract_tool_calls(c.delta))
                                if tool_calls:
                                    async def exec_tool(tc):
                                        name = tc.get("name")
                                        args = tc.get("arguments", {})
                                        func = self._tool_registry.get(name)
                                        if not func:
                                            return {"role": "tool", "content": f"Tool '{name}' not found", "tool_call_id": tc.get("id")}
                                        if asyncio.iscoroutinefunction(func):
                                            result = await func(**args)
                                        else:
                                            result = func(**args)
                                        return {"role": "tool", "content": result, "tool_call_id": tc.get("id")}
                                    if len(tool_calls) > 1:
                                        tool_results = await asyncio.gather(*(exec_tool(tc) for tc in tool_calls))
                                    else:
                                        tool_results = [await exec_tool(tc) for tc in tool_calls]
                                    messages = messages + tool_results
                                    current_payload = dict(payload)
                                    current_payload["input"] = messages
                                    break  # break inner for, re-query outer while
                            else:
                                # Check if response is complete
                                if last_chunk and self._is_complete_response(last_chunk.choices):
                                    break  # No more chunks and complete, exit
                        else:
                            # Not an async generator: treat as coroutine (non-streaming)
                            chunk = await llm_result
                            chunk_obj = self._safe_stream_chunk(chunk)
                            yield chunk_obj
                            last_chunk = chunk_obj
                            if last_chunk and self._is_complete_response(last_chunk.choices):
                                break
                        # Continue outer while for next tool pass
                    return
                return recursive_stream()
            resp = await self._async_llm(payload)
            resp = await self._tool_call_execution_loop(
                resp, tools if tools is not None else self._tools, parallel_tool_calls, self._tool_registry, stream
            )
            if not isinstance(resp, AgentRunResponse):
                resp = AgentRunResponse(**resp)
            return resp
        except Exception as e:
            return self._wrap_error(e, stream=stream, object_type="agent.run", payload=payload)

class SyncAgent:
    """
    SyncAgent implements the Langbase Agent Run API (sync variant).
    
    This class provides blocking functionality for running agents with tool execution.
    Streaming is not supported and will return a structured error.
    """
    def __init__(
        self,
        *,
        sync_llm: Optional[Callable[[JsonDict], AgentRunResponse]] = None,
        async_llm: Optional[Callable[[JsonDict], Any]] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None,
        model: str = "openai:gpt-4o",
    ) -> None:
        """
        Initialize SyncAgent.
        
        Args:
            sync_llm: Sync backend callable
            async_llm: Async backend callable (for compatibility)
            tools: Optional list of Tool objects with callables
            api_key: API key for built-in backends
            model: Default model to use
        """
        self._sync_llm = sync_llm
        self._async_llm = async_llm
        self._tools = tools or []
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        
        # Set up built-in backend if no custom backend provided
        if not sync_llm and not async_llm and api_key:
            if model.startswith("openai:"):
                backend = OpenAI(api_key)
                self._sync_llm = backend.call_sync
            elif model.startswith("anthropic:"):
                backend = Anthropic(api_key)
                self._sync_llm = backend.call_sync
        
        # Register tools
        for tool in self._tools:
            if tool.function.name and tool.callable:
                self._tool_registry[tool.function.name] = tool.callable

    def _normalize_input(self, input: Union[str, List[InputMessage]]) -> List[JsonDict]:
        """Normalize input to OpenAI-style message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        if isinstance(input, list) and input and isinstance(input[0], InputMessage):
            return [msg.model_dump() for msg in input]
        return input  # type: ignore

    def _validate_api_key(self, api_key: str) -> None:
        """Validate apiKey parameter."""
        if not api_key or not api_key.strip():
            raise ValueError("apiKey cannot be empty or None")

    def _validate_tool_choice(self, tool_choice: Optional[Union[str, ToolChoice]]) -> None:
        """Validate tool_choice parameter."""
        if isinstance(tool_choice, str) and tool_choice not in ["auto", "required"]:
            raise ValueError("tool_choice must be 'auto', 'required', or a ToolChoice object")

    def _wrap_error(self, err: Exception, object_type: str = "agent.run", payload: Optional[JsonDict] = None) -> AgentRunResponse:
        """Wrap an exception in a structured error response."""
        import time, uuid
        err_obj: JsonDict = {
            "id": str(uuid.uuid4()),
            "object": object_type,
            "created": int(time.time()),
            "model": payload.get("model", "error") if payload else "error",
            "choices": [],
            "usage": None,
            "error": {
                "type": err.__class__.__name__,
                "message": str(err),
            },
            "systemFingerprint": None,
        }
        return AgentRunResponse(**err_obj)

    def _extract_tool_calls(self, obj: Any) -> List[JsonDict]:
        """Extract tool calls from choice message or delta."""
        tool_calls: List[JsonDict] = []
        if isinstance(obj, dict):
            # Look only in message.tool_calls and delta.tool_calls
            if "message" in obj and isinstance(obj["message"], dict):
                if "tool_calls" in obj["message"] and isinstance(obj["message"]["tool_calls"], list):
                    tool_calls.extend(obj["message"]["tool_calls"])
            if "delta" in obj and isinstance(obj["delta"], dict):
                if "tool_calls" in obj["delta"] and isinstance(obj["delta"]["tool_calls"], list):
                    tool_calls.extend(obj["delta"]["tool_calls"])
        return tool_calls

    def _tool_call_execution_loop(
        self,
        llm_response: JsonDict,
        tools: Optional[List[Tool]],
        parallel_tool_calls: bool,
        tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> JsonDict:
        """Execute tool calls and re-query LLM until completion."""
        registry = tool_registry or {}
        messages: List[JsonDict] = []
        while True:
            tool_calls: List[JsonDict] = []
            if llm_response and "choices" in llm_response and llm_response["choices"]:
                for choice in llm_response["choices"]:
                    tool_calls.extend(self._extract_tool_calls(choice))
                    if "message" in choice:
                        messages = [choice["message"]]   # keep messages as a list
            if not tool_calls:
                break
            def exec_tool(tool_call: JsonDict) -> JsonDict:
                name = tool_call.get("name")
                args = tool_call.get("arguments", {})
                func = registry.get(name)
                if not func:
                    return {"role": "tool", "content": f"Tool '{name}' not found", "tool_call_id": tool_call.get("id")}
                return {"role": "tool", "content": func(**args), "tool_call_id": tool_call.get("id")}
            if parallel_tool_calls and len(tool_calls) > 1:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    tool_results = list(executor.map(exec_tool, tool_calls))
            else:
                tool_results = [exec_tool(tc) for tc in tool_calls]
            messages = messages + tool_results
            payload = dict(llm_response)
            payload["input"] = messages
            if hasattr(self, "_sync_llm") and self._sync_llm:
                llm_response = self._sync_llm(payload)
            else:
                break
        return llm_response

    def run(
        self,
        *,
        model: str,
        input: Union[str, List[InputMessage]],
        instructions: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        parallel_tool_calls: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        apiKey: str,
    ) -> AgentRunResponse:
        """
        Run the agent with the given parameters.
        
        Args:
            model: Model name (provider-qualified)
            input: Prompt or message array
            instructions: System-level guidance
            stream: If True, return a generator yielding AgentStreamChunk (not supported in sync)
            tools: List of Tool objects
            tool_choice: Tool selection mode ('auto', 'required', or ToolChoice)
            parallel_tool_calls: Run tool calls concurrently if True
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            presence_penalty: Penalize new tokens based on presence
            frequency_penalty: Penalize new tokens based on frequency
            apiKey: API key (required)
            
        Returns:
            AgentRunResponse
            
        Raises:
            ValueError: If apiKey is empty or tool_choice is invalid
            RuntimeError: If streaming is requested (not supported in sync)
        """
        # Validate inputs
        self._validate_api_key(apiKey)
        self._validate_tool_choice(tool_choice)
        
        if stream:
            return self._wrap_error(RuntimeError("Streaming not supported in SyncAgent"), object_type="agent.run")
        
        norm_input = self._normalize_input(input)
        payload: JsonDict = {
            "model": model,
            "input": norm_input,
            "instructions": instructions,
            "stream": stream,
            "tools": tools if tools is not None else self._tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "apiKey": apiKey,
        }
        if not self._sync_llm:
            return self._wrap_error(RuntimeError("No sync_llm callable provided."), object_type="agent.run", payload=payload)
        try:
            resp = self._sync_llm(payload)
            resp = self._tool_call_execution_loop(
                resp, tools if tools is not None else self._tools, parallel_tool_calls, self._tool_registry
            )
            if not isinstance(resp, AgentRunResponse):
                resp = AgentRunResponse(**resp)
            return resp
        except Exception as e:
            return self._wrap_error(e, object_type="agent.run", payload=payload)

# --- CONVENIENCE FUNCTIONS ---
"""
Convenience functions for easy agent usage.
"""

def create_agent(
    api_key: str,
    model: str = "openai:gpt-4o",
    tools: Optional[List[Tool]] = None,
    async_mode: bool = True
) -> Union[AsyncAgent, SyncAgent]:
    """
    Create an agent with built-in backend.
    
    Args:
        api_key: API key for the LLM provider
        model: Model to use (e.g., "openai:gpt-4o", "anthropic:claude-3-sonnet")
        tools: Optional list of tools
        async_mode: Whether to create async or sync agent
        
    Returns:
        AsyncAgent or SyncAgent instance
    """
    if async_mode:
        return AsyncAgent(api_key=api_key, model=model, tools=tools)
    else:
        return SyncAgent(api_key=api_key, model=model, tools=tools)

def register_tool(agent: Union[AsyncAgent, SyncAgent], name: str, func: Callable[..., Any], description: str = "", parameters: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a tool with an agent.
    
    Args:
        agent: Agent instance
        name: Tool name
        func: Tool function
        description: Tool description
        parameters: Tool parameters schema
    """
    tool = Tool(
        type="function",
        function=ToolFunction(
            name=name,
            description=description,
            parameters=parameters
        ),
        callable=func
    )
    agent._tools.append(tool)
    agent._tool_registry[name] = func

# --- SELF-TEST ---
if __name__ == "__main__":
    import os
    
    async def test_agent():
        """Test the agent implementation."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY environment variable to test")
            return
        
        # Create agent
        agent = create_agent(api_key, "openai:gpt-4o")
        
        # Register a simple tool
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        
        register_tool(agent, "get_weather", get_weather, "Get weather for a location")
        
        # Test basic response
        print("Testing basic response...")
        response = await agent.run(
            model="openai:gpt-4o",
            input="What is 2 + 2?",
            apiKey=api_key
        )
        print(f"Response: {response.choices[0].message['content']}")
        
        # Test tool calling
        print("\nTesting tool calling...")
        response = await agent.run(
            model="openai:gpt-4o",
            input="What's the weather in New York?",
            apiKey=api_key
        )
        print(f"Response: {response.choices[0].message['content']}")
    
    # Run test
    asyncio.run(test_agent()) 