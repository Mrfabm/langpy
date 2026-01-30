from typing import Callable, AsyncGenerator, Awaitable, Literal, Union, Optional, List, Dict, Any
from pydantic import BaseModel, Field
import asyncio
import time
import uuid

# Type alias for JSON dictionaries
JsonDict = Dict[str, Any]

# --- Pydantic models matching Langbase Agent API ---

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

# --- End models ---

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
        tools: Optional[List[Tool]] = None,
    ) -> None:
        """
        Initialize AsyncAgent.
        
        Args:
            async_llm: Async backend callable
            tools: Optional list of Tool objects with callables
        """
        self._async_llm = async_llm
        self._tools = tools or []
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
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

        # Convert AgentRunResponse to dict if needed
        if isinstance(llm_response, AgentRunResponse):
            llm_response = {
                "id": llm_response.id,
                "object": llm_response.object,
                "created": llm_response.created,
                "model": llm_response.model,
                "choices": [{"message": c.message, "finish_reason": c.finish_reason, "index": c.index} for c in llm_response.choices],
                "usage": llm_response.usage,
            }

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
                import json as json_module
                # Handle OpenAI's nested function format
                func_data = tool_call.get("function", {})
                name = func_data.get("name") if func_data else tool_call.get("name")
                args_raw = func_data.get("arguments", "{}") if func_data else tool_call.get("arguments", {})
                # Parse arguments if they're a JSON string
                if isinstance(args_raw, str):
                    try:
                        args = json_module.loads(args_raw)
                    except json_module.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw or {}
                func = registry.get(name)
                if not func:
                    return {"role": "tool", "content": f"Tool '{name}' not found", "tool_call_id": tool_call.get("id")}
                if asyncio.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    result = func(**args)
                return {"role": "tool", "content": str(result), "tool_call_id": tool_call.get("id")}
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
            mcp_servers: Optional MCP routing
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            presence_penalty: Penalize new tokens based on presence
            frequency_penalty: Penalize new tokens based on frequency
            customModelParams: Additional model params
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
                        llm_result = await self._async_llm(current_payload)
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
                            
                            # After streaming is complete, check if we need to continue
                            if last_chunk and self._is_complete_response(last_chunk.choices):
                                # No tool calls and response is complete, exit the loop
                                break
                            elif not tool_calls:
                                # No tool calls found, exit after one response
                                break
                        else:
                            # Not an async generator: treat as coroutine (non-streaming)
                            chunk = llm_result
                            chunk_obj = self._safe_stream_chunk(chunk)
                            yield chunk_obj
                            last_chunk = chunk_obj
                            if last_chunk and self._is_complete_response(last_chunk.choices):
                                break
                        
                        # If we get here and there are no tool calls, exit
                        if not tool_calls:
                            break
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

 