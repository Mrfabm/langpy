# This is a full copy of the pipe primitive, including all LLM backends and context management.
# It is a standalone file for reference or integration purposes.

# --- MODELS (from pipe/schema.py) ---
"""
Pipe Models - Core data structures for Langbase-style pipe primitive.
"""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# Type aliases
JsonDict = Dict[str, Any]
Message = Dict[str, Any]
Tool = Dict[str, Any]
ToolChoice = Dict[str, Any]

class PipePreset(BaseModel):
    """Pipe configuration preset."""
    name: str = Field(..., description="Unique pipe id")
    description: str = ""
    status: Literal["public", "private"] = "private"
    upsert: bool = False
    
    # Model and generation params
    model: str = "openai:gpt-4o"
    stream: bool = False
    json_output: bool = False
    store: bool = True
    moderate: bool = False
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    
    # Tools
    tools: List[Tool] = []
    tool_choice: Optional[Literal["auto", "required"] | ToolChoice] = None
    parallel_tool_calls: bool = False
    
    # Prompt and variables
    messages: List[Message] = []
    variables: Dict[str, str] = {}
    
    # Memory and extras
    memory: List[Dict[str, str]] = []
    response_format: Optional[Dict[str, Any]] = None
    few_shot: Optional[List[Message]] = None
    safety_prompt: Optional[str] = None

# --- LLM PROVIDERS (from pipe/adapters/) ---
"""
Built-in LLM providers for the pipe primitive.
"""
import asyncio
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    async def run(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run the LLM provider."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")
    
    async def run(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run OpenAI provider."""
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", "gpt-4o"),
                messages=messages,
                tools=tools,
                tool_choice=kwargs.get("tool_choice", "auto"),
                temperature=kwargs.get("temperature", 1.0),
                max_tokens=kwargs.get("max_tokens"),
                stream=stream,
                **{k: v for k, v in kwargs.items() if k not in ["model", "tool_choice"]}
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")
    
    async def run(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run Anthropic provider."""
        try:
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
                model=kwargs.get("model", "claude-3-sonnet-20240229"),
                messages=anthropic_messages,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 1.0),
                stream=stream
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.type == "content_block_delta" and chunk.delta.text:
                            yield chunk.delta.text
                return stream_generator()
            else:
                return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

class MistralProvider(LLMProvider):
    """Mistral provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from mistralai.async_client import MistralAsyncClient
            self.client = MistralAsyncClient(api_key=api_key)
        except ImportError:
            raise ImportError("Mistral not installed. Run: pip install mistralai")
    
    async def run(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run Mistral provider."""
        try:
            response = await self.client.chat(
                model=kwargs.get("model", "mistral-large-latest"),
                messages=messages,
                temperature=kwargs.get("temperature", 1.0),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=stream
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Mistral API error: {str(e)}")

# Provider registry
PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "mistral": MistralProvider,
}

# --- PIPE IMPLEMENTATION (from pipe/pipe_primitive.py) ---
"""
Complete pipe implementation with LLM integration and context management.
"""

class PipeRegistry:
    """Registry for pipe presets."""
    
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.pipes_file = Path(storage_path) / "pipes.json"
        else:
            self.pipes_file = Path.home() / ".langpy" / "pipes.json"
        
        self.pipes_file.parent.mkdir(parents=True, exist_ok=True)
        self._pipes: Dict[str, PipePreset] = {}
        self._load_pipes()
    
    def _load_pipes(self) -> None:
        """Load pipes from JSON file."""
        if self.pipes_file.exists():
            try:
                with open(self.pipes_file, 'r') as f:
                    pipes_data = json.load(f)
                    for pipe_data in pipes_data:
                        pipe = PipePreset(**pipe_data)
                        self._pipes[pipe.name] = pipe
            except Exception as e:
                print(f"Warning: Could not load pipes.json: {e}")
    
    def _save_pipes(self) -> None:
        """Save pipes to JSON file."""
        pipes_data = [pipe.model_dump() for pipe in self._pipes.values()]
        with open(self.pipes_file, 'w') as f:
            json.dump(pipes_data, f, indent=2)
    
    def create(
        self, 
        name: str, 
        model: str, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> None:
        """
        Create a new pipe.
        
        Args:
            name: Pipe name
            model: Model to use
            messages: Message template
            tools: Optional tools
            stream: Whether to stream
            **kwargs: Additional configuration
        """
        pipe = PipePreset(
            name=name,
            model=model,
            messages=messages,
            tools=tools or [],
            stream=stream,
            **kwargs
        )
        
        self._pipes[name] = pipe
        self._save_pipes()
    
    def update(self, name: str, new_config: Dict[str, Any]) -> None:
        """
        Update an existing pipe.
        
        Args:
            name: Pipe name
            new_config: New configuration
        """
        if name not in self._pipes:
            raise ValueError(f"Pipe '{name}' not found")
        
        pipe = self._pipes[name]
        for key, value in new_config.items():
            if hasattr(pipe, key):
                setattr(pipe, key, value)
        
        self._save_pipes()
    
    async def run(
        self, 
        name: str, 
        variables: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Run a pipe.
        
        Args:
            name: Pipe name
            variables: Variables to interpolate
            api_key: API key for the LLM provider
            
        Returns:
            Response string or generator
        """
        if name not in self._pipes:
            raise ValueError(f"Pipe '{name}' not found")
        
        pipe = self._pipes[name]
        variables = variables or {}
        
        # Format messages with variables
        formatted_messages = self._format_messages(pipe.messages, variables)
        
        # Get provider
        provider = self._get_provider(pipe.model, api_key)
        
        # Prepare kwargs
        kwargs = {
            "model": pipe.model.split(":")[-1],  # Remove provider prefix
            "temperature": pipe.temperature,
            "top_p": pipe.top_p,
            "max_tokens": pipe.max_tokens,
            "presence_penalty": pipe.presence_penalty,
            "frequency_penalty": pipe.frequency_penalty,
            "stop": pipe.stop,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Handle tools
        tools = pipe.tools if pipe.tools else None
        if tools:
            kwargs["tool_choice"] = pipe.tool_choice
        
        # Handle JSON response format
        if pipe.json_output and pipe.response_format:
            kwargs["response_format"] = pipe.response_format
        
        # Execute with tool calling if needed
        if tools and pipe.tool_choice:
            return await self._execute_with_tools(
                provider, formatted_messages, tools, pipe.parallel_tool_calls, **kwargs
            )
        else:
            return await provider.run(formatted_messages, tools, pipe.stream, **kwargs)
    
    def _format_messages(self, messages: List[Message], variables: Dict[str, str]) -> List[Message]:
        """Format messages with variable interpolation."""
        formatted = []
        for msg in messages:
            formatted_msg = msg.copy()
            if isinstance(msg.get("content"), str):
                try:
                    formatted_msg["content"] = msg["content"].format(**variables)
                except KeyError as e:
                    raise ValueError(f"Missing variable: {e}")
            formatted.append(formatted_msg)
        return formatted
    
    def _get_provider(self, model: str, api_key: str) -> LLMProvider:
        """Get provider for model."""
        vendor = model.split(":")[0]
        provider_cls = PROVIDER_REGISTRY.get(vendor)
        if not provider_cls:
            raise ValueError(f"Unknown provider: {vendor}")
        
        return provider_cls(api_key)
    
    async def _execute_with_tools(
        self,
        provider: LLMProvider,
        messages: List[Message],
        tools: List[Tool],
        parallel_tool_calls: bool,
        **kwargs
    ) -> str:
        """Execute pipe with tool calling (single LLM call, no execution loops)."""
        # Single LLM call with tool definitions - no execution loops
        # This matches Langbase pipe behavior: tools are defined but not executed
        response = await provider.run(messages, tools, stream=False, **kwargs)
        if isinstance(response, str):
            return response
        else:
            # Handle streaming response
            result = ""
            async for chunk in response:
                result += chunk
            return result

# --- ENHANCED PIPE (from pipe/async_pipe.py) ---
"""
Enhanced pipe with memory and thread integration.
"""

class PipeRunMetadata:
    """Metadata for pipe execution runs."""
    
    def __init__(self, run_id: str, pipe_name: str, start_time: float):
        self.run_id = run_id
        self.pipe_name = pipe_name
        self.start_time = start_time
        self.end_time = None
        self.tokens_used = None
        self.cost = None
        self.error = None
        self.retries = 0
        self.tool_calls = []
    
    def complete(self, tokens_used: int = None, cost: float = None):
        """Mark run as completed."""
        self.end_time = time.time()
        self.tokens_used = tokens_used
        self.cost = cost
    
    def fail(self, error: str):
        """Mark run as failed."""
        self.end_time = time.time()
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "pipe_name": self.pipe_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "error": self.error,
            "retries": self.retries,
            "tool_calls": self.tool_calls
        }

class AsyncPipe:
    """Enhanced pipe with full Langbase-style capabilities (single LLM call)."""

    def __init__(self, *, default_model: str = "openai:gpt-4o", **defaults):
        self.default_model = default_model
        self.defaults = defaults
        self._registered_functions: Dict[str, Dict[str, Any]] = {}
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        self._memory_interface = None
        self._thread_interface = None
        self._agent_interface = None
        self._run_logs: List[PipeRunMetadata] = []
        self._registry = PipeRegistry()
    
    @classmethod
    def register_function(cls, name: str, func: Callable, model: str = None, **config):
        """Register a function as a pipe."""
        cls._registered_functions[name] = {
            'function': func,
            'model': model,
            'config': config
        }
    
    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool function (for tool definitions only, not execution)."""
        self._tool_registry[name] = func
    
    def set_memory_interface(self, memory_interface):
        """Set memory interface for pipe."""
        self._memory_interface = memory_interface
    
    def set_thread_interface(self, thread_interface):
        """Set thread interface for pipe."""
        self._thread_interface = thread_interface
    
    def set_agent_interface(self, agent_interface):
        """Set agent interface for pipe."""
        self._agent_interface = agent_interface

    async def run(
        self,
        *,
        name: str | None = None,          # optional preset id
        apiKey: str,                      # Always required
        messages: list | None = None,     # Langbase messages
        input: Any | None = None,         # Alias for simple completion
        model: str | None = None,
        stream: bool | None = None,
        # Integration parameters
        memory=None,
        thread=None,
        agent=None,
        tools=None,
        tool_choice="auto",
        parallel_tool_calls=False,
        # Advanced features
        json_output=False,
        response_format=None,
        few_shot=None,
        safety_prompt=None,
        moderate=False,
        store=True,
        # Error handling
        max_retries=3,
        retry_delay=1.0,
        timeout=30.0,
        **overrides,
    ) -> str | AsyncGenerator[str, None]:
        """Enhanced run with full integration capabilities (single LLM call)."""

        # Create run metadata
        run_id = str(uuid.uuid4())
        run_meta = PipeRunMetadata(run_id, name or "unnamed", time.time())
        
        try:
            # Use registry for preset management
            if name:
                return await self._registry.run(name, overrides, apiKey)
            
            # Direct execution without preset
            final_messages = await self._prepare_messages(
                messages or input or [],
                memory=memory,
                thread=thread,
                few_shot=few_shot,
                safety_prompt=safety_prompt,
                run_meta=run_meta
            )

            # Tool preparation (definitions only, no execution)
            final_tools = await self._prepare_tools(
                tools=tools,
                agent=agent,
                run_meta=run_meta
            )

            # Build payload for provider
            payload = {
                "model": model or self.default_model,
                "messages": final_messages,
                "tools": final_tools,
                "tool_choice": tool_choice,
                "stream": stream or False,
                "temperature": overrides.get("temperature", 1.0),
                "max_tokens": overrides.get("max_tokens"),
                "apiKey": apiKey,
                **{k: v for k, v in overrides.items() if k not in ["model", "messages", "tools", "tool_choice", "stream", "temperature", "max_tokens", "apiKey"]}
            }

            # Execute with retries
            result = await self._execute_with_retries(
                payload, max_retries, retry_delay, timeout, run_meta
            )

            # Post-process result
            result = await self._post_process(
                result, thread=thread, memory=memory, json_output=json_output, 
                response_format=response_format, run_meta=run_meta
            )

            # Save run metadata
            await self._save_run_metadata(run_meta)
            
            return result

        except Exception as e:
            run_meta.fail(str(e))
            await self._save_run_metadata(run_meta)
            raise

    async def _prepare_messages(
        self, 
        messages: List[Message], 
        memory=None, 
        thread=None, 
        few_shot=None, 
        safety_prompt=None,
        run_meta: PipeRunMetadata = None
    ) -> List[Message]:
        """Prepare messages with all integrations."""
        final_messages = []
        
        # Add safety prompt if provided
        if safety_prompt:
            final_messages.append({"role": "system", "content": safety_prompt})
        
        # Add memory context if available
        if memory and self._memory_interface:
            try:
                # Extract user query from messages
                user_query = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        break
                
                if user_query:
                    memory_results = await self._memory_interface.query(user_query, k=3)
                    if memory_results:
                        memory_context = "\n\n".join([r.get("text", "") for r in memory_results])
                        final_messages.append({
                            "role": "system", 
                            "content": f"Relevant context from memory:\n{memory_context}"
                        })
            except Exception as e:
                if run_meta:
                    run_meta.tool_calls.append(f"Memory integration failed: {str(e)}")
        
        # Add thread context if available
        if thread and self._thread_interface:
            try:
                thread_messages = await self._thread_interface.get_messages(thread.id)
                final_messages.extend(thread_messages)
            except Exception as e:
                if run_meta:
                    run_meta.tool_calls.append(f"Thread integration failed: {str(e)}")
        
        # Add few-shot examples if provided
        if few_shot:
            final_messages.extend(few_shot)
        
        # Add main messages
        final_messages.extend(messages)
        
        return final_messages

    async def _prepare_tools(
        self, 
        tools: List[Tool] = None, 
        agent=None,
        run_meta: PipeRunMetadata = None
    ) -> List[Tool]:
        """Prepare tools for execution."""
        final_tools = tools or []
        
        # Add agent tools if agent provided
        if agent and self._agent_interface:
            try:
                agent_tools = await self._agent_interface.get_tools(agent.id)
                final_tools.extend(agent_tools)
            except Exception as e:
                if run_meta:
                    run_meta.tool_calls.append(f"Agent integration failed: {str(e)}")
        
        return final_tools

    async def _execute_with_retries(
        self, 
        payload: JsonDict, 
        max_retries: int, 
        retry_delay: float, 
        timeout: float,
        run_meta: PipeRunMetadata
    ) -> str | AsyncGenerator[str, None]:
        """Execute with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                run_meta.retries = attempt
                
                # Get provider
                model = payload["model"]
                api_key = payload["apiKey"]
                vendor = model.split(":")[0]
                provider_cls = PROVIDER_REGISTRY.get(vendor)
                
                if not provider_cls:
                    raise ValueError(f"Unknown provider: {vendor}")
                
                provider = provider_cls(api_key)
                
                # Execute
                result = await provider.run(
                    messages=payload["messages"],
                    tools=payload.get("tools"),
                    stream=payload.get("stream", False),
                    model=model.split(":")[-1],
                    temperature=payload.get("temperature", 1.0),
                    max_tokens=payload.get("max_tokens")
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
        
        raise last_exception

    async def _post_process(
        self, 
        result: str, 
        thread=None, 
        memory=None, 
        json_output=False, 
        response_format=None,
        run_meta: PipeRunMetadata = None
    ) -> str:
        """Post-process the result."""
        # Store in thread if available
        if thread and self._thread_interface:
            try:
                await self._thread_interface.add_message(thread.id, {
                    "role": "assistant",
                    "content": result
                })
            except Exception as e:
                if run_meta:
                    run_meta.tool_calls.append(f"Thread storage failed: {str(e)}")
        
        # Store in memory if available
        if memory and self._memory_interface:
            try:
                await self._memory_interface.add_text(
                    text=result,
                    source="pipe_response",
                    tags=["pipe", "response"]
                )
            except Exception as e:
                if run_meta:
                    run_meta.tool_calls.append(f"Memory storage failed: {str(e)}")
        
        return result

    def _calculate_cost(self, usage: Dict[str, int], model: str) -> float:
        """Calculate cost based on usage."""
        # Simplified cost calculation
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            # Rough cost estimates (these would be more accurate in production)
            if "gpt-4" in model:
                return (usage["prompt_tokens"] * 0.03 + usage["completion_tokens"] * 0.06) / 1000
            elif "gpt-3.5" in model:
                return (usage["prompt_tokens"] * 0.0015 + usage["completion_tokens"] * 0.002) / 1000
        return 0.0

    async def _save_run_metadata(self, run_meta: PipeRunMetadata):
        """Save run metadata."""
        self._run_logs.append(run_meta)

    def get_run_logs(self) -> List[PipeRunMetadata]:
        """Get run logs."""
        return self._run_logs.copy()

    def clear_run_logs(self):
        """Clear run logs."""
        self._run_logs.clear()

# --- CONVENIENCE FUNCTIONS ---
"""
Convenience functions for easy pipe usage.
"""

def create_pipe(name: str, model: str, messages: List[Message], **kwargs) -> None:
    """Create a pipe."""
    registry = PipeRegistry()
    registry.create(name, model, messages, **kwargs)

def update_pipe(name: str, new_config: Dict[str, Any]) -> None:
    """Update a pipe."""
    registry = PipeRegistry()
    registry.update(name, new_config)

async def run_pipe(name: str, variables: Optional[Dict[str, str]] = None, api_key: Optional[str] = None) -> Union[str, AsyncGenerator[str, None]]:
    """Run a pipe."""
    registry = PipeRegistry()
    return await registry.run(name, variables, api_key)

# --- SELF-TEST ---
if __name__ == "__main__":
    import os
    
    async def test_pipe():
        """Test the pipe implementation."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY environment variable to test")
            return
        
        # Create a test pipe
        create_pipe(
            "test_pipe",
            "openai:gpt-4o",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, {name}!"}
            ]
        )
        
        # Run with variables
        try:
            result = await run_pipe("test_pipe", {"name": "John"}, api_key)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Test failed: {e}")
            print("Make sure you have OPENAI_API_KEY set and openai installed")
        
        # Update pipe
        update_pipe("test_pipe", {"temperature": 0.7})
        print("Pipe updated successfully")
    
    # Run test
    asyncio.run(test_pipe()) 