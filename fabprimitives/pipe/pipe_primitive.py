"""
Pipe Primitive - Langbase-style pipe implementation

pip install openai anthropic mistralai jsonschema
"""

from __future__ import annotations
import asyncio
import json
import os
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

class LLMProvider:
    """Base class for LLM providers."""
    
    async def run(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run the LLM with given messages and tools."""
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
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
        """Run OpenAI completion."""
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                tools=tools,
                stream=stream,
                **kwargs
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content or ""
                
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
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
        """Run Anthropic completion."""
        try:
            response = await self.client.messages.create(
                messages=messages,
                tools=tools,
                stream=stream,
                **kwargs
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.delta.text:
                            yield chunk.delta.text
                return stream_generator()
            else:
                return response.content[0].text
                
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

class MistralProvider(LLMProvider):
    """Mistral provider implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import mistralai
            self.client = mistralai.AsyncMistralClient(api_key=api_key)
        except ImportError:
            raise ImportError("Mistral not installed. Run: pip install mistralai")
    
    async def run(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Run Mistral completion."""
        try:
            response = await self.client.chat(
                messages=messages,
                tools=tools,
                stream=stream,
                **kwargs
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content or ""
                
        except Exception as e:
            raise RuntimeError(f"Mistral API error: {e}")

# Provider registry
PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "mistral": MistralProvider,
}

# Tool registry for function calling
TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_tool(name: str, func: Callable[..., Any]) -> None:
    """Register a tool function."""
    TOOL_REGISTRY[name] = func

class PipeRegistry:
    """Registry for pipe definitions and execution."""
    
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.cwd()
        
        self.pipes_file = self.storage_path / "pipes.json"
        self._pipes: Dict[str, PipePreset] = {}
        self._load_pipes()
    
    def _load_pipes(self) -> None:
        """Load pipes from JSON file."""
        if self.pipes_file.exists():
            try:
                with open(self.pipes_file, 'r') as f:
                    data = json.load(f)
                    for pipe_data in data:
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
        variables: Optional[Dict[str, str]] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Run a pipe.
        
        Args:
            name: Pipe name
            variables: Variables to interpolate
            
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
        provider = self._get_provider(pipe.model, pipe.api_key)
        
        # Prepare kwargs
        kwargs = {
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

# Global registry instance
pipe_registry = PipeRegistry()

# Convenience functions
def create_pipe(name: str, model: str, messages: List[Message], **kwargs) -> None:
    """Create a pipe."""
    pipe_registry.create(name, model, messages, **kwargs)

def update_pipe(name: str, new_config: Dict[str, Any]) -> None:
    """Update a pipe."""
    pipe_registry.update(name, new_config)

async def run_pipe(name: str, variables: Optional[Dict[str, str]] = None) -> Union[str, AsyncGenerator[str, None]]:
    """Run a pipe."""
    return await pipe_registry.run(name, variables)

if __name__ == "__main__":
    # Self-test
    async def test():
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
            result = await run_pipe("test_pipe", {"name": "John"})
            print(f"Result: {result}")
        except Exception as e:
            print(f"Test failed: {e}")
            print("Make sure you have OPENAI_API_KEY set and openai installed")
        
        # Update pipe
        update_pipe("test_pipe", {"temperature": 0.7})
        print("Pipe updated successfully")
    
    asyncio.run(test()) 