"""
Enhanced AsyncPipe - Full Langbase-style pipe primitive for Langpy
────────────────────────────────────────────────────────────────
• Single LLM call primitive (no tool execution loops)
• Memory integration with RAG capabilities
• Agent and thread integration
• Structured output validation
• Advanced prompt features
• Run metadata and logging
• Robust error handling and retries
"""

from __future__ import annotations
import os, json, time, uuid, asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Callable
import jsonschema
from pydantic import ValidationError

from agent.async_agent import AgentRunResponse, AgentStreamChunk
from pipe.adapters import get_adapter
from pipe.schema import PipePreset
from pipe.store import load as _load_preset, create as _save_preset

# folder where presets live
_BASE = Path(os.getenv("LANGPY_HOME", Path.home() / ".langpy")) / "pipes"
_BASE.mkdir(parents=True, exist_ok=True)

Json = Dict[str, Any]
Message = Dict[str, Any]
Tool = Dict[str, Any]

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
        # Memory integration tracking
        self.memory_retrieved = 0
        self.memory_error = None
    
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
            "tool_calls": self.tool_calls,
            "memory_retrieved": self.memory_retrieved,
            "memory_error": self.memory_error
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
    ) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
        """Enhanced run with full integration capabilities (single LLM call)."""

        # Create run metadata
        run_id = str(uuid.uuid4())
        run_meta = PipeRunMetadata(run_id, name or "unnamed", time.time())
        
        try:
            # ────────────────────────────────────────────────────────────
            # 1. auto-create preset file on first call (if name supplied)
            # ────────────────────────────────────────────────────────────
            if name:
                preset_path = _BASE / f"{name}.json"
                if not preset_path.exists():
                    preset = PipePreset(
                        name=name,
                        model=model or self.default_model,
                        messages=messages or [],
                        **{k: v for k, v in overrides.items() if v is not None},
                    )
                    _save_preset(preset)

            # ────────────────────────────────────────────────────────────
            # 2. merge defaults < preset < call-time overrides
            # ────────────────────────────────────────────────────────────
            preset_data = _load_preset(name) if name else {}
            preset = PipePreset(**preset_data) if preset_data else None
            base = preset.model_dump() if preset else {}

            call_args = {
                "model": model,
                "stream": stream,
                "messages": messages,
                "input": input,
            }
            merged = {**self.defaults, **base,
                      **{k: v for k, v in call_args.items() if v is not None},
                      **overrides}

            # ────────────────────────────────────────────────────────────
            # 3. Integration preprocessing
            # ────────────────────────────────────────────────────────────
            final_messages = await self._prepare_messages(
                merged.get("messages") or merged.get("input"),
                memory=memory,
                thread=thread,
                few_shot=few_shot,
                safety_prompt=safety_prompt,
                run_meta=run_meta
            )

            # ────────────────────────────────────────────────────────────
            # 4. Tool preparation (definitions only, no execution)
            # ────────────────────────────────────────────────────────────
            final_tools = await self._prepare_tools(
                tools=tools,
                agent=agent,
                run_meta=run_meta
            )

            # ────────────────────────────────────────────────────────────
            # 5. build payload for adapter
            # ────────────────────────────────────────────────────────────
            payload: Json = {
                "model": merged.get("model", self.default_model),
                "input": final_messages,
                "apiKey": apiKey,
                "stream": merged.get("stream", False),

                # generation / tool knobs
                "top_p": merged.get("top_p"),
                "max_tokens": merged.get("max_tokens"),
                "temperature": merged.get("temperature"),
                "presence_penalty": merged.get("presence_penalty"),
                "frequency_penalty": merged.get("frequency_penalty"),
                "stop": merged.get("stop"),

                # Tool integration (definitions only)
                "tools": final_tools,
                "tool_choice": tool_choice if final_tools else None,
                "parallel_tool_calls": parallel_tool_calls,

                # Structured output
                "json_output": json_output,
                "response_format": response_format,

                # Other features
                "store": store,
                "moderate": moderate,
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            # ────────────────────────────────────────────────────────────
            # 6. Execute with retry logic (single LLM call)
            # ────────────────────────────────────────────────────────────
            result = await self._execute_with_retries(
                payload, max_retries, retry_delay, timeout, run_meta
            )

            # ────────────────────────────────────────────────────────────
            # 7. Post-processing
            # ────────────────────────────────────────────────────────────
            result = await self._post_process(
                result, thread, memory, json_output, response_format, run_meta
            )

            # ────────────────────────────────────────────────────────────
            # 8. Complete run metadata
            # ────────────────────────────────────────────────────────────
            if hasattr(result, 'usage') and result.usage:
                run_meta.complete(
                    tokens_used=result.usage.get('total_tokens'),
                    cost=self._calculate_cost(result.usage, payload['model'])
                )
            else:
                run_meta.complete()

            # Store run metadata if requested
            if store:
                self._run_logs.append(run_meta)
                await self._save_run_metadata(run_meta)

            return result

        except Exception as e:
            run_meta.fail(str(e))
            if store:
                self._run_logs.append(run_meta)
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
        """Prepare messages with integrations."""
        final_messages = messages.copy() if messages else []
        
        # Add few-shot examples
        if few_shot:
            final_messages = few_shot + final_messages
        
        # Add safety prompt
        if safety_prompt:
            final_messages.insert(0, {"role": "system", "content": safety_prompt})
        
        # Add memory context (Langbase-style automatic retrieval)
        if memory and self._memory_interface:
            try:
                # Extract query from last message
                query = ""
                if final_messages:
                    last_message = final_messages[-1]
                    if isinstance(last_message, dict):
                        query = last_message.get("content", "")
                    elif hasattr(last_message, 'content'):
                        query = last_message.content
                
                # Use SIMPLE memory query (no advanced features)
                memory_results = await self._memory_interface.query(
                    query=query,
                    k=5
                )
                
                if memory_results:
                    # Format memory context
                    context_parts = []
                    for i, result in enumerate(memory_results[:3], 1):  # Top 3 results
                        if isinstance(result, dict):
                            text = result.get('text', '')
                            score = result.get('score', 0.0)
                        else:
                            text = result.text
                            score = result.score
                        
                        context_parts.append(f"{i}. {text} (relevance: {score:.2f})")
                    
                    memory_context = "\n\nRelevant context from memory:\n" + "\n".join(context_parts)
                    final_messages.insert(0, {"role": "system", "content": memory_context})
                    
                    # Log memory usage
                    if run_meta:
                        run_meta.memory_retrieved = len(memory_results)
                        
            except Exception as e:
                print(f"Memory search error: {e}")
                if run_meta:
                    run_meta.memory_error = str(e)
        
        # Add thread context
        if thread and self._thread_interface:
            try:
                thread_messages = await self._thread_interface.get_messages(thread.id)
                if thread_messages:
                    final_messages = thread_messages + final_messages
            except Exception as e:
                print(f"Thread retrieval error: {e}")
        
        return final_messages

    async def _prepare_tools(
        self, 
        tools: List[Tool] = None, 
        agent=None,
        run_meta: PipeRunMetadata = None
    ) -> List[Tool]:
        """Prepare tools (definitions only, no execution)."""
        final_tools = []
        
        # Add tools from agent if provided
        if agent and hasattr(agent, '_tools'):
            final_tools.extend(agent._tools)
        
        # Add direct tools
        if tools:
            final_tools.extend(tools)
        
        # Convert registered functions to tools
        for name, config in self._registered_functions.items():
            if config.get('function'):
                # Create tool definition from registered function
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Registered function: {name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                final_tools.append(tool_def)
        
        return final_tools

    async def _execute_with_retries(
        self, 
        payload: Json, 
        max_retries: int, 
        retry_delay: float, 
        timeout: float,
        run_meta: PipeRunMetadata
    ) -> AgentRunResponse | AsyncGenerator[AgentStreamChunk, None]:
        """Execute with retry logic (single LLM call)."""
        
        for attempt in range(max_retries + 1):
            try:
                adapter = get_adapter(payload["model"])
                return await adapter(payload)
                    
            except Exception as e:
                run_meta.retries = attempt
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise e

    async def _post_process(
        self, 
        result: AgentRunResponse, 
        thread=None, 
        memory=None, 
        json_output=False, 
        response_format=None,
        run_meta: PipeRunMetadata = None
    ) -> AgentRunResponse:
        """Post-process the result."""
        
        # Validate JSON output if requested
        if json_output and response_format:
            try:
                content = result.choices[0].message.get("content", "")
                parsed_content = json.loads(content)
                jsonschema.validate(parsed_content, response_format)
            except (json.JSONDecodeError, ValidationError) as e:
                # Re-prompt with validation error
                raise ValueError(f"Invalid JSON output: {e}")
        
        # Store in thread if provided
        if thread and self._thread_interface:
            try:
                content = result.choices[0].message.get("content", "")
                await self._thread_interface.add_message(
                    thread_id=thread.id,
                    role="assistant",
                    content=content
                )
            except Exception as e:
                print(f"Thread storage error: {e}")
        
        # Store in memory if provided
        if memory and self._memory_interface:
            try:
                content = result.choices[0].message.get("content", "")
                await self._memory_interface.add_text(
                    text=content,
                    source=f"pipe_{run_meta.pipe_name if run_meta else 'unknown'}",
                    tags=["pipe_output"]
                )
            except Exception as e:
                print(f"Memory storage error: {e}")
        
        return result

    def _calculate_cost(self, usage: Dict[str, int], model: str) -> float:
        """Calculate cost based on token usage and model."""
        # Simplified cost calculation - would need actual pricing data
        total_tokens = usage.get('total_tokens', 0)
        
        # Rough cost estimates (per 1K tokens)
        costs = {
            'gpt-4': 0.03,
            'gpt-4o': 0.005,
            'gpt-3.5-turbo': 0.002,
            'claude-3': 0.015,
            'gemini': 0.001
        }
        
        for model_prefix, cost_per_1k in costs.items():
            if model_prefix in model.lower():
                return (total_tokens / 1000) * cost_per_1k
        
        return 0.0  # Unknown model

    async def _save_run_metadata(self, run_meta: PipeRunMetadata):
        """Save run metadata to storage."""
        try:
            # Save to file for now - could be database in production
            runs_dir = _BASE / "runs"
            runs_dir.mkdir(exist_ok=True)
            
            run_file = runs_dir / f"{run_meta.run_id}.json"
            with open(run_file, 'w') as f:
                json.dump(run_meta.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Failed to save run metadata: {e}")

    def get_run_logs(self) -> List[PipeRunMetadata]:
        """Get run logs."""
        return self._run_logs.copy()

    def clear_run_logs(self):
        """Clear run logs."""
        self._run_logs.clear()
    
    async def extract_context(self, query: str, memory, k: int = 5) -> str:
        """
        Extract context from memory for a given query using SIMPLE approach.
        Uses basic memory query without advanced features (reranker/BM25).
        
        Args:
            query: Search query
            memory: Memory instance to query
            k: Number of chunks to retrieve
            
        Returns:
            Combined context string from memory chunks
        """
        if not memory:
            return ""
        
        try:
            # Use SIMPLE memory query (no advanced features)
            memory_results = await memory.query(query, k=k)
            
            if not memory_results:
                return ""
            
            # Extract context from memory results
            context_parts = []
            for result in memory_results:
                if hasattr(result, 'text'):
                    context_parts.append(result.text)
                elif isinstance(result, dict):
                    text = result.get('text') or result.get('content') or result.get('chunk_text', '')
                    if text:
                        context_parts.append(text)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"Memory query error: {e}")
            return ""
    
    async def extract_response(self, query: str, context: str, **pipe_kwargs) -> str:
        """
        Extract response using context with FULL Pipe integrations.
        
        Args:
            query: User question
            context: Retrieved context from memory
            **pipe_kwargs: Additional arguments for pipe.run() (including thread, agent, tools, etc.)
            
        Returns:
            Generated response string
        """
        if not context:
            return "I don't have enough information to answer that question."
        
        # Create messages with context
        messages = [
            {
                "role": "system",
                "content": f"Use the following context to answer the user's question:\n\n{context}"
            },
            {
                "role": "user", 
                "content": query
            }
        ]
        
        # Run pipe with FULL integrations (thread, agent, tools, streaming, etc.)
        result = await self.run(
            messages=messages,
            **pipe_kwargs  # This preserves ALL integrations: thread, agent, tools, streaming, etc.
        )
        
        # Handle streaming response
        if hasattr(result, '__aiter__'):
            # Streaming response - collect chunks
            full_response = ""
            async for chunk in result:
                if hasattr(chunk, 'choices') and chunk.choices:
                    content = chunk.choices[0].delta.get('content', '')
                    if content:
                        full_response += content
            return full_response
        
        # Handle regular response
        if hasattr(result, 'choices') and result.choices:
            choice = result.choices[0]
            if hasattr(choice, 'message'):
                return choice.message.get('content', '')
            elif hasattr(choice, 'text'):
                return choice.text
            else:
                return str(result)
        else:
            return str(result)
    
    async def rag_with_memory(self, query: str, memory, k: int = 5, **pipe_kwargs):
        """
        Complete RAG workflow: extract context and generate response with FULL Pipe integrations.
        
        Args:
            query: User question
            memory: Memory instance to query
            k: Number of chunks to retrieve
            **pipe_kwargs: Additional arguments for pipe.run() (including thread, agent, tools, streaming, etc.)
            
        Returns:
            Generated response (string for regular, AsyncGenerator for streaming)
        """
        # Extract context
        context = await self.extract_context(query, memory, k)
        
        # Check if streaming is requested
        if pipe_kwargs.get('stream', False):
            # For streaming, we need to handle it differently
            messages = [
                {
                    "role": "system",
                    "content": f"Use the following context to answer the user's question:\n\n{context}"
                },
                {
                    "role": "user", 
                    "content": query
                }
            ]
            
            # Return streaming response directly
            return await self.run(
                messages=messages,
                **pipe_kwargs  # Preserves ALL integrations
            )
        else:
            # Generate response (handles streaming internally)
            return await self.extract_response(query, context, **pipe_kwargs)