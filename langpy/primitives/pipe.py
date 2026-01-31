"""
Pipe Primitive - Langbase-compatible Pipe API.

The Pipe primitive provides a single LLM call with prompt templates,
variables, memory integration, and structured outputs.

Unlike Agent (which has tool execution loops), Pipe is a single LLM call.

Usage:
    # Direct API (Langbase-compatible)
    response = await pipe.run(
        name="summarizer",
        messages=[{"role": "user", "content": "Summarize: {text}"}],
        variables={"text": "Long document..."}
    )

    # Pipeline composition
    pipeline = memory | pipe
    result = await pipeline.process(ctx)
"""

from __future__ import annotations
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, TYPE_CHECKING

from langpy.core.primitive import BasePrimitive, PipeResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


class Pipe(BasePrimitive):
    """
    Pipe primitive - Single LLM call with templates and variables.

    Pipes are the core building blocks for AI features. Unlike Agents,
    Pipes make a single LLM call without tool execution loops.

    Features:
    - Prompt templates with variable substitution
    - Memory integration for RAG
    - Structured output (JSON mode)
    - Few-shot examples
    - Response validation

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Create a pipe
        await lb.pipe.create(
            name="summarizer",
            model="openai:gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarizer."},
                {"role": "user", "content": "Summarize: {text}"}
            ]
        )

        # Run the pipe
        response = await lb.pipe.run(
            name="summarizer",
            variables={"text": "Long document here..."}
        )
        print(response.output)
    """

    def __init__(self, client: Any = None, name: str = "pipe"):
        """
        Initialize the Pipe primitive.

        Args:
            client: Parent Langpy client (for shared config)
            name: Primitive name for tracing
        """
        super().__init__(name=name, client=client)
        self._async_pipe = None  # Lazy initialization

    def _get_async_pipe(self):
        """Get or create the underlying AsyncPipe."""
        if self._async_pipe is None:
            from pipe.async_pipe import AsyncPipe
            self._async_pipe = AsyncPipe()
        return self._async_pipe

    # ========================================================================
    # Langbase-compatible API
    # ========================================================================

    async def create(
        self,
        name: str,
        model: str = "openai:gpt-4",
        messages: List[Dict[str, str]] = None,
        description: str = "",
        variables: Dict[str, str] = None,
        tools: List[Dict[str, Any]] = None,
        memory: List[str] = None,
        json_output: bool = False,
        response_format: Dict[str, Any] = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> PipeResponse:
        """
        Create a new pipe definition.

        Args:
            name: Unique pipe identifier
            model: LLM model to use
            messages: Message template with {variable} placeholders
            description: Pipe description
            variables: Default variable values
            tools: Tool definitions (not executed, just passed to LLM)
            memory: Memory names to attach
            json_output: Enable JSON output mode
            response_format: JSON schema for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            PipeResponse with creation status
        """
        try:
            from pipe.store import create as save_preset
            from pipe.schema import PipePreset

            preset = PipePreset(
                name=name,
                model=model,
                messages=messages or [],
                description=description,
                variables=variables or {},
                tools=tools or [],
                memory=memory or [],
                json_output=json_output,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            save_preset(preset)

            return PipeResponse(
                success=True,
                output=f"Pipe '{name}' created successfully"
            )
        except Exception as e:
            return PipeResponse(success=False, error=str(e))

    async def update(
        self,
        name: str,
        **updates
    ) -> PipeResponse:
        """
        Update an existing pipe.

        Args:
            name: Pipe name to update
            **updates: Fields to update

        Returns:
            PipeResponse with update status
        """
        try:
            from pipe.store import load as load_preset, create as save_preset

            preset = load_preset(name)
            if not preset:
                return PipeResponse(success=False, error=f"Pipe '{name}' not found")

            # Update fields
            for key, value in updates.items():
                if hasattr(preset, key):
                    setattr(preset, key, value)

            save_preset(preset)

            return PipeResponse(
                success=True,
                output=f"Pipe '{name}' updated successfully"
            )
        except Exception as e:
            return PipeResponse(success=False, error=str(e))

    async def _run(
        self,
        name: str = None,
        messages: List[Dict[str, str]] = None,
        variables: Dict[str, str] = None,
        model: str = None,
        stream: bool = False,
        memory: Any = None,
        thread: Any = None,
        json_output: bool = False,
        response_format: Dict[str, Any] = None,
        api_key: str = None,
        **kwargs
    ) -> PipeResponse:
        """
        Run a pipe with Langbase-compatible parameters.

        Args:
            name: Pipe name (loads preset) or None for inline
            messages: Message array (overrides preset)
            variables: Variables to substitute in messages
            model: Model override
            stream: Enable streaming
            memory: Memory to query for RAG
            thread: Thread for conversation history
            json_output: Enable JSON output
            response_format: JSON schema
            api_key: LLM provider API key

        Returns:
            PipeResponse with output
        """
        try:
            # Get API key
            resolved_api_key = api_key
            if not resolved_api_key and self._client:
                resolved_api_key = getattr(self._client, '_api_key', None)
            if not resolved_api_key:
                resolved_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LANGPY_API_KEY")

            if not resolved_api_key:
                return PipeResponse(
                    success=False,
                    error="API key required"
                )

            # Load preset if name provided
            preset_messages = messages
            preset_model = model or "openai:gpt-4"
            preset_variables = variables or {}

            if name:
                try:
                    from pipe.store import load as load_preset
                    preset = load_preset(name)
                    if preset:
                        preset_messages = preset_messages or preset.messages
                        preset_model = model or preset.model
                        preset_variables = {**preset.variables, **(variables or {})}
                except Exception:
                    pass  # Use provided messages if preset not found

            if not preset_messages:
                return PipeResponse(
                    success=False,
                    error="Messages required (provide messages or valid pipe name)"
                )

            # Substitute variables in messages
            formatted_messages = self._format_messages(preset_messages, preset_variables)

            # Handle memory integration (RAG)
            if memory:
                # Query memory and prepend context
                try:
                    if hasattr(memory, 'retrieve'):
                        query = preset_variables.get('query') or preset_variables.get('input', '')
                        docs = await memory.retrieve(query=query, top_k=5)
                        if docs and docs.documents:
                            context = "\n\n".join([d.content for d in docs.documents])
                            # Insert context into system message
                            formatted_messages = self._inject_context(formatted_messages, context)
                except Exception as e:
                    # Log but don't fail
                    pass

            # Handle thread integration
            if thread:
                try:
                    if hasattr(thread, 'list'):
                        history = await thread.list()
                        if history and history.messages:
                            # Prepend thread history
                            formatted_messages = history.messages + formatted_messages
                except Exception:
                    pass

            # Get adapter and make LLM call
            from pipe.adapters import get_adapter

            provider = preset_model.split(":")[0] if ":" in preset_model else "openai"
            model_name = preset_model.split(":")[-1] if ":" in preset_model else preset_model

            adapter = get_adapter(provider)

            payload = {
                "model": model_name,
                "messages": formatted_messages,
                "stream": stream,
                "apiKey": resolved_api_key,
                **kwargs
            }

            if json_output or response_format:
                payload["response_format"] = response_format or {"type": "json_object"}

            response = await adapter.run(payload)

            # Handle streaming
            if stream:
                return PipeResponse(
                    success=True,
                    output=None,
                    _stream=response
                )

            # Extract output
            output = None
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message'):
                    msg = choice.message
                    output = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)

            return PipeResponse(
                success=True,
                output=output,
                messages=formatted_messages,
                variables=preset_variables,
                usage=getattr(response, 'usage', None)
            )

        except Exception as e:
            return PipeResponse(success=False, error=str(e))

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        variables: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Substitute variables in message templates."""
        formatted = []
        for msg in messages:
            formatted_msg = msg.copy()
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    formatted_msg["content"] = content.format(**variables)
                except KeyError:
                    # Leave unformatted if variable missing
                    pass
            formatted.append(formatted_msg)
        return formatted

    def _inject_context(
        self,
        messages: List[Dict[str, str]],
        context: str
    ) -> List[Dict[str, str]]:
        """Inject RAG context into messages."""
        if not messages:
            return messages

        # Find or create system message
        result = list(messages)
        system_idx = None
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                system_idx = i
                break

        context_block = f"\n\n<context>\n{context}\n</context>"

        if system_idx is not None:
            result[system_idx] = {
                **result[system_idx],
                "content": result[system_idx].get("content", "") + context_block
            }
        else:
            result.insert(0, {
                "role": "system",
                "content": f"Use the following context to answer questions:{context_block}"
            })

        return result

    # ========================================================================
    # Pipeline API: process()
    # ========================================================================

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Process context for pipeline composition.

        Reads from ctx.query and ctx.documents, writes to ctx.response.

        Args:
            ctx: Input context

        Returns:
            Result[Context] with response populated
        """
        # Build messages from context
        messages = []

        # Add system message if instructions provided
        instructions = ctx.get("instructions") or ctx.get("system_prompt")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Add context from documents (RAG)
        if ctx.documents:
            context = ctx.format_documents()
            if messages:
                messages[0]["content"] += f"\n\nContext:\n{context}"
            else:
                messages.append({
                    "role": "system",
                    "content": f"Use this context to answer:\n{context}"
                })

        # Add user query
        if ctx.query:
            messages.append({"role": "user", "content": ctx.query})

        # Add conversation history
        if ctx.messages:
            # Insert before the last user message
            for msg in ctx.messages:
                messages.insert(-1, msg.to_dict())

        # Run the pipe
        response = await self._run(
            messages=messages,
            model=ctx.get("model", "openai:gpt-4"),
            api_key=ctx.get("api_key"),
            **{k: v for k, v in ctx.variables.items() if k not in ["model", "api_key", "instructions", "system_prompt"]}
        )

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error or "Pipe execution failed",
                primitive=self._name
            ))

        # Update context
        new_ctx = ctx.with_response(response.output)

        if response.usage:
            from langpy.core.context import TokenUsage
            usage = TokenUsage(
                prompt_tokens=response.usage.get('prompt_tokens', 0),
                completion_tokens=response.usage.get('completion_tokens', 0),
                total_tokens=response.usage.get('total_tokens', 0)
            )
            new_ctx = new_ctx.add_usage(usage)

        return Success(new_ctx)

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    async def generate(
        self,
        prompt: str,
        model: str = "openai:gpt-4",
        **kwargs
    ) -> str:
        """
        Simple text generation.

        Args:
            prompt: Input prompt
            model: LLM model
            **kwargs: Additional options

        Returns:
            Generated text
        """
        response = await self.run(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            **kwargs
        )
        return response.output if response.success else f"Error: {response.error}"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai:gpt-4",
        **kwargs
    ) -> str:
        """
        Chat completion.

        Args:
            messages: Conversation messages
            model: LLM model
            **kwargs: Additional options

        Returns:
            Response text
        """
        response = await self.run(
            messages=messages,
            model=model,
            **kwargs
        )
        return response.output if response.success else f"Error: {response.error}"
