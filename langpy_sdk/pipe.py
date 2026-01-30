"""
LangPy Pipe - Clean SDK wrapper for LLM calls.

Simple, intuitive interface for making LLM calls without tool execution.
Pipes are simpler than Agents - they just call the LLM once.

Supports both the original API and the new composable architecture:

Original API:
    from langpy_sdk import Pipe

    pipe = Pipe(model="gpt-4o-mini")
    response = await pipe.run("Translate 'hello' to French")
    print(response.content)  # "Bonjour"

New Composable API:
    from langpy.core import Context
    from langpy_sdk import Pipe

    pipe = Pipe(model="gpt-4o-mini", system_prompt="You are a translator.")
    result = await pipe.process(Context(query="Translate 'hello' to French"))
    print(result.unwrap().response)  # "Bonjour"
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any, Union, AsyncGenerator, TYPE_CHECKING
from dataclasses import dataclass

# Import core types for the new architecture
try:
    from langpy.core.context import Context, TokenUsage, CostInfo, Document
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode
    from langpy.core.primitive import BasePrimitive
    from langpy.core.observability import CostCalculator
    _NEW_ARCH_AVAILABLE = True
except ImportError:
    _NEW_ARCH_AVAILABLE = False
    # Define minimal stubs for type checking
    Context = Any
    Result = Any
    BasePrimitive = object


@dataclass
class PipeResponse:
    """
    Response from a pipe call.

    Attributes:
        content: The text response
        model: Model used
        usage: Token usage stats
        raw: Raw response object
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw: Any = None

    def __str__(self) -> str:
        return self.content


class Pipe:
    """
    Clean, simple Pipe interface for LLM calls.

    Pipes are simpler than Agents - they make a single LLM call
    without tool execution loops. Use for:
    - Simple Q&A
    - Text transformation
    - Classification
    - Summarization

    Implements IPrimitive for composable pipelines.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-4")
        system: Default system prompt (also accepts system_prompt)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens (default: 1000)
        api_key: API key (defaults to OPENAI_API_KEY env var)
        provider: LLM provider (default: "openai")
        name: Primitive name for tracing (default: "Pipe")

    Example:
        # Simple usage (original API)
        pipe = Pipe(model="gpt-4o-mini")
        response = await pipe.run("What is 2+2?")

        # With system prompt
        pipe = Pipe(
            model="gpt-4o-mini",
            system="You are a helpful translator."
        )
        response = await pipe.run("Translate 'hello' to Spanish")

        # Composable API (new)
        from langpy.core import Context
        pipe = Pipe(system_prompt="Answer using context.")
        result = await pipe.process(Context(query="What is LangPy?"))

        # Pipeline composition
        rag = memory | pipe | validator
        result = await rag.process(ctx)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        provider: str = "openai",
        # New parameters for composable architecture
        system_prompt: Optional[str] = None,
        name: str = "Pipe",
        include_documents: bool = True,
        document_template: Optional[str] = None
    ):
        self.model = model
        self.system = system or system_prompt  # Support both names
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.provider = provider
        self._client = None

        # New architecture fields
        self._name = name
        self._include_documents = include_documents
        self._document_template = document_template or "Context:\n{context}\n\nQuestion: {query}"

    @property
    def name(self) -> str:
        """Return the primitive name for tracing."""
        return self._name

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def quick(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
    ) -> str:
        """
        Quick helper for simple prompts. Returns just the response string.

        This is the recommended way for simple, non-streaming LLM calls.
        For composition with other primitives, use .process(ctx) instead.

        Args:
            prompt: User prompt string
            system: Override default system prompt

        Returns:
            Response string

        Example:
            pipe = Pipe(model="gpt-4o-mini")
            response = await pipe.quick("What is 2+2?")
            print(response)  # "4"
        """
        if not _NEW_ARCH_AVAILABLE:
            # Fallback to legacy run if new architecture not available
            response = await self._legacy_run(prompt, system=system)
            return response.content

        from langpy.core import Context

        ctx = Context(query=prompt)
        if system:
            # Store system override in context
            old_system = self.system
            self.system = system

        try:
            result = await self.process(ctx)
            if result.is_success():
                return result.unwrap().response or ""
            else:
                raise RuntimeError(str(result.error))
        finally:
            if system:
                self.system = old_system

    async def run(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        system: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> Union[PipeResponse, AsyncGenerator[str, None]]:
        """
        DEPRECATED: Use .quick() for simple prompts or .process() for composition.

        Run the pipe with a prompt.

        Args:
            prompt: User prompt or list of messages
            system: Override default system prompt
            stream: Stream the response (default: False)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            json_mode: Force JSON output (default: False)

        Returns:
            PipeResponse object (or async generator if streaming)
        """
        import warnings
        warnings.warn(
            "Pipe.run() is deprecated. Use .quick() for simple prompts or .process() for composition.",
            DeprecationWarning,
            stacklevel=2
        )
        return await self._legacy_run(
            prompt, system=system, stream=stream,
            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )

    async def _legacy_run(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        system: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> Union[PipeResponse, AsyncGenerator[str, None]]:
        """
        Internal implementation of run (kept for backward compatibility).
        """
        client = self._get_client()

        # Build messages
        messages = []

        # Add system prompt
        sys_prompt = system or self.system
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Add user prompt
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)

        # Build request params
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        # Make the call
        if stream:
            response = await client.chat.completions.create(**params)

            async def stream_generator():
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            return stream_generator()
        else:
            response = await client.chat.completions.create(**params)

            content = response.choices[0].message.content or ""
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return PipeResponse(
                content=content,
                model=self.model,
                usage=usage,
                raw=response
            )

    async def classify(
        self,
        text: str,
        categories: List[str],
        allow_multiple: bool = False
    ) -> Union[str, List[str]]:
        """
        Classify text into categories.

        Args:
            text: Text to classify
            categories: List of possible categories
            allow_multiple: Allow multiple categories (default: False)

        Returns:
            Category string or list of categories

        Example:
            category = await pipe.classify(
                "I love this product!",
                ["positive", "negative", "neutral"]
            )
            # Returns: "positive"
        """
        if allow_multiple:
            prompt = f"""Classify the following text into one or more of these categories: {', '.join(categories)}

Text: {text}

Return only the matching category names, comma-separated."""
        else:
            prompt = f"""Classify the following text into exactly one of these categories: {', '.join(categories)}

Text: {text}

Return only the category name, nothing else."""

        response = await self._legacy_run(prompt, temperature=0.0)

        if allow_multiple:
            return [c.strip() for c in response.content.split(",")]
        else:
            return response.content.strip()

    async def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> str:
        """
        Summarize text.

        Args:
            text: Text to summarize
            max_length: Maximum length in words (optional)
            style: Summary style - "concise", "detailed", "bullet" (default: "concise")

        Returns:
            Summary string

        Example:
            summary = await pipe.summarize(long_article, style="bullet")
        """
        length_instruction = ""
        if max_length:
            length_instruction = f" Keep it under {max_length} words."

        style_instructions = {
            "concise": "Provide a brief, concise summary.",
            "detailed": "Provide a detailed summary covering all main points.",
            "bullet": "Provide a bullet-point summary of the key points."
        }

        prompt = f"""{style_instructions.get(style, style_instructions['concise'])}{length_instruction}

Text to summarize:
{text}"""

        response = await self._legacy_run(prompt, temperature=0.3)
        return response.content

    async def extract(
        self,
        text: str,
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        Extract structured data from text.

        Args:
            text: Text to extract from
            fields: List of field names to extract

        Returns:
            Dictionary with extracted fields

        Example:
            data = await pipe.extract(
                "John Smith, age 30, lives in New York",
                ["name", "age", "city"]
            )
            # Returns: {"name": "John Smith", "age": "30", "city": "New York"}
        """
        import json as json_module

        prompt = f"""Extract the following fields from the text: {', '.join(fields)}

Text: {text}

Return as JSON object with the field names as keys. Use null for missing fields."""

        response = await self._legacy_run(prompt, temperature=0.0, json_mode=True)

        try:
            return json_module.loads(response.content)
        except json_module.JSONDecodeError:
            return {field: None for field in fields}

    def __repr__(self) -> str:
        return f"Pipe(model='{self.model}')"

    # ========================================================================
    # New Composable Architecture Methods
    # ========================================================================

    async def process(self, ctx: "Context") -> "Result[Context]":
        """
        Process a context (IPrimitive interface).

        This method enables composable pipelines with the | operator.

        Args:
            ctx: Input context with query and optional documents

        Returns:
            Result[Context] - Success with response or Failure with error

        Example:
            from langpy.core import Context

            pipe = Pipe(system_prompt="Answer questions using the context.")
            ctx = Context(query="What is Python?")
            result = await pipe.process(ctx)

            if result.is_success():
                print(result.unwrap().response)
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError(
                "New architecture not available. "
                "Make sure langpy.core is properly installed."
            )

        from langpy.core.context import Context, TokenUsage, CostInfo
        from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode
        from langpy.core.observability import CostCalculator

        # Start span for tracing
        ctx = ctx.start_span(self._name, {"model": self.model})

        try:
            # Build the prompt
            if self._include_documents and ctx.documents:
                # RAG mode: include documents in prompt
                prompt = self._document_template.format(
                    context=ctx.format_documents(),
                    query=ctx.query or ""
                )
            else:
                # Simple mode: just use the query
                prompt = ctx.query or ""

            # Build messages from context
            messages = []

            # Add system prompt
            if self.system:
                messages.append({"role": "system", "content": self.system})

            # Add conversation history if present
            for msg in ctx.messages:
                messages.append(msg.to_dict())

            # Add the current query
            messages.append({"role": "user", "content": prompt})

            # Make the LLM call
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract response
            content = response.choices[0].message.content or ""

            # Track token usage
            usage = TokenUsage()
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            # Calculate cost
            calculator = CostCalculator()
            cost = calculator.calculate(self.model, usage)

            # Build result context
            result_ctx = ctx.with_response(content)
            result_ctx = result_ctx.add_usage(usage)
            result_ctx = result_ctx.add_cost(cost)

            # Store raw response in variables
            result_ctx = result_ctx.set("_last_model", self.model)
            result_ctx = result_ctx.set("_last_usage", {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            })

            # End span successfully
            result_ctx = result_ctx.end_span("ok")

            return Success(result_ctx)

        except Exception as e:
            # End span with error
            ctx = ctx.end_span("error", str(e))

            # Determine error code
            error_name = type(e).__name__.lower()
            if "rate" in error_name or "ratelimit" in error_name:
                code = ErrorCode.LLM_RATE_LIMIT
            elif "timeout" in error_name:
                code = ErrorCode.TIMEOUT
            elif "context" in error_name or "length" in error_name:
                code = ErrorCode.LLM_CONTEXT_LENGTH
            else:
                code = ErrorCode.LLM_API_ERROR

            return Failure(PrimitiveError(
                code=code,
                message=str(e),
                primitive=self._name,
                cause=e
            ))

    def __or__(self, other) -> "Any":
        """
        Sequential composition with the | operator.

        Example:
            pipeline = memory | pipe | validator
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import Pipeline
        return Pipeline([self, other])

    def __and__(self, other) -> "Any":
        """
        Parallel composition with the & operator.

        Example:
            parallel = optimist & pessimist
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import ParallelPrimitives
        return ParallelPrimitives([self, other])

    def __rshift__(self, other) -> "Any":
        """
        Alternative sequential composition with >> operator.

        Example:
            pipeline = memory >> pipe >> validator
        """
        return self.__or__(other)
