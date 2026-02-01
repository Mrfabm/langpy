"""
Tools Primitive - Langbase-compatible Tools API.

The Tools primitive provides pre-built and custom tool integrations.

Usage:
    # Direct API
    result = await lb.tools.run(tool="web_search", query="Python tutorials")

    # Register custom tools
    lb.tools.register("calculator", calculator_fn)
"""

from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from langpy.core.primitive import BasePrimitive, ToolResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


class Tools(BasePrimitive):
    """
    Tools primitive - Pre-built and custom tool integrations.

    Provides tools for:
    - Web search (via search APIs)
    - Web crawling
    - Custom functions

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Web search
        result = await lb.tools.run(
            tool="web_search",
            query="Latest Python news"
        )

        # Register custom tool
        def calculator(expression: str) -> str:
            return str(eval(expression))

        lb.tools.register("calculator", calculator)

        result = await lb.tools.run(
            tool="calculator",
            expression="2 + 2"
        )
    """

    def __init__(self, client: Any = None, name: str = "tools"):
        super().__init__(name=name, client=client)
        self._registry: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}

        # Register built-in tools
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools."""
        # Web search (placeholder - needs API key)
        async def web_search(query: str, num_results: int = 5) -> str:
            try:
                # Try using available search APIs
                import httpx

                # Try SerpAPI
                serpapi_key = os.getenv("SERPAPI_API_KEY")
                if serpapi_key:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            "https://serpapi.com/search",
                            params={
                                "q": query,
                                "api_key": serpapi_key,
                                "num": num_results
                            }
                        )
                        data = response.json()
                        results = data.get("organic_results", [])
                        return "\n".join([
                            f"- {r.get('title')}: {r.get('snippet')}"
                            for r in results[:num_results]
                        ])

                # Try Tavily
                tavily_key = os.getenv("TAVILY_API_KEY")
                if tavily_key:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.tavily.com/search",
                            json={
                                "query": query,
                                "api_key": tavily_key,
                                "max_results": num_results
                            }
                        )
                        data = response.json()
                        results = data.get("results", [])
                        return "\n".join([
                            f"- {r.get('title')}: {r.get('content')}"
                            for r in results[:num_results]
                        ])

                return "Web search requires SERPAPI_API_KEY or TAVILY_API_KEY"

            except Exception as e:
                return f"Search error: {e}"

        self._registry["web_search"] = web_search
        self._tool_schemas["web_search"] = {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }

        # Web crawl
        async def web_crawl(url: str) -> str:
            try:
                import httpx
                from bs4 import BeautifulSoup

                async with httpx.AsyncClient() as client:
                    response = await client.get(url, follow_redirects=True, timeout=30.0)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Remove scripts and styles
                    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()

                    text = soup.get_text(separator='\n', strip=True)
                    # Limit length
                    return text[:10000]

            except ImportError:
                return "Web crawl requires: pip install httpx beautifulsoup4"
            except Exception as e:
                return f"Crawl error: {e}"

        self._registry["web_crawl"] = web_crawl
        self._tool_schemas["web_crawl"] = {
            "name": "web_crawl",
            "description": "Fetch and extract text from a web page",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to crawl"}
                },
                "required": ["url"]
            }
        }

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = None,
        parameters: Dict[str, Any] = None
    ):
        """
        Register a custom tool.

        Args:
            name: Tool name
            fn: Tool function (sync or async)
            description: Tool description
            parameters: JSON Schema for parameters
        """
        self._registry[name] = fn
        self._tool_schemas[name] = {
            "name": name,
            "description": description or fn.__doc__ or f"Tool: {name}",
            "parameters": parameters or {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema for LLM function calling."""
        return self._tool_schemas.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas."""
        return [
            {"type": "function", "function": schema}
            for schema in self._tool_schemas.values()
        ]

    async def _run(
        self,
        tool: str = None,
        **params
    ) -> ToolResponse:
        """
        Run a tool.

        Args:
            tool: Tool name
            **params: Tool parameters

        Returns:
            ToolResponse with output
        """
        try:
            if not tool:
                return ToolResponse(
                    success=False,
                    error="tool name required"
                )

            if tool not in self._registry:
                return ToolResponse(
                    success=False,
                    error=f"Unknown tool: {tool}. Available: {list(self._registry.keys())}"
                )

            fn = self._registry[tool]

            # Execute (handle sync/async)
            import asyncio
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**params)
            else:
                result = fn(**params)

            return ToolResponse(
                success=True,
                tool=tool,
                output=result,
                metadata=params
            )

        except Exception as e:
            return ToolResponse(success=False, error=str(e), tool=tool)

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Process context - run tool from context variables."""
        tool = ctx.get("tool")

        if not tool:
            return Failure(PrimitiveError(
                code=ErrorCode.VALIDATION_ERROR,
                message="tool name required in context",
                primitive=self._name
            ))

        # Get tool params from context
        params = {k: v for k, v in ctx.variables.items() if k != "tool"}

        response = await self._run(tool=tool, **params)

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error,
                primitive=self._name
            ))

        new_ctx = ctx.set("tool_output", response.output)
        new_ctx = new_ctx.set("tool_name", tool)

        # If output is string, add as response
        if isinstance(response.output, str):
            new_ctx = new_ctx.with_response(response.output)

        return Success(new_ctx)

    # Convenience methods
    async def search(self, query: str, **kwargs) -> str:
        """Web search shortcut."""
        response = await self._run(tool="web_search", query=query, **kwargs)
        return response.output if response.success else f"Error: {response.error}"

    async def crawl(self, url: str) -> str:
        """Web crawl shortcut."""
        response = await self._run(tool="web_crawl", url=url)
        return response.output if response.success else f"Error: {response.error}"

    async def execute(self, tool: str, **params) -> Any:
        """Execute tool and return output."""
        response = await self._run(tool=tool, **params)
        return response.output if response.success else None
