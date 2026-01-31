"""
Parser Primitive - Langbase-compatible Document Parser API.

The Parser primitive extracts text from various document formats.

Usage:
    # Direct API
    result = await lb.parser.run(document="file.pdf")
    print(result.text)

    # Pipeline composition
    pipeline = parser | chunker | embed
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

from langpy.core.primitive import BasePrimitive, ParserResponse
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


class Parser(BasePrimitive):
    """
    Parser primitive - Document text extraction.

    Supports:
    - PDF documents
    - Office files (docx, xlsx, pptx)
    - Images (with OCR)
    - HTML/XML
    - Plain text
    - And more

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Parse a PDF
        result = await lb.parser.run(document="report.pdf")
        print(result.text)

        # Parse with options
        result = await lb.parser.run(
            document="image.png",
            ocr=True
        )
    """

    def __init__(self, client: Any = None, name: str = "parser"):
        super().__init__(name=name, client=client)
        self._async_parser = None

    def _get_async_parser(self):
        if self._async_parser is None:
            try:
                from parser.async_parser import AsyncParser
                self._async_parser = AsyncParser()
            except ImportError:
                from parser.simple_parser import SimpleParser
                self._async_parser = SimpleParser()
        return self._async_parser

    async def _run(
        self,
        document: Union[str, Path, bytes] = None,
        content: Union[str, bytes] = None,
        format: str = "auto",
        ocr: bool = False,
        **kwargs
    ) -> ParserResponse:
        """
        Parse a document.

        Args:
            document: File path or URL
            content: Raw content bytes
            format: Format hint ("auto", "pdf", "docx", etc.)
            ocr: Enable OCR for images

        Returns:
            ParserResponse with extracted text
        """
        try:
            parser = self._get_async_parser()

            # Determine input
            if document:
                if isinstance(document, (str, Path)):
                    path = Path(document)
                    if path.exists():
                        result = await parser.parse_file(path)
                    else:
                        return ParserResponse(
                            success=False,
                            error=f"File not found: {document}"
                        )
                else:
                    result = await parser.parse_bytes(document, format=format)
            elif content:
                if isinstance(content, str):
                    result = await parser.parse_text(content)
                else:
                    result = await parser.parse_bytes(content, format=format)
            else:
                return ParserResponse(
                    success=False,
                    error="Provide document or content"
                )

            # Extract text from result
            text = ""
            pages = 0
            metadata = {}

            if hasattr(result, 'pages'):
                text = "\n\n".join(result.pages)
                pages = len(result.pages)
            elif hasattr(result, 'text'):
                text = result.text
                pages = 1
            elif isinstance(result, str):
                text = result
                pages = 1

            if hasattr(result, 'metadata'):
                metadata = result.metadata

            return ParserResponse(
                success=True,
                text=text,
                pages=pages,
                format=format,
                metadata=metadata
            )

        except Exception as e:
            return ParserResponse(success=False, error=str(e))

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Process context - parse document and add text to context."""
        document = ctx.get("document") or ctx.get("file")
        content = ctx.get("content")

        if not document and not content:
            return Failure(PrimitiveError(
                code=ErrorCode.VALIDATION_ERROR,
                message="document or content required",
                primitive=self._name
            ))

        response = await self._run(document=document, content=content)

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error,
                primitive=self._name
            ))

        new_ctx = ctx.set("parsed_text", response.text)
        new_ctx = new_ctx.set("parsed_pages", response.pages)
        if not ctx.query:
            new_ctx = new_ctx.with_query(response.text)

        return Success(new_ctx)

    # Convenience methods
    async def parse(self, document: Union[str, Path, bytes], **kwargs) -> str:
        """Parse and return text."""
        response = await self._run(document=document, **kwargs)
        return response.text if response.success else ""

    async def parse_file(self, path: Union[str, Path], **kwargs) -> str:
        """Parse a file."""
        return await self.parse(document=path, **kwargs)
