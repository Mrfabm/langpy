"""
Image Parser - Basic OCR for image files.

Handles JPEG, PNG, and GIF images with OCR text extraction.
"""

import time
from typing import Dict, Any, Optional
from ..models import ParsedDocument, ParseRequest, ParseResult, ParserError, OcrFailureError


class ImageParser:
    """Parser for image files with OCR."""
    
    def __init__(self):
        """Initialize image parser."""
        pass
    
    async def parse(self, request: ParseRequest) -> ParseResult:
        """
        Parse image with OCR.
        
        Args:
            request: Parse request with image content
            
        Returns:
            Parse result with extracted text
        """
        start_time = time.time()
        
        try:
            # Get content as bytes
            if isinstance(request.content, str):
                content = request.content.encode('utf-8')
            else:
                content = request.content
            
            # Extract text using OCR
            text, metadata = await self._extract_text(content, request)
            
            # Create document
            document = ParsedDocument(text=text, metadata=metadata)
            
            parse_time = time.time() - start_time
            return ParseResult(document=document, parse_time=parse_time)
            
        except Exception as e:
            raise ParserError(f"Image parsing failed: {str(e)}")
    
    async def _extract_text(self, content: bytes, request: ParseRequest) -> tuple[str, Dict[str, Any]]:
        """Extract text from image using OCR."""
        try:
            from PIL import Image
            import pytesseract
            from io import BytesIO
            
            # Load image
            image = Image.open(BytesIO(content))
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Create metadata
            metadata = {
                'content_type': request.content_type or 'image/unknown',
                'filename': request.filename,
                'text_length': len(text),
                'image_size': image.size,
                'image_mode': image.mode,
                'ocr_used': True,
                'ocr_method': 'pytesseract'
            }
            
            return text.strip(), metadata
            
        except Exception as e:
            raise OcrFailureError(request.filename or "image", str(e)) 