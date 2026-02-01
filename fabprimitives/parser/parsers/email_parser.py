"""
Email Parser - Basic email text extraction.

Handles EML and other email formats with simple text extraction.
"""

import time
import email
from typing import Dict, Any, Optional
from email import policy
from email.parser import BytesParser
from ..models import ParsedDocument, ParseRequest, ParseResult, ParserError


class EmailParser:
    """Parser for email files."""
    
    def __init__(self):
        """Initialize email parser."""
        pass
    
    async def parse(self, request: ParseRequest) -> ParseResult:
        """
        Parse email document.
        
        Args:
            request: Parse request with email content
            
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
            
            # Extract text from email
            text, metadata = await self._extract_text(content, request)
            
            # Create document
            document = ParsedDocument(text=text, metadata=metadata)
            
            parse_time = time.time() - start_time
            return ParseResult(document=document, parse_time=parse_time)
            
        except Exception as e:
            raise ParserError(f"Email parsing failed: {str(e)}")
    
    async def _extract_text(self, content: bytes, request: ParseRequest) -> tuple[str, Dict[str, Any]]:
        """Extract text from email content."""
        try:
            # Parse email
            msg = BytesParser(policy=policy.default).parsebytes(content)
            
            # Extract headers
            headers_text = self._extract_headers(msg)
            
            # Extract body
            body_text = self._extract_body(msg)
            
            # Combine text
            full_text = ""
            if headers_text:
                full_text += f"Headers:\n{headers_text}\n\n"
            if body_text:
                full_text += f"Body:\n{body_text}"
            
            # Create metadata
            metadata = {
                'content_type': request.content_type or 'message/rfc822',
                'filename': request.filename,
                'text_length': len(full_text),
                'has_headers': bool(headers_text),
                'has_body': bool(body_text)
            }
            
            return full_text, metadata
            
        except Exception as e:
            raise ParserError(f"Email extraction failed: {str(e)}")
    
    def _extract_headers(self, msg) -> str:
        """Extract email headers."""
        headers = []
        
        # Common headers to extract
        header_fields = [
            'From', 'To', 'Cc', 'Bcc', 'Subject', 'Date', 'Message-ID',
            'In-Reply-To', 'References', 'Reply-To', 'Sender', 'Return-Path'
        ]
        
        for field in header_fields:
            value = msg.get(field)
            if value:
                headers.append(f"{field}: {value}")
        
        return '\n'.join(headers)
    
    def _extract_body(self, msg) -> str:
        """Extract email body."""
        body_parts = []
        
        if msg.is_multipart():
            # Handle multipart messages
            for part in msg.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                
                if part.get_content_maintype() == 'text':
                    content = part.get_content()
                    if content:
                        body_parts.append(content)
        else:
            # Handle simple text messages
            content = msg.get_content()
            if content:
                body_parts.append(content)
        
        if not body_parts:
            return ""
        
        # Join all text parts
        return '\n\n'.join(body_parts) 