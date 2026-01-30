"""
Text Parser - Basic text extraction for various formats.

Handles plain text, markdown, HTML, XML, JSON, YAML, CSV, and code files.
"""

import re
import time
from typing import Dict, Any, Optional
from ..models import ParsedDocument, ParseRequest, ParseResult, ParserError


class TextParser:
    """Parser for text-based formats."""
    
    def __init__(self):
        """Initialize text parser."""
        pass
    
    async def parse(self, request: ParseRequest) -> ParseResult:
        """
        Parse text content.
        
        Args:
            request: Parse request with content and metadata
            
        Returns:
            Parse result with extracted text
        """
        start_time = time.time()
        
        try:
            # Convert content to string if needed
            if isinstance(request.content, bytes):
                content = request.content.decode('utf-8', errors='replace')
            else:
                content = str(request.content)
            
            # Extract text based on content type
            text = self._extract_text(content, request.content_type)
            
            # Create metadata
            metadata = {
                'content_type': request.content_type,
                'filename': request.filename,
                'text_length': len(text),
                'original_length': len(content)
            }
            
            # Create document
            document = ParsedDocument(text=text, metadata=metadata)
            
            parse_time = time.time() - start_time
            return ParseResult(document=document, parse_time=parse_time)
            
        except Exception as e:
            raise ParserError(f"Text parsing failed: {str(e)}")
    
    def _extract_text(self, content: str, content_type: Optional[str]) -> str:
        """Extract text from content based on type."""
        if not content_type:
            return content
        
        # Handle different text formats
        if content_type == 'text/html':
            return self._extract_from_html(content)
        elif content_type in ['text/xml', 'application/xml']:
            return self._extract_from_xml(content)
        elif content_type == 'application/json':
            return self._extract_from_json(content)
        elif content_type in ['application/x-yaml', 'text/yaml']:
            return self._extract_from_yaml(content)
        elif content_type == 'text/csv':
            return self._extract_from_csv(content)
        else:
            # Plain text, markdown, code files
            return content
    
    def _extract_from_html(self, content: str) -> str:
        """Extract text from HTML content."""
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Decode HTML entities
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        content = content.replace('&#39;', "'")
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _extract_from_xml(self, content: str) -> str:
        """Extract text from XML content with better structure preservation."""
        try:
            # Try to pretty-print the XML for better readability
            import xml.etree.ElementTree as ET
            from xml.dom import minidom
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Convert to string with pretty formatting
            rough_string = ET.tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Remove the XML declaration line
            lines = pretty_xml.split('\n')
            if lines and lines[0].startswith('<?xml'):
                lines = lines[1:]
            
            return '\n'.join(lines).strip()
            
        except Exception:
            # Fallback: just return the original content
            return content
    
    def _extract_from_json(self, content: str) -> str:
        """Extract text from JSON content."""
        try:
            import json
            data = json.loads(content)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except:
            return content
    
    def _extract_from_yaml(self, content: str) -> str:
        """Extract text from YAML content."""
        try:
            import yaml
            data = yaml.safe_load(content)
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except:
            return content
    
    def _extract_from_csv(self, content: str) -> str:
        """Extract text from CSV content."""
        lines = content.split('\n')
        if not lines:
            return content
        
        # Join with spaces for readability
        return ' '.join(lines) 