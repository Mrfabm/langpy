"""
Parser Registry - Core parsers for Langbase compatibility.

Provides essential parsers for text, PDF, office documents, images, archives, and emails.
"""

from typing import Dict, Type, Optional
from .text_parser import TextParser
from .pdf_parser import PdfParser
from .office_parser import OfficeParser
from .image_parser import ImageParser
from .archive_parser import ArchiveParser
from .email_parser import EmailParser


class ParserRegistry:
    """Registry for available parsers."""
    
    def __init__(self):
        """Initialize parser registry."""
        self._parsers: Dict[str, Type] = {}
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """Register default parsers."""
        # Text formats
        self.register('text/plain', TextParser)
        self.register('text/markdown', TextParser)
        self.register('text/html', TextParser)
        self.register('text/xml', TextParser)
        self.register('application/xml', TextParser)
        self.register('application/json', TextParser)
        self.register('application/x-yaml', TextParser)
        self.register('text/csv', TextParser)
        
        # Code files
        self.register('application/x-python', TextParser)
        self.register('application/javascript', TextParser)
        self.register('application/typescript', TextParser)
        self.register('text/x-java-source', TextParser)
        self.register('text/x-c++src', TextParser)
        self.register('text/x-csrc', TextParser)
        self.register('text/x-go', TextParser)
        self.register('text/x-rust', TextParser)
        self.register('application/x-php', TextParser)
        self.register('text/x-ruby', TextParser)
        self.register('application/sql', TextParser)
        self.register('application/x-sh', TextParser)
        self.register('text/css', TextParser)
        
        # Office documents
        self.register('application/pdf', PdfParser)
        self.register('application/vnd.openxmlformats-officedocument.wordprocessingml.document', OfficeParser)
        self.register('application/vnd.openxmlformats-officedocument.presentationml.presentation', OfficeParser)
        self.register('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', OfficeParser)
        
        # Images
        self.register('image/jpeg', ImageParser)
        self.register('image/png', ImageParser)
        self.register('image/gif', ImageParser)
        
        # Archives
        self.register('application/zip', ArchiveParser)
        
        # Email
        self.register('message/rfc822', EmailParser)
        self.register('application/vnd.ms-outlook', EmailParser)
        self.register('application/mbox', EmailParser)
    
    def register(self, mime_type: str, parser_class: Type):
        """Register a parser for a MIME type."""
        self._parsers[mime_type] = parser_class
    
    def get_parser(self, mime_type: str):
        """Get parser class for MIME type."""
        return self._parsers.get(mime_type)
    
    def is_supported(self, mime_type: str) -> bool:
        """Check if MIME type is supported."""
        return mime_type in self._parsers
    
    def get_supported_types(self):
        """Get list of supported MIME types."""
        return list(self._parsers.keys())


# Global parser registry
parser_registry = ParserRegistry() 