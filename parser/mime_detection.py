"""
MIME Detection - Basic format detection for parser primitive.

Provides simple MIME type detection compatible with Langbase.
"""

import mimetypes
from pathlib import Path
from typing import Optional


class MimeDetector:
    """Basic MIME type detection."""
    
    def detect_from_bytes(self, content: bytes, filename: Optional[str] = None) -> str:
        """
        Detect MIME type from content bytes with fallback to filename.
        
        Args:
            content: File content as bytes
            filename: Optional filename for extension fallback
            
        Returns:
            Detected MIME type
        """
        # Try magic bytes for common formats
        mime_type = self._detect_magic_bytes(content)
        if mime_type:
            return mime_type
        
        # Fallback to filename extension
        if filename:
            mime_type = self._detect_from_extension(filename)
            if mime_type:
                return mime_type
        
        # Default fallback
        return 'application/octet-stream'
    
    def _detect_magic_bytes(self, content: bytes) -> Optional[str]:
        """Detect MIME type from magic bytes signatures."""
        # Basic magic bytes for common formats
        if content.startswith(b'%PDF'):
            return 'application/pdf'
        elif content.startswith(b'PK\x03\x04'):
            return 'application/zip'
        elif content.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif content.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif content.startswith(b'{\x22'):
            return 'application/json'
        elif content.startswith(b'<html') or content.startswith(b'<!DOCTYPE'):
            return 'text/html'
        elif content.startswith(b'<?xml'):
            return 'application/xml'
        return None
    
    def _detect_from_extension(self, filename: str) -> Optional[str]:
        """Detect MIME type from file extension."""
        path = Path(filename)
        extension = path.suffix.lower()
        
        # Common extensions
        extension_map = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.xml': 'application/xml',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.csv': 'text/csv',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.zip': 'application/zip',
            '.py': 'application/x-python',
            '.js': 'application/javascript',
            '.ts': 'application/typescript',
            '.java': 'text/x-java-source',
            '.cpp': 'text/x-c++src',
            '.c': 'text/x-csrc',
            '.go': 'text/x-go',
            '.rs': 'text/x-rust',
            '.php': 'application/x-php',
            '.rb': 'text/x-ruby',
            '.sql': 'application/sql',
            '.sh': 'application/x-sh',
            '.css': 'text/css',
        }
        
        if extension in extension_map:
            return extension_map[extension]
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type


# Global MIME detector instance
mime_detector = MimeDetector() 