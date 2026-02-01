"""
Archive Parser - Basic ZIP archive text extraction.

Handles ZIP archives with simple text extraction from contained files.
"""

import time
import zipfile
from typing import Dict, Any, Optional
from io import BytesIO
from ..models import ParsedDocument, ParseRequest, ParseResult, ParserError


class ArchiveParser:
    """Parser for ZIP archives."""
    
    def __init__(self):
        """Initialize archive parser."""
        pass
    
    async def parse(self, request: ParseRequest) -> ParseResult:
        """
        Parse ZIP archive.
        
        Args:
            request: Parse request with archive content
            
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
            
            # Extract text from archive
            text, metadata = await self._extract_text(content, request)
            
            # Create document
            document = ParsedDocument(text=text, metadata=metadata)
            
            parse_time = time.time() - start_time
            return ParseResult(document=document, parse_time=parse_time)
            
        except Exception as e:
            raise ParserError(f"Archive parsing failed: {str(e)}")
    
    async def _extract_text(self, content: bytes, request: ParseRequest) -> tuple[str, Dict[str, Any]]:
        """Extract text from ZIP archive content."""
        try:
            # Open ZIP file
            zip_file = BytesIO(content)
            
            with zipfile.ZipFile(zip_file, 'r') as archive:
                # Get file list
                file_list = archive.namelist()
                
                # Extract text from text files
                text_parts = []
                file_count = len(file_list)
                text_file_count = 0
                
                for filename in file_list:
                    # Skip directories and non-text files
                    if filename.endswith('/') or self._is_binary_file(filename):
                        continue
                    
                    try:
                        # Read file content
                        with archive.open(filename) as file:
                            file_content = file.read()
                            
                            # Try to decode as text
                            try:
                                text_content = file_content.decode('utf-8')
                                if text_content.strip():
                                    text_parts.append(f"File: {filename}\n{text_content}")
                                    text_file_count += 1
                            except UnicodeDecodeError:
                                # Skip binary files
                                continue
                                
                    except Exception:
                        # Skip problematic files
                        continue
                
                full_text = '\n\n'.join(text_parts)
                
                metadata = {
                    'content_type': 'application/zip',
                    'filename': request.filename,
                    'text_length': len(full_text),
                    'file_count': file_count,
                    'text_file_count': text_file_count
                }
                
                return full_text, metadata
                
        except Exception as e:
            raise ParserError(f"ZIP extraction failed: {str(e)}")
    
    def _is_binary_file(self, filename: str) -> bool:
        """Check if file is likely binary based on extension."""
        binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
            '.mp3', '.wav', '.flac', '.ogg', '.mp4', '.avi', '.mov', '.mkv',
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z',
            '.class', '.pyc', '.o', '.a', '.lib'
        }
        
        return any(filename.lower().endswith(ext) for ext in binary_extensions) 