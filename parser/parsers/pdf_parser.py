"""
PDF Parser - Basic PDF text extraction with OCR support.

Handles PDF documents with text extraction and OCR for scanned documents.
"""

import time
from typing import Dict, Any, Optional
from ..models import ParsedDocument, ParseRequest, ParseResult, ParserError, OcrFailureError


class PdfParser:
    """Parser for PDF documents."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []
        
        try:
            import PyPDF2
        except ImportError:
            missing_deps.append("PyPDF2")
        
        try:
            import fitz  # PyMuPDF
        except ImportError:
            missing_deps.append("PyMuPDF")
        
        try:
            import pytesseract
        except ImportError:
            missing_deps.append("pytesseract")
        
        try:
            from PIL import Image
        except ImportError:
            missing_deps.append("Pillow")
        
        if missing_deps:
            raise ParserError(f"PDF parser requires missing dependencies: {', '.join(missing_deps)}. Install with: pip install {' '.join(missing_deps)}")
    
    async def parse(self, request: ParseRequest) -> ParseResult:
        """
        Parse PDF document.
        
        Args:
            request: Parse request with PDF content
            
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
            
            # Extract text from PDF
            text, metadata = await self._extract_text(content, request)
            
            # Create document
            document = ParsedDocument(text=text, metadata=metadata)
            
            parse_time = time.time() - start_time
            return ParseResult(document=document, parse_time=parse_time)
            
        except Exception as e:
            raise ParserError(f"PDF parsing failed: {str(e)}")
    
    async def _extract_text(self, content: bytes, request: ParseRequest) -> tuple[str, Dict[str, Any]]:
        """Extract text from PDF content."""
        try:
            import PyPDF2
            from io import BytesIO
            
            # Create PDF reader
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_parts = []
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    # Page extraction failed, might be scanned
                    if request.options and request.options.enable_ocr:
                        try:
                            ocr_text = await self._ocr_page(content, page_num)
                            if ocr_text:
                                text_parts.append(ocr_text)
                        except Exception as ocr_error:
                            # OCR failed, continue with next page
                            continue
            
            # Combine all text
            full_text = '\n\n'.join(text_parts)
            
            # Create metadata
            metadata = {
                'content_type': 'application/pdf',
                'filename': request.filename,
                'page_count': page_count,
                'text_length': len(full_text),
                'has_text': bool(full_text.strip()),
                'ocr_used': len(text_parts) > page_count  # Indicates OCR was used
            }
            
            return full_text, metadata
            
        except Exception as e:
            # Try OCR if enabled
            if request.options and request.options.enable_ocr:
                try:
                    return await self._ocr_full_pdf(content, request)
                except Exception as ocr_error:
                    raise ParserError(f"PDF parsing failed: {str(e)}. OCR also failed: {str(ocr_error)}")
            else:
                raise ParserError(f"PDF text extraction failed: {str(e)}")
    
    async def _ocr_page(self, content: bytes, page_num: int) -> Optional[str]:
        """Perform OCR on a single page."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import pytesseract
            from io import BytesIO
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=content, filetype="pdf")
            
            if page_num <= len(pdf_document):
                page = pdf_document[page_num - 1]
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # Perform OCR
                text = pytesseract.image_to_string(image)
                return text.strip()
            
            return None
            
        except Exception as e:
            raise OcrFailureError(f"page {page_num}", str(e))
    
    async def _ocr_full_pdf(self, content: bytes, request: ParseRequest) -> tuple[str, Dict[str, Any]]:
        """Perform OCR on entire PDF."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import pytesseract
            from io import BytesIO
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=content, filetype="pdf")
            
            text_parts = []
            page_count = len(pdf_document)
            
            for page_num in range(page_count):
                page = pdf_document[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # Perform OCR
                text = pytesseract.image_to_string(image)
                if text.strip():
                    text_parts.append(text.strip())
            
            # Combine all text
            full_text = '\n\n'.join(text_parts)
            
            # Create metadata
            metadata = {
                'content_type': 'application/pdf',
                'filename': request.filename,
                'page_count': page_count,
                'text_length': len(full_text),
                'has_text': bool(full_text.strip()),
                'ocr_used': True,
                'ocr_method': 'pytesseract'
            }
            
            return full_text, metadata
            
        except Exception as e:
            raise OcrFailureError(request.filename or "PDF", str(e)) 