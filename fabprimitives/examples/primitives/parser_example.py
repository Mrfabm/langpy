"""
Parser Primitive
================
Extract text from various document formats.

Parser supports:
    - PDF documents
    - Text files
    - Office documents (docx, xlsx, pptx)
    - Images with OCR
    - HTML/XML
    - And more

Architecture:
    Document -> Parser -> Extracted Text

    +--------------------------------------+
    |             Parser                   |
    |  +------------+   +---------------+  |
    |  |   Input    | -> |   Extracted   |  |
    |  | (PDF/docx/ |   |     Text      |  |
    |  |  image...) |   |               |  |
    |  +------------+   +---------------+  |
    +--------------------------------------+
"""

import asyncio
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Simple Parser class for standalone example
# (The langpy.primitives.Parser expects a client parameter)
class Parser:
        """Simple parser implementation."""

        SUPPORTED_FORMATS = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".xml": "application/xml",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }

        def __init__(self):
            pass

        async def parse(self, file_path: str) -> str:
            """Parse a file and extract text."""
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            ext = path.suffix.lower()

            if ext in [".txt", ".md"]:
                return path.read_text(encoding="utf-8")

            elif ext == ".json":
                import json
                data = json.loads(path.read_text(encoding="utf-8"))
                return json.dumps(data, indent=2)

            elif ext == ".csv":
                return path.read_text(encoding="utf-8")

            elif ext in [".html", ".htm"]:
                text = path.read_text(encoding="utf-8")
                # Simple HTML tag stripping
                import re
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            elif ext == ".pdf":
                return "[PDF parsing requires pypdf or pdfplumber library]"

            elif ext == ".docx":
                return "[DOCX parsing requires python-docx library]"

            elif ext in [".png", ".jpg", ".jpeg"]:
                return "[Image OCR requires pytesseract library]"

            else:
                # Try to read as text
                try:
                    return path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    return f"[Cannot parse binary file: {ext}]"

        def supports(self, format_or_path: str) -> bool:
            """Check if a format is supported."""
            if format_or_path.startswith("."):
                ext = format_or_path.lower()
            else:
                ext = Path(format_or_path).suffix.lower()
            return ext in self.SUPPORTED_FORMATS


# =============================================================================
# SAMPLE FILES FOR DEMO
# =============================================================================

SAMPLE_TEXT = """
Introduction to AI

Artificial intelligence (AI) has transformed many industries.
Machine learning, a subset of AI, enables computers to learn from data.

Key concepts:
- Neural networks
- Deep learning
- Natural language processing
- Computer vision

The future of AI looks promising with advances in:
1. Large language models
2. Robotics
3. Autonomous systems
"""

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Overview</title>
    <style>body { font-family: Arial; }</style>
</head>
<body>
    <h1>What is AI?</h1>
    <p>Artificial intelligence simulates human thinking.</p>
    <script>console.log("ignore this");</script>
    <ul>
        <li>Machine Learning</li>
        <li>Deep Learning</li>
    </ul>
</body>
</html>
"""

SAMPLE_CSV = """name,age,city
Alice,30,New York
Bob,25,San Francisco
Carol,35,Chicago
"""


# =============================================================================
# BASIC PARSING
# =============================================================================

async def basic_parsing_demo():
    """Demonstrate basic text parsing."""
    print("=" * 60)
    print("   BASIC PARSING - Text Extraction")
    print("=" * 60)
    print()

    parser = Parser()

    # Create temp files for demo
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample text file
        txt_path = Path(tmpdir) / "sample.txt"
        txt_path.write_text(SAMPLE_TEXT)

        print("1. Parsing text file:")
        print("-" * 40)

        text = await parser.parse(str(txt_path))
        preview = text[:200].replace("\n", " ")
        print(f"   Source: {txt_path.name}")
        print(f"   Length: {len(text)} characters")
        print(f"   Preview: '{preview}...'")
        print()


# =============================================================================
# HTML PARSING
# =============================================================================

async def html_parsing_demo():
    """Demonstrate HTML parsing with tag stripping."""
    print("=" * 60)
    print("   HTML PARSING - Clean Text Extraction")
    print("=" * 60)
    print()

    parser = Parser()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "page.html"
        html_path.write_text(SAMPLE_HTML)

        print("1. Original HTML (with tags):")
        print("-" * 40)
        print(f"   {SAMPLE_HTML[:150]}...")
        print()

        print("2. Parsed text (tags stripped):")
        print("-" * 40)
        text = await parser.parse(str(html_path))
        print(f"   {text}")
        print()


# =============================================================================
# CSV PARSING
# =============================================================================

async def csv_parsing_demo():
    """Demonstrate CSV parsing."""
    print("=" * 60)
    print("   CSV PARSING - Structured Data")
    print("=" * 60)
    print()

    parser = Parser()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        csv_path.write_text(SAMPLE_CSV)

        print("Parsing CSV file:")
        print("-" * 40)

        text = await parser.parse(str(csv_path))
        print(text)
        print()

        print("Tip: For structured processing, use Python's csv module")
        print("     Parser gives you the raw text to feed to LLMs")
        print()


# =============================================================================
# FORMAT SUPPORT CHECKING
# =============================================================================

async def format_support_demo():
    """Demonstrate checking supported formats."""
    print("=" * 60)
    print("   FORMAT SUPPORT - Checking Compatibility")
    print("=" * 60)
    print()

    parser = Parser()

    formats = [
        ".txt", ".pdf", ".docx", ".xlsx",
        ".html", ".md", ".json", ".csv",
        ".png", ".jpg", ".mp4", ".zip"
    ]

    print("Supported formats:")
    print("-" * 40)

    for fmt in formats:
        supported = parser.supports(fmt)
        status = "YES" if supported else "NO"
        print(f"   {fmt:10} {status}")

    print()


# =============================================================================
# OCR PREVIEW
# =============================================================================

async def ocr_preview_demo():
    """Preview OCR capabilities for images."""
    print("=" * 60)
    print("   OCR PREVIEW - Image Text Extraction")
    print("=" * 60)
    print()

    print("Image OCR capabilities:")
    print("-" * 40)
    print("""
    Parser can extract text from images using OCR (Optical Character Recognition).

    Supported image formats:
    - PNG
    - JPG/JPEG
    - TIFF
    - BMP

    Usage:
        parser = Parser()
        text = await parser.parse("document_scan.png")

    Requirements:
        - pytesseract library
        - Tesseract OCR engine installed

    Best practices:
        - Use high-resolution images (300+ DPI)
        - Ensure good contrast
        - Straighten skewed documents
        - Pre-process noisy images
    """)
    print()


# =============================================================================
# PDF PARSING PREVIEW
# =============================================================================

async def pdf_preview_demo():
    """Preview PDF parsing capabilities."""
    print("=" * 60)
    print("   PDF PREVIEW - Document Extraction")
    print("=" * 60)
    print()

    print("PDF parsing capabilities:")
    print("-" * 40)
    print("""
    Parser extracts text from PDF documents:

    Features:
    - Text extraction from all pages
    - Handling of multi-column layouts
    - Table detection and extraction
    - Metadata extraction (title, author, etc.)

    Usage:
        parser = Parser()
        text = await parser.parse("report.pdf")
        print(text)

    Requirements:
        - pypdf or pdfplumber library

    Limitations:
        - Scanned PDFs need OCR
        - Complex layouts may need special handling
        - Some PDFs have copy-protection
    """)
    print()


# =============================================================================
# TABLE EXTRACTION PREVIEW
# =============================================================================

async def table_extraction_demo():
    """Preview table extraction capabilities."""
    print("=" * 60)
    print("   TABLE EXTRACTION - Structured Data from Documents")
    print("=" * 60)
    print()

    print("Table extraction from documents:")
    print("-" * 40)
    print("""
    Parser can extract tables from various formats:

    Formats with table support:
    - PDF (with pdfplumber)
    - DOCX (with python-docx)
    - HTML (with beautifulsoup)
    - Excel (with openpyxl)

    Output format:
        Tables are extracted as structured data that can be
        converted to CSV, JSON, or markdown for LLM processing.

    Example output:
        | Name  | Age | City          |
        |-------|-----|---------------|
        | Alice | 30  | New York      |
        | Bob   | 25  | San Francisco |

    Use case:
        Extract tables -> Convert to text -> Feed to LLM for analysis
    """)
    print()


# =============================================================================
# PARSER IN RAG PIPELINE
# =============================================================================

async def rag_pipeline_demo():
    """Demonstrate Parser in a RAG pipeline context."""
    print("=" * 60)
    print("   PARSER IN RAG - Document Processing Pipeline")
    print("=" * 60)
    print()

    print("Parser in a RAG pipeline:")
    print("-" * 40)
    print("""
    Document -> Parser -> Chunker -> Embed -> Memory -> Search

    +----------+   +--------+   +---------+   +-------+   +--------+
    |   PDF    | -> | Parser | -> | Chunker | -> | Embed | -> | Memory |
    |   DOCX   |   |        |   |         |   |       |   |        |
    |   HTML   |   |        |   |         |   |       |   |        |
    +----------+   +--------+   +---------+   +-------+   +--------+

    Example code:
    """)

    print("""
    from langpy_sdk import Memory, Pipe

    # Step 1: Parse documents
    parser = Parser()
    text = await parser.parse("research_paper.pdf")

    # Step 2: Chunk the text
    chunker = Chunker(chunk_size=500)
    chunks = await chunker.chunk(text)

    # Step 3: Store in memory
    memory = Memory(name="research")
    await memory.add_many(chunks)

    # Step 4: Search and generate
    results = await memory.search("key findings")
    context = "\\n".join([r.text for r in results])

    pipe = Pipe(system="Answer based on the research paper.")
    answer = await pipe.quick(f"Context: {context}\\n\\nQuestion: What are the key findings?")
    """)
    print()


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Parser demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 17 + "PARSER PRIMITIVE DEMO" + " " * 17 + "*")
    print("*" * 60)
    print()

    await basic_parsing_demo()
    await html_parsing_demo()
    await csv_parsing_demo()
    await format_support_demo()
    await ocr_preview_demo()
    await pdf_preview_demo()
    await table_extraction_demo()
    await rag_pipeline_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
