import os
from pathlib import Path
from parser.docling_parser import DoclingParser
from parser.models import ParseRequest, ParserOptions

def test_file(filepath: Path):
    print(f"Testing {filepath.name} ...", end=" ")
    try:
        content = filepath.read_bytes()
        request = ParseRequest(
            content=content,
            filename=filepath.name,
            options=ParserOptions(
                enable_ocr=False,
                parse_timeout=30,
                table_strategy="none"
            )
        )
        parser = DoclingParser()
        result = parser.parse_sync(request)
        print(f"✅ Success | Pages: {result.metadata.page_count} | Chars: {result.metadata.char_count}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    test_dir = Path("demos/parser/test_files")
    files = list(test_dir.glob("*"))
    results = []
    for f in sorted(files):
        results.append(test_file(f))
    print("\nSummary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    if all(results):
        print("🎉 All allowed formats parsed successfully!")
    else:
        print("❌ Some formats failed.")

if __name__ == "__main__":
    main() 