#!/usr/bin/env python3

# Minimal test for the chunker demo issue
from chunker import SyncChunker

print("Testing minimal demo...")

try:
    chunker1 = SyncChunker(chunk_max_length=1500, chunk_overlap=300)
    print("✓ SyncChunker created successfully")
    
    text = "This is a sample text that will be chunked into smaller pieces. " * 10
    chunks = chunker1.chunk(text)
    print(f"✓ Chunking successful: {len(chunks)} chunks created")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.") 