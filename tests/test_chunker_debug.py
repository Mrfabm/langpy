#!/usr/bin/env python3

# Debug script for chunker
from chunker.settings import ChunkerSettings
from chunker.base import _ChunkerCore
from chunker.sync_chunker import SyncChunker

print("Testing step by step...")

# Step 1: Test settings
print("1. Testing ChunkerSettings...")
try:
    cfg = ChunkerSettings(chunk_max_length=1500, chunk_overlap=300)
    print("   ✓ ChunkerSettings created successfully")
except Exception as e:
    print(f"   ✗ ChunkerSettings failed: {e}")

# Step 2: Test base class
print("2. Testing _ChunkerCore...")
try:
    cfg = ChunkerSettings(chunk_max_length=1500, chunk_overlap=300)
    base = _ChunkerCore(cfg)
    print("   ✓ _ChunkerCore created successfully")
except Exception as e:
    print(f"   ✗ _ChunkerCore failed: {e}")

# Step 3: Test SyncChunker
print("3. Testing SyncChunker...")
try:
    sync = SyncChunker(chunk_max_length=1500, chunk_overlap=300)
    print("   ✓ SyncChunker created successfully")
except Exception as e:
    print(f"   ✗ SyncChunker failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug complete.") 