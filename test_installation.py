"""
Test script to verify LangPy installation works.
"""
import asyncio
import sys

print("=" * 70)
print("TESTING LANGPY INSTALLATION")
print("=" * 70)
print()

# Test 1: Import core modules
print("[TEST 1] Importing core modules...")
try:
    from langpy import Langpy
    from langpy.core import Context
    print("  [OK] Core modules imported successfully")
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[TEST 2] Checking dependencies...")
dependencies = {
    'pydantic': None,
    'openai': None,
    'httpx': None,
    'aiohttp': None,
    'faiss': None,
    'numpy': None,
    'aiofiles': None
}

for dep, actual_name in dependencies.items():
    try:
        __import__(actual_name or dep)
        print(f"  [OK] {dep} is installed")
    except ImportError as e:
        print(f"  [FAIL] {dep} is missing: {e}")
        sys.exit(1)

# Test 3: Create Langpy client
print("\n[TEST 3] Creating Langpy client...")
try:
    lb = Langpy(api_key="test-key")
    print("  [OK] Langpy client created")
    print(f"  [OK] Client has agent: {hasattr(lb, 'agent')}")
    print(f"  [OK] Client has memory: {hasattr(lb, 'memory')}")
    print(f"  [OK] Client has pipe: {hasattr(lb, 'pipe')}")
    print(f"  [OK] Client has thread: {hasattr(lb, 'thread')}")
    print(f"  [OK] Client has workflow: {hasattr(lb, 'workflow')}")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create Context
print("\n[TEST 4] Creating Context...")
try:
    ctx = Context(query="test query")
    print(f"  [OK] Context created with query: {ctx.query}")
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# Test 5: Test FAISS vector store
print("\n[TEST 5] Testing FAISS vector store...")
try:
    import faiss
    import numpy as np

    # Create simple index
    dimension = 128
    index = faiss.IndexFlatL2(dimension)

    # Add some vectors
    vectors = np.random.random((10, dimension)).astype('float32')
    index.add(vectors)

    # Search
    query = np.random.random((1, dimension)).astype('float32')
    distances, indices = index.search(query, 3)

    print(f"  [OK] FAISS working: {index.ntotal} vectors indexed")
    print(f"  [OK] Search returned {len(indices[0])} results")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("Installation verified successfully!")
print("  [OK] All core modules working")
print("  [OK] All dependencies installed")
print("  [OK] Langpy client functional")
print("  [OK] Context creation working")
print("  [OK] FAISS vector storage working")
print()
print("You can now use LangPy!")
print("=" * 70)
