import asyncio
import subprocess
import pytest
from sdk.memory_interface import MemoryInterface
from memory.models import MemorySettings

@pytest.mark.asyncio
async def test_pgvector_status():
    """Test the current status of pgvector implementation."""
    print("🧪 Testing pgvector Status")
    print("=" * 40)
    
    # Check if PostgreSQL is running
    print("1. Checking PostgreSQL container...")
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=postgres-pgvector'], 
                              capture_output=True, text=True)
        if 'postgres-pgvector' in result.stdout:
            print("✅ PostgreSQL container is running")
        else:
            print("❌ PostgreSQL container is not running")
            return
    except Exception as e:
        print(f"❌ Error checking container: {e}")
        return
    
    # Check database connection
    print("\n2. Testing database connection...")
    try:
        result = subprocess.run([
            'docker', 'exec', 'postgres-pgvector',
            'psql', '-U', 'postgres', '-d', 'memory_db', '-c', 'SELECT version();'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database connection successful")
        else:
            print(f"❌ Database connection failed: {result.stderr}")
            return
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return
    
    # Check vector extension
    print("\n3. Checking vector extension...")
    try:
        result = subprocess.run([
            'docker', 'exec', 'postgres-pgvector',
            'psql', '-U', 'postgres', '-d', 'memory_db', '-c', 'SELECT * FROM pg_extension WHERE extname = \'vector\';'
        ], capture_output=True, text=True)
        
        if 'vector' in result.stdout:
            print("✅ Vector extension is installed")
        else:
            print("❌ Vector extension is not installed")
            return
    except Exception as e:
        print(f"❌ Error checking vector extension: {e}")
        return
    
    # Test memory interface creation
    print("\n4. Testing memory interface creation...")
    try:
        settings = MemorySettings(
            name="test_memory",
            store_backend="pgvector",
            store_uri="postgresql://postgres:password@localhost:5432/memory_db",
            embed_model="openai:text-embedding-3-large"
        )
        
        mem_interface = MemoryInterface(settings=settings)
        print("✅ Memory interface created successfully")
    except Exception as e:
        print(f"❌ Error creating memory interface: {e}")
        return
    
    # Test embedding generation
    print("\n5. Testing embedding generation...")
    try:
        from embed import get_embedder
        embedder = get_embedder("openai:text-embedding-3-large")
        test_embedding = await embedder.embed(["test"])
        dimension = len(test_embedding[0])
        print(f"✅ Embedding generated successfully: {dimension} dimensions")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return
    
    print("\n🎉 All basic tests passed!")
    print("The pgvector setup appears to be working correctly.")

if __name__ == "__main__":
    asyncio.run(test_pgvector_status()) 