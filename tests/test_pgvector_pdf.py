import asyncio
import pytest
from sdk.memory_interface import MemoryInterface
from memory.models import MemorySettings
from dotenv import load_dotenv
import os

load_dotenv()

@pytest.mark.asyncio
async def test_pgvector_with_pdf():
    """Test pgvector memory primitive with the specific PDF file."""
    print("🐘 Testing pgvector Memory with PDF File")
    print("=" * 50)
    
    # Use the specific PDF file path
    pdf_path = r"C:\Users\USER\Desktop\AGENTS\Important\Dump Here\Amakoro Lodge Fact Sheet.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"📄 Using PDF file: {pdf_path}")
    
    # Create memory interface with pgvector backend
    settings = MemorySettings(
        name="amakoro_lodge_pgvector",
        store_backend="pgvector",
        store_uri="postgresql://postgres:password@localhost:5432/memory_db",
        embed_model="openai:text-embedding-3-large",
        chunk_max_length=8000,
        chunk_overlap=256
    )
    
    mem_interface = MemoryInterface(settings=settings)
    print("✅ MemoryInterface created with pgvector backend")
    
    # Upload the PDF file
    print("\n📝 Uploading PDF to pgvector...")
    try:
        job_id = await mem_interface.upload(
            content=pdf_path,
            source="Amakoro Lodge Fact Sheet"
        )
        print(f"✅ Upload job created: {job_id}")
        
        # Wait for completion
        print("⏳ Waiting for processing...")
        for i in range(30):  # Longer timeout for PDF processing
            try:
                job = await mem_interface.get_job_status(job_id)
                if isinstance(job, dict):
                    status = job.get('status', 'unknown')
                else:
                    status = job.status.value if hasattr(job, 'status') else 'unknown'
                
                print(f"📊 Job status: {status}")
                
                if status in ['completed', 'failed']:
                    break
            except Exception as e:
                print(f"⚠️ Error checking job status: {e}")
                break
            await asyncio.sleep(2)  # Check every 2 seconds
        
        # Query the content
        print("\n🔍 Querying pgvector database...")
        queries = [
            "Amakoro Lodge accommodation",
            "safari lodge features",
            "lodge amenities",
            "wildlife viewing",
            "African cuisine"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = await mem_interface.query(query, k=3)
            if not results:
                print("❌ No results found.")
                continue
            print(f"✅ Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                print(f"  --- Result {i} ---")
                print(f"  Score: {result.get('score', 0):.3f}")
                print(f"  Text: {result.get('text', '')[:150]}...")
                metadata = result.get('metadata', {})
                if metadata:
                    print(f"  Source: {metadata.get('source', 'unknown')}")
                    print(f"  Metadata: {metadata}")
        
        # Get statistics
        try:
            stats = await mem_interface.get_stats()
            print(f"\n📊 Database Statistics:")
            print(f"Total documents: {stats.get('total_documents', 0)}")
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            print(f"Storage backend: {stats.get('storage_backend', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Could not get stats: {e}")
        
        print("\n🎉 pgvector PDF Test Completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pgvector_with_pdf()) 