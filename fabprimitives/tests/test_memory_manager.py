"""
Unit tests for MemoryManager - Langbase-style container management.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import os

from memory.memory_manager import MemoryManager
from memory import MemorySettings


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    @pytest.fixture
    def temp_registry_file(self, tmp_path):
        """Create a temporary registry file for testing."""
        reg_file = tmp_path / ".langpy_memories.json"
        with open(reg_file, 'w') as f:
            json.dump({'memories': ['test-memory-1', 'test-memory-2']}, f)
        yield str(reg_file)
        if reg_file.exists():
            reg_file.unlink()
    
    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a MemoryManager instance for testing."""
        with patch('memory.memory_manager.Path.home', return_value=tmp_path):
            default_settings = MemorySettings(
                store_backend="faiss",
                chunk_max_length=5000,
                embed_model="openai:text-embedding-3-small"
            )
            return MemoryManager(default_settings=default_settings)
    
    @pytest.mark.asyncio
    async def test_create_memory(self, memory_manager):
        """Test creating a new memory instance."""
        # Test basic creation
        slug = await memory_manager.create("My Test Memory")
        assert slug == "my-test-memory"
        assert slug in memory_manager._registry
        
        # Test creation with overrides
        slug2 = await memory_manager.create(
            "Another Memory",
            chunk_max_length=10000,
            embed_model="openai:text-embedding-3-large"
        )
        assert slug2 == "another-memory"
        
        # Verify memory instance was created
        memory = memory_manager._registry[slug2]
        assert memory is not None
        assert memory.settings.chunk_max_length == 10000
        assert memory.settings.embed_model == "openai:text-embedding-3-large"
    
    @pytest.mark.asyncio
    async def test_create_memory_duplicate_names(self, memory_manager):
        """Test creating memories with duplicate names generates unique slugs."""
        slug1 = await memory_manager.create("Test Memory")
        slug2 = await memory_manager.create("Test Memory")
        slug3 = await memory_manager.create("Test Memory")
        
        assert slug1 == "test-memory"
        assert slug2 == "test-memory-1"
        assert slug3 == "test-memory-2"
    
    @pytest.mark.asyncio
    async def test_update_memory_settings(self, memory_manager):
        """Test updating memory settings."""
        # Create a memory
        slug = await memory_manager.create("Test Memory")
        
        # Update settings
        await memory_manager.update(slug, chunk_max_length=15000)
        
        # Verify update
        memory = memory_manager._registry[slug]
        assert memory.settings.chunk_max_length == 15000
        
        # Update multiple settings
        await memory_manager.update(
            slug,
            chunk_overlap=512,
            embed_model="openai:text-embedding-3-large"
        )
        
        assert memory.settings.chunk_overlap == 512
        assert memory.settings.embed_model == "openai:text-embedding-3-large"
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_memory(self, memory_manager):
        """Test updating a memory that doesn't exist."""
        with pytest.raises(ValueError, match="Memory 'nonexistent' not found"):
            await memory_manager.update("nonexistent", chunk_max_length=10000)
    
    @pytest.mark.asyncio
    async def test_delete_memory_purge_vectors(self, memory_manager):
        """Test deleting memory with vector purging."""
        # Create a memory
        slug = await memory_manager.create("Test Memory")
        assert slug in memory_manager._registry
        
        # Mock the clear method
        memory = memory_manager._registry[slug]
        memory.clear = AsyncMock()
        
        # Delete with purge
        await memory_manager.delete(slug, purge_vectors=True)
        
        # Verify clear was called and memory removed
        memory.clear.assert_awaited_once()
        assert slug not in memory_manager._registry
    
    @pytest.mark.asyncio
    async def test_delete_memory_no_purge(self, memory_manager):
        """Test deleting memory without vector purging."""
        # Create a memory
        slug = await memory_manager.create("Test Memory")
        assert slug in memory_manager._registry
        
        # Mock the clear method
        memory = memory_manager._registry[slug]
        memory.clear = AsyncMock()
        
        # Delete without purge
        await memory_manager.delete(slug, purge_vectors=False)
        
        # Verify clear was not called but memory removed
        memory.clear.assert_not_awaited()
        assert slug not in memory_manager._registry
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, memory_manager):
        """Test deleting a memory that doesn't exist."""
        with pytest.raises(ValueError, match="Memory 'nonexistent' not found"):
            await memory_manager.delete("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_memory(self, memory_manager):
        """Test retrieving a memory instance."""
        # Create a memory
        slug = await memory_manager.create("Test Memory")
        
        # Get the memory
        memory = await memory_manager.get(slug)
        
        # Verify it's the same instance
        assert memory is memory_manager._registry[slug]
        assert memory.settings.name == "Test Memory"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, memory_manager):
        """Test retrieving a memory that doesn't exist."""
        with pytest.raises(ValueError, match="Memory 'nonexistent' not found"):
            await memory_manager.get("nonexistent")
    
    @pytest.mark.asyncio
    async def test_list_memories(self, memory_manager):
        """Test listing all memories with metadata."""
        # Create multiple memories
        slug1 = await memory_manager.create("Memory 1")
        slug2 = await memory_manager.create("Memory 2")
        
        # Mock get_stats to return predictable data
        mock_stats1 = MagicMock()
        mock_stats1.total_documents = 10
        mock_stats1.total_chunks = 50
        mock_stats1.total_tokens = 1000
        
        mock_stats2 = MagicMock()
        mock_stats2.total_documents = 5
        mock_stats2.total_chunks = 25
        mock_stats2.total_tokens = 500
        
        memory1 = memory_manager._registry[slug1]
        memory2 = memory_manager._registry[slug2]
        memory1.get_stats = AsyncMock(return_value=mock_stats1)
        memory2.get_stats = AsyncMock(return_value=mock_stats2)
        
        # List memories
        memories = await memory_manager.list()
        
        # Verify results
        assert len(memories) == 2
        
        # Find each memory in results
        mem1_data = next(m for m in memories if m['slug'] == slug1)
        mem2_data = next(m for m in memories if m['slug'] == slug2)
        
        assert mem1_data['name'] == "Memory 1"
        assert mem1_data['backend'] == "faiss"
        assert mem1_data['total_docs'] == 10
        assert mem1_data['total_chunks'] == 50
        assert mem1_data['total_tokens'] == 1000
        
        assert mem2_data['name'] == "Memory 2"
        assert mem2_data['backend'] == "faiss"
        assert mem2_data['total_docs'] == 5
        assert mem2_data['total_chunks'] == 25
        assert mem2_data['total_tokens'] == 500
    
    def test_generate_slug(self, memory_manager):
        """Test slug generation from names."""
        # Test basic slug generation
        assert memory_manager._generate_slug("My Test Memory") == "my-test-memory"
        assert memory_manager._generate_slug("Another-Memory") == "another-memory"
        assert memory_manager._generate_slug("Memory with Special Chars!@#") == "memory-with-special-chars"
        
        # Test uniqueness
        memory_manager._registry = {"my-test-memory": None}
        assert memory_manager._generate_slug("My Test Memory") == "my-test-memory-1"
    
    def test_build_store_uri(self, memory_manager):
        """Test building namespaced store URIs."""
        settings = MemorySettings(store_backend="faiss")
        
        # Test FAISS backend
        uri = memory_manager._build_store_uri("test-memory", settings)
        assert uri == "./vector_indexes/mem_test-memory.faiss"
        
        # Test FAISS with custom path
        settings.store_uri = "./custom/path.faiss"
        uri = memory_manager._build_store_uri("test-memory", settings)
        assert uri == "./custom/mem_test-memory.faiss"
        
        # Test pgvector backend
        settings.store_backend = "pgvector"
        settings.store_uri = "postgresql://localhost/langpy"
        uri = memory_manager._build_store_uri("test-memory", settings)
        assert uri == "postgresql://localhost/langpy?table=mem_test-memory"
        
        # Test pgvector with existing query params
        settings.store_uri = "postgresql://localhost/langpy?user=test"
        uri = memory_manager._build_store_uri("test-memory", settings)
        assert uri == "postgresql://localhost/langpy?user=test&table=mem_test-memory"
    
    @pytest.mark.asyncio
    async def test_registry_persistence(self, memory_manager):
        """Test that registry is persisted to disk."""
        # Create memories
        await memory_manager.create("Memory 1")
        await memory_manager.create("Memory 2")
        
        # Verify registry file was created
        assert memory_manager._registry_file.exists()
        
        # Read registry file
        with open(memory_manager._registry_file, 'r') as f:
            data = json.load(f)
        
        # Verify content
        assert 'memories' in data
        assert 'memory-1' in data['memories']
        assert 'memory-2' in data['memories']
        assert 'updated_at' in data
    
    @pytest.mark.asyncio
    async def test_load_existing_registry(self, temp_registry_file):
        """Test loading existing registry from disk."""
        with patch('langpy.memory_manager.Path.home') as mock_home:
            mock_home.return_value = Path(temp_registry_file).parent
            
            # Create manager (should load existing registry)
            manager = MemoryManager()
            
            # Verify registry was loaded
            assert 'test-memory-1' in manager._registry
            assert 'test-memory-2' in manager._registry
            assert manager._registry['test-memory-1'] is None  # Lazy loaded
            assert manager._registry['test-memory-2'] is None  # Lazy loaded 