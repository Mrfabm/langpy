"""
Unit tests for Memory and Pipe integration with registry system.

Tests the new registry-based integration between Memory and Pipe primitives.
"""

import pytest
import tempfile
import os
from pathlib import Path

from sdk import Memory, Pipe
from sdk.registry import get, list_registered, clear


class TestRegistrySystem:
    """Test the global registry system."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear()
    
    def test_memory_auto_registration(self):
        """Test that Memory instances are automatically registered."""
        # Create memory instance
        mem = Memory(name="test_memory", backend="faiss")
        
        # Check it's registered
        assert get("test_memory") is not None
        assert get("test_memory") == mem._async_memory
        
        # Check registry listing
        registered = list_registered()
        assert "test_memory" in registered
        assert registered["test_memory"] == "AsyncMemory"
    
    def test_pipe_auto_registration(self):
        """Test that Pipe instances are automatically registered."""
        # Create pipe instance
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini")
        
        # Check it's registered
        assert get("test_pipe") is not None
        assert get("test_pipe") == pipe
        
        # Check registry listing
        registered = list_registered()
        assert "test_pipe" in registered
        assert registered["test_pipe"] == "Pipe"
    
    def test_multiple_memory_registration(self):
        """Test creating multiple memories with distinct names."""
        # Create two memories
        mem1 = Memory(name="memory1", backend="faiss")
        mem2 = Memory(name="memory2", backend="faiss")
        
        # Both should be registered
        assert get("memory1") == mem1._async_memory
        assert get("memory2") == mem2._async_memory
        
        # Registry should contain both
        registered = list_registered()
        assert "memory1" in registered
        assert "memory2" in registered
    
    def test_duplicate_name_error(self):
        """Test that duplicate names raise an error."""
        # Create first memory
        Memory(name="duplicate", backend="faiss")
        
        # Creating another with same name should fail
        with pytest.raises(ValueError, match="already registered"):
            Memory(name="duplicate", backend="faiss")


class TestMemoryPipeIntegration:
    """Test Memory and Pipe integration."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear()
    
    def test_pipe_with_memory_string_reference(self):
        """Test creating a Pipe with memory as string slug."""
        # Create memory first
        mem = Memory(name="test_memory", backend="faiss")
        
        # Create pipe with string reference
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini", memory="test_memory")
        
        # Memory should be resolved correctly
        assert pipe.memory is not None
        assert pipe.memory == mem._async_memory
    
    def test_pipe_with_memory_instance(self):
        """Test creating a Pipe with direct memory instance."""
        # Create memory
        mem = Memory(name="test_memory", backend="faiss")
        
        # Create pipe with direct instance
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini", memory=mem)
        
        # Memory should be accessible
        assert pipe.memory is not None
        assert pipe.memory == mem
    
    def test_pipe_with_invalid_memory_slug(self):
        """Test that invalid memory slug raises error."""
        # Create pipe with non-existent memory
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini", memory="nonexistent")
        
        # Accessing memory should raise error
        with pytest.raises(ValueError, match="not found in registry"):
            _ = pipe.memory
    
    def test_pipe_without_memory(self):
        """Test pipe without memory integration."""
        # Create pipe without memory
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini")
        
        # Memory should be None
        assert pipe.memory is None
    
    def test_lazy_memory_resolution(self):
        """Test that memory is resolved lazily on first access."""
        # Create pipe with memory slug before memory exists
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini", memory="future_memory")
        
        # Memory should not be resolved yet
        assert pipe._memory is None
        
        # Create the memory
        mem = Memory(name="future_memory", backend="faiss")
        
        # Now memory should resolve
        assert pipe.memory == mem._async_memory


class TestMemoryOperations:
    """Test Memory class operations."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear()
    
    def test_memory_creation_faiss(self):
        """Test creating Memory with FAISS backend."""
        mem = Memory(
            name="test_faiss",
            backend="faiss",
            chunk_max_length=300,
            chunk_overlap=50
        )
        
        assert mem.settings.name == "test_faiss"
        assert mem.settings.store_backend == "faiss"
        assert mem.settings.chunk_max_length == 300
        assert mem.settings.chunk_overlap == 50
    
    def test_memory_creation_pgvector(self):
        """Test creating Memory with pgvector backend."""
        dsn = "postgresql://user:pass@localhost:5432/test"
        
        mem = Memory(
            name="test_pgvector",
            backend="pgvector",
            dsn=dsn,
            chunk_max_length=300,
            chunk_overlap=50
        )
        
        assert mem.settings.name == "test_pgvector"
        assert mem.settings.store_backend == "pgvector"
        assert mem.settings.store_uri == dsn
    
    def test_memory_pgvector_without_dsn(self):
        """Test that pgvector without DSN raises error."""
        with pytest.raises(ValueError, match="PostgreSQL DSN required"):
            Memory(name="test", backend="pgvector")
    
    def test_memory_upload_method_exists(self):
        """Test that Memory has upload method."""
        mem = Memory(name="test", backend="faiss")
        assert hasattr(mem, 'upload')
        assert callable(mem.upload)
    
    def test_memory_query_method_exists(self):
        """Test that Memory has query method."""
        mem = Memory(name="test", backend="faiss")
        assert hasattr(mem, 'query')
        assert callable(mem.query)


class TestPipeOperations:
    """Test Pipe class operations."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear()
    
    def test_pipe_creation(self):
        """Test creating Pipe instance."""
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini")
        
        assert pipe.name == "test_pipe"
        assert pipe.model == "gpt-4o-mini"
        assert pipe.settings.name == "test_pipe"
    
    def test_pipe_run_method_exists(self):
        """Test that Pipe has run method."""
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini")
        assert hasattr(pipe, 'run')
        assert callable(pipe.run)
    
    def test_pipe_run_requires_api_key(self):
        """Test that Pipe.run requires API key."""
        pipe = Pipe(name="test_pipe", model="gpt-4o-mini")
        
        # Temporarily unset API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if original_key:
            del os.environ["OPENAI_API_KEY"]
        
        try:
            # Should raise error without API key
            with pytest.raises(ValueError, match="API key required"):
                pipe.run("test input")
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear()
    
    def test_legacy_memory_interface_still_works(self):
        """Test that legacy MemoryInterface still works."""
        from sdk import MemoryInterface
        
        # Just test that it can be imported
        assert MemoryInterface is not None
        assert hasattr(MemoryInterface, '__init__')
        
        # Test that the class exists and has expected methods
        assert hasattr(MemoryInterface, 'upload')
        assert hasattr(MemoryInterface, 'query')
    
    def test_legacy_pipe_interface_still_works(self):
        """Test that legacy PipeInterface still works."""
        from sdk import PipeInterface
        
        # Should still be importable and instantiable
        interface = PipeInterface()
        assert interface is not None
    
    def test_factory_functions_still_work(self):
        """Test that factory functions still work."""
        from sdk import memory, pipe
        
        # Should still be callable
        mem_factory = memory()
        pipe_factory = pipe()
        
        assert mem_factory is not None
        assert pipe_factory is not None


if __name__ == "__main__":
    pytest.main([__file__]) 