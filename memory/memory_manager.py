"""
Memory Manager - Langbase-style container management for memory primitives.

Provides a unified interface for creating, managing, and accessing multiple
memory instances with persistent registry and namespaced storage.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import re

from .async_memory import AsyncMemory
from .models import MemorySettings


class MemoryManager:
    """
    Langbase-style container manager for memory primitives.
    
    Maintains a registry of memory instances with persistent storage
    and namespaced vector storage backends.
    """
    
    def __init__(self, default_settings: Optional[MemorySettings] = None):
        """
        Initialize the memory manager.
        
        Args:
            default_settings: Default settings for new memory instances
        """
        self.default_settings = default_settings or MemorySettings()
        self._registry: Dict[str, AsyncMemory] = {}
        self._lock = asyncio.Lock()
        self._registry_file = Path.home() / ".langpy_memories.json"
        
        # Load existing registry from disk
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load memory registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                    # Registry will be lazy-loaded when accessed
                    self._registry = {slug: None for slug in data.get('memories', [])}
            except (json.JSONDecodeError, IOError):
                self._registry = {}
        else:
            self._registry = {}
    
    def _save_registry(self) -> None:
        """Save memory registry to disk."""
        try:
            data = {
                'memories': list(self._registry.keys()),
                'updated_at': datetime.now().isoformat()
            }
            with open(self._registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            # Silently fail if we can't save registry
            pass
    
    def _generate_slug(self, name: str) -> str:
        """
        Generate a unique slug from a name.
        
        Args:
            name: Human-readable name
            
        Returns:
            URL-safe slug
        """
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', name.lower())
        slug = re.sub(r'\s+', '-', slug.strip())
        
        # Ensure uniqueness
        base_slug = slug
        counter = 1
        while slug in self._registry:
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug
    
    def _build_store_uri(self, slug: str, settings: MemorySettings) -> str:
        """
        Build namespaced store URI for the memory instance.
        
        Args:
            slug: Memory slug
            settings: Memory settings
            
        Returns:
            Namespaced store URI
        """
        if settings.store_backend == "pgvector":
            # Add table parameter for pgvector
            base_uri = settings.store_uri or "postgresql://localhost/langpy"
            separator = "&" if "?" in base_uri else "?"
            return f"{base_uri}{separator}table=mem_{slug}"
        elif settings.store_backend == "faiss":
            # Use namespaced directory for FAISS
            base_uri = settings.store_uri or "./memory_index.faiss"
            if base_uri.endswith('.faiss'):
                # Replace filename with namespaced version
                base_path = Path(base_uri)
                return str(base_path.parent / f"mem_{slug}.faiss")
            else:
                # Use directory structure
                return f"./vector_indexes/mem_{slug}.faiss"
        else:
            # For other backends, append slug to URI
            base_uri = settings.store_uri or f"./memory_{slug}"
            return f"{base_uri}_{slug}"
    
    async def create(self, name: str, **override_settings) -> str:
        """
        Create a new memory instance.
        
        Args:
            name: Human-readable name for the memory
            **override_settings: Settings to override defaults
            
        Returns:
            Memory slug for future access
        """
        async with self._lock:
            # Generate unique slug
            slug = self._generate_slug(name)
            
            # Merge settings
            settings = self.default_settings.model_copy(update=override_settings)
            settings.name = name
            
            # Build namespaced store URI
            settings.store_uri = self._build_store_uri(slug, settings)
            
            # Create memory instance
            memory = AsyncMemory(settings)
            
            # Store in registry
            self._registry[slug] = memory
            
            # Persist registry
            self._save_registry()
            
            return slug
    
    async def update(self, slug: str, **override_settings) -> None:
        """
        Update settings for an existing memory instance.
        
        Args:
            slug: Memory slug
            **override_settings: Settings to update
        """
        async with self._lock:
            if slug not in self._registry:
                raise ValueError(f"Memory '{slug}' not found")
            
            memory = self._registry[slug]
            if memory is None:
                # Lazy load if needed
                memory = await self._load_memory(slug)
                self._registry[slug] = memory
            
            # Update settings (shallow merge)
            for key, value in override_settings.items():
                if hasattr(memory.settings, key):
                    setattr(memory.settings, key, value)
    
    async def delete(self, slug: str, purge_vectors: bool = True) -> None:
        """
        Delete a memory instance.
        
        Args:
            slug: Memory slug
            purge_vectors: Whether to clear vector data
        """
        async with self._lock:
            if slug not in self._registry:
                raise ValueError(f"Memory '{slug}' not found")
            
            memory = self._registry[slug]
            if memory is not None:
                if purge_vectors:
                    await memory.clear()
            
            # Remove from registry
            del self._registry[slug]
            
            # Persist registry
            self._save_registry()
    
    async def get(self, slug: str) -> AsyncMemory:
        """
        Retrieve a memory instance.
        
        Args:
            slug: Memory slug
            
        Returns:
            AsyncMemory instance
        """
        async with self._lock:
            if slug not in self._registry:
                raise ValueError(f"Memory '{slug}' not found")
            
            memory = self._registry[slug]
            if memory is None:
                # Lazy load if needed
                memory = await self._load_memory(slug)
                self._registry[slug] = memory
            
            return memory
    
    async def list(self) -> List[Dict[str, Any]]:
        """
        List all memory instances with metadata.
        
        Returns:
            List of memory metadata dictionaries
        """
        async with self._lock:
            results = []
            
            for slug in self._registry.keys():
                try:
                    memory = await self.get(slug)
                    stats = await memory.get_stats()
                    
                    results.append({
                        'slug': slug,
                        'name': memory.settings.name,
                        'backend': memory.settings.store_backend,
                        'total_docs': stats.total_documents,
                        'total_chunks': stats.total_chunks,
                        'total_tokens': stats.total_tokens,
                        'created_at': memory.settings.name,  # Use name as proxy for now
                        'updated_at': datetime.now().isoformat()
                    })
                except Exception as e:
                    # Skip memories that can't be loaded
                    results.append({
                        'slug': slug,
                        'name': 'Unknown',
                        'backend': 'Unknown',
                        'total_docs': 0,
                        'total_chunks': 0,
                        'total_tokens': 0,
                        'error': str(e)
                    })
            
            return results
    
    async def _load_memory(self, slug: str) -> AsyncMemory:
        """
        Load a memory instance from disk.
        
        Args:
            slug: Memory slug
            
        Returns:
            AsyncMemory instance
        """
        # This would need to be implemented based on how we want to
        # reconstruct memory instances from disk. For now, we'll
        # create a new instance with default settings.
        settings = self.default_settings.model_copy()
        settings.name = slug
        settings.store_uri = self._build_store_uri(slug, settings)
        
        return AsyncMemory(settings) 

    # Langbase-style method aliases
    async def retrieve(self, slug: str, *args, **kwargs):
        """Langbase alias for query on a memory instance."""
        memory = await self.get(slug)
        return await memory.query(*args, **kwargs)

    async def stats(self, slug: str, *args, **kwargs):
        """Langbase alias for get_stats on a memory instance."""
        memory = await self.get(slug)
        return await memory.get_stats(*args, **kwargs)

    async def info(self, slug: str, *args, **kwargs):
        """Langbase alias for get_stats (info) on a memory instance."""
        memory = await self.get(slug)
        return await memory.get_stats(*args, **kwargs)

    async def list_memories(self, *args, **kwargs):
        """Langbase alias for list."""
        return await self.list(*args, **kwargs) 