from __future__ import annotations
import typing as t, abc, asyncio
from typing import List, Dict, Any, Optional

class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores."""
    
    @abc.abstractmethod
    async def add(self, texts: List[str], metas: List[Dict[str, Any]]) -> None:
        """Add texts with metadata to store."""
        
    @abc.abstractmethod
    async def query(self, query: str, k: int, filt: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query store for similar content."""
        
    @abc.abstractmethod
    async def token_usage(self) -> int:
        """Get total token count."""
    
    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all stored data."""
    
    @abc.abstractmethod
    async def delete_by_filter(self, filt: Dict[str, Any]) -> int:
        """Delete documents matching filter criteria."""
    
    @abc.abstractmethod
    async def update_metadata(self, filt: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """Update metadata for documents matching filter criteria."""
    
    @abc.abstractmethod
    async def get_metadata_stats(self) -> Dict[str, Any]:
        """Get statistics about stored metadata.""" 
    
    async def get_all_texts(self) -> List[str]:
        """Get all stored texts (for hybrid search)."""
        # Default implementation - override in subclasses
        return []
    
    async def get_texts_by_filter(self, filt: Dict[str, Any]) -> List[str]:
        """Get texts matching filter criteria (for hybrid search)."""
        # Default implementation - override in subclasses
        return [] 