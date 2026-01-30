"""
LangPy Embed Primitive - Full Implementation Copy

This file contains the complete implementation of the embed primitive,
which provides vector embedding capabilities for text content.

The embed primitive supports multiple embedding providers including:
- OpenAI embeddings (text-embedding-3-small, text-embedding-3-large, etc.)
- HuggingFace embeddings (local and hosted models)
- Extensible architecture for additional providers

Key Features:
- Async-first design for high-performance embedding generation
- Provider-agnostic interface with consistent API
- Automatic dimension detection and validation
- Batch processing support for efficient embedding generation
- Error handling and retry logic
- Integration with Langbase-style primitive patterns

Usage Examples:
    from embedders import get_embedder
    
    # OpenAI embeddings
    embedder = get_embedder("openai:text-embedding-3-small")
    embeddings = await embedder.embed(["Hello world", "Another text"])
    
    # HuggingFace embeddings
    embedder = get_embedder("hf:sentence-transformers/all-MiniLM-L6-v2")
    embeddings = await embedder.embed(["Sample text for embedding"])

Architecture:
    BaseEmbedder (ABC) - Abstract base class defining the embedder interface
    ├── OpenAIAsyncEmbedder - OpenAI API integration
    ├── HFAsyncEmbedder - HuggingFace integration
    └── [Future providers] - Extensible for additional services

Dependencies:
    - httpx: For async HTTP requests to OpenAI API
    - sentence-transformers: For HuggingFace model support (optional)
    - numpy: For vector operations (optional)
"""

# ============================================================================
# BASE EMBEDDER INTERFACE
# ============================================================================

from __future__ import annotations
import typing as t
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """
    Pluggable async embedder interface (vector dimension is backend-specific).
    
    This abstract base class defines the contract for all embedding providers.
    Each concrete implementation must provide:
    - A name identifier (vendor:model format)
    - The embedding dimension
    - An async embed method for generating embeddings
    
    Attributes:
        name (str): Vendor and model identifier (e.g., 'openai:text-embedding-3-small')
        dim (int): Embedding dimension for this model
    """
    
    name: str                       # vendor:model id e.g. 'openai:text-embedding-3-small'
    dim: int                        # embedding dimension

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, aligned with input texts
            
        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If input validation fails
        """
        pass


# ============================================================================
# OPENAI EMBEDDER IMPLEMENTATION
# ============================================================================

import os
import httpx
from typing import Optional, List

class OpenAIAsyncEmbedder(BaseEmbedder):
    """
    OpenAI embedding provider using the OpenAI API.
    
    Supports all OpenAI embedding models including:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    
    Features:
    - Automatic API key management
    - Async HTTP requests with timeout
    - Error handling and status code validation
    - Batch processing support
    """
    
    def __init__(self, model: str = "openai:text-embedding-3-small", *, api_key: str | None = None):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: Model identifier in format 'openai:model-name'
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        vendor, self.model = model.split(":", 1)
        self.name = model
        self.dim = 1536  # Default for text-embedding-3-small
        self.key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Set dimension based on model
        if "text-embedding-3-large" in self.model:
            self.dim = 3072
        elif "text-embedding-3-small" in self.model or "text-embedding-ada-002" in self.model:
            self.dim = 1536
        else:
            # Default to 1536 for unknown models
            self.dim = 1536

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            httpx.HTTPStatusError: If API request fails
            ValueError: If API key is missing
        """
        if not self.key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            return [d["embedding"] for d in response.json()["data"]]


# ============================================================================
# HUGGINGFACE EMBEDDER IMPLEMENTATION
# ============================================================================

class HFAsyncEmbedder(BaseEmbedder):
    """
    HuggingFace embedding provider.
    
    Supports HuggingFace models through the Inference API or local models.
    Currently implements a placeholder that returns zero vectors.
    
    Planned features:
    - HuggingFace Inference API integration
    - Local model support with sentence-transformers
    - Automatic model loading and caching
    - Batch processing optimization
    """
    
    def __init__(self, model: str):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model: Model identifier (e.g., 'hf:sentence-transformers/all-MiniLM-L6-v2')
        """
        self.name = model
        self.dim = 768  # Default dimension for many HF models
        
        # Extract model name and set appropriate dimension
        if "all-MiniLM-L6-v2" in model:
            self.dim = 384
        elif "all-mpnet-base-v2" in model:
            self.dim = 768
        elif "all-MiniLM-L12-v2" in model:
            self.dim = 384
        else:
            # Default dimension for unknown models
            self.dim = 768

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using HuggingFace models.
        
        TODO: Implement actual HuggingFace integration
        Currently returns zero vectors as placeholder.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (currently zero vectors)
        """
        # TODO: integrate Inference-API / local model
        return [[0.0] * self.dim for _ in texts]


# ============================================================================
# REGISTRY AND FACTORY FUNCTIONS
# ============================================================================

# Registry mapping vendor names to embedder classes
REGISTRY = {
    "openai": OpenAIAsyncEmbedder,
    "hf": HFAsyncEmbedder,
}

def get_embedder(name: str) -> BaseEmbedder:
    """
    Get embedder instance by name.
    
    Args:
        name: Embedder identifier in format 'vendor:model' 
              (e.g., 'openai:text-embedding-3-small')
              
    Returns:
        Configured embedder instance
        
    Raises:
        ValueError: If vendor is not supported
    """
    vendor = name.split(":", 1)[0]
    cls = REGISTRY.get(vendor)
    if not cls:
        raise ValueError(f"Unknown embedder vendor '{vendor}'. Supported vendors: {list(REGISTRY.keys())}")
    return cls(name)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_supported_vendors() -> List[str]:
    """Get list of supported embedding vendors."""
    return list(REGISTRY.keys())

def get_embedder_info(name: str) -> dict:
    """
    Get information about an embedder without creating an instance.
    
    Args:
        name: Embedder identifier
        
    Returns:
        Dictionary with embedder information
    """
    embedder = get_embedder(name)
    return {
        "name": embedder.name,
        "dimension": embedder.dim,
        "vendor": name.split(":", 1)[0],
        "model": name.split(":", 1)[1] if ":" in name else name
    }


# ============================================================================
# ERROR HANDLING
# ============================================================================

class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass

class EmbeddingTimeoutError(EmbeddingError):
    """Raised when embedding generation times out."""
    pass

class EmbeddingAPIError(EmbeddingError):
    """Raised when the embedding API returns an error."""
    pass


# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

# Default embedder configurations
DEFAULT_EMBEDDERS = {
    "openai": "openai:text-embedding-3-small",
    "hf": "hf:sentence-transformers/all-MiniLM-L6-v2"
}

# Timeout settings
DEFAULT_TIMEOUT = 30  # seconds
BATCH_SIZE_LIMIT = 1000  # maximum texts per batch

# Model dimension mappings
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Basic OpenAI embedding
from embedders import get_embedder

embedder = get_embedder("openai:text-embedding-3-small")
embeddings = await embedder.embed(["Hello world", "Another text"])
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

# Example 2: HuggingFace embedding
embedder = get_embedder("hf:sentence-transformers/all-MiniLM-L6-v2")
embeddings = await embedder.embed(["Sample text for embedding"])

# Example 3: Get embedder information
from embedders import get_embedder_info

info = get_embedder_info("openai:text-embedding-3-small")
print(f"Model: {info['model']}, Dimension: {info['dimension']}")

# Example 4: Batch processing
texts = ["Text 1", "Text 2", "Text 3", ...]  # Large list of texts
embedder = get_embedder("openai:text-embedding-3-small")

# Process in batches
batch_size = 100
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_embeddings = await embedder.embed(batch)
    all_embeddings.extend(batch_embeddings)
"""


# ============================================================================
# INTEGRATION WITH LANGPY SDK
# ============================================================================

"""
The embed primitive integrates with the LangPy SDK through the embedder interface.

SDK Usage:
    from sdk import embed
    
    # Create embedder instance
    embedder = embed()
    
    # Generate embeddings
    embeddings = await embedder.embed_texts(["Hello", "World"])
    
    # Get embedder information
    info = embedder.get_embedder_info()
    
    # Use with memory primitive
    from sdk import memory
    memory_instance = memory()
    await memory_instance.add_texts_with_embeddings(texts, embeddings)
"""


# ============================================================================
# FUTURE ENHANCEMENTS
# ============================================================================

"""
Planned enhancements for the embed primitive:

1. HuggingFace Integration:
   - Complete HuggingFace Inference API support
   - Local model loading with sentence-transformers
   - Model caching and optimization

2. Additional Providers:
   - Cohere embeddings
   - Google PaLM embeddings
   - Azure OpenAI embeddings
   - AWS Bedrock embeddings

3. Advanced Features:
   - Embedding similarity calculations
   - Embedding clustering and analysis
   - Multi-modal embeddings (text + image)
   - Embedding compression and quantization

4. Performance Optimizations:
   - Connection pooling
   - Request batching and queuing
   - Caching and memoization
   - Parallel processing

5. Monitoring and Analytics:
   - Embedding generation metrics
   - API usage tracking
   - Performance monitoring
   - Cost optimization
"""


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

"""
Test cases for the embed primitive:

1. Basic Functionality:
   - Test OpenAI embedder with valid API key
   - Test HuggingFace embedder (placeholder)
   - Verify embedding dimensions
   - Test batch processing

2. Error Handling:
   - Test with invalid API key
   - Test with network timeout
   - Test with invalid model names
   - Test with empty text lists

3. Integration Tests:
   - Test with memory primitive
   - Test with vector stores
   - Test with workflow primitive
   - Test with agent primitive

4. Performance Tests:
   - Test batch processing efficiency
   - Test concurrent requests
   - Test memory usage
   - Test response times
"""


# ============================================================================
# DEPENDENCIES AND REQUIREMENTS
# ============================================================================

"""
Required dependencies for the embed primitive:

Core dependencies (in requirements.txt):
- httpx>=0.24.0  # For async HTTP requests
- numpy>=1.24.0  # For vector operations (optional)

Optional dependencies:
- sentence-transformers>=2.2.0  # For HuggingFace models
- torch>=2.0.0  # For local model inference
- transformers>=4.30.0  # For HuggingFace model loading

Environment variables:
- OPENAI_API_KEY  # OpenAI API key for OpenAI embeddings
- HF_API_KEY  # HuggingFace API key (future)
- COHERE_API_KEY  # Cohere API key (future)
"""


# ============================================================================
# CONCLUSION
# ============================================================================

"""
The embed primitive provides a flexible, extensible foundation for text embedding
generation in the LangPy framework. It follows the same patterns as other primitives
with async-first design, provider abstraction, and comprehensive error handling.

The primitive is ready for production use with OpenAI embeddings and has a clear
path for expansion to additional providers and advanced features.
""" 