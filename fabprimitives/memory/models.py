"""
Memory Models - Core data structures for the memory orchestrator primitive.

Defines the models and settings for the memory primitive that coordinates
parser, chunker, embed, and store operations.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from enum import Enum
from pathlib import Path


class DocumentStatus(str, Enum):
    """Status of document processing."""
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"

class MemoryTier(str, Enum):
    """Tier/priority for memory chunks and documents."""
    GENERAL = "general"
    IMPORTANT = "important"
    CRITICAL = "critical"


class FilterExpression(BaseModel):
    """Individual filter expression for memory queries (Langbase-style)."""
    field: str = Field(..., description="Field name to filter on")
    operator: Literal["eq", "neq", "in", "nin", "gt", "gte", "lt", "lte", "contains"] = Field(
        ..., description="Comparison operator"
    )
    value: Any = Field(..., description="Value to compare against")


class CompoundFilter(BaseModel):
    """Compound filter with AND/OR logic (Langbase-style)."""
    and_conditions: Optional[List[Union[FilterExpression, "CompoundFilter"]]] = Field(
        None, alias="and", description="AND conditions"
    )
    or_conditions: Optional[List[Union[FilterExpression, "CompoundFilter"]]] = Field(
        None, alias="or", description="OR conditions"
    )
    
    @model_validator(mode='after')
    def validate_compound_filter(self):
        """Ensure either 'and' or 'or' is specified, but not both."""
        if self.and_conditions is None and self.or_conditions is None:
            raise ValueError("CompoundFilter must specify either 'and' or 'or'")
        if self.and_conditions is not None and self.or_conditions is not None:
            raise ValueError("CompoundFilter cannot specify both 'and' and 'or'")
        return self


# Update the type alias for backward compatibility
FilterType = Union[Dict[str, Any], FilterExpression, CompoundFilter]


class MemorySettings(BaseModel):
    """Configuration settings for memory operations."""
    
    # Storage backend configuration
    store_backend: Literal["faiss", "pgvector", "docling"] = Field(
        "faiss", 
        description="Vector store backend: 'faiss', 'pgvector', or 'docling'"
    )
    store_uri: Optional[str] = Field(
        None, 
        description="URI for vector store (file path for FAISS, DSN for pgvector)"
    )
    
    # Parser configuration
    parser_enable_ocr: bool = Field(True, description="Enable OCR for images")
    parser_ocr_languages: List[str] = Field(
        ["eng"], 
        description="OCR languages to use"
    )
    parser_max_file_size: int = Field(
        50 * 1024 * 1024,  # 50MB
        description="Maximum file size for parsing"
    )
    
    # Chunker configuration
    chunk_max_length: int = Field(
        10000, 
        description="Maximum length of each chunk"
    )
    chunk_overlap: int = Field(
        256, 
        description="Overlap between consecutive chunks"
    )
    
    # Embedding configuration
    embed_model: str = Field(
        "openai:text-embedding-3-large",
        description="Embedding model to use"
    )
    
    # Reranking configuration
    enable_reranking: bool = Field(
        True, 
        description="Enable cross-encoder reranking for better results"
    )
    reranker_model: str = Field(
        "BAAI/bge-reranker-large",
        description="Cross-encoder model for reranking"
    )
    rerank_top_k: int = Field(
        20, 
        description="Number of candidates to rerank"
    )
    
    # Hybrid search configuration
    enable_hybrid_search: bool = Field(
        True, 
        description="Enable hybrid search (ANN + BM25)"
    )
    hybrid_weight: float = Field(
        0.7, 
        description="Weight for vector similarity vs keyword matching (0-1)"
    )
    
    # Search configuration
    default_k: int = Field(5, description="Number of results to return by default")
    similarity_threshold: float = Field(0.7, description="Similarity threshold for search")
    
    # Memory name for identification
    name: str = Field("default", description="Memory name for identification")
    
    # Metadata configuration
    include_source: bool = Field(True, description="Include source in metadata")
    include_timestamp: bool = Field(True, description="Include timestamp in metadata")
    include_tokens: bool = Field(True, description="Include token count in metadata")
    
    @model_validator(mode='after')
    def validate_settings(self):
        """Validate configuration settings."""
        # Validate chunk settings
        if self.chunk_max_length < 100:  # Reduced minimum to allow smaller chunks
            raise ValueError("chunk_max_length must be at least 100")
        if self.chunk_overlap >= self.chunk_max_length:
            raise ValueError("chunk_overlap must be less than chunk_max_length")
        
        # Validate hybrid weight
        if not 0 <= self.hybrid_weight <= 1:
            raise ValueError("hybrid_weight must be between 0 and 1")
        
        # Validate store URI for specific backends
        if self.store_backend == "pgvector" and not self.store_uri:
            raise ValueError("store_uri is required for pgvector backend")
        if self.store_backend == "faiss" and not self.store_uri:
            # Set default FAISS path
            self.store_uri = "./memory_index.faiss"
        
        return self


class DocumentMetadata(BaseModel):
    """Metadata for a document in memory."""
    source: Optional[str] = Field(None, description="Source of the document")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    tokens: Optional[int] = Field(None, description="Token count")
    file_size: Optional[int] = Field(None, description="Original file size")
    mime_type: Optional[str] = Field(None, description="MIME type of original file")
    chunk_count: Optional[int] = Field(None, description="Number of chunks created")
    processing_time: Optional[float] = Field(None, description="Total processing time")
    
    # Additional custom metadata
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class MemoryChunk(BaseModel):
    """A chunk of text with its embedding and metadata."""
    text: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    metadata: DocumentMetadata = Field(..., description="Chunk metadata")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    score: Optional[float] = Field(None, description="Similarity score (for search results)")


class DocumentProcessingJob(BaseModel):
    """A document processing job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: DocumentStatus = Field(DocumentStatus.PENDING, description="Processing status")
    file_path: Optional[str] = Field(None, description="Path to the file being processed")
    content: Optional[Union[str, bytes]] = Field(None, description="Raw content being processed")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Processing results
    parsed_text: Optional[str] = Field(None, description="Parsed text content")
    chunks: List[MemoryChunk] = Field(default_factory=list, description="Generated chunks")
    embeddings: List[List[float]] = Field(default_factory=list, description="Generated embeddings")
    
    # Processing timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get total processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class MemoryQuery(BaseModel):
    """A memory query with filters and options."""
    query: str = Field(..., description="Search query text")
    k: int = Field(5, description="Number of results to return")
    source: Optional[str] = Field(None, description="Filter by source")
    min_score: Optional[float] = Field(None, description="Minimum similarity score")
    metadata_filter: Optional[FilterType] = Field(None, description="Additional metadata filters")
    enable_reranking: Optional[bool] = Field(None, description="Override reranking setting")
    enable_hybrid_search: Optional[bool] = Field(None, description="Override hybrid search setting")


class MemorySearchResult(BaseModel):
    """A search result from memory."""
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Normalized similarity score (0-1)")
    metadata: DocumentMetadata = Field(..., description="Chunk metadata")
    chunk_index: int = Field(..., description="Chunk index in original document")
    raw_score: Optional[float] = Field(None, description="Raw similarity score from store")


class MemoryStats(BaseModel):
    """Memory statistics."""
    total_documents: int = Field(0, description="Total number of documents")
    total_chunks: int = Field(0, description="Total number of chunks")
    total_tokens: int = Field(0, description="Total token count")
    storage_backend: str = Field(..., description="Storage backend being used")
    
    # Distribution by source
    source_distribution: Dict[str, int] = Field(default_factory=dict, description="Chunks per source")
    
    # Processing statistics
    avg_processing_time: Optional[float] = Field(None, description="Average processing time per document")
    success_rate: Optional[float] = Field(None, description="Success rate of document processing")


class ProcessingProgress(BaseModel):
    """Progress tracking for document processing."""
    job_id: str = Field(..., description="Job identifier")
    status: DocumentStatus = Field(..., description="Current status")
    progress_percent: float = Field(0.0, description="Progress percentage (0-100)")
    current_step: str = Field("", description="Current processing step")
    estimated_time: Optional[float] = Field(None, description="Estimated time remaining")
    error_message: Optional[str] = Field(None, description="Error message if any") 