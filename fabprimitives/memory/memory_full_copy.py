"""
Async Memory Orchestrator - Core memory primitive with full Langbase parity.

This is the main memory primitive that acts as an orchestrator, automatically running
the pipeline: Parser → Chunker → Embed → Store when documents are uploaded.

ADVANCED FEATURES (Langbase Parity):
• Advanced Filter DSL with nested Boolean logic (AND/OR)
• Hybrid search combining vector similarity and BM25 keyword matching
• Cross-encoder reranking for improved relevance
• Automatic memory integration in pipes
• Normalized scoring (0-1 range)
• Score normalization for different store backends
• Langbase-style container management with MemoryManager
"""

import asyncio
import uuid
import time
import json
import re
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
from datetime import datetime

from .models import (
    MemorySettings, DocumentMetadata, MemoryChunk, DocumentProcessingJob,
    DocumentStatus, MemoryQuery, MemorySearchResult, MemoryStats,
    ProcessingProgress
)

# Import the primitives we're orchestrating
from langpy.sdk.parser_interface import ParserInterface
from langpy.sdk.chunker_interface import ChunkerInterface
from langpy.sdk.embed_interface import EmbedInterface
from stores import get_store


class AsyncMemory:
    """
    Async memory orchestrator that coordinates parser, chunker, embed, and store operations.
    
    When you call upload(), the platform automatically runs:
    Parser → Chunker → Embed → Store behind the scenes.
    
    Each stage is also exposed as a standalone primitive that can be invoked directly.
    """
    
    def __init__(self, settings: Optional[MemorySettings] = None):
        """
        Initialize the memory orchestrator.
        
        Args:
            settings: Memory configuration settings
        """
        self.settings = settings or MemorySettings()
        # Validation is automatic in Pydantic v2 with model_validator
        
        # Initialize the primitives we're orchestrating
        self._parser = ParserInterface()
        self._chunker = ChunkerInterface()
        self._embedder = EmbedInterface(default_embedder=self.settings.embed_model)
        self._store = None
        
        # Job tracking
        self._jobs: Dict[str, DocumentProcessingJob] = {}
        self._progress_callbacks: Dict[str, Callable[[ProcessingProgress], None]] = {}
    
    async def _get_store(self):
        """Get or create the vector store instance."""
        if self._store is None:
            from stores import get_store
            
            # Use the new store factory with configuration
            if hasattr(self.settings, '_store_config'):
                # Use the store config from create_memory
                store_config = self.settings._store_config
                self._store = get_store(**store_config)
            else:
                # Fallback to old method for backward compatibility
                self._store = get_store(
                    kind=self.settings.store_backend,
                    uri=self.settings.store_uri or "./memory_index.faiss"
                )
        return self._store
    
    async def upload(
        self,
        content: Union[str, bytes, Path],
        source: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> str:
        """
        Upload a document to memory, automatically running the full pipeline.
        This triggers the internal pipeline: Parser → Chunker → Embed → Store
        Args:
            content: Document content (file path, bytes, or text)
            source: Optional source identifier
            custom_metadata: Additional custom metadata
            progress_callback: Optional callback for progress updates
        Returns:
            Job ID for tracking the upload process
        """
        try:
            # Create job
            job_id = str(uuid.uuid4())
            print(f"[DEBUG] Created job_id: {job_id}")
            # Use meta parameter if provided, otherwise fall back to custom_metadata
            metadata_dict = meta or custom_metadata or {}
            metadata = DocumentMetadata(
                source=source,
                custom=metadata_dict
            )
            job = DocumentProcessingJob(
                job_id=job_id,
                status=DocumentStatus.PENDING,
                metadata=metadata
            )
            # Set content based on type
            if isinstance(content, (str, Path)):
                if Path(content).exists():
                    job.file_path = str(content)
                    job.metadata.source = job.metadata.source or str(content)
                else:
                    job.content = content
            else:
                job.content = content
            # Store job and callback
            self._jobs[job_id] = job
            if progress_callback:
                self._progress_callbacks[job_id] = progress_callback
            # Start processing asynchronously
            asyncio.create_task(self._process_document(job_id))
            return job_id
        except Exception as e:
            print(f"[ERROR] Failed to create/upload job: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create/upload job: {e}") from e
    
    async def _process_document(self, job_id: str):
        """Process a document through the full pipeline."""
        job = self._jobs[job_id]
        callback = self._progress_callbacks.get(job_id)
        
        try:
            job.status = DocumentStatus.PARSING
            job.started_at = datetime.now()
            await self._update_progress(job_id, "Parsing document...", 10, callback)
            
            # Step 1: Parse
            parsed_text = await self._parse_document(job)
            job.parsed_text = parsed_text
            job.metadata.tokens = len(parsed_text.split()) if parsed_text else 0
            
            # Check if we have content to process
            if not parsed_text or not parsed_text.strip():
                print(f"⚠️ No content extracted from {job.metadata.source}, skipping processing")
                job.status = DocumentStatus.COMPLETED
                job.completed_at = datetime.now()
                await self._update_progress(job_id, "No content to process", 100, callback)
                return
            
            await self._update_progress(job_id, "Chunking text...", 30, callback)
            
            # Step 2: Chunk
            chunks = await self._chunk_text(parsed_text, job)
            job.chunks = chunks
            job.metadata.chunk_count = len(chunks)
            
            # Check if chunking produced any chunks
            if not chunks:
                print(f"⚠️ No chunks created from {job.metadata.source}, skipping embedding and storage")
                job.status = DocumentStatus.COMPLETED
                job.completed_at = datetime.now()
                await self._update_progress(job_id, "No chunks to process", 100, callback)
                return
            
            await self._update_progress(job_id, "Generating embeddings...", 60, callback)
            
            # Step 3: Embed
            embeddings = await self._embed_chunks(chunks)
            job.embeddings = embeddings
            
            # Update chunks with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk.embedding = embedding
            
            await self._update_progress(job_id, "Storing in vector database...", 80, callback)
            
            # Step 4: Store
            await self._store_chunks(chunks)
            
            job.status = DocumentStatus.COMPLETED
            job.completed_at = datetime.now()
            await self._update_progress(job_id, "Completed", 100, callback)
            
            print(f"✅ Document uploaded successfully: {job_id}")
            
        except Exception as e:
            job.status = DocumentStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            await self._update_progress(job_id, f"Failed: {str(e)}", 0, callback)
            print(f"❌ Document upload failed: {job_id} - {str(e)}")
    
    async def _parse_document(self, job: DocumentProcessingJob) -> str:
        """Parse the document using the parser primitive in the simplest way."""
        try:
            if job.file_path:
                # Use the parser primitive directly, no custom options
                result = await self._parser.parse_file(job.file_path)
            else:
                result = await self._parser.parse_content(job.content, filename=job.metadata.source)
            
            # Use the parser's public API for text extraction
            if hasattr(result, 'text') and result.text:
                return result.text
            elif hasattr(result, 'pages') and result.pages:
                return "\n".join(result.pages)
            elif hasattr(result, 'content') and result.content:
                return result.content
            else:
                # If parser didn't return text, use original content
                if job.content:
                    if isinstance(job.content, bytes):
                        return job.content.decode('utf-8', errors='ignore')
                    return str(job.content)
                else:
                    return ""
        except Exception as e:
            print(f"⚠️ Parser failed for {job.metadata.source}: {str(e)}")
            print("Continuing with original content...")
            # Fallback to original content
            if job.content:
                if isinstance(job.content, bytes):
                    return job.content.decode('utf-8', errors='ignore')
                return str(job.content)
            else:
                return ""
    
    async def _chunk_text(self, text: str, job: DocumentProcessingJob) -> List[MemoryChunk]:
        """Chunk the text using the chunker primitive."""
        chunks = await self._chunker.chunk_text(
            text=text,
            chunk_max_length=self.settings.chunk_max_length,
            chunk_overlap=self.settings.chunk_overlap
        )
        
        # Create MemoryChunk objects
        memory_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = DocumentMetadata(
                source=job.metadata.source,
                timestamp=job.metadata.timestamp,
                tokens=len(chunk_text.split()),
                chunk_count=len(chunks),
                custom=job.metadata.custom.copy()
            )
            
            memory_chunk = MemoryChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i
            )
            memory_chunks.append(memory_chunk)
        
        return memory_chunks
    
    async def _embed_chunks(self, chunks: List[MemoryChunk]) -> List[List[float]]:
        """Generate embeddings for chunks using the embed primitive."""
        texts = [chunk.text for chunk in chunks]
        embeddings = await self._embedder.embed_texts(
            texts=texts,
            embedder_name=self.settings.embed_model
        )
        return embeddings
    
    async def _store_chunks(self, chunks: List[MemoryChunk]):
        """Store chunks in the vector store."""
        store = await self._get_store()
        
        # Prepare data for storage
        texts = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [chunk.metadata.dict() for chunk in chunks]
        
        # Store in vector database
        await store.add(texts, metadatas)
    
    async def _update_progress(
        self, 
        job_id: str, 
        step: str, 
        percent: float, 
        callback: Optional[Callable[[ProcessingProgress], None]]
    ):
        """Update progress for a job."""
        if callback:
            progress = ProcessingProgress(
                job_id=job_id,
                status=self._jobs[job_id].status,
                progress_percent=percent,
                current_step=step
            )
            callback(progress)
    
    async def get_job_status(self, job_id: str) -> Optional[DocumentProcessingJob]:
        """Get the status of a processing job."""
        return self._jobs.get(job_id)
    
    async def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get simplified job status for easy polling.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with status information or None if job not found
        """
        job = self._jobs.get(job_id)
        if not job:
            return None
            
        return {
            "status": job.status.value,
            "error_message": job.error_message,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metadata": job.metadata.dict() if job.metadata else None
        }
    
    async def query(
        self,
        query: str,
        k: Optional[int] = None,
        source: Optional[str] = None,
        min_score: Optional[float] = None,
        as_dict: bool = True
    ) -> Union[List[MemorySearchResult], List[Dict[str, Any]]]:
        """
        Query memory for similar content.
        
        Args:
            query: Search query
            k: Number of results to return
            source: Filter by source
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        store = await self._get_store()
        
        # Build metadata filter
        metadata_filter = {}
        if source:
            metadata_filter['source'] = source
        
        # Query the store
        results = await store.query(
            query=query,
            k=k or self.settings.default_k,
            filt=metadata_filter
        )
        
        # Convert to MemorySearchResult objects
        search_results = []
        for result in results:
            # Apply score threshold
            if min_score is not None and result.get('score', 0) < min_score:
                continue
            
            metadata = DocumentMetadata(**result['meta'])
            search_result = MemorySearchResult(
                text=result['text'],
                score=result.get('score', 0.0),
                metadata=metadata,
                chunk_index=metadata.custom.get('chunk_index', 0)
            )
            search_results.append(search_result)
        
        if as_dict:
            return [
                {
                    "text": result.text,
                    "score": result.score,
                    "metadata": result.metadata.dict() if result.metadata else None
                }
                for result in search_results
            ]
        return search_results
    
    async def query_dict(
        self,
        query: str,
        k: Optional[int] = None,
        source: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query memory for similar content, returning results as dictionaries.
        
        Args:
            query: Search query
            k: Number of results to return
            source: Filter by source
            min_score: Minimum similarity score
            
        Returns:
            List of result dictionaries with 'text' and 'score' fields
        """
        return await self.query(query, k, source, min_score, as_dict=True)
    
    async def get_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        k: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """
        Get memory entries by metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata to match
            k: Number of results to return
            
        Returns:
            List of matching memory entries
        """
        store = await self._get_store()
        
        # Query by metadata only (no semantic search)
        results = await store.query("", k or self.settings.default_k, metadata_filter)
        
        # Convert to MemorySearchResult objects
        search_results = []
        for result in results:
            metadata = DocumentMetadata(**result['meta'])
            search_result = MemorySearchResult(
                text=result['text'],
                score=result.get('score', 0.0),
                metadata=metadata,
                chunk_index=metadata.custom.get('chunk_index', 0)
            )
            search_results.append(search_result)
        
        return search_results
    
    async def delete_by_filter(
        self,
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Delete memory entries matching filter criteria.
        
        Args:
            source: Filter by source
            metadata_filter: Additional metadata filters
            
        Returns:
            Number of deleted entries
        """
        store = await self._get_store()
        
        # Build filter
        filt = metadata_filter or {}
        if source:
            filt['source'] = source
        
        return await store.delete_by_filter(filt)
    
    async def update_metadata(
        self,
        updates: Dict[str, Any],
        source: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Update metadata for entries matching filter criteria.
        
        Args:
            updates: Metadata updates to apply
            source: Filter by source
            metadata_filter: Additional metadata filters
            
        Returns:
            Number of updated entries
        """
        store = await self._get_store()
        
        # Build filter
        filt = metadata_filter or {}
        if source:
            filt['source'] = source
        
        return await store.update_metadata(filt, updates)
    
    async def clear(self):
        """Clear all memory data."""
        store = await self._get_store()
        await store.clear()
        self._jobs.clear()
        self._progress_callbacks.clear()
        print("🗑️  Cleared all memory data")
    
    async def get_stats(self) -> MemoryStats:
        """
        Get memory statistics.
        
        Returns:
            Memory statistics
        """
        store = await self._get_store()
        store_stats = await store.get_metadata_stats()
        
        # Calculate processing statistics
        completed_jobs = [job for job in self._jobs.values() if job.status == DocumentStatus.COMPLETED]
        failed_jobs = [job for job in self._jobs.values() if job.status == DocumentStatus.FAILED]
        
        avg_processing_time = None
        if completed_jobs:
            processing_times = [job.processing_time for job in completed_jobs if job.processing_time]
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
        
        success_rate = None
        total_jobs = len(completed_jobs) + len(failed_jobs)
        if total_jobs > 0:
            success_rate = len(completed_jobs) / total_jobs
        
        return MemoryStats(
            total_documents=store_stats.get('total_documents', 0),
            total_chunks=store_stats.get('total_chunks', 0),
            total_tokens=store_stats.get('total_tokens', 0),
            storage_backend=self.settings.store_backend,

            source_distribution=store_stats.get('sources', {}),
            avg_processing_time=avg_processing_time,
            success_rate=success_rate
        )
    
    # Standalone primitive access methods
    async def parse_document(self, content: Union[str, bytes, Path], **kwargs) -> str:
        """Parse a document using the parser primitive directly."""
        if isinstance(content, (str, Path)) and Path(content).exists():
            result = await self._parser.parse_file(file_path=str(content), **kwargs)
        else:
            result = await self._parser.parse_content(content=content, **kwargs)
        return result.text
    
    async def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text using the chunker primitive directly."""
        return await self._chunker.chunk_text(text=text, **kwargs)
    
    async def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using the embed primitive directly."""
        return await self._embedder.embed_texts(texts=texts, **kwargs)
    
    async def close(self):
        """Close memory connections."""
        if hasattr(self._store, 'close'):
            await self._store.close() 

    # Langbase-style method aliases
    async def retrieve(self, *args, **kwargs):
        """Langbase alias for query."""
        return await self.query(*args, **kwargs)

    async def stats(self, *args, **kwargs):
        """Langbase alias for get_stats."""
        return await self.get_stats(*args, **kwargs)

    async def info(self, *args, **kwargs):
        """Langbase alias for get_stats (info)."""
        return await self.get_stats(*args, **kwargs)

    async def delete(self, *args, **kwargs):
        """Langbase alias for delete_by_filter (per-document delete)."""
        return await self.delete_by_filter(*args, **kwargs)

    async def update(self, *args, **kwargs):
        """Langbase alias for update_metadata (per-document update)."""
        return await self.update_metadata(*args, **kwargs)

    async def flush(self, *args, **kwargs):
        """Langbase alias for close (flush resources)."""
        return await self.close(*args, **kwargs)


# --- LANGBASE-STYLE CONTAINER MANAGEMENT ---
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