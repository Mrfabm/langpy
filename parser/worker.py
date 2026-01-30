"""
Parser Worker - Background job processor for document parsing.

Provides async job processing for the parser service, handling
file uploads, URL downloads, and document parsing with Docling.
"""

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from parser.models import (
    ParseJob, ParseResult, ParseRequest, ParserOptions, JobStatus,
    ParserError, FileTooLargeError, UnsupportedFormatError
)
from parser.docling_parser import DoclingParser


class ParserWorker:
    """Background worker for processing parse jobs."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize parser worker.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path) if storage_path else Path("./data/parser")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize parser
        self.parser = DoclingParser()
        
        # Job queue (in production, use Redis or similar)
        self.job_queue: List[ParseJob] = []
        self.processing_jobs: Dict[str, ParseJob] = {}
        self.completed_jobs: Dict[str, ParseJob] = {}
        
        # Worker state
        self.running = False
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", 5))
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def add_job(self, job: ParseJob) -> bool:
        """Add job to processing queue."""
        try:
            self.job_queue.append(job)
            logger.info(f"Added job {job.id} to queue")
            return True
        except Exception as e:
            logger.error(f"Failed to add job {job.id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[ParseJob]:
        """Get job by ID from any state."""
        # Check processing jobs
        if job_id in self.processing_jobs:
            return self.processing_jobs[job_id]
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check queue
        for job in self.job_queue:
            if job.id == job_id:
                return job
        
        return None
    
    def list_jobs(self, status: Optional[str] = None) -> List[ParseJob]:
        """List jobs with optional status filter."""
        all_jobs = []
        
        # Add queued jobs
        all_jobs.extend(self.job_queue)
        
        # Add processing jobs
        all_jobs.extend(self.processing_jobs.values())
        
        # Add completed jobs
        all_jobs.extend(self.completed_jobs.values())
        
        # Filter by status if provided
        if status:
            all_jobs = [j for j in all_jobs if j.status.value == status]
        
        return sorted(all_jobs, key=lambda x: x.created_at, reverse=True)
    
    async def _process_job(self, job: ParseJob):
        """Process a single job."""
        try:
            logger.info(f"Starting to process job {job.id}")
            
            # Update job status to processing
            job.status = JobStatus.PROCESSING
            job.updated_at = datetime.utcnow()
            
            # Load job data from storage
            job_data = await self._load_job_data(job.id)
            if not job_data:
                raise ParserError(f"Job data not found for {job.id}")
            
            # Create parse request
            parse_request = ParseRequest(**job_data)
            
            # Process the job
            result = await self.parser._parse_with_docling(
                parse_request.content,
                job.mime_type or "application/octet-stream",
                parse_request.options or ParserOptions()
            )
            
            # Update job with results
            job.status = JobStatus.READY
            job.updated_at = datetime.utcnow()
            job.stats = result.metadata
            job.result_url = f"/results/{job.id}"
            
            # Save result
            await self._save_result(job.id, result)
            
            logger.info(f"Successfully processed job {job.id}")
            
        except Exception as e:
            logger.error(f"Failed to process job {job.id}: {e}")
            
            # Update job with error
            job.status = JobStatus.FAILED
            job.updated_at = datetime.utcnow()
            job.error_code = type(e).__name__
            job.error_message = str(e)
        
        finally:
            # Remove from processing and add to completed
            if job.id in self.processing_jobs:
                del self.processing_jobs[job.id]
            
            self.completed_jobs[job.id] = job
    
    async def _load_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job data from storage."""
        job_file = self.storage_path / "jobs" / f"{job_id}.json"
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load job data for {job_id}: {e}")
            return None
    
    async def _save_result(self, job_id: str, result: ParseResult):
        """Save result to storage."""
        result_file = self.storage_path / "results" / f"{job_id}.json"
        result_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            logger.info(f"Saved result for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to save result for job {job_id}: {e}")
    
    async def _worker_loop(self):
        """Main worker loop."""
        logger.info("Starting parser worker loop")
        
        while self.running:
            try:
                # Check if we can process more jobs
                if len(self.processing_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(1)
                    continue
                
                # Get next job from queue
                if not self.job_queue:
                    await asyncio.sleep(1)
                    continue
                
                job = self.job_queue.pop(0)
                
                # Add to processing
                self.processing_jobs[job.id] = job
                
                # Start processing in background
                asyncio.create_task(self._process_job(job))
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)
    
    async def start(self):
        """Start the worker."""
        logger.info("Starting parser worker")
        self.running = True
        
        # Start worker loop
        await self._worker_loop()
    
    async def stop(self):
        """Stop the worker."""
        logger.info("Stopping parser worker")
        self.running = False
        
        # Wait for processing jobs to complete
        while self.processing_jobs:
            logger.info(f"Waiting for {len(self.processing_jobs)} jobs to complete...")
            await asyncio.sleep(5)
        
        logger.info("Parser worker stopped")


# Celery integration (optional)
try:
    from celery import Celery
    
    # Create Celery app
    celery_app = Celery('parser_worker')
    celery_app.config_from_object({
        'broker_url': os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        'result_backend': os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        'task_serializer': 'json',
        'accept_content': ['json'],
        'result_serializer': 'json',
        'timezone': 'UTC',
        'enable_utc': True,
    })
    
    @celery_app.task
    def process_parse_job(job_id: str):
        """Celery task for processing parse jobs."""
        async def _process():
            worker = ParserWorker()
            job = worker.get_job(job_id)
            if job:
                await worker._process_job(job)
        
        asyncio.run(_process())
    
    CELERY_AVAILABLE = True
    
except ImportError:
    CELERY_AVAILABLE = False
    logger.info("Celery not available, running in standalone mode")


async def main():
    """Main entry point."""
    # Create worker
    worker = ParserWorker()
    
    # Start worker
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await worker.stop()


if __name__ == "__main__":
    # Run standalone worker
    asyncio.run(main()) 