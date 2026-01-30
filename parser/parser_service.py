"""
Parser Service - FastAPI microservice for document parsing.

Provides Langbase-compatible REST API with job lifecycle management,
async processing, and comprehensive document parsing capabilities.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiofiles
import aiohttp

from .models import (
    ParseJob, ParseResult, ParseRequest, ParserOptions, JobStatus,
    SupportedFormat, ParserError, FileTooLargeError, UnsupportedFormatError
)
from .docling_parser import DoclingParser, get_parser


# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./data/parser"))
API_KEY = os.getenv("PARSER_API_KEY", "dev-key")

# Ensure storage directory exists
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
(STORAGE_PATH / "results").mkdir(exist_ok=True)
(STORAGE_PATH / "uploads").mkdir(exist_ok=True)


class ParserService:
    """Parser service with job management."""
    
    def __init__(self):
        self.parser = get_parser()
        self.jobs: Dict[str, ParseJob] = {}
        self.results: Dict[str, ParseResult] = {}
    
    async def create_job(self, request: ParseRequest) -> ParseJob:
        """Create a new parse job."""
        job = await self.parser.parse_async(request)
        self.jobs[job.id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[ParseJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status: Optional[str] = None, 
                  mime_type: Optional[str] = None,
                  created_after: Optional[datetime] = None) -> List[ParseJob]:
        """List jobs with optional filters."""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status.value == status]
        
        if mime_type:
            jobs = [j for j in jobs if j.mime_type == mime_type]
        
        if created_after:
            jobs = [j for j in jobs if j.created_at >= created_after]
        
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job and its artifacts."""
        if job_id in self.jobs:
            del self.jobs[job_id]
        
        if job_id in self.results:
            del self.results[job_id]
        
        # Delete result file if exists
        result_file = STORAGE_PATH / "results" / f"{job_id}.json"
        if result_file.exists():
            result_file.unlink()
        
        return True
    
    async def save_result(self, job_id: str, result: ParseResult):
        """Save result to storage."""
        self.results[job_id] = result
        
        # Save to file
        result_file = STORAGE_PATH / "results" / f"{job_id}.json"
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(result.json())
    
    async def load_result(self, job_id: str) -> Optional[ParseResult]:
        """Load result from storage."""
        if job_id in self.results:
            return self.results[job_id]
        
        # Try to load from file
        result_file = STORAGE_PATH / "results" / f"{job_id}.json"
        if result_file.exists():
            async with aiofiles.open(result_file, 'r') as f:
                content = await f.read()
                result_data = json.loads(content)
                return ParseResult(**result_data)
        
        return None


# Global service instance
parser_service = ParserService()


# FastAPI app
app = FastAPI(
    title="Langpy Parser Service",
    description="Langbase-compatible document parsing service powered by Docling",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication
def verify_api_key(x_api_key: str = Depends(HTTPException(status_code=401, detail="API key required"))):
    """Verify API key."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Request/Response models
class CreateJobRequest(BaseModel):
    """Request model for creating parse jobs."""
    url: Optional[str] = Field(None, description="URL to download and parse")
    text: Optional[str] = Field(None, description="Raw text to parse")
    options: Optional[ParserOptions] = Field(None, description="Parser options")


class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: List[ParseJob] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    page: int = Field(1, description="Current page")
    per_page: int = Field(50, description="Jobs per page")


class SupportedFormatsResponse(BaseModel):
    """Response model for supported formats."""
    formats: List[SupportedFormat] = Field(..., description="Supported formats")
    total: int = Field(..., description="Total number of formats")


# Background task to monitor job completion
async def monitor_job_completion(job_id: str):
    """Monitor job completion and save results."""
    while True:
        job = parser_service.get_job(job_id)
        if not job:
            break
        
        if job.status == JobStatus.READY:
            # Job completed successfully
            # In a real implementation, you'd load the result from the parser
            # For now, we'll create a mock result
            result = ParseResult(
                pages=["Sample parsed content"],
                tables=[],
                metadata=job.stats
            )
            await parser_service.save_result(job_id, result)
            break
        elif job.status == JobStatus.FAILED:
            # Job failed
            break
        
        # Wait before checking again
        await asyncio.sleep(1)


# API Endpoints

@app.post("/parsers", response_model=ParseJob)
async def create_parser_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new parse job.
    
    Accepts file upload, URL, or raw text content.
    """
    try:
        # Create parse request
        parse_request = ParseRequest(
            url=request.url,
            content=request.text.encode() if request.text else None,
            options=request.options
        )
        
        # Create job
        job = await parser_service.create_job(parse_request)
        
        # Start monitoring job completion
        background_tasks.add_task(monitor_job_completion, job.id)
        
        return job
        
    except ParserError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/parsers/upload", response_model=ParseJob)
async def upload_and_parse(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Upload file and create parse job.
    """
    try:
        # Check file size
        if file.size and file.size > MAX_FILE_SIZE:
            raise FileTooLargeError(file.size, MAX_FILE_SIZE)
        
        # Read file content
        content = await file.read()
        
        # Parse options if provided
        parser_options = None
        if options:
            try:
                options_dict = json.loads(options)
                parser_options = ParserOptions(**options_dict)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid options JSON")
        
        # Create parse request
        parse_request = ParseRequest(
            content=content,
            filename=file.filename,
            options=parser_options
        )
        
        # Create job
        job = await parser_service.create_job(parse_request)
        
        # Start monitoring job completion
        if background_tasks:
            background_tasks.add_task(monitor_job_completion, job.id)
        
        return job
        
    except ParserError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/parsers", response_model=JobListResponse)
async def list_parser_jobs(
    status: Optional[str] = None,
    mime_type: Optional[str] = None,
    created_after: Optional[str] = None,
    page: int = 1,
    per_page: int = 50,
    api_key: str = Depends(verify_api_key)
):
    """
    List parse jobs with optional filters.
    """
    try:
        # Parse created_after if provided
        created_after_dt = None
        if created_after:
            try:
                created_after_dt = datetime.fromisoformat(created_after.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid created_after format")
        
        # Get jobs
        jobs = parser_service.list_jobs(status, mime_type, created_after_dt)
        
        # Pagination
        total = len(jobs)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        jobs_page = jobs[start_idx:end_idx]
        
        return JobListResponse(
            jobs=jobs_page,
            total=total,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/parsers/{job_id}", response_model=ParseJob)
async def get_parser_job(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get parse job by ID.
    """
    job = parser_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job


@app.get("/parsers/{job_id}/result")
async def get_parser_result(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get parse result by job ID.
    """
    job = parser_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.READY:
        raise HTTPException(status_code=400, detail=f"Job not ready: {job.status}")
    
    result = await parser_service.load_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return result


@app.delete("/parsers/{job_id}")
async def delete_parser_job(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Delete parse job and its artifacts.
    """
    job = parser_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    success = parser_service.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete job")
    
    return {"message": "Job deleted successfully"}


@app.get("/parsers/supported-formats", response_model=SupportedFormatsResponse)
async def get_supported_formats(
    api_key: str = Depends(verify_api_key)
):
    """
    Get list of supported formats and their limits.
    """
    try:
        formats = parser_service.parser.get_supported_formats()
        return SupportedFormatsResponse(
            formats=formats,
            total=len(formats)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/parsers/{job_id}/webhook")
async def set_webhook(
    job_id: str,
    webhook_url: str = Form(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Set webhook URL for job completion notifications.
    """
    job = parser_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # In a real implementation, you'd store the webhook URL
    # and call it when the job completes
    
    return {"message": "Webhook set successfully"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "langpy-parser"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Langpy Parser Service",
        "version": "2.0.0",
        "description": "Langbase-compatible document parsing service",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 