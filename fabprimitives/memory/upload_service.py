"""
Upload Service - Signed URL upload functionality for memory.

Provides signed URL generation for secure file uploads, matching Langbase's API.
For local development, this is a stub implementation.
"""

import uuid
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta


class UploadService:
    """Service for handling signed URL uploads."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the upload service.
        
        Args:
            base_url: Base URL for the upload service
        """
        self.base_url = base_url.rstrip('/')
        self.uploads: Dict[str, Dict[str, Any]] = {}
    
    def generate_upload_url(
        self,
        filename: str,
        content_type: Optional[str] = None,
        expires_in: int = 3600,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a signed URL for file upload.
        
        Args:
            filename: Name of the file to upload
            content_type: MIME type of the file
            expires_in: URL expiration time in seconds
            metadata: Additional metadata for the upload
            
        Returns:
            Dictionary containing upload URL and metadata
        """
        upload_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        # Create upload record
        upload_info = {
            'id': upload_id,
            'filename': filename,
            'content_type': content_type or 'application/octet-stream',
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat(),
            'metadata': metadata or {},
            'status': 'pending'
        }
        
        self.uploads[upload_id] = upload_info
        
        # Generate signed URL (stub implementation)
        signed_url = f"{self.base_url}/upload/{upload_id}"
        
        return {
            'upload_id': upload_id,
            'upload_url': signed_url,
            'expires_at': expires_at.isoformat(),
            'filename': filename,
            'content_type': content_type
        }
    
    def get_upload_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an upload.
        
        Args:
            upload_id: Upload identifier
            
        Returns:
            Upload status information or None if not found
        """
        if upload_id not in self.uploads:
            return None
        
        upload_info = self.uploads[upload_id].copy()
        
        # Check if expired
        expires_at = datetime.fromisoformat(upload_info['expires_at'])
        if datetime.now() > expires_at:
            upload_info['status'] = 'expired'
        
        return upload_info
    
    def complete_upload(
        self,
        upload_id: str,
        file_path: Optional[str] = None,
        content: Optional[bytes] = None
    ) -> bool:
        """
        Complete an upload by providing the file content.
        
        Args:
            upload_id: Upload identifier
            file_path: Path to the uploaded file
            content: File content as bytes
            
        Returns:
            True if upload was completed successfully
        """
        if upload_id not in self.uploads:
            return False
        
        upload_info = self.uploads[upload_id]
        
        # Check if expired
        expires_at = datetime.fromisoformat(upload_info['expires_at'])
        if datetime.now() > expires_at:
            upload_info['status'] = 'expired'
            return False
        
        # Update upload status
        upload_info['status'] = 'completed'
        upload_info['completed_at'] = datetime.now().isoformat()
        
        if file_path:
            upload_info['file_path'] = file_path
        if content:
            upload_info['content_size'] = len(content)
        
        return True
    
    def delete_upload(self, upload_id: str) -> bool:
        """
        Delete an upload.
        
        Args:
            upload_id: Upload identifier
            
        Returns:
            True if upload was deleted successfully
        """
        if upload_id in self.uploads:
            del self.uploads[upload_id]
            return True
        return False
    
    def list_uploads(self, status: Optional[str] = None) -> list:
        """
        List uploads with optional status filter.
        
        Args:
            status: Filter by upload status
            
        Returns:
            List of upload information
        """
        uploads = []
        
        for upload_id, upload_info in self.uploads.items():
            if status is None or upload_info['status'] == status:
                uploads.append({
                    'upload_id': upload_id,
                    **upload_info
                })
        
        return uploads
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired uploads.
        
        Returns:
            Number of uploads cleaned up
        """
        now = datetime.now()
        expired_ids = []
        
        for upload_id, upload_info in self.uploads.items():
            expires_at = datetime.fromisoformat(upload_info['expires_at'])
            if now > expires_at:
                expired_ids.append(upload_id)
        
        for upload_id in expired_ids:
            del self.uploads[upload_id]
        
        return len(expired_ids)


# Global upload service instance
_upload_service = None


def get_upload_service() -> UploadService:
    """Get the global upload service instance."""
    global _upload_service
    if _upload_service is None:
        _upload_service = UploadService()
    return _upload_service


def generate_upload_url(
    filename: str,
    content_type: Optional[str] = None,
    expires_in: int = 3600,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a signed URL for file upload.
    
    Args:
        filename: Name of the file to upload
        content_type: MIME type of the file
        expires_in: URL expiration time in seconds
        metadata: Additional metadata for the upload
        
    Returns:
        Dictionary containing upload URL and metadata
    """
    service = get_upload_service()
    return service.generate_upload_url(filename, content_type, expires_in, metadata)


def get_upload_status(upload_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of an upload.
    
    Args:
        upload_id: Upload identifier
        
    Returns:
        Upload status information or None if not found
    """
    service = get_upload_service()
    return service.get_upload_status(upload_id) 