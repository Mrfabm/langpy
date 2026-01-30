"""
Workflow exception types for LangPy - Enhanced error taxonomy matching Langbase.

This module provides a comprehensive error hierarchy for workflow execution,
with proper error context and debugging information.
"""

from typing import Optional, Dict, Any, List
import traceback
import time


class WorkflowError(Exception):
    """Base exception for workflow errors."""
    
    def __init__(self, message: str, step_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize WorkflowError.
        
        Args:
            message: Error message
            step_id: ID of the step that failed
            context: Additional context information
        """
        super().__init__(message)
        self.step_id = step_id
        self.context = context or {}
        self.timestamp = int(time.time())
        self.trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "step_id": self.step_id,
            "context": self.context,
            "timestamp": self.timestamp,
            "trace": self.trace
        }


class TimeoutError(WorkflowError):
    """Raised when a step times out."""
    
    def __init__(self, step_id: str, timeout_ms: int, elapsed_ms: int):
        """
        Initialize TimeoutError.
        
        Args:
            step_id: ID of the step that timed out
            timeout_ms: Configured timeout in milliseconds
            elapsed_ms: Actual elapsed time in milliseconds
        """
        message = f"Step '{step_id}' timed out after {elapsed_ms}ms (limit: {timeout_ms}ms)"
        super().__init__(message, step_id)
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms


class RetryExhaustedError(WorkflowError):
    """Raised when a step exhausts all retry attempts."""
    
    def __init__(self, step_id: str, attempts: int, last_error: Optional[Exception] = None):
        """
        Initialize RetryExhaustedError.
        
        Args:
            step_id: ID of the step that exhausted retries
            attempts: Number of attempts made
            last_error: The last error that occurred
        """
        message = f"Step '{step_id}' exhausted all {attempts} retry attempts"
        if last_error:
            message += f". Last error: {last_error}"
        super().__init__(message, step_id)
        self.attempts = attempts
        self.last_error = last_error


class StepError(WorkflowError):
    """Raised when a step fails during execution."""
    
    def __init__(self, step_id: str, original_error: Exception, attempt: int = 1):
        """
        Initialize StepError.
        
        Args:
            step_id: ID of the step that failed
            original_error: The original exception that caused the failure
            attempt: Current attempt number
        """
        message = f"Step '{step_id}' failed on attempt {attempt}: {original_error}"
        super().__init__(message, step_id)
        self.original_error = original_error
        self.attempt = attempt


class DependencyError(WorkflowError):
    """Raised when there are circular dependencies or missing dependencies."""
    
    def __init__(self, step_id: str, dependency_chain: List[str]):
        """
        Initialize DependencyError.
        
        Args:
            step_id: ID of the step with dependency issues
            dependency_chain: Chain of dependencies that caused the issue
        """
        message = f"Dependency error in step '{step_id}': {' -> '.join(dependency_chain)}"
        super().__init__(message, step_id)
        self.dependency_chain = dependency_chain


class SecretError(WorkflowError):
    """Raised when there are issues with secret access."""
    
    def __init__(self, step_id: str, secret_name: str, reason: str):
        """
        Initialize SecretError.
        
        Args:
            step_id: ID of the step that failed to access secret
            secret_name: Name of the secret that couldn't be accessed
            reason: Reason for the failure
        """
        message = f"Secret '{secret_name}' not available for step '{step_id}': {reason}"
        super().__init__(message, step_id)
        self.secret_name = secret_name
        self.reason = reason


class PrimitiveError(WorkflowError):
    """Raised when a primitive (pipe, agent, tool) fails."""
    
    def __init__(self, step_id: str, primitive_type: str, primitive_ref: str, original_error: Exception):
        """
        Initialize PrimitiveError.
        
        Args:
            step_id: ID of the step that failed
            primitive_type: Type of primitive (pipe, agent, tool)
            primitive_ref: Reference to the primitive
            original_error: The original exception from the primitive
        """
        message = f"Primitive '{primitive_type}:{primitive_ref}' failed in step '{step_id}': {original_error}"
        super().__init__(message, step_id)
        self.primitive_type = primitive_type
        self.primitive_ref = primitive_ref
        self.original_error = original_error


class ContextError(WorkflowError):
    """Raised when there are issues with workflow context."""
    
    def __init__(self, step_id: str, missing_keys: List[str]):
        """
        Initialize ContextError.
        
        Args:
            step_id: ID of the step that failed
            missing_keys: Keys that were missing from context
        """
        message = f"Step '{step_id}' missing required context keys: {', '.join(missing_keys)}"
        super().__init__(message, step_id)
        self.missing_keys = missing_keys


# Legacy aliases for backward compatibility
StepTimeout = TimeoutError
StepRetryExhausted = RetryExhaustedError 