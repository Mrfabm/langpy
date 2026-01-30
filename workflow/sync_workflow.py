"""
SyncWorkflow - Synchronous workflow execution system.

This module provides a synchronous wrapper around AsyncWorkflow
for compatibility with sync codebases.
"""

from __future__ import annotations
import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field

from .async_workflow import (
    StepConfig, 
    WorkflowContext, 
    WorkflowRun, 
    WorkflowRegistry,
    AsyncWorkflow
)

# Type aliases
JsonDict = Dict[str, Any]


class SyncWorkflow:
    """
    Synchronous workflow execution system similar to Langbase.
    
    This is a wrapper around AsyncWorkflow that provides synchronous
    access to all workflow functionality.
    """
    
    def __init__(self, registry: Optional[WorkflowRegistry] = None):
        """
        Initialize SyncWorkflow.
        
        Args:
            registry: Workflow registry (creates default if None)
        """
        self._async_workflow = AsyncWorkflow(registry=registry)
    
    def register_runner(self, primitive_type: str, runner: callable) -> None:
        """
        Register a primitive runner.
        
        Args:
            primitive_type: Type of primitive (pipe, agent, tool)
            runner: Sync function to run the primitive
        """
        # Convert sync runner to async
        async def async_runner(ref: str, context: Dict[str, Any]) -> Any:
            return runner(ref, context)
        
        self._async_workflow.register_runner(primitive_type, async_runner)
    
    def set_memory(self, memory_interface) -> None:
        """Set memory interface for workflow context."""
        self._async_workflow.set_memory(memory_interface)
    
    def set_thread(self, thread_interface) -> None:
        """Set thread interface for workflow context."""
        self._async_workflow.set_thread(thread_interface)
    
    def run(
        self, 
        name: str, 
        inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a workflow synchronously.
        
        Args:
            name: Workflow name to run
            inputs: Input variables for the workflow
            
        Returns:
            Workflow outputs
            
        Raises:
            ValueError: If workflow not found
        """
        return asyncio.run(self._async_workflow.run(name=name, inputs=inputs))
    
    def get_run_history(
        self,
        workflow_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get workflow run history.
        
        Args:
            workflow_name: Optional workflow name to filter by
            limit: Maximum number of runs to return
            
        Returns:
            List of run records
        """
        return asyncio.run(self._async_workflow.get_run_history(
            workflow_name=workflow_name,
            limit=limit
        ))
    
    @property
    def registry(self) -> WorkflowRegistry:
        """Get the workflow registry."""
        return self._async_workflow.registry 