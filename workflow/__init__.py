"""
Workflows module for LangPy primitives.

This module provides workflow orchestration similar to Langbase.
"""

from .async_workflow import AsyncWorkflow, StepConfig, WorkflowRegistry, WorkflowContext, WorkflowRun
from .exceptions import (
    WorkflowError, TimeoutError, RetryExhaustedError, StepError,
    SecretError, PrimitiveError, ContextError, DependencyError,
    # Legacy aliases
    StepTimeout, StepRetryExhausted
)
from .retry import RetryConfig, RetryEngine
from .logging import get_workflow_logger
from .run_registry import get_run_registry
from .core import WorkflowEngine, get_workflow_engine
import functools
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-local storage for current workflow instance
_workflow_context = threading.local()

class WorkflowContext:
    """Context manager for collecting steps during workflow execution."""
    
    def __init__(self, workflow_instance):
        self.workflow = workflow_instance
        self.previous_workflow = None
    
    def __enter__(self):
        self.previous_workflow = getattr(_workflow_context, 'current_workflow', None)
        _workflow_context.current_workflow = self.workflow
        return self.workflow
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _workflow_context.current_workflow = self.previous_workflow

# Enhanced decorator-based workflow API
def workflow(name: Optional[str] = None, debug: bool = False):
    """
    Decorator to define a workflow.
    
    Args:
        name: Workflow name (defaults to function name)
        debug: Enable debug mode
    """
    def decorator(func):
        workflow_name = name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create workflow instance
            wf = Workflow()
            
            # Use context manager to collect steps during execution
            with WorkflowContext(wf):
                # Call the function to register steps
                result = func(*args, **kwargs)
                if hasattr(result, '__await__'):
                    result = await result
            
            # Run the workflow with collected steps
            return await wf.run(**kwargs)
        
        wrapper.workflow_name = workflow_name
        wrapper.debug = debug
        return wrapper
    
    return decorator

def step(workflow_instance=None, id=None, after=None, when=None, timeout=None, retries=None, group=None, **kwargs):
    """
    Enhanced decorator to define a workflow step.
    
    Args:
        workflow_instance: The workflow instance (can be None for inline workflows)
        id: Step identifier (defaults to function name)
        after: List of step IDs this step depends on
        when: Conditional function that takes context and returns bool
        timeout: Timeout in milliseconds
        retries: Retry configuration {"limit": 3, "delay": 1000, "backoff": 2}
        group: Group for parallel execution
        **kwargs: Additional StepConfig parameters
    """
    def decorator(func):
        step_id = id or func.__name__
        
        # Create a wrapper that handles context conversion and logging
        async def step_wrapper(context_dict):
            run_id = context_dict.get('_run_id', 'unknown')
            logger.info(f"[{run_id}] Starting step '{step_id}'")
            
            try:
                # Convert dict context to object context
                ctx = ContextObject(context_dict)
                
                # Call the function (sync or async)
                result = func(ctx)
                
                # Handle async functions
                if hasattr(result, '__await__'):
                    result = await result
                
                logger.info(f"[{run_id}] Completed step '{step_id}'")
                return result
                
            except Exception as e:
                logger.error(f"[{run_id}] Failed step '{step_id}': {str(e)}")
                raise
        
        # Create StepConfig with enhanced parameters
        step_config = StepConfig(
            id=step_id,
            run=step_wrapper,
            type="function",
            timeout=timeout,
            retries=retries,
            group=group,
            **kwargs
        )
        
        # Handle after dependencies
        if after:
            step_config.after = after
        
        # Handle conditional execution
        if when:
            step_config.condition = when
        
        # Always try to register with the current workflow context if present
        current_wf = getattr(_workflow_context, 'current_workflow', None)
        if workflow_instance:
            if not hasattr(workflow_instance, '_inline_steps'):
                workflow_instance._inline_steps = []
            workflow_instance._inline_steps.append(step_config)
        elif current_wf:
            if not hasattr(current_wf, '_inline_steps'):
                current_wf._inline_steps = []
            current_wf._inline_steps.append(step_config)
        else:
            # Store on the function for static workflows
            if not hasattr(func, '_step_configs'):
                func._step_configs = []
            func._step_configs.append(step_config)
        
        return func
    
    return decorator

class ContextObject:
    """Enhanced context object that allows both attribute and dict access."""
    
    def __init__(self, data):
        self._data = data
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def to_dict(self):
        return self._data.copy()
    
    def __repr__(self):
        return f"ContextObject({list(self._data.keys())})"
    
    def __dir__(self):
        """Make context keys available for autocomplete."""
        return list(self._data.keys()) + list(super().__dir__())

# Enhanced AsyncWorkflow with decorator support
class Workflow(AsyncWorkflow):
    """Enhanced workflow with decorator support and simplified API."""
    
    def __init__(self, registry=None):
        super().__init__(registry)
        self._pending_steps = []
        self._workflow_name = None
    
    def run(self, *args, **inputs):
        """Run the workflow with the given inputs."""
        # Merge inline steps if present
        if hasattr(self, '_inline_steps') and self._inline_steps:
            self._pending_steps.extend(self._inline_steps)
            self._inline_steps = []
        if not self._pending_steps:
            raise ValueError("No steps defined. Use @step decorator to define steps.")
        # Create workflow if not already created
        if not self._workflow_name:
            self._workflow_name = f"workflow_{id(self)}"
            self.registry.create(self._workflow_name, self._pending_steps)
        # Add run ID for logging
        import uuid
        run_id = str(uuid.uuid4())[:8]
        inputs['_run_id'] = run_id
        # Flatten positional args if present (for *args signature)
        if args:
            for i, arg in enumerate(args):
                inputs[f'arg{i}'] = arg
        return super().run(self._workflow_name, inputs)

# Loop helper for fan-out operations
def foreach(items: List[Any], step_func: Callable, **step_kwargs):
    """
    Create multiple parallel steps for each item in a list.
    
    Args:
        items: List of items to process
        step_func: Function to apply to each item
        **step_kwargs: Additional step configuration
    
    Returns:
        List of step configurations
    """
    steps = []
    for i, item in enumerate(items):
        step_id = f"{step_func.__name__}_{i}"
        
        async def item_step(ctx, item=item):
            return await step_func(ctx, item)
        
        step_config = StepConfig(
            id=step_id,
            run=item_step,
            type="function",
            group=f"foreach_{step_func.__name__}",
            **step_kwargs
        )
        steps.append(step_config)
    
    return steps 

# Export for import convenience
__all__ = [
    "workflow",
    "step",
    "Workflow",
    "AsyncWorkflow",
    "StepConfig",
    "WorkflowRegistry",
    "WorkflowContext",
    "WorkflowRun",
    "WorkflowEngine",
    "get_workflow_engine",
    "WorkflowError",
    "TimeoutError",
    "RetryExhaustedError",
    "StepError",
    "SecretError",
    "PrimitiveError",
    "ContextError",
    "DependencyError",
    "StepTimeout",  # Legacy alias
    "StepRetryExhausted",  # Legacy alias
    "RetryConfig",
    "RetryEngine",
    "get_workflow_logger",
    "get_run_registry",
] 