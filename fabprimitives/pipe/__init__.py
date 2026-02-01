from pipe.async_pipe import AsyncPipe, PipeRunMetadata
from pipe.sync_pipe import SyncPipe
from pipe.pipe_primitive import PipeRegistry, create_pipe, update_pipe, run_pipe
from pipe.store import create, update, delete, list_all   # re-export helpers

# Re-export existing functionality
__all__ = [
    "AsyncPipe", "SyncPipe", "PipeRegistry", "create_pipe", "update_pipe", "run_pipe",
    "create", "update", "delete", "list_all", "PipeRunMetadata"
]

# Decorator support
def pipe(name: str, model: str = None, **config):
    """Decorator for creating pipes with integration."""
    def decorator(func):
        # Register with the core pipe engine
        AsyncPipe.register_function(name, func, model, **config)
        return func
    return decorator

# Integration decorators
def with_memory(memory_config: dict):
    """Decorator to add memory integration to pipe."""
    def decorator(func):
        func._memory_config = memory_config
        return func
    return decorator

def with_thread(thread_config: dict):
    """Decorator to add thread integration to pipe."""
    def decorator(func):
        func._thread_config = thread_config
        return func
    return decorator

def with_tools(tools_config: list):
    """Decorator to add tool integration to pipe."""
    def decorator(func):
        func._tools_config = tools_config
        return func
    return decorator

def with_agent(agent_config: dict):
    """Decorator to add agent integration to pipe."""
    def decorator(func):
        func._agent_config = agent_config
        return func
    return decorator

# Add decorators to __all__
__all__.extend([
    "pipe", "with_memory", "with_thread", "with_tools", "with_agent"
])
