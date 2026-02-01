from __future__ import annotations
import asyncio
import json
import sqlite3
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import inspect

# Configure logging
logger = logging.getLogger(__name__)

# Import custom exceptions
from .exceptions import WorkflowError, StepTimeout, StepRetryExhausted, DependencyError

# Type aliases
JsonDict = Dict[str, Any]

class ContextObject:
    """Context object for step execution."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data.copy()
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access like ctx.email."""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict access like ctx['email']."""
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'key in ctx' syntax."""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in context."""
        self._data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return self._data.copy()

class StepConfig(BaseModel):
    """Enhanced configuration for a workflow step matching Langbase's StepConfig."""
    id: str
    run: Optional[Callable[[Dict[str, Any]], Any]] = None
    timeout: Optional[int] = None  # milliseconds
    retries: Optional[Union[Dict[str, Any], Any]] = None  # {"limit": 3, "delay": 1000, "backoff": "exponential"|"linear"|"fixed"} or RetryConfig
    group: Optional[List[str]] = None  # parallel execution groups (changed from str to List[str])
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    after: Optional[List[str]] = None  # list of step IDs this step depends on
    use_secrets: Optional[List[str]] = None  # GitHub Actions-style secrets
    type: Literal["function", "pipe", "tool", "agent", "memory", "thread"] = "function"
    ref: Optional[str] = None  # reference to primitive name
    config: Optional[Dict[str, Any]] = None  # configuration for primitive steps

class WorkflowContext:
    """Enhanced context for workflow execution with thread_id, memory, tool_results, secrets."""
    
    def __init__(self, inputs: Dict[str, Any], memory_interface=None, thread_interface=None, agent_interface=None, pipe_interface=None):
        self._data = inputs.copy()
        self._memory_interface = memory_interface
        self._thread_interface = thread_interface
        self._agent_interface = agent_interface
        self._pipe_interface = pipe_interface
        
        # Enhanced context data
        self._thread_id: Optional[str] = None
        self._memory: Dict[str, Any] = {}
        self._tool_results: Dict[str, Any] = {}
        self._secrets: Dict[str, Any] = {}
        self._step_outputs: Dict[str, Any] = {}
    
    @property
    def thread_id(self) -> Optional[str]:
        """Get the current thread ID."""
        return self._thread_id
    
    @thread_id.setter
    def thread_id(self, value: str):
        """Set the thread ID."""
        self._thread_id = value
    
    @property
    def memory(self) -> Dict[str, Any]:
        """Get memory data."""
        return self._memory
    
    @property
    def tool_results(self) -> Dict[str, Any]:
        """Get tool results."""
        return self._tool_results
    
    @property
    def secrets(self) -> Dict[str, Any]:
        """Get available secrets for current step."""
        return self._secrets
    
    def set_secrets(self, secrets: Dict[str, str]) -> None:
        """Set secrets for current step."""
        self._secrets = secrets
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        # Check step outputs first
        if key in self._step_outputs:
            return self._step_outputs[key]
        
        # Check original data
        if key in self._data:
            return self._data[key]
        
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in context."""
        self._step_outputs[key] = value
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update memory data."""
        self._memory[key] = value
    
    def update_tool_results(self, key: str, value: Any) -> None:
        """Update tool results."""
        self._tool_results[key] = value
    
    @property
    def memory_interface(self):
        """Get memory interface."""
        return self._memory_interface
    
    @property
    def thread_interface(self):
        """Get thread interface."""
        return self._thread_interface
    
    @property
    def agent_interface(self):
        """Get agent interface."""
        return self._agent_interface
    
    @property
    def pipe_interface(self):
        """Get pipe interface."""
        return self._pipe_interface
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            **self._data,
            **self._step_outputs,
            "thread_id": self._thread_id,
            "memory": self._memory,
            "tool_results": self._tool_results,
            "secrets": self._secrets
        }
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in context."""
        return key in self._step_outputs or key in self._data

class WorkflowRun(BaseModel):
    """A workflow execution run."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str
    status: str = "running"  # running, completed, failed
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: int = Field(default_factory=lambda: int(time.time()))
    completed_at: Optional[int] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)

class WorkflowRegistry:
    """Registry for workflow definitions."""
    
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".langpy" / "workflows"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._workflows: Dict[str, List[StepConfig]] = {}
        self._step_registry: Dict[str, Callable] = {}  # Registry for named steps
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for run history."""
        db_path = self.storage_path / "workflows.db"
        self.db_path = db_path
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs TEXT,
                    outputs TEXT,
                    error TEXT,
                    started_at INTEGER NOT NULL,
                    completed_at INTEGER,
                    steps TEXT
                )
            """)
            conn.commit()
    
    def register_step(self, name: str, step_func: Callable) -> None:
        """Register a named step function for workflow restoration."""
        self._step_registry[name] = step_func
    
    def create(
        self, 
        name: str, 
        steps: List[StepConfig], 
        debug: bool = False
    ) -> None:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            steps: List of step configurations
            debug: Enable debug mode
        """
        self._workflows[name] = steps
        
        # Save to disk with step function names for restoration
        workflow_file = self.storage_path / f"{name}.json"
        workflow_data = {
            "name": name,
            "debug": debug,
            "steps": []
        }
        
        for step in steps:
            step_data = {
                "id": step.id,
                "timeout": step.timeout,
                "retries": step.retries,
                "group": step.group,
                "type": step.type,
                "ref": step.ref,
                "config": step.config
            }
            
            # For function steps, try to save function name for restoration
            if step.type == "function" and step.run:
                # Try to get function name
                if hasattr(step.run, '__name__'):
                    step_data["function_name"] = step.run.__name__
                elif hasattr(step.run, '__qualname__'):
                    step_data["function_name"] = step.run.__qualname__
            
            workflow_data["steps"].append(step_data)
        
        with open(workflow_file, 'w') as f:
            json.dump(workflow_data, f, indent=2)
    
    def get_workflow(self, name: str) -> Optional[List[StepConfig]]:
        """Get workflow by name."""
        if name in self._workflows:
            return self._workflows[name]
        
        # Try to load from disk
        workflow_file = self.storage_path / f"{name}.json"
        if workflow_file.exists():
            with open(workflow_file, 'r') as f:
                data = json.load(f)
            
            # Restore workflow from JSON
            steps = []
            for step_data in data.get("steps", []):
                step_config = {
                    "id": step_data["id"],
                    "timeout": step_data.get("timeout"),
                    "retries": step_data.get("retries"),
                    "group": step_data.get("group"),
                    "type": step_data["type"],
                    "ref": step_data.get("ref"),
                    "config": step_data.get("config")
                }
                
                # Try to restore function if it's a function step
                if step_data["type"] == "function" and "function_name" in step_data:
                    func_name = step_data["function_name"]
                    if func_name in self._step_registry:
                        step_config["run"] = self._step_registry[func_name]
                
                steps.append(StepConfig(**step_config))
            
            self._workflows[name] = steps
            return steps
        
        return None
    
    def list_workflows(self) -> List[str]:
        """List all workflow names."""
        workflow_files = list(self.storage_path.glob("*.json"))
        return [f.stem for f in workflow_files if f.name != "workflows.db"]
    
    def delete_workflow(self, name: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            name: Workflow name to delete
            
        Returns:
            True if deleted, False if not found
        """
        workflow_file = self.storage_path / f"{name}.json"
        if workflow_file.exists():
            workflow_file.unlink()
            if name in self._workflows:
                del self._workflows[name]
            return True
        return False
    
    def update_workflow(
        self, 
        name: str, 
        steps: List[StepConfig], 
        debug: Optional[bool] = None
    ) -> bool:
        """
        Update an existing workflow.
        
        Args:
            name: Workflow name to update
            steps: New list of step configurations
            debug: Optional debug mode override
            
        Returns:
            True if updated, False if not found
        """
        workflow_file = self.storage_path / f"{name}.json"
        if not workflow_file.exists():
            return False
        
        # Load existing workflow to preserve debug setting if not overridden
        with open(workflow_file, 'r') as f:
            existing_data = json.load(f)
        
        debug_setting = debug if debug is not None else existing_data.get("debug", False)
        
        # Update workflow
        self._workflows[name] = steps
        
        # Save updated workflow
        workflow_data = {
            "name": name,
            "debug": debug_setting,
            "steps": []
        }
        
        for step in steps:
            step_data = {
                "id": step.id,
                "timeout": step.timeout,
                "retries": step.retries,
                "group": step.group,
                "type": step.type,
                "ref": step.ref,
                "config": step.config
            }
            
            # For function steps, try to save function name for restoration
            if step.type == "function" and step.run:
                if hasattr(step.run, '__name__'):
                    step_data["function_name"] = step.run.__name__
                elif hasattr(step.run, '__qualname__'):
                    step_data["function_name"] = step.run.__qualname__
            
            workflow_data["steps"].append(step_data)
        
        with open(workflow_file, 'w') as f:
            json.dump(workflow_data, f, indent=2)
        
        return True

class AsyncWorkflow:
    """
    Async workflow execution system similar to Langbase.
    
    Provides:
    - Multi-step workflow execution
    - Multi-primitive step execution (agent, pipe, memory, thread)
    - Timeout and retry handling
    - Parallel step execution
    - Conditional branching
    - SQLite-based run history
    """
    
    def __init__(self, registry: Optional[WorkflowRegistry] = None):
        """
        Initialize AsyncWorkflow.
        
        Args:
            registry: Workflow registry (creates default if None)
        """
        self.registry = registry or WorkflowRegistry()
        self._runners = {}  # Primitive runners
        self._memory = None  # Memory interface
        self._thread = None  # Thread interface
        self._agent = None  # Agent interface
        self._pipe = None  # Pipe interface
        self._inline_steps: List[StepConfig] = []  # Inline workflow steps
    
    async def run(
        self, 
        name: Optional[str] = None, 
        inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a workflow.
        
        Args:
            name: Workflow name to run (None for inline workflows)
            inputs: Input variables for the workflow
            
        Returns:
            Workflow outputs
            
        Raises:
            ValueError: If workflow not found or no steps defined
        """
        inputs = inputs or {}
        
        # Determine which steps to execute
        if name is None:
            # Inline workflow - use _inline_steps
            if not self._inline_steps:
                raise ValueError("No inline steps defined. Use @step decorator to define steps.")
            workflow_steps = self._inline_steps
            workflow_name = "inline_workflow"
        else:
            # Traditional workflow - get from registry
            workflow_steps = self.registry.get_workflow(name)
            if not workflow_steps:
                raise ValueError(f"Workflow '{name}' not found")
            workflow_name = name
        
        context = WorkflowContext(
            inputs=inputs,
            memory_interface=self._memory,
            thread_interface=self._thread,
            agent_interface=self._agent,
            pipe_interface=self._pipe
        )
        run_id = str(uuid.uuid4())
        
        # Create run record
        run = WorkflowRun(
            id=run_id,
            workflow_name=workflow_name,
            inputs=inputs
        )
        
        try:
            # Execute workflow
            outputs = await self._execute_workflow(workflow_steps, context, run)
            run.outputs = outputs
            run.status = "completed"
            run.completed_at = int(time.time())
            
        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.completed_at = int(time.time())
            raise
        
        finally:
            # Save run to database
            await self._save_run(run)
        
        return run.outputs
    
    def register_runner(self, primitive_type: str, runner: callable) -> None:
        """
        Register a primitive runner.
        
        Args:
            primitive_type: Type of primitive (pipe, agent, tool, memory, thread)
            runner: Async function to run the primitive
        """
        self._runners[primitive_type] = runner
    
    def set_memory(self, memory_interface) -> None:
        """Set memory interface for workflow context."""
        self._memory = memory_interface
    
    def set_thread(self, thread_interface) -> None:
        """Set thread interface for workflow context."""
        self._thread = thread_interface
    
    def set_agent(self, agent_interface) -> None:
        """Set agent interface for workflow context."""
        self._agent = agent_interface
    
    def set_pipe(self, pipe_interface) -> None:
        """Set pipe interface for workflow context."""
        self._pipe = pipe_interface
    
    async def get_run_history(
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
        with sqlite3.connect(self.registry.db_path) as conn:
            query = "SELECT * FROM workflow_runs"
            params = []
            
            if workflow_name:
                query += " WHERE workflow_name = ?"
                params.append(workflow_name)
            
            query += " ORDER BY started_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            runs = []
            for row in cursor.fetchall():
                run_dict = dict(zip(columns, row))
                # Parse JSON fields
                for field in ['inputs', 'outputs', 'steps']:
                    if run_dict[field]:
                        run_dict[field] = json.loads(run_dict[field])
                runs.append(run_dict)
            
            return runs
    
    async def step(self, step_config: Dict[str, Any]) -> Any:
        """
        Execute a single step.
        
        Args:
            step_config: Step configuration dictionary
            
        Returns:
            Step result
        """
        step = StepConfig(**step_config)
        context = WorkflowContext(inputs={})
        run = WorkflowRun(workflow_name="single_step", inputs={})
        return await self._execute_step(step, context, run)

    def _build_dependency_graph(self, workflow: List[StepConfig]):
        """Build dependency graph for workflow steps."""
        graph = {}
        for step in workflow:
            graph[step.id] = []
            if step.after:
                for dep_id in step.after:
                    if dep_id in graph:
                        graph[dep_id].append(step.id)
                    else:
                        # Dependency not found, create empty list
                        graph[dep_id] = [step.id]
        return graph

    def _topological_sort(self, workflow: List[StepConfig]):
        """Topologically sort workflow steps based on dependencies."""
        graph = self._build_dependency_graph(workflow)
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving step: {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            for neighbor in graph.get(node, []):
                visit(neighbor)
            
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for step in workflow:
            if step.id not in visited:
                visit(step.id)
        
        return result

    def _group_steps_ordered(self, workflow: List[StepConfig]):
        """Group steps by execution order while preserving dependencies."""
        # First, get topological order
        ordered_ids = self._topological_sort(workflow)
        
        # Create step lookup
        step_lookup = {step.id: step for step in workflow}
        
        # Group steps by their execution group
        groups = {}
        for step_id in ordered_ids:
            step = step_lookup[step_id]
            group_name = step.group or "sequential"
            
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(step)
        
        return groups

    async def _execute_workflow(
        self, 
        workflow: List[StepConfig], 
        context: WorkflowContext,
        run: WorkflowRun
    ) -> Dict[str, Any]:
        """Execute a workflow with dependency resolution and parallel execution."""
        outputs = {}
        executed_steps = set()
        
        # Group steps by execution order
        step_groups = self._group_steps_ordered(workflow)
        
        # Execute groups in order
        for group_name, group_steps in step_groups.items():
            if group_name == "sequential":
                # Execute steps sequentially
                for step in group_steps:
                    if step.id in executed_steps:
                        continue
                    
                    # Check if all dependencies are satisfied
                    dependencies_satisfied = True
                    if step.after:
                        for dep_id in step.after:
                            if dep_id not in executed_steps:
                                dependencies_satisfied = False
                                break
                    
                    if not dependencies_satisfied:
                        continue
                    
                    # Check condition after dependencies are satisfied
                    if step.condition:
                        ctx_obj = ContextObject(context._data)
                        try:
                            if not step.condition(ctx_obj):
                                logger.debug(f"Skipping step '{step.id}' due to condition")
                                executed_steps.add(step.id)
                                continue
                        except (AttributeError, KeyError):
                            # Missing context key, skip for now
                            logger.debug(f"Deferring step '{step.id}' until all context keys are present for condition.")
                            continue
                    
                    logger.debug(f"Executing sequential step '{step.id}'")
                    result = await self._execute_step(step, context, run)
                    outputs[step.id] = result
                    context.set(step.id, result)
                    executed_steps.add(step.id)
                    logger.debug(f"Set context['{step.id}'] = {str(result)[:100]}...")
            
            else:
                # Execute steps in parallel
                ready_steps = []
                for step in group_steps:
                    if step.id in executed_steps:
                        continue
                    
                    # Check if all dependencies are satisfied
                    dependencies_satisfied = True
                    if step.after:
                        for dep_id in step.after:
                            if dep_id not in executed_steps:
                                dependencies_satisfied = False
                                break
                    
                    if not dependencies_satisfied:
                        continue
                    
                    # Check condition after dependencies are satisfied
                    if step.condition:
                        ctx_obj = ContextObject(context._data)
                        try:
                            if not step.condition(ctx_obj):
                                logger.debug(f"Skipping parallel step '{step.id}' due to condition")
                                executed_steps.add(step.id)
                                continue
                        except (AttributeError, KeyError):
                            # Missing context key, skip for now
                            logger.debug(f"Deferring parallel step '{step.id}' until all context keys are present for condition.")
                            continue
                    
                    ready_steps.append(step)
                
                if ready_steps:
                    # Execute ready steps in parallel
                    tasks = []
                    task_ids = []
                    for step in ready_steps:
                        logger.debug(f"Adding parallel step '{step.id}' to tasks")
                        task = self._execute_step(step, context, run)
                        tasks.append(task)
                        task_ids.append(step.id)
                    
                    logger.debug(f"Executing {len(tasks)} parallel tasks")
                    results = await asyncio.gather(*tasks)
                    for step_id, result in zip(task_ids, results):
                        outputs[step_id] = result
                        context.set(step_id, result)
                        executed_steps.add(step_id)
                        logger.debug(f"Set context['{step_id}'] = {str(result)[:100]}...")
        
        logger.debug(f"Final context keys: {list(context._data.keys())}")
        return outputs
    
    def _group_steps_by_group(self, steps: List[StepConfig]) -> Dict[str, List[StepConfig]]:
        """Group steps by their execution group."""
        groups = {"sequential": []}
        
        for step in steps:
            if step.group:
                if step.group not in groups:
                    groups[step.group] = []
                groups[step.group].append(step)
            else:
                groups["sequential"].append(step)
        
        return groups
    
    async def _execute_step(
        self, 
        step: StepConfig, 
        context: WorkflowContext,
        run: WorkflowRun
    ) -> Any:
        """Execute a single step with retry and timeout logic."""
        step_record = {
            "id": step.id,
            "type": step.type,
            "started_at": int(time.time()),
            "status": "running"
        }
        run.steps.append(step_record)
        
        try:
            if step.timeout:
                result = await asyncio.wait_for(
                    self._execute_step_with_retries(step, context),
                    timeout=step.timeout / 1000.0
                )
            else:
                result = await self._execute_step_with_retries(step, context)
            
            step_record["status"] = "completed"
            step_record["completed_at"] = int(time.time())
            return result
            
        except Exception as e:
            step_record["status"] = "failed"
            step_record["error"] = str(e)
            step_record["completed_at"] = int(time.time())
            raise
    
    async def _execute_step_with_retries(
        self, 
        step: StepConfig, 
        context: WorkflowContext
    ) -> Any:
        """Execute step with retry logic."""
        if not step.retries:
            return await self._execute_step_inner(step, context)
        
        limit = step.retries.get("limit", 3)
        delay = step.retries.get("delay", 1000) / 1000.0
        backoff = step.retries.get("backoff", 2)
        
        last_exception = None
        
        for attempt in range(limit + 1):
            try:
                return await self._execute_step_inner(step, context)
            except Exception as e:
                last_exception = e
                if attempt < limit:
                    await asyncio.sleep(delay * (backoff ** attempt))
        
        raise last_exception
    
    async def _execute_step_inner(
        self, 
        step: StepConfig, 
        context: WorkflowContext
    ) -> Any:
        """Execute the actual step logic."""
        if step.type == "function":
            if not step.run:
                raise ValueError(f"Function step '{step.id}' has no run function")
            result = step.run(context.to_dict())
            if inspect.iscoroutine(result):
                result = await result
            return result
        
        elif step.type == "pipe":
            if not step.ref:
                raise ValueError(f"Pipe step '{step.id}' has no reference")
            if "pipe" not in self._runners:
                raise NotImplementedError("Pipe runner not registered")
            return await self._runners["pipe"](step.ref, context.to_dict())
        
        elif step.type == "tool":
            if not step.ref:
                raise ValueError(f"Tool step '{step.id}' has no reference")
            if "tool" not in self._runners:
                raise NotImplementedError("Tool runner not registered")
            return await self._runners["tool"](step.ref, context.to_dict())
        
        elif step.type == "agent":
            if not step.ref:
                raise ValueError(f"Agent step '{step.id}' has no reference")
            if "agent" not in self._runners:
                raise NotImplementedError("Agent runner not registered")
            return await self._runners["agent"](step.ref, context.to_dict())
        
        elif step.type == "memory":
            if not step.ref:
                raise ValueError(f"Memory step '{step.id}' has no reference")
            if "memory" not in self._runners:
                raise NotImplementedError("Memory runner not registered")
            return await self._runners["memory"](step.ref, context.to_dict())
        
        elif step.type == "thread":
            if not step.ref:
                raise ValueError(f"Thread step '{step.id}' has no reference")
            if "thread" not in self._runners:
                raise NotImplementedError("Thread runner not registered")
            return await self._runners["thread"](step.ref, context.to_dict())
        
        else:
            raise ValueError(f"Unknown step type: {step.type}")
    
    async def _save_run(self, run: WorkflowRun) -> None:
        """Save run to SQLite database."""
        with sqlite3.connect(self.registry.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflow_runs 
                (id, workflow_name, status, inputs, outputs, error, started_at, completed_at, steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.id,
                run.workflow_name,
                run.status,
                json.dumps(run.inputs),
                json.dumps(run.outputs),
                run.error,
                run.started_at,
                run.completed_at,
                json.dumps(run.steps)
            ))
            conn.commit() 