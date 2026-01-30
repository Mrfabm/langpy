"""
Core Workflow Engine with Enhanced Features

This module provides the core workflow execution engine with full Langbase parity
including await-able builder pattern, enhanced error handling, retry logic,
secret scoping, thread handoff, and parallel execution.
"""

import asyncio
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Union
from .async_workflow import StepConfig, WorkflowContext
from .exceptions import (
    WorkflowError, TimeoutError, RetryExhaustedError, StepError,
    SecretError, PrimitiveError, ContextError, DependencyError
)
from .retry import RetryConfig, RetryEngine
from .logging import WorkflowLogger, get_workflow_logger
from .run_registry import WorkflowRun, RunRegistry, RunFilter
from .template_engine import render_step_config


class WorkflowEngine:
    """Core workflow engine with enhanced features."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = WorkflowLogger(debug=debug)
        self.run_registry = RunRegistry()
        
        # Primitive runners
        self._runners: Dict[str, Callable] = {}
        
        # Secret managers
        self._secret_managers: Dict[str, Callable] = {
            'env': os.getenv
        }
        
        # Current execution state
        self._current_run: Optional[WorkflowRun] = None
        self._current_context: Optional[WorkflowContext] = None
        self._steps: List[StepConfig] = []
    
    def register_runner(self, primitive_type: str, runner: Callable) -> None:
        """Register a primitive runner."""
        self._runners[primitive_type] = runner
    
    def register_secret_manager(self, name: str, manager: Callable) -> None:
        """Register a secret manager."""
        self._secret_managers[name] = manager
    
    async def step(self, **kwargs) -> Any:
        """
        Await-able builder pattern for step execution.
        This matches Langbase's `await workflow.step(**StepConfig)` pattern.
        """
        step_config = StepConfig(**kwargs)
        
        # If we're in a workflow context, collect the step
        if self._current_context is not None:
            self._steps.append(step_config)
            return f"{{{{ {step_config.id} }}}}"  # Placeholder for template resolution
        
        # Otherwise, execute the step directly
        return await self._execute_step(step_config, WorkflowContext({}))
    
    async def run(
        self,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        steps: Optional[List[StepConfig]] = None
    ) -> Dict[str, Any]:
        """Run a workflow with enhanced features."""
        if name is None:
            name = f"workflow_{int(time.time())}"
        
        if inputs is None:
            inputs = {}
        
        if steps is None:
            steps = []
        
        # Create run record
        run = WorkflowRun(
            id=str(uuid.uuid4()),
            workflow_name=name,
            status="running",
            inputs=inputs,
            outputs={},
            error=None,
            started_at=int(time.time()),
            completed_at=None,
            duration_ms=None,
            steps=[],
            context={},
            metadata={}
        )
        
        # Set current run
        self._current_run = run
        self._current_context = WorkflowContext(inputs)
        
        try:
            # Log workflow start
            start_time = time.time()
            self.logger.print_workflow_banner(name, run.id, inputs)
            
            # Execute workflow
            outputs = await self._execute_workflow(steps, self._current_context, run)
            
            # Mark run as completed
            run.complete(outputs)
            
            # Log workflow completion
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.print_workflow_completion(name, run.id, outputs, elapsed_ms)
            
            return outputs
            
        except Exception as e:
            # Mark run as failed
            run.fail(str(e))
            
            # Log workflow error
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.print_workflow_error(name, run.id, e, elapsed_ms)
            
            raise
        
        finally:
            # Save run to registry
            self.run_registry.save_run(run)
            
            # Clean up
            self._current_run = None
            self._current_context = None
    
    def _render_step_config(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Render step configuration templates."""
        try:
            # Use the template engine to render the config
            return render_step_config(config, context.to_dict())
        except Exception as e:
            # Fall back to original config if template rendering fails
            if self.debug:
                print(f"Template rendering failed: {e}")
            return config
    
    async def _execute_workflow(self, 
                               steps: List[StepConfig],
                               context: WorkflowContext,
                               run: WorkflowRun) -> Dict[str, Any]:
        """Execute a complete workflow."""
        outputs = {}
        
        # Group steps by execution order
        step_groups = self._group_steps_by_execution_order(steps)
        executed_steps: Set[str] = set()
        
        # Execute step groups
        for group_type, group_steps in step_groups.items():
            if group_type == "sequential":
                # Execute sequentially
                for step in group_steps:
                    if step.id in executed_steps:
                        continue
                    if not self._check_dependencies(step, executed_steps):
                        continue
                    if not self._check_condition(step, context):
                        executed_steps.add(step.id)
                        continue
                    
                    result = await self._execute_step(step, context)
                    outputs[step.id] = result
                    context.set(step.id, result)
                    executed_steps.add(step.id)
            
            else:
                # Execute in parallel
                await self._execute_parallel_group(group_type, group_steps, context, outputs, executed_steps)
        
        return outputs
    
    async def _execute_parallel_group(self,
                                    group_name: str,
                                    steps: List[StepConfig],
                                    context: WorkflowContext,
                                    outputs: Dict[str, Any],
                                    executed_steps: set) -> None:
        """Execute a group of steps in parallel."""
        # Filter ready steps
        ready_steps = []
        for step in steps:
            if step.id in executed_steps:
                continue
            if not self._check_dependencies(step, executed_steps):
                continue
            if not self._check_condition(step, context):
                executed_steps.add(step.id)
                continue
            ready_steps.append(step)
        
        if not ready_steps:
            return
        
        # Log parallel execution start
        step_ids = [step.id for step in ready_steps]
        self.logger.log_parallel_start(group_name, step_ids)
        
        # Execute steps in parallel
        start_time = time.time()
        
        async def execute_single_step(step: StepConfig) -> tuple[str, Any]:
            result = await self._execute_step(step, context)
            return step.id, result
        
        # Run all steps in parallel
        results = await asyncio.gather(
            *[execute_single_step(step) for step in ready_steps],
            return_exceptions=True
        )
        
        # Process results
        for i, result in enumerate(results):
            step = ready_steps[i]
            if isinstance(result, Exception):
                # Handle step failure
                raise result
            else:
                step_id, step_result = result
                outputs[step_id] = step_result
                context.set(step_id, step_result)
                executed_steps.add(step_id)
        
        # Log parallel execution completion
        elapsed_ms = int((time.time() - start_time) * 1000)
        self.logger.log_parallel_complete(group_name, elapsed_ms)
    
    async def _execute_step(self, step: StepConfig, context: WorkflowContext) -> Any:
        """Execute a single step with timeout and retry."""
        start_time = time.time()
        
        try:
            # Prepare secrets
            self._prepare_secrets(step, context)
            
            # Log step start
            self.logger.log_step_start(step.id, step.type, step.config)
            
            # Execute with timeout and retry
            if step.timeout:
                result = await asyncio.wait_for(
                    self._execute_step_with_retry(step, context),
                    timeout=step.timeout / 1000.0
                )
            else:
                result = await self._execute_step_with_retry(step, context)
            
            # Check for thread handoff
            self._check_thread_handoff(step, result, context)
            
            # Log step success
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.log_step_success(step.id, result)
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.log_step_failure(step.id, TimeoutError(step.id, step.timeout, elapsed_ms))
            raise TimeoutError(step.id, step.timeout, elapsed_ms)
        
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.log_step_failure(step.id, e)
            raise
    
    async def _execute_step_with_retry(self, step: StepConfig, context: WorkflowContext) -> Any:
        """Execute step with retry logic."""
        # Parse retry configuration
        retry_config = self._parse_retry_config(step.retries)
        
        if retry_config is None:
            # No retry, execute once
            return await self._execute_step_inner(step, context)
        
        # Execute with retry
        return await RetryEngine.execute_with_retry(
            lambda: self._execute_step_inner(step, context),
            retry_config,
            step.id,
            self.logger.logger
        )
    
    async def _execute_step_inner(self, step: StepConfig, context: WorkflowContext) -> Any:
        """Execute the actual step logic."""
        if step.type == "function":
            if step.run is None:
                raise StepError(step.id, ValueError("Function step requires 'run' parameter"))
            
            # Execute function
            if asyncio.iscoroutinefunction(step.run):
                return await step.run(context)
            else:
                return step.run(context)
        
        elif step.type in ["pipe", "agent", "tool", "memory", "thread"]:
            # Execute primitive
            if step.type not in self._runners:
                raise PrimitiveError(step.id, step.type, step.ref, ValueError(f"No runner registered for {step.type}"))
            
            runner = self._runners[step.type]
            
            try:
                # Render config templates before execution
                rendered_config = self._render_step_config(step.config or {}, context)
                return await runner(step.ref, context.to_dict(), rendered_config)
            except Exception as e:
                raise PrimitiveError(step.id, step.type, step.ref, e)
        
        else:
            raise StepError(step.id, ValueError(f"Unknown step type: {step.type}"))
    
    def _prepare_secrets(self, step: StepConfig, context: WorkflowContext) -> None:
        """Prepare secrets for step execution."""
        if not step.use_secrets:
            return
        
        secrets = {}
        for secret_name in step.use_secrets:
            # Try to get secret from environment
            secret_value = self._secret_managers['env'](secret_name)
            if secret_value is not None:
                secrets[secret_name] = secret_value
                self.logger.log_secret_access(step.id, secret_name)
            else:
                raise SecretError(step.id, secret_name, "Secret not found")
        
        context.set_secrets(secrets)
    
    def _check_thread_handoff(self, step: StepConfig, result: Any, context: WorkflowContext) -> None:
        """Check for thread handoff in step result."""
        if isinstance(result, dict) and 'lb-thread-id' in result:
            thread_id = result['lb-thread-id']
            context.thread_id = thread_id
            self.logger.log_thread_handoff(step.id, thread_id)
    
    def _check_dependencies(self, step: StepConfig, executed_steps: Set[str]) -> bool:
        """Check if step dependencies are met."""
        if not step.after:
            return True
        
        missing_deps = [dep for dep in step.after if dep not in executed_steps]
        if missing_deps:
            return False
        
        return True
    
    def _check_condition(self, step: StepConfig, context: WorkflowContext) -> bool:
        """Check if step condition is met."""
        if step.condition is None:
            return True
        
        try:
            return bool(step.condition(context.to_dict()))
        except Exception:
            return False
    
    def _group_steps_by_execution_order(self, steps: List[StepConfig]) -> Dict[str, List[StepConfig]]:
        """Group steps by execution order."""
        groups = {"sequential": []}
        
        for step in steps:
            if step.group:
                for group_name in step.group:
                    if group_name not in groups:
                        groups[group_name] = []
                    groups[group_name].append(step)
            else:
                groups["sequential"].append(step)
        
        return groups
    
    def _parse_retry_config(self, retries: Optional[Union[Dict[str, Any], RetryConfig]]) -> Optional[RetryConfig]:
        """Parse retry configuration."""
        if retries is None:
            return None
        
        if isinstance(retries, RetryConfig):
            return retries
        
        if isinstance(retries, dict):
            return RetryConfig(**retries)
        
        return None
    
    def list_run_history(self,
                        workflow_name: Optional[str] = None,
                        status: Optional[str] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List run history with filtering."""
        filter_obj = RunFilter(
            workflow_name=workflow_name,
            status=status,
            limit=limit
        )
        
        runs = self.run_registry.list_runs(filter_obj)
        return [run.to_dict() for run in runs]


# Global workflow engine instance
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine(debug: bool = False) -> WorkflowEngine:
    """Get the global workflow engine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine(debug=debug)
    return _workflow_engine


def set_workflow_engine(engine: WorkflowEngine) -> None:
    """Set the global workflow engine instance."""
    global _workflow_engine
    _workflow_engine = engine 