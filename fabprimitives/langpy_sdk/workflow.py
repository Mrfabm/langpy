"""
LangPy Workflow - Clean SDK wrapper for multi-step orchestration.

Simple, intuitive interface for creating multi-step AI workflows.

Supports both the original API and the new composable architecture:

Original API:
    from langpy_sdk import Workflow, Step

    workflow = Workflow("research-pipeline")
    workflow.add_step(Step(name="search", handler=search_web))
    workflow.add_step(Step(name="summarize", handler=summarize_results, depends_on=["search"]))
    result = await workflow.run({"query": "AI trends 2024"})

New Composable API:
    from langpy.core import Context, pipeline
    from langpy_sdk import Workflow

    # Convert workflow to a primitive
    workflow_primitive = workflow.as_primitive()
    result = await workflow_primitive.process(Context(query="AI trends 2024"))

    # Or use workflow to wrap a pipeline
    pipeline = memory | pipe
    workflow = Workflow.from_pipeline("rag-workflow", pipeline)
"""

from __future__ import annotations
import asyncio
import time
import uuid
from typing import Optional, List, Dict, Any, Callable, Union, Awaitable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

# Import core types for the new architecture
try:
    from langpy.core.context import Context
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode
    from langpy.core.primitive import BasePrimitive, IPrimitive
    from langpy.core.pipeline import Pipeline
    _NEW_ARCH_AVAILABLE = True
except ImportError:
    _NEW_ARCH_AVAILABLE = False
    Context = Any
    Result = Any
    IPrimitive = Any


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """
    A single step in a workflow.

    Attributes:
        name: Unique step name
        handler: Function to execute (sync or async)
        depends_on: List of step names this step depends on
        retry: Number of retries on failure (default: 0)
        timeout: Timeout in seconds (optional)
        condition: Optional function that returns True to run step

    Example:
        Step(
            name="fetch_data",
            handler=fetch_from_api,
            retry=3,
            timeout=30
        )
    """
    name: str
    handler: Callable[[Dict[str, Any]], Any]
    depends_on: List[str] = field(default_factory=list)
    retry: int = 0
    timeout: Optional[float] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass
class StepResult:
    """
    Result of a step execution.

    Attributes:
        name: Step name
        status: Execution status
        output: Step output (if successful)
        error: Error message (if failed)
        duration: Execution time in seconds
    """
    name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class WorkflowResult:
    """
    Result of a workflow execution.

    Attributes:
        workflow_id: Unique workflow run ID
        status: Overall status
        steps: Results for each step
        outputs: Final outputs from all steps
        duration: Total execution time
    """
    workflow_id: str
    status: str
    steps: Dict[str, StepResult]
    outputs: Dict[str, Any]
    duration: float

    def get(self, step_name: str) -> Any:
        """Get output from a specific step."""
        if step_name in self.steps:
            return self.steps[step_name].output
        return None

    @property
    def success(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == "completed"


class Workflow:
    """
    Clean, simple Workflow interface for multi-step orchestration.

    Workflows coordinate multiple steps with dependencies,
    parallel execution, and error handling.

    Args:
        name: Workflow name
        max_parallel: Maximum parallel steps (default: 5)
        persist: Enable persistence and run history (default: False)
        storage_path: Custom storage path for persistence

    Example:
        workflow = Workflow("data-pipeline")

        # Add steps
        workflow.add_step(Step(
            name="fetch",
            handler=fetch_data
        ))

        workflow.add_step(Step(
            name="process",
            handler=process_data,
            depends_on=["fetch"]
        ))

        workflow.add_step(Step(
            name="save",
            handler=save_results,
            depends_on=["process"]
        ))

        # Run
        result = await workflow.run({"url": "https://api.example.com"})
        print(result.outputs)

        # With persistence (uses internal workflow engine)
        workflow = Workflow("data-pipeline", persist=True)
        result = await workflow.run(inputs)
        history = await workflow.get_run_history()
    """

    def __init__(
        self,
        name: str,
        max_parallel: int = 5,
        persist: bool = False,
        storage_path: Optional[str] = None
    ):
        self.name = name
        self.max_parallel = max_parallel
        self._steps: Dict[str, Step] = {}
        self._order: List[str] = []
        self._persist = persist
        self._storage_path = storage_path
        self._internal_workflow = None
        self._internal_registry = None

    def add_step(self, step: Step) -> "Workflow":
        """
        Add a step to the workflow.

        Args:
            step: Step to add

        Returns:
            Self for chaining

        Example:
            workflow.add_step(Step("a", handler_a)).add_step(Step("b", handler_b))
        """
        self._steps[step.name] = step
        if step.name not in self._order:
            self._order.append(step.name)
        return self

    def step(
        self,
        name: str,
        depends_on: Optional[List[str]] = None,
        retry: int = 0,
        timeout: Optional[float] = None
    ) -> Callable:
        """
        Decorator to add a step.

        Args:
            name: Step name
            depends_on: Dependencies
            retry: Retry count
            timeout: Timeout in seconds

        Example:
            @workflow.step("fetch", retry=3)
            async def fetch(inputs):
                return await fetch_data(inputs["url"])

            @workflow.step("process", depends_on=["fetch"])
            def process(inputs):
                return transform(inputs["fetch"])
        """
        def decorator(func: Callable) -> Callable:
            self.add_step(Step(
                name=name,
                handler=func,
                depends_on=depends_on or [],
                retry=retry,
                timeout=timeout
            ))
            return func
        return decorator

    async def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            inputs: Initial inputs available to all steps
            parallel: Run independent steps in parallel (default: True)

        Returns:
            WorkflowResult with all step outputs

        Example:
            result = await workflow.run({"api_key": "xxx", "query": "test"})

            if result.success:
                print(result.outputs["final_step"])
            else:
                for name, step in result.steps.items():
                    if step.status == StepStatus.FAILED:
                        print(f"{name} failed: {step.error}")
        """
        # If persistence is enabled, delegate to internal workflow engine
        if self._persist:
            return await self._run_with_persistence(inputs or {})

        workflow_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Initialize context with inputs
        context: Dict[str, Any] = dict(inputs or {})
        results: Dict[str, StepResult] = {}
        completed: set = set()

        # Build dependency graph
        def get_ready_steps() -> List[str]:
            """Get steps whose dependencies are satisfied."""
            ready = []
            for name in self._order:
                if name in completed:
                    continue
                step = self._steps[name]
                if all(dep in completed for dep in step.depends_on):
                    ready.append(name)
            return ready

        # Execute workflow
        while len(completed) < len(self._steps):
            ready = get_ready_steps()

            if not ready:
                # No steps ready but not all complete = deadlock
                remaining = set(self._steps.keys()) - completed
                for name in remaining:
                    results[name] = StepResult(
                        name=name,
                        status=StepStatus.SKIPPED,
                        error="Dependency not satisfied"
                    )
                    completed.add(name)
                break

            if parallel and len(ready) > 1:
                # Run ready steps in parallel
                tasks = []
                for name in ready[:self.max_parallel]:
                    tasks.append(self._run_step(name, context))

                step_results = await asyncio.gather(*tasks, return_exceptions=True)

                for name, result in zip(ready[:self.max_parallel], step_results):
                    if isinstance(result, Exception):
                        results[name] = StepResult(
                            name=name,
                            status=StepStatus.FAILED,
                            error=str(result)
                        )
                    else:
                        results[name] = result
                        if result.status == StepStatus.COMPLETED:
                            context[name] = result.output

                    completed.add(name)
            else:
                # Run one step at a time
                name = ready[0]
                result = await self._run_step(name, context)
                results[name] = result

                if result.status == StepStatus.COMPLETED:
                    context[name] = result.output

                completed.add(name)

        # Determine overall status
        failed = any(r.status == StepStatus.FAILED for r in results.values())
        status = "failed" if failed else "completed"

        # Build outputs (only from completed steps)
        outputs = {
            name: result.output
            for name, result in results.items()
            if result.status == StepStatus.COMPLETED
        }

        return WorkflowResult(
            workflow_id=workflow_id,
            status=status,
            steps=results,
            outputs=outputs,
            duration=time.time() - start_time
        )

    async def _run_step(self, name: str, context: Dict[str, Any]) -> StepResult:
        """Execute a single step with retries."""
        step = self._steps[name]
        start_time = time.time()

        # Check condition
        if step.condition and not step.condition(context):
            return StepResult(
                name=name,
                status=StepStatus.SKIPPED,
                duration=time.time() - start_time
            )

        last_error = None
        attempts = step.retry + 1

        for attempt in range(attempts):
            try:
                # Run handler (sync or async)
                if asyncio.iscoroutinefunction(step.handler):
                    if step.timeout:
                        output = await asyncio.wait_for(
                            step.handler(context),
                            timeout=step.timeout
                        )
                    else:
                        output = await step.handler(context)
                else:
                    output = step.handler(context)

                return StepResult(
                    name=name,
                    status=StepStatus.COMPLETED,
                    output=output,
                    duration=time.time() - start_time
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout}s"
            except Exception as e:
                last_error = str(e)

            # Wait before retry
            if attempt < attempts - 1:
                await asyncio.sleep(0.5 * (attempt + 1))

        return StepResult(
            name=name,
            status=StepStatus.FAILED,
            error=last_error,
            duration=time.time() - start_time
        )

    async def _run_with_persistence(self, inputs: Dict[str, Any]) -> WorkflowResult:
        """Run workflow using internal engine with persistence."""
        from workflow.async_workflow import AsyncWorkflow, WorkflowRegistry, StepConfig

        # Initialize internal workflow on first use
        if self._internal_registry is None:
            self._internal_registry = WorkflowRegistry(storage_path=self._storage_path)
            self._internal_workflow = AsyncWorkflow(self._internal_registry)

        # Convert SDK Steps to internal StepConfig
        step_configs = []
        for name in self._order:
            step = self._steps[name]
            step_config = StepConfig(
                id=step.name,
                run=step.handler,
                timeout=int(step.timeout * 1000) if step.timeout else None,
                retries={"limit": step.retry, "delay": 500, "backoff": 2} if step.retry > 0 else None,
                after=step.depends_on if step.depends_on else None,
                condition=step.condition,
                type="function"
            )
            step_configs.append(step_config)

        # Register workflow
        self._internal_registry.create(self.name, step_configs)

        # Run workflow
        start_time = time.time()
        workflow_id = str(uuid.uuid4())[:8]

        try:
            outputs = await self._internal_workflow.run(self.name, inputs)

            # Convert to WorkflowResult
            step_results = {}
            for name in self._order:
                if name in outputs:
                    step_results[name] = StepResult(
                        name=name,
                        status=StepStatus.COMPLETED,
                        output=outputs[name]
                    )

            return WorkflowResult(
                workflow_id=workflow_id,
                status="completed",
                steps=step_results,
                outputs=outputs,
                duration=time.time() - start_time
            )

        except Exception as e:
            return WorkflowResult(
                workflow_id=workflow_id,
                status="failed",
                steps={},
                outputs={},
                duration=time.time() - start_time
            )

    async def get_run_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get workflow run history (requires persist=True).

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run records with id, status, inputs, outputs, timestamps

        Example:
            workflow = Workflow("my-workflow", persist=True)
            await workflow.run({"query": "test"})
            history = await workflow.get_run_history(limit=10)
            for run in history:
                print(f"{run['id']}: {run['status']}")
        """
        if not self._persist:
            raise RuntimeError("Run history requires persist=True")

        if self._internal_workflow is None:
            from workflow.async_workflow import AsyncWorkflow, WorkflowRegistry
            self._internal_registry = WorkflowRegistry(storage_path=self._storage_path)
            self._internal_workflow = AsyncWorkflow(self._internal_registry)

        return await self._internal_workflow.get_run_history(
            workflow_name=self.name,
            limit=limit
        )

    def visualize(self) -> str:
        """
        Get a text visualization of the workflow.

        Returns:
            ASCII representation of the workflow

        Example:
            print(workflow.visualize())
        """
        lines = [f"Workflow: {self.name}", "=" * 40]

        for name in self._order:
            step = self._steps[name]
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            retry = f" [retry: {step.retry}]" if step.retry else ""
            lines.append(f"  - {name}{deps}{retry}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Workflow(name='{self.name}', steps={len(self._steps)})"

    # ========================================================================
    # New Composable Architecture Methods
    # ========================================================================

    def as_primitive(self) -> "WorkflowPrimitive":
        """
        Convert this Workflow to an IPrimitive for use in pipelines.

        Returns:
            WorkflowPrimitive that can be composed with other primitives

        Example:
            workflow = Workflow("my-workflow")
            workflow.add_step(Step("process", handler=process_func))

            primitive = workflow.as_primitive()
            result = await primitive.process(Context(query="Hello"))
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Composable architecture requires langpy.core")
        return WorkflowPrimitive(self)

    @classmethod
    def from_pipeline(
        cls,
        name: str,
        pipeline: "IPrimitive",
        step_name: str = "pipeline"
    ) -> "Workflow":
        """
        Create a Workflow from an existing pipeline primitive.

        This allows wrapping composable pipelines in Workflow's
        retry/timeout infrastructure.

        Args:
            name: Workflow name
            pipeline: Pipeline primitive to wrap
            step_name: Name for the pipeline step

        Returns:
            Workflow with the pipeline as a single step

        Example:
            from langpy.core import pipeline

            p = memory | pipe | validator
            workflow = Workflow.from_pipeline("rag-workflow", p)

            # Now you can add retry, timeout, etc.
            result = await workflow.run({"query": "Hello"})
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Composable architecture requires langpy.core")

        workflow = cls(name)

        async def pipeline_handler(inputs: Dict[str, Any]) -> Any:
            from langpy.core.context import Context

            # Create context from inputs
            ctx = Context(
                query=inputs.get("query"),
                variables=inputs
            )

            # Run the pipeline
            result = await pipeline.process(ctx)

            if result.is_success():
                result_ctx = result.unwrap()
                return {
                    "response": result_ctx.response,
                    "documents": [d.content for d in result_ctx.documents],
                    "variables": result_ctx.variables,
                    "cost": result_ctx.cost.total_cost,
                    "tokens": result_ctx.token_usage.total_tokens
                }
            else:
                raise Exception(str(result.error))

        workflow.add_step(Step(
            name=step_name,
            handler=pipeline_handler
        ))

        return workflow

    def __or__(self, other: "IPrimitive") -> "IPrimitive":
        """
        Sequential composition with the | operator.

        Converts this Workflow to a primitive and composes it.

        Example:
            result_pipeline = workflow | post_processor
        """
        if not _NEW_ARCH_AVAILABLE:
            raise ImportError("Pipeline composition requires langpy.core")

        from langpy.core.pipeline import Pipeline
        return Pipeline([self.as_primitive(), other])


# ============================================================================
# Workflow Primitive Class (for composable architecture)
# ============================================================================

if _NEW_ARCH_AVAILABLE:
    from langpy.core.primitive import BasePrimitive
    from langpy.core.context import Context
    from langpy.core.result import Result, Success, Failure, PrimitiveError, ErrorCode

    class WorkflowPrimitive(BasePrimitive):
        """
        Primitive that wraps a Workflow for use in composable pipelines.

        This allows Workflows to be used with the | operator.
        """

        def __init__(self, workflow: "Workflow"):
            super().__init__(f"Workflow({workflow.name})")
            self._workflow = workflow

        async def _process(self, ctx: Context) -> Result[Context]:
            """Run the workflow with context data."""
            try:
                # Build inputs from context
                inputs = dict(ctx.variables)
                inputs["query"] = ctx.query
                inputs["documents"] = [d.content for d in ctx.documents]

                # Run the workflow
                result = await self._workflow.run(inputs)

                if result.success:
                    # Build result context
                    result_ctx = ctx

                    # If any step returned a "response", use it
                    for step_name, output in result.outputs.items():
                        if isinstance(output, dict) and "response" in output:
                            result_ctx = result_ctx.with_response(output["response"])
                        elif isinstance(output, str):
                            result_ctx = result_ctx.with_response(output)

                    # Store all outputs in variables
                    for step_name, output in result.outputs.items():
                        result_ctx = result_ctx.set(f"workflow_{step_name}", output)

                    result_ctx = result_ctx.set("_workflow_duration", result.duration)
                    result_ctx = result_ctx.set("_workflow_id", result.workflow_id)

                    return Success(result_ctx)
                else:
                    # Workflow failed - collect errors
                    errors = []
                    for step_name, step_result in result.steps.items():
                        if step_result.status == StepStatus.FAILED:
                            errors.append(f"{step_name}: {step_result.error}")

                    return Failure(PrimitiveError(
                        code=ErrorCode.PIPELINE_ERROR,
                        message=f"Workflow failed: {'; '.join(errors)}",
                        primitive=self._name
                    ))

            except Exception as e:
                return Failure(PrimitiveError(
                    code=ErrorCode.UNKNOWN,
                    message=str(e),
                    primitive=self._name,
                    cause=e
                ))

else:
    class WorkflowPrimitive:
        def __init__(self, *args, **kwargs):
            raise ImportError("Composable architecture requires langpy.core")
