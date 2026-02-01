"""
Workflow Primitive - Langbase-compatible Workflow Orchestration API.

The Workflow primitive orchestrates multi-step agent pipelines
with dependencies, retries, and conditional execution.

Usage:
    # Build a workflow with primitives as steps
    wf = lb.workflow(name="rag-agent")
    wf.step(id="retrieve", primitive=lb.memory)
    wf.step(id="generate", primitive=lb.agent, after=["retrieve"])
    result = await wf.run(query="What is Python?")

    # Workflow is also a primitive - can be nested
    pipeline = workflow1 | workflow2
"""

from __future__ import annotations
import asyncio
import uuid
import time
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field

from langpy.core.primitive import BasePrimitive, WorkflowResponse, IPrimitive
from langpy.core.result import Success, Failure, PrimitiveError, ErrorCode

if TYPE_CHECKING:
    from langpy.core.context import Context
    from langpy.core.result import Result


@dataclass
class StepConfig:
    """Configuration for a workflow step."""
    id: str
    primitive: Optional[IPrimitive] = None
    run: Optional[Callable] = None  # Alternative: raw function
    after: List[str] = field(default_factory=list)
    when: Optional[Callable] = None  # Condition: (context) -> bool
    timeout: Optional[float] = None  # Seconds
    retries: int = 0
    retry_delay: float = 1.0

    def __post_init__(self):
        if not self.primitive and not self.run:
            raise ValueError(f"Step '{self.id}' requires 'primitive' or 'run'")


@dataclass
class StepResult:
    """Result of a step execution."""
    id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    retries: int = 0


class Workflow(BasePrimitive):
    """
    Workflow primitive - Multi-step orchestration with primitives.

    Workflows connect primitives with:
    - Dependency ordering (after=[...])
    - Conditional execution (when=lambda ctx: ...)
    - Timeout and retries
    - Parallel execution for independent steps

    Example:
        from langpy import Langpy

        lb = Langpy()

        # Create workflow
        wf = lb.workflow(name="rag-agent")

        # Add steps using primitives
        wf.step(id="load_thread", primitive=lb.thread.loader)
        wf.step(id="retrieve", primitive=lb.memory, after=["load_thread"])
        wf.step(id="generate", primitive=lb.agent, after=["retrieve"])
        wf.step(id="save_thread", primitive=lb.thread.saver, after=["generate"])

        # Run
        result = await wf.run(query="What is Python?", thread_id="abc123")

        # Or use workflow in a pipeline (workflow IS a primitive)
        pipeline = preprocess | wf | postprocess
    """

    def __init__(
        self,
        client: Any = None,
        name: str = "workflow",
        debug: bool = False
    ):
        super().__init__(name=name, client=client)
        self._steps: List[StepConfig] = []
        self._debug = debug

    def step(
        self,
        id: str,
        primitive: IPrimitive = None,
        run: Callable = None,
        after: List[str] = None,
        when: Callable = None,
        timeout: float = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        **kwargs
    ) -> "Workflow":
        """
        Add a step to the workflow.

        Args:
            id: Unique step identifier
            primitive: Primitive to execute (preferred)
            run: Alternative: raw async function
            after: Step IDs that must complete first
            when: Condition function (ctx) -> bool
            timeout: Step timeout in seconds
            retries: Number of retry attempts
            retry_delay: Delay between retries

        Returns:
            Self for chaining
        """
        # If run is provided but not primitive, and run is a primitive, use it
        if run and isinstance(run, IPrimitive):
            primitive = run
            run = None

        config = StepConfig(
            id=id,
            primitive=primitive,
            run=run,
            after=after or [],
            when=when,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay
        )

        self._steps.append(config)
        return self

    def _topological_sort(self) -> List[List[StepConfig]]:
        """Sort steps into execution layers respecting dependencies."""
        # Build dependency graph
        step_map = {s.id: s for s in self._steps}
        in_degree = {s.id: len(s.after) for s in self._steps}
        dependents = {s.id: [] for s in self._steps}

        for s in self._steps:
            for dep in s.after:
                if dep in dependents:
                    dependents[dep].append(s.id)

        # Find layers using Kahn's algorithm
        layers = []
        remaining = set(step_map.keys())

        while remaining:
            # Find steps with no remaining dependencies
            ready = [sid for sid in remaining if in_degree[sid] == 0]

            if not ready:
                # Circular dependency
                raise ValueError(f"Circular dependency detected in workflow. Remaining: {remaining}")

            layers.append([step_map[sid] for sid in ready])

            # Update degrees
            for sid in ready:
                remaining.remove(sid)
                for dep_id in dependents[sid]:
                    in_degree[dep_id] -= 1

        return layers

    async def _execute_step(
        self,
        step: StepConfig,
        context: "Context",
        results: Dict[str, StepResult]
    ) -> StepResult:
        """Execute a single step with retries and timeout."""
        from langpy.core.context import Context

        start_time = time.time()
        attempts = 0
        last_error = None

        while attempts <= step.retries:
            attempts += 1

            try:
                # Check condition
                if step.when and not step.when(context):
                    return StepResult(
                        id=step.id,
                        success=True,
                        output=None,
                        duration_ms=(time.time() - start_time) * 1000
                    )

                # Prepare context with previous results
                step_ctx = context.clone()
                step_ctx = step_ctx.set("_step_id", step.id)
                step_ctx = step_ctx.set("_step_results", results)

                # Add outputs from dependencies
                for dep_id in step.after:
                    if dep_id in results and results[dep_id].success:
                        step_ctx = step_ctx.set(dep_id, results[dep_id].output)

                # Execute
                if step.primitive:
                    # Use primitive's process() method
                    if step.timeout:
                        result = await asyncio.wait_for(
                            step.primitive.process(step_ctx),
                            timeout=step.timeout
                        )
                    else:
                        result = await step.primitive.process(step_ctx)

                    if result.is_success():
                        output_ctx = result.unwrap()
                        return StepResult(
                            id=step.id,
                            success=True,
                            output=output_ctx.response or output_ctx.variables,
                            duration_ms=(time.time() - start_time) * 1000,
                            retries=attempts - 1
                        )
                    else:
                        last_error = str(result.error)

                elif step.run:
                    # Use raw function
                    if asyncio.iscoroutinefunction(step.run):
                        if step.timeout:
                            output = await asyncio.wait_for(
                                step.run(step_ctx),
                                timeout=step.timeout
                            )
                        else:
                            output = await step.run(step_ctx)
                    else:
                        output = step.run(step_ctx)

                    return StepResult(
                        id=step.id,
                        success=True,
                        output=output,
                        duration_ms=(time.time() - start_time) * 1000,
                        retries=attempts - 1
                    )

            except asyncio.TimeoutError:
                last_error = f"Step '{step.id}' timed out after {step.timeout}s"
            except Exception as e:
                last_error = str(e)

            # Retry delay
            if attempts <= step.retries:
                await asyncio.sleep(step.retry_delay)

        return StepResult(
            id=step.id,
            success=False,
            error=last_error,
            duration_ms=(time.time() - start_time) * 1000,
            retries=attempts - 1
        )

    async def _run(
        self,
        **inputs
    ) -> WorkflowResponse:
        """
        Run the workflow.

        Args:
            **inputs: Initial context variables (query, thread_id, etc.)

        Returns:
            WorkflowResponse with step outputs
        """
        from langpy.core.context import Context

        if not self._steps:
            return WorkflowResponse(
                success=False,
                error="No steps defined",
                status="failed"
            )

        try:
            # Create initial context
            context = Context(
                query=inputs.get("query") or inputs.get("input"),
                variables=inputs
            )

            # Sort steps
            layers = self._topological_sort()

            # Execute layers
            results: Dict[str, StepResult] = {}
            step_outputs = []

            for layer in layers:
                # Execute layer in parallel
                tasks = [
                    self._execute_step(step, context, results)
                    for step in layer
                ]

                layer_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(layer_results):
                    step = layer[i]

                    if isinstance(result, Exception):
                        result = StepResult(
                            id=step.id,
                            success=False,
                            error=str(result)
                        )

                    results[step.id] = result
                    step_outputs.append({
                        "id": result.id,
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                        "duration_ms": result.duration_ms
                    })

                    # Update context with step output
                    if result.success and result.output:
                        if isinstance(result.output, dict):
                            for k, v in result.output.items():
                                context = context.set(k, v)
                        context = context.set(step.id, result.output)

                        # If output has response, propagate it
                        if isinstance(result.output, str):
                            context = context.with_response(result.output)

            # Check for failures
            failed = [r for r in results.values() if not r.success]

            if failed:
                return WorkflowResponse(
                    success=False,
                    status="failed",
                    outputs={k: v.output for k, v in results.items()},
                    steps=step_outputs,
                    error=f"Steps failed: {[f.id for f in failed]}"
                )

            # Get final output
            final_outputs = {k: v.output for k, v in results.items() if v.success}

            return WorkflowResponse(
                success=True,
                status="completed",
                outputs=final_outputs,
                steps=step_outputs
            )

        except Exception as e:
            return WorkflowResponse(
                success=False,
                status="failed",
                error=str(e)
            )

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """
        Process context - run workflow with context as input.

        This makes Workflow a composable primitive.
        """
        # Extract inputs from context
        inputs = {"query": ctx.query, **ctx.variables}

        response = await self._run(**inputs)

        if not response.success:
            return Failure(PrimitiveError(
                code=ErrorCode.PRIMITIVE_ERROR,
                message=response.error or "Workflow failed",
                primitive=self._name
            ))

        # Update context with outputs
        new_ctx = ctx.clone()

        if response.outputs:
            for key, value in response.outputs.items():
                new_ctx = new_ctx.set(key, value)

            # Get last step's output as response
            last_step = self._steps[-1] if self._steps else None
            if last_step and last_step.id in response.outputs:
                output = response.outputs[last_step.id]
                if isinstance(output, str):
                    new_ctx = new_ctx.with_response(output)

        new_ctx = new_ctx.set("workflow_results", response.outputs)
        new_ctx = new_ctx.set("workflow_steps", response.steps)

        return Success(new_ctx)

    # ========================================================================
    # Builder Methods
    # ========================================================================

    def clear(self) -> "Workflow":
        """Clear all steps."""
        self._steps = []
        return self

    def clone(self) -> "Workflow":
        """Create a copy of this workflow."""
        wf = Workflow(client=self._client, name=self._name, debug=self._debug)
        wf._steps = list(self._steps)
        return wf

    @property
    def steps(self) -> List[StepConfig]:
        """Get step configurations."""
        return list(self._steps)


def workflow(name: str = None, debug: bool = False) -> Workflow:
    """Create a new workflow builder."""
    return Workflow(name=name or f"workflow_{uuid.uuid4().hex[:8]}", debug=debug)
