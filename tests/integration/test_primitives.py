"""
Integration tests for LangPy primitives.

Tests that all primitives:
1. Implement the correct interface (run() and process())
2. Can be composed with pipeline operators (| and &)
3. Work with the unified Langpy client
4. Integrate with workflows
"""

import pytest
import asyncio
from typing import List

# Import the unified client
from langpy import Langpy, Context, Success, Failure


class TestPrimitiveInterface:
    """Test that all primitives implement the correct interface."""

    def test_langpy_client_has_all_primitives(self):
        """Verify Langpy client exposes all 9 primitives."""
        lb = Langpy()

        # All primitives should be accessible
        assert hasattr(lb, 'agent')
        assert hasattr(lb, 'pipe')
        assert hasattr(lb, 'memory')
        assert hasattr(lb, 'thread')
        assert hasattr(lb, 'parser')
        assert hasattr(lb, 'chunker')
        assert hasattr(lb, 'embed')
        assert hasattr(lb, 'tools')

        # workflow is a method that returns a Workflow
        assert callable(lb.workflow)

    def test_primitives_have_run_method(self):
        """All primitives should have async run() method."""
        lb = Langpy()

        primitives = [
            lb.agent,
            lb.pipe,
            lb.memory,
            lb.thread,
            lb.parser,
            lb.chunker,
            lb.embed,
            lb.tools,
        ]

        for prim in primitives:
            assert hasattr(prim, 'run'), f"{prim.name} missing run()"
            assert asyncio.iscoroutinefunction(prim.run), f"{prim.name}.run() not async"

    def test_primitives_have_process_method(self):
        """All primitives should have async process() method."""
        lb = Langpy()

        primitives = [
            lb.agent,
            lb.pipe,
            lb.memory,
            lb.thread,
            lb.parser,
            lb.chunker,
            lb.embed,
            lb.tools,
        ]

        for prim in primitives:
            assert hasattr(prim, 'process'), f"{prim.name} missing process()"
            assert asyncio.iscoroutinefunction(prim.process), f"{prim.name}.process() not async"

    def test_primitives_have_name_property(self):
        """All primitives should have name property."""
        lb = Langpy()

        primitives = [
            lb.agent,
            lb.pipe,
            lb.memory,
            lb.thread,
            lb.parser,
            lb.chunker,
            lb.embed,
            lb.tools,
        ]

        for prim in primitives:
            assert hasattr(prim, 'name'), f"Missing name property"
            assert isinstance(prim.name, str), f"{prim.name} name not string"


class TestPipelineComposition:
    """Test pipeline composition with | and & operators."""

    def test_sequential_composition(self):
        """Test | operator creates pipeline."""
        lb = Langpy()

        # Should not raise
        pipeline = lb.memory | lb.pipe
        assert pipeline is not None
        assert hasattr(pipeline, 'process')

    def test_parallel_composition(self):
        """Test & operator creates parallel execution."""
        lb = Langpy()

        # Should not raise
        parallel = lb.agent & lb.pipe
        assert parallel is not None
        assert hasattr(parallel, 'process')

    def test_chained_composition(self):
        """Test multiple composition."""
        lb = Langpy()

        pipeline = lb.memory | lb.pipe | lb.thread.saver
        assert pipeline is not None

    def test_mixed_composition(self):
        """Test mixed sequential and parallel."""
        lb = Langpy()

        # (memory | pipe) then save
        pipeline = (lb.memory | lb.pipe) | lb.thread.saver
        assert pipeline is not None


class TestWorkflowWithPrimitives:
    """Test workflow accepts primitives as steps."""

    def test_workflow_step_with_primitive(self):
        """Workflow.step() accepts primitive parameter."""
        lb = Langpy()

        wf = lb.workflow(name="test-wf")
        wf.step(id="step1", primitive=lb.memory)

        assert len(wf.steps) == 1
        assert wf.steps[0].id == "step1"
        assert wf.steps[0].primitive is not None

    def test_workflow_step_dependencies(self):
        """Workflow respects step dependencies."""
        lb = Langpy()

        wf = lb.workflow(name="test-wf")
        wf.step(id="retrieve", primitive=lb.memory)
        wf.step(id="generate", primitive=lb.agent, after=["retrieve"])

        assert len(wf.steps) == 2
        assert wf.steps[1].after == ["retrieve"]

    def test_workflow_chaining(self):
        """Workflow.step() returns self for chaining."""
        lb = Langpy()

        wf = (
            lb.workflow(name="test-wf")
            .step(id="step1", primitive=lb.memory)
            .step(id="step2", primitive=lb.pipe, after=["step1"])
        )

        assert len(wf.steps) == 2

    def test_workflow_is_primitive(self):
        """Workflow itself is a primitive (can be composed)."""
        lb = Langpy()

        wf = lb.workflow(name="inner")
        wf.step(id="s1", primitive=lb.pipe)

        # Workflow should have process() method
        assert hasattr(wf, 'process')
        assert asyncio.iscoroutinefunction(wf.process)

        # Should be composable
        pipeline = wf | lb.thread.saver
        assert pipeline is not None


class TestContextFlow:
    """Test context flows through primitives correctly."""

    @pytest.mark.asyncio
    async def test_context_creation(self):
        """Context can be created with query."""
        ctx = Context(query="test query")
        assert ctx.query == "test query"

    @pytest.mark.asyncio
    async def test_context_with_response(self):
        """Context.with_response creates new context."""
        ctx = Context(query="test")
        new_ctx = ctx.with_response("answer")

        assert ctx.response is None  # Original unchanged
        assert new_ctx.response == "answer"

    @pytest.mark.asyncio
    async def test_context_variables(self):
        """Context supports custom variables."""
        ctx = Context(query="test")
        ctx = ctx.set("custom_key", "custom_value")

        assert ctx.get("custom_key") == "custom_value"


class TestResponseTypes:
    """Test response types from primitives."""

    def test_agent_response_type(self):
        """AgentResponse has expected fields."""
        from langpy.core.primitive import AgentResponse

        response = AgentResponse(success=True, output="Hello")
        assert response.success is True
        assert response.output == "Hello"

    def test_memory_response_type(self):
        """MemoryResponse has expected fields."""
        from langpy.core.primitive import MemoryResponse

        response = MemoryResponse(
            success=True,
            documents=[{"content": "test"}],
            count=1
        )
        assert response.success is True
        assert response.count == 1

    def test_workflow_response_type(self):
        """WorkflowResponse has expected fields."""
        from langpy.core.primitive import WorkflowResponse

        response = WorkflowResponse(
            success=True,
            status="completed",
            outputs={"step1": "output"}
        )
        assert response.status == "completed"


class TestThreadHelpers:
    """Test Thread loader and saver primitives."""

    def test_thread_has_loader(self):
        """Thread has loader property."""
        lb = Langpy()
        assert hasattr(lb.thread, 'loader')
        assert hasattr(lb.thread.loader, 'process')

    def test_thread_has_saver(self):
        """Thread has saver property."""
        lb = Langpy()
        assert hasattr(lb.thread, 'saver')
        assert hasattr(lb.thread.saver, 'process')


class TestToolsRegistry:
    """Test Tools primitive tool registration."""

    def test_register_custom_tool(self):
        """Tools.register() adds custom tool."""
        lb = Langpy()

        def my_tool(x: int) -> int:
            return x * 2

        lb.tools.register("double", my_tool, description="Doubles a number")

        schema = lb.tools.get_schema("double")
        assert schema is not None
        assert schema["name"] == "double"

    def test_builtin_tools_exist(self):
        """Built-in tools are registered."""
        lb = Langpy()

        assert lb.tools.get_schema("web_search") is not None
        assert lb.tools.get_schema("web_crawl") is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
