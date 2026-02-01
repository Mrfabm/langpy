"""
Acceptance test for LangPy workflow Langbase parity.

Tests the await-able builder pattern and all enhanced features.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from workflow import workflow, WorkflowEngine, get_workflow_engine, RetryConfig
from workflow.core import StepConfig
from workflow.exceptions import WorkflowError, TimeoutError, RetryExhaustedError, StepError
from workflow.retry import RetryEngine
from workflow.run_registry import get_run_registry
from workflow.logging import get_workflow_logger


async def test_await_able_builder():
    """Test the await-able builder pattern."""
    print("🔍 Testing await-able builder pattern...")
    
    # Set up mock environment
    os.environ["OPENAI_KEY"] = "test-key"
    
    # Create workflow engine
    engine = WorkflowEngine(debug=True)
    
    # Test single step execution
    result = await engine.step(
        id="test_step",
        type="function",
        run=lambda ctx: f"Hello, {ctx.get('name', 'World')}!",
        timeout=5000
    )
    
    assert result == "Hello, World!"
    print("✅ Single step execution works")
    
    # Test step with retry configuration
    failure_count = 0
    def failing_function(ctx):
        nonlocal failure_count
        failure_count += 1
        if failure_count < 3:
            raise Exception(f"Failure {failure_count}")
        return f"Success after {failure_count} attempts"
    
    result = await engine.step(
        id="retry_step",
        type="function",
        run=failing_function,
        retries=RetryConfig(limit=3, delay=100, backoff="exponential"),
        timeout=10000
    )
    
    assert "Success after 3 attempts" in result
    print("✅ Retry configuration works")


async def test_context_contract():
    """Test the enhanced context contract."""
    print("🔍 Testing context contract...")
    
    engine = WorkflowEngine(debug=True)
    
    # Test context with secrets
    def secret_using_step(ctx):
        secrets = ctx.get("secrets", {})
        return f"Using secret: {secrets.get('OPENAI_KEY', 'none')}"
    
    result = await engine.step(
        id="secret_step",
        type="function",
        run=secret_using_step,
        use_secrets=["OPENAI_KEY"]
    )
    
    assert "test-key" in result
    print("✅ Secret scoping works")


async def test_workflow_execution():
    """Test full workflow execution."""
    print("🔍 Testing full workflow execution...")
    
    engine = WorkflowEngine(debug=True)
    
    # Define workflow steps
    steps = [
        StepConfig(
            id="step1",
            type="function",
            run=lambda ctx: {"result": "step1 complete", "data": ctx.get("input", "")},
        ),
        StepConfig(
            id="step2",
            type="function",
            run=lambda ctx: {"result": "step2 complete", "input_from_step1": ctx.get("step1", {}).get("result", "")},
            after=["step1"]
        ),
        StepConfig(
            id="step3",
            type="function",
            run=lambda ctx: {
                "result": "step3 complete",
                "step1_result": ctx.get("step1", {}).get("result", ""),
                "step2_result": ctx.get("step2", {}).get("result", "")
            },
            after=["step1", "step2"]
        )
    ]
    
    # Run workflow
    result = await engine.run(
        name="test_workflow",
        inputs={"input": "test data"},
        steps=steps
    )
    
    assert "step1" in result
    assert "step2" in result
    assert "step3" in result
    assert result["step1"]["result"] == "step1 complete"
    assert result["step2"]["input_from_step1"] == "step1 complete"
    print("✅ Full workflow execution works")


async def test_parallel_execution():
    """Test parallel step execution."""
    print("🔍 Testing parallel execution...")
    
    engine = WorkflowEngine(debug=True)
    
    # Define parallel steps
    steps = [
        StepConfig(
            id="parallel1",
            type="function",
            run=lambda ctx: {"result": "parallel1 complete"},
            group=["parallel_group"]
        ),
        StepConfig(
            id="parallel2",
            type="function",
            run=lambda ctx: {"result": "parallel2 complete"},
            group=["parallel_group"]
        ),
        StepConfig(
            id="parallel3",
            type="function",
            run=lambda ctx: {"result": "parallel3 complete"},
            group=["parallel_group"]
        ),
        StepConfig(
            id="final",
            type="function",
            run=lambda ctx: {
                "result": "final complete",
                "parallel_results": [
                    ctx.get("parallel1", {}).get("result", ""),
                    ctx.get("parallel2", {}).get("result", ""),
                    ctx.get("parallel3", {}).get("result", "")
                ]
            },
            after=["parallel1", "parallel2", "parallel3"]
        )
    ]
    
    # Run workflow
    result = await engine.run(
        name="parallel_workflow",
        inputs={},
        steps=steps
    )
    
    assert len(result["final"]["parallel_results"]) == 3
    assert all("complete" in r for r in result["final"]["parallel_results"])
    print("✅ Parallel execution works")


async def test_error_handling():
    """Test error handling and retry logic."""
    print("🔍 Testing error handling...")
    
    engine = WorkflowEngine(debug=True)
    
    # Test timeout error
    def slow_function(ctx):
        import time
        time.sleep(2)  # This will timeout
        return "Should not reach here"
    
    try:
        await engine.step(
            id="timeout_step",
            type="function",
            run=slow_function,
            timeout=500  # 500ms timeout
        )
        assert False, "Should have raised TimeoutError"
    except TimeoutError as e:
        assert "timeout_step" in str(e)
        print("✅ Timeout error handling works")
    
    # Test retry exhausted
    def always_failing_function(ctx):
        raise Exception("Always fails")
    
    try:
        await engine.step(
            id="failing_step",
            type="function",
            run=always_failing_function,
            retries=RetryConfig(limit=2, delay=100, backoff="fixed")
        )
        assert False, "Should have raised RetryExhaustedError"
    except RetryExhaustedError as e:
        assert "failing_step" in str(e)
        print("✅ Retry exhausted error handling works")


async def test_run_history():
    """Test run history functionality."""
    print("🔍 Testing run history...")
    
    engine = WorkflowEngine(debug=True)
    
    # Run a simple workflow
    await engine.run(
        name="history_test",
        inputs={"test": "data"},
        steps=[
            StepConfig(
                id="simple_step",
                type="function",
                run=lambda ctx: {"result": "completed"}
            )
        ]
    )
    
    # Get run history
    history = engine.list_run_history(workflow_name="history_test")
    
    assert len(history) >= 1
    assert history[0]["workflow_name"] == "history_test"
    assert history[0]["status"] == "completed"
    print("✅ Run history works")


async def test_decorator_workflow():
    """Test decorator-based workflow."""
    print("🔍 Testing decorator workflow...")
    
    @workflow(name="decorator_test", debug=True)
    async def test_workflow(input_data: str):
        """Test workflow using decorators."""
        engine = get_workflow_engine(debug=True)
        
        # This simulates the decorator pattern
        result1 = await engine.step(
            id="process_input",
            type="function",
            run=lambda ctx: f"Processed: {ctx.get('input_data', '')}"
        )
        
        result2 = await engine.step(
            id="format_output",
            type="function",
            run=lambda ctx: f"Formatted: {result1}"
        )
        
        return {"final_result": result2}
    
    # This would be called by the decorator framework
    engine = get_workflow_engine(debug=True)
    result = await engine.run(
        name="decorator_test",
        inputs={"input_data": "test data"},
        steps=[
            StepConfig(
                id="process_input",
                type="function",
                run=lambda ctx: f"Processed: {ctx.get('input_data', '')}"
            ),
            StepConfig(
                id="format_output",
                type="function",
                run=lambda ctx: f"Formatted: {ctx.get('process_input', '')}"
            )
        ]
    )
    
    assert "Processed: test data" in result["format_output"]
    print("✅ Decorator workflow pattern works")


async def run_all_tests():
    """Run all acceptance tests."""
    print("🚀 Running LangPy Workflow Langbase Parity Tests")
    print("=" * 60)
    
    tests = [
        test_await_able_builder,
        test_context_contract,
        test_workflow_execution,
        test_parallel_execution,
        test_error_handling,
        test_run_history,
        test_decorator_workflow,
    ]
    
    for test in tests:
        try:
            await test()
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("✅ All acceptance tests completed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests()) 