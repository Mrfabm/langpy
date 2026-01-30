# Workflow SDK Options - Complete Reference

**Enhanced with Full Langbase Parity** - This document provides a comprehensive reference for all available options when working with the enhanced Workflow SDK in LangPy.

## Table of Contents

1. [Enhanced Workflow Creation](#enhanced-workflow-creation)
2. [Await-able Builder Pattern](#await-able-builder-pattern)
3. [Enhanced Step Configuration](#enhanced-step-configuration)
4. [Retry Configuration](#retry-configuration)
5. [Error Handling](#error-handling)
6. [Secret Scoping](#secret-scoping)
7. [Thread Handoff](#thread-handoff)
8. [Parallel Execution](#parallel-execution)
9. [Run History Management](#run-history-management)
10. [Primitive Runner Registration](#primitive-runner-registration)
11. [CLI Support](#cli-support)
12. [Step Types](#step-types)
13. [Advanced Features](#advanced-features)
14. [Decorator Support](#decorator-support)
15. [Examples](#examples)

## Enhanced Workflow Creation

### Factory Method (Recommended)

```python
from sdk import workflow

# Basic workflow interface
workflow_interface = workflow()

# Enhanced workflow interface with debug mode
workflow_interface = workflow(debug=True)

# With custom storage path
workflow_interface = workflow(
    storage_path="/path/to/workflows",
    debug=True
)
```

### Direct Creation Methods

```python
# Using WorkflowInterface directly
from sdk.workflow_interface import WorkflowInterface

workflow_interface = WorkflowInterface(
    async_backend=async_backend_func,
    sync_backend=sync_backend_func,
    storage_path="/path/to/workflows",
    debug=True  # Enable debug mode
)

# Using enhanced WorkflowEngine directly
from workflow import WorkflowEngine

engine = WorkflowEngine(debug=True)
```

## Await-able Builder Pattern

**NEW**: The SDK now supports Langbase's await-able builder pattern for step execution.

### Basic Step Execution

```python
from sdk import workflow

# Create workflow instance
workflow_interface = workflow(debug=True)

# Execute a single step (matches Langbase exactly)
result = await workflow_interface.step(
    id="demo_step",
    type="function",
    run=lambda ctx: "Hello from step!",
    timeout=5000
)

# Execute pipe step
result = await workflow_interface.step(
    id="pipe_step",
    type="pipe",
    ref="my-pipe",
    config={"input": "Hello"},
    timeout=10000,
    retries={"limit": 3, "delay": 1000, "backoff": "exponential"},
    use_secrets=["OPENAI_KEY"]
)
```

### Available Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `str` | **Required** - Unique step identifier |
| `type` | `str` | Step type: "function", "pipe", "agent", "tool", "memory", "thread" |
| `ref` | `str` | Reference to primitive (required for non-function types) |
| `run` | `Callable` | Function to execute (required for function type) |
| `config` | `Dict[str, Any]` | Configuration for the step |
| `timeout` | `int` | Timeout in milliseconds |
| `retries` | `RetryConfig` or `Dict` | Retry configuration |
| `group` | `List[str]` | Parallel execution groups |
| `condition` | `Callable` | Condition function for step execution |
| `use_secrets` | `List[str]` | Secrets to inject for this step |
| `after` | `List[str]` | Step dependencies |

## Enhanced Step Configuration

### StepConfig Fields

The enhanced `StepConfig` now includes all Langbase parity features:

```python
from sdk import workflow
from workflow import StepConfig

# Create step with all options
step = StepConfig(
    id="enhanced_step",
    type="pipe",
    ref="my-pipe",
    config={
        "input": "Hello",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    },
    timeout=15000,  # 15 seconds
    retries={
        "limit": 3,
        "delay": 1000,
        "backoff": "exponential"
    },
    group=["processing"],  # Parallel execution group
    condition=lambda ctx: ctx.get("should_run", True),
    use_secrets=["OPENAI_KEY", "ANTHROPIC_KEY"],
    after=["previous_step"]  # Dependencies
)
```

### Step Creation Methods

The SDK provides enhanced step creation methods:

```python
workflow_interface = workflow()

# Function step with all features
function_step = workflow_interface.create_function_step(
    step_id="process_data",
    func=lambda ctx: f"Processed: {ctx.get('input', '')}",
    timeout=5000,
    retries=workflow_interface.create_retry_config(
        limit=3,
        delay=1000,
        backoff="exponential"
    ),
    group=["processing"],
    use_secrets=["API_KEY"],
    after=["input_validation"]
)

# Pipe step with enhanced features
pipe_step = workflow_interface.create_pipe_step(
    step_id="llm_processing",
    pipe_ref="my-pipe",
    timeout=30000,
    retries={"limit": 3, "delay": 2000, "backoff": "exponential"},
    use_secrets=["OPENAI_KEY"],
    config={
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000
    }
)

# Agent step with enhanced features
agent_step = workflow_interface.create_agent_step(
    step_id="ai_agent",
    agent_ref="my-agent",
    timeout=60000,
    retries=workflow_interface.create_retry_config(
        limit=2,
        delay=5000,
        backoff="linear"
    ),
    use_secrets=["OPENAI_KEY"],
    config={
        "tools": ["calculator", "search"],
        "instructions": "Help the user with their query"
    }
)

# Memory step with enhanced features
memory_step = workflow_interface.create_memory_step(
    step_id="search_memory",
    memory_ref="document-store",
    timeout=10000,
    config={
        "query": "artificial intelligence",
        "k": 5,
        "similarity_threshold": 0.8
    }
)

# Tool step with enhanced features
tool_step = workflow_interface.create_tool_step(
    step_id="web_search",
    tool_ref="search-tool",
    timeout=20000,
    config={
        "query": "latest AI news",
        "max_results": 10
    }
)
```

## Retry Configuration

### RetryConfig Options

```python
from sdk import workflow

workflow_interface = workflow()

# Create retry configuration
retry_config = workflow_interface.create_retry_config(
    limit=3,                    # Maximum retry attempts
    delay=1000,                 # Base delay in milliseconds
    backoff="exponential",      # Backoff strategy
    max_delay=30000,           # Maximum delay cap
    jitter=True                # Add randomness to delays
)

# Backoff strategies
strategies = {
    "fixed": "Same delay for all attempts",
    "linear": "Delay increases linearly (delay * attempt)",
    "exponential": "Delay doubles each attempt (delay * 2^attempt)"
}

# Use in step
result = await workflow_interface.step(
    id="retry_step",
    type="function",
    run=lambda ctx: potentially_failing_function(),
    retries=retry_config,
    timeout=10000
)
```

### Retry Configuration as Dictionary

```python
# Alternative: Use dictionary format
retry_dict = {
    "limit": 3,
    "delay": 1000,
    "backoff": "exponential",
    "max_delay": 30000,
    "jitter": True
}

result = await workflow_interface.step(
    id="retry_step",
    type="function",
    run=lambda ctx: potentially_failing_function(),
    retries=retry_dict
)
```

## Error Handling

### Enhanced Error Types

```python
from sdk.workflow_interface import (
    WorkflowError,
    TimeoutError,
    RetryExhaustedError,
    StepError,
    SecretError,
    PrimitiveError,
    ContextError,
    DependencyError
)

try:
    result = await workflow_interface.step(
        id="risky_step",
            type="function",
        run=lambda ctx: risky_operation(),
        timeout=5000,
        retries={"limit": 3, "delay": 1000, "backoff": "exponential"}
    )
except TimeoutError as e:
    print(f"Step timed out: {e}")
    print(f"Timeout: {e.timeout_ms}ms, Elapsed: {e.elapsed_ms}ms")
except RetryExhaustedError as e:
    print(f"Retry exhausted: {e}")
    print(f"Attempts: {e.attempts}, Last error: {e.last_error}")
except StepError as e:
    print(f"Step failed: {e}")
    print(f"Original error: {e.original_error}")
except SecretError as e:
    print(f"Secret error: {e}")
    print(f"Secret: {e.secret_name}, Reason: {e.reason}")
except PrimitiveError as e:
    print(f"Primitive error: {e}")
    print(f"Type: {e.primitive_type}, Ref: {e.primitive_ref}")
except WorkflowError as e:
    print(f"Workflow error: {e}")
    print(f"Step ID: {e.step_id}, Context: {e.context}")
```

### Error Context Information

```python
try:
    result = await workflow_interface.step(...)
except WorkflowError as e:
    # Enhanced error context
    error_info = {
        "step_id": e.step_id,
        "context": e.context,
        "timestamp": e.timestamp,
        "trace": e.trace
    }
    print(f"Error details: {error_info}")
```

## Secret Scoping

### GitHub Actions-Style Secret Injection

```python
import os

# Set up environment secrets
os.environ["OPENAI_KEY"] = "sk-your-openai-key"
os.environ["ANTHROPIC_KEY"] = "sk-ant-your-anthropic-key"

# Use secrets in steps
result = await workflow_interface.step(
    id="secret_step",
    type="pipe",
    ref="my-pipe",
    config={"input": "Hello"},
    use_secrets=["OPENAI_KEY"]  # Only inject specified secrets
)

# Multiple secrets
result = await workflow_interface.step(
    id="multi_secret_step",
    type="agent",
    ref="my-agent",
    config={"input": "Hello"},
    use_secrets=["OPENAI_KEY", "ANTHROPIC_KEY"]
)
```

### Secret Manager Registration

```python
# Register custom secret manager
def my_secret_manager(secret_name: str) -> str:
    # Custom secret retrieval logic
    return get_secret_from_vault(secret_name)

workflow_interface.register_secret_manager("vault", my_secret_manager)
```

## Thread Handoff

### Automatic Thread ID Interception

```python
# Thread handoff happens automatically
# When a primitive returns 'lb-thread-id', it's captured

result = await workflow_interface.step(
    id="pipe_step",
    type="pipe",
    ref="my-pipe",
    config={"input": "Hello"}
)

# If pipe returns: {"output": "response", "lb-thread-id": "thread-123"}
# The thread ID is automatically captured and passed to next steps

# Access thread ID in subsequent steps
next_result = await workflow_interface.step(
    id="next_step",
    type="function",
    run=lambda ctx: f"Thread ID: {ctx.thread_id}"
)
```

## Parallel Execution

### Group-Based Parallel Execution

```python
# Create parallel steps
parallel_steps = [
    workflow_interface.create_function_step(
        step_id=f"parallel_task_{i}",
        func=lambda ctx, i=i: f"Task {i} completed",
        group=["parallel_group"],  # Same group = parallel execution
        timeout=5000
    )
    for i in range(3)
]

# Add aggregation step
aggregation_step = workflow_interface.create_function_step(
    step_id="aggregate_results",
    func=lambda ctx: {
        "results": [
            ctx.get(f"parallel_task_{i}", "") for i in range(3)
        ]
    },
    after=[f"parallel_task_{i}" for i in range(3)]  # Wait for all parallel tasks
)

# Execute workflow
result = await workflow_interface.run(
    name="parallel_workflow",
    inputs={},
    steps=parallel_steps + [aggregation_step]
)
```

### Multiple Parallel Groups

```python
steps = [
    # Group A - parallel execution
    workflow_interface.create_function_step(
        step_id="groupA_task1",
        func=lambda ctx: "A1 done",
        group=["groupA"]
    ),
    workflow_interface.create_function_step(
        step_id="groupA_task2",
        func=lambda ctx: "A2 done",
        group=["groupA"]
    ),
    
    # Group B - parallel execution (after Group A)
    workflow_interface.create_function_step(
        step_id="groupB_task1",
        func=lambda ctx: "B1 done",
        group=["groupB"],
        after=["groupA_task1", "groupA_task2"]
    ),
    workflow_interface.create_function_step(
        step_id="groupB_task2",
        func=lambda ctx: "B2 done",
        group=["groupB"],
        after=["groupA_task1", "groupA_task2"]
    )
]
```

## Run History Management

### Accessing Run History

```python
# List all runs
history = workflow_interface.list_run_history()

# Filter by workflow name
history = workflow_interface.list_run_history(
    workflow_name="my_workflow"
)

# Filter by status
history = workflow_interface.list_run_history(
    status="completed"
)

# Combined filters with limit
history = workflow_interface.list_run_history(
    workflow_name="my_workflow",
    status="completed",
    limit=10
)

# Process run history
for run in history:
    print(f"Run: {run['id']}")
    print(f"Workflow: {run['workflow_name']}")
    print(f"Status: {run['status']}")
    print(f"Duration: {run.get('duration_ms', 0)}ms")
    print(f"Started: {run['started_at']}")
    print("---")
```

### Run Record Structure

```python
# Each run record contains:
run_record = {
    "id": "uuid-string",
    "workflow_name": "my_workflow",
    "status": "completed",  # "running", "completed", "failed"
    "inputs": {"key": "value"},
    "outputs": {"step1": "result1", "step2": "result2"},
    "error": None,  # Error message if failed
    "started_at": 1234567890,  # Unix timestamp
    "completed_at": 1234567891,  # Unix timestamp
    "duration_ms": 1000,  # Duration in milliseconds
    "steps": [{"id": "step1", "status": "completed"}],
    "context": {"thread_id": "thread-123"},
    "metadata": {"version": "1.0"}
}
```

## Primitive Runner Registration

### Registering Primitive Runners

```python
# Register primitive runners for workflow steps
workflow_interface = workflow()

# Register pipe runner
async def my_pipe_runner(ref: str, context: dict, config: dict):
    # Your pipe execution logic
    return {"output": f"Pipe {ref} processed: {config.get('input', '')}"}

workflow_interface.register_runner("pipe", my_pipe_runner)

# Register agent runner
async def my_agent_runner(ref: str, context: dict, config: dict):
    # Your agent execution logic
    return {"output": f"Agent {ref} responded: {config.get('input', '')}"}

workflow_interface.register_runner("agent", my_agent_runner)

# Register memory runner
async def my_memory_runner(ref: str, context: dict, config: dict):
    # Your memory execution logic
    return {"results": [{"id": "doc1", "content": "Found document"}]}

workflow_interface.register_runner("memory", my_memory_runner)

# Register tool runner
async def my_tool_runner(ref: str, context: dict, config: dict):
    # Your tool execution logic
    return {"result": f"Tool {ref} executed with {config}"}

workflow_interface.register_runner("tool", my_tool_runner)

# Register thread runner
async def my_thread_runner(ref: str, context: dict, config: dict):
    # Your thread execution logic
    return {"thread_id": "thread-123", "messages": []}

workflow_interface.register_runner("thread", my_thread_runner)
```

### Runner Function Signature

```python
async def primitive_runner(
    ref: str,                    # Primitive reference
    context: dict,               # Workflow context
    config: dict                 # Step configuration
) -> Any:                        # Return value
    """
    Args:
        ref: The primitive reference (e.g., "my-pipe")
        context: Current workflow context including:
            - Step outputs
            - Thread ID
            - Memory data
            - Tool results
            - Secrets (scoped to current step)
        config: Step-specific configuration
    
    Returns:
        Any: Result that will be stored in workflow context
    """
    pass
```

## CLI Support

### Command Line Interface

```bash
# Run workflow from file
python -m workflow run demos/my_workflow.py

# Run with debug mode
python -m workflow --debug run demos/my_workflow.py

# Run with inputs
python -m workflow run demos/my_workflow.py --inputs '{"name": "John"}'

# Save output to file
python -m workflow run demos/my_workflow.py --output result.json

# List recent runs
python -m workflow list

# Filter runs by workflow
python -m workflow list --workflow my_workflow

# Filter runs by status
python -m workflow list --status completed --limit 10

# Show workflow statistics
python -m workflow stats

# Show stats for specific workflow
python -m workflow stats my_workflow

# Cleanup old runs
python -m workflow cleanup --days 30
```

### Workflow File Structure

```python
# demos/my_workflow.py
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflow import StepConfig

# Define workflow steps
steps = [
    StepConfig(
        id="greeting",
        type="function",
        run=lambda ctx: f"Hello, {ctx.get('name', 'World')}!"
    ),
    StepConfig(
        id="process_greeting",
        type="function",
        run=lambda ctx: f"Processed: {ctx.get('greeting', 'No greeting')}"
    )
]

# Optional workflow configuration
workflow_config = {
    "name": "my_workflow",
    "description": "A simple greeting workflow",
    "version": "1.0.0"
}
```

## Step Types

### Function Steps

```python
# Basic function step
function_step = workflow_interface.create_function_step(
    step_id="my_function",
    func=lambda ctx: f"Processing {ctx.get('input', '')}"
)

# Async function step
async def async_function(ctx):
    await asyncio.sleep(0.1)
    return f"Async processing {ctx.get('input', '')}"

async_step = workflow_interface.create_function_step(
    step_id="async_function",
    func=async_function
)

# Function with context access
def context_function(ctx):
    # Access workflow context
    input_data = ctx.get("input", "")
    thread_id = ctx.thread_id
    secrets = ctx.secrets
    
        return {
        "processed_input": input_data,
        "thread_id": thread_id,
        "has_secrets": bool(secrets)
    }

context_step = workflow_interface.create_function_step(
    step_id="context_function",
    func=context_function
)
```

### Pipe Steps

```python
# Basic pipe step
pipe_step = workflow_interface.create_pipe_step(
    step_id="llm_call",
    pipe_ref="my-pipe",
    config={
        "input": "Hello, how are you?",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
)

# Pipe step with template resolution
pipe_step = workflow_interface.create_pipe_step(
    step_id="templated_pipe",
    pipe_ref="my-pipe",
    config={
        "input": "{{previous_step.output}}",  # Template from previous step
        "model": "gpt-4o-mini"
    },
    after=["previous_step"]
)
```

### Agent Steps

```python
# Basic agent step
agent_step = workflow_interface.create_agent_step(
    step_id="ai_agent",
    agent_ref="my-agent",
    config={
        "input": "Help me with this task",
        "tools": ["calculator", "search"],
        "instructions": "You are a helpful assistant"
    }
)

# Agent step with tool results
agent_step = workflow_interface.create_agent_step(
    step_id="tool_agent",
    agent_ref="tool-agent",
    config={
        "input": "{{user_input}}",
        "tool_results": "{{search_tool.results}}"
    },
    after=["search_tool"]
)
```

### Memory Steps

```python
# Memory search step
memory_step = workflow_interface.create_memory_step(
    step_id="search_docs",
    memory_ref="document-store",
    config={
        "query": "artificial intelligence",
        "k": 5,
        "similarity_threshold": 0.8
    }
)

# Memory add step
memory_add_step = workflow_interface.create_memory_step(
    step_id="add_to_memory",
    memory_ref="document-store",
    config={
        "operation": "add",
        "text": "{{processed_content}}",
        "metadata": {"source": "user_input"}
    }
)
```

### Tool Steps

```python
# Web search tool
search_step = workflow_interface.create_tool_step(
    step_id="web_search",
    tool_ref="search-tool",
    config={
        "query": "latest AI developments",
        "max_results": 10
    }
)

# Calculator tool
calc_step = workflow_interface.create_tool_step(
    step_id="calculate",
    tool_ref="calculator",
    config={
        "expression": "{{math_problem}}"
    }
)
```

### Thread Steps

```python
# Create thread step
thread_step = workflow_interface.create_tool_step(
    step_id="create_thread",
    tool_ref="thread-manager",
    config={
        "operation": "create",
        "name": "User Conversation"
    }
)

# Add message to thread
message_step = workflow_interface.create_tool_step(
    step_id="add_message",
    tool_ref="thread-manager",
    config={
        "operation": "add_message",
        "thread_id": "{{create_thread.thread_id}}",
        "role": "user",
        "content": "{{user_input}}"
    }
)
```

## Advanced Features

### Conditional Step Execution

```python
# Step with condition
conditional_step = workflow_interface.create_function_step(
    step_id="conditional_step",
    func=lambda ctx: "Condition met!",
    condition=lambda ctx: ctx.get("should_run", False)
)

# Complex condition
def complex_condition(ctx):
    user_type = ctx.get("user_type", "")
    score = ctx.get("score", 0)
    return user_type == "premium" and score > 80

premium_step = workflow_interface.create_function_step(
    step_id="premium_feature",
    func=lambda ctx: "Premium feature activated",
    condition=complex_condition
)
```

### Step Dependencies

```python
# Sequential dependencies
steps = [
    workflow_interface.create_function_step(
        step_id="step1",
        func=lambda ctx: "Step 1 complete"
    ),
    workflow_interface.create_function_step(
        step_id="step2",
        func=lambda ctx: "Step 2 complete",
        after=["step1"]  # Wait for step1
    ),
    workflow_interface.create_function_step(
        step_id="step3",
        func=lambda ctx: "Step 3 complete",
        after=["step1", "step2"]  # Wait for both step1 and step2
    )
]
```

### Template Resolution

```python
# Template resolution in configs
pipe_step = workflow_interface.create_pipe_step(
    step_id="templated_pipe",
    pipe_ref="my-pipe",
    config={
        "input": "Process this: {{user_input}}",
        "context": "Previous result: {{previous_step.output}}",
        "metadata": {
            "user_id": "{{user_id}}",
            "timestamp": "{{current_time}}"
        }
    }
)
```

### Workflow Composition

```python
# Compose complex workflows
async def create_complex_workflow():
    # Input validation phase
    validation_steps = [
        workflow_interface.create_function_step(
            step_id="validate_input",
            func=lambda ctx: validate_user_input(ctx.get("input", "")),
            timeout=5000
        ),
        workflow_interface.create_function_step(
            step_id="sanitize_input",
            func=lambda ctx: sanitize_data(ctx.get("validate_input", {})),
            after=["validate_input"]
        )
    ]
    
    # Processing phase
    processing_steps = [
        workflow_interface.create_pipe_step(
            step_id="llm_analysis",
            pipe_ref="analysis-pipe",
            config={"input": "{{sanitize_input.clean_data}}"},
            after=["sanitize_input"],
            timeout=30000,
            retries={"limit": 3, "delay": 2000, "backoff": "exponential"}
        ),
        workflow_interface.create_agent_step(
            step_id="ai_enhancement",
            agent_ref="enhancement-agent",
            config={"input": "{{llm_analysis.output}}"},
            after=["llm_analysis"],
            use_secrets=["OPENAI_KEY"]
        )
    ]
    
    # Storage phase
    storage_steps = [
        workflow_interface.create_memory_step(
            step_id="store_result",
            memory_ref="result-store",
            config={
                "operation": "add",
                "data": "{{ai_enhancement.output}}"
            },
            after=["ai_enhancement"]
        )
    ]
    
    # Combine all steps
    all_steps = validation_steps + processing_steps + storage_steps
    
    # Execute workflow
    result = await workflow_interface.run(
        name="complex_workflow",
        inputs={"input": "user data"},
        steps=all_steps
    )
    
    return result
```

## Decorator Support

### Workflow Decorator

```python
from workflow import workflow, step

@workflow(name="decorator_workflow", debug=True)
async def my_workflow(ctx):
    # Use the workflow engine within the decorated function
    engine = get_workflow_engine()
    
    # Execute steps
    result1 = await engine.step(
        id="step1",
        type="function",
        run=lambda ctx: f"Processing {ctx.get('input', '')}"
    )
    
    result2 = await engine.step(
        id="step2",
        type="function",
        run=lambda ctx: f"Final result: {ctx.get('step1', '')}"
    )
    
    return {"final": result2}

# Execute decorated workflow
result = await my_workflow(input="test data")
```

### Step Decorator

```python
@step("my_step", timeout=5000, retries={"limit": 3})
async def my_step_function(ctx):
    # Step logic here
    return f"Processed: {ctx.get('input', '')}"

# Use in workflow
steps = [
        StepConfig(
        id="decorated_step",
            type="function",
        run=my_step_function
    )
]
```

## Examples

### Complete Example: Customer Support Workflow

```python
import asyncio
from sdk import workflow

async def customer_support_workflow():
    """Complete customer support workflow with all features."""
    
    # Create workflow interface
    workflow_interface = workflow(debug=True)
    
    # Register mock runners
    async def mock_pipe_runner(ref, context, config):
        return {"output": f"Pipe {ref} processed: {config.get('input', '')}"}
    
    async def mock_agent_runner(ref, context, config):
        return {"output": f"Agent {ref} responded: {config.get('input', '')}"}
    
    workflow_interface.register_runner("pipe", mock_pipe_runner)
    workflow_interface.register_runner("agent", mock_agent_runner)
    
    # Create workflow steps
    steps = [
        # Input validation
        workflow_interface.create_function_step(
            step_id="validate_ticket",
            func=lambda ctx: {
                "valid": True,
                "ticket_id": ctx.get("ticket_id", ""),
                "priority": ctx.get("priority", "medium")
            },
            timeout=5000
        ),
        
        # Parallel processing
        workflow_interface.create_pipe_step(
            step_id="sentiment_analysis",
            pipe_ref="sentiment-pipe",
            config={"input": "{{validate_ticket.message}}"},
            group=["analysis"],
            timeout=10000,
            retries={"limit": 2, "delay": 1000, "backoff": "exponential"}
        ),
        
        workflow_interface.create_pipe_step(
            step_id="category_detection",
            pipe_ref="category-pipe",
            config={"input": "{{validate_ticket.message}}"},
            group=["analysis"],
            timeout=10000
        ),
        
        # Agent response
        workflow_interface.create_agent_step(
            step_id="generate_response",
            agent_ref="support-agent",
            config={
                "input": "{{validate_ticket.message}}",
                "sentiment": "{{sentiment_analysis.output}}",
                "category": "{{category_detection.output}}",
                "priority": "{{validate_ticket.priority}}"
            },
            after=["sentiment_analysis", "category_detection"],
            use_secrets=["OPENAI_KEY"],
            timeout=30000,
            retries={"limit": 3, "delay": 2000, "backoff": "exponential"}
        ),
        
        # Final processing
        workflow_interface.create_function_step(
            step_id="format_response",
            func=lambda ctx: {
                "ticket_id": ctx.get("validate_ticket", {}).get("ticket_id", ""),
                "response": ctx.get("generate_response", {}).get("output", ""),
                "metadata": {
                    "sentiment": ctx.get("sentiment_analysis", {}).get("output", ""),
                    "category": ctx.get("category_detection", {}).get("output", ""),
                    "priority": ctx.get("validate_ticket", {}).get("priority", "")
                }
            },
            after=["generate_response"]
        )
    ]
    
    # Execute workflow
    result = await workflow_interface.run(
        name="customer_support",
        inputs={
            "ticket_id": "TICKET-123",
            "message": "I'm having trouble with my account login",
            "priority": "high"
        },
        steps=steps
    )
    
    return result

# Run the example
if __name__ == "__main__":
    result = asyncio.run(customer_support_workflow())
    print(f"Workflow result: {result}")
```

### Example: Data Processing Pipeline

```python
async def data_processing_pipeline():
    """Data processing pipeline with parallel execution."""
    
    workflow_interface = workflow(debug=True)
    
    # Create processing steps
    steps = [
        # Data ingestion
        workflow_interface.create_function_step(
            step_id="ingest_data",
            func=lambda ctx: {
                "raw_data": ctx.get("data_source", []),
                "count": len(ctx.get("data_source", []))
            }
        ),
        
        # Parallel processing
        workflow_interface.create_function_step(
            step_id="process_chunk_1",
            func=lambda ctx: {"processed": "chunk_1_done"},
            group=["processing"],
            timeout=10000
        ),
        
        workflow_interface.create_function_step(
            step_id="process_chunk_2",
            func=lambda ctx: {"processed": "chunk_2_done"},
            group=["processing"],
            timeout=10000
        ),
        
        workflow_interface.create_function_step(
            step_id="process_chunk_3",
            func=lambda ctx: {"processed": "chunk_3_done"},
            group=["processing"],
            timeout=10000
        ),
        
        # Aggregation
        workflow_interface.create_function_step(
            step_id="aggregate_results",
            func=lambda ctx: {
                "results": [
                    ctx.get("process_chunk_1", {}).get("processed", ""),
                    ctx.get("process_chunk_2", {}).get("processed", ""),
                    ctx.get("process_chunk_3", {}).get("processed", "")
                ],
                "total_count": ctx.get("ingest_data", {}).get("count", 0)
            },
            after=["process_chunk_1", "process_chunk_2", "process_chunk_3"]
        )
    ]
    
    # Execute workflow
    result = await workflow_interface.run(
        name="data_processing",
        inputs={"data_source": [1, 2, 3, 4, 5]},
        steps=steps
    )
    
    return result
```

### Example: AI Agent Workflow

```python
async def ai_agent_workflow():
    """AI agent workflow with memory and tools."""
    
    workflow_interface = workflow(debug=True)
    
    # Register AI runners
    async def ai_pipe_runner(ref, context, config):
        return {
            "output": f"AI response: {config.get('input', '')}",
            "lb-thread-id": "thread-ai-123"
        }
    
    async def memory_runner(ref, context, config):
        return {
            "results": [
                {"content": "Previous conversation about AI", "score": 0.9},
                {"content": "User preferences", "score": 0.8}
            ]
        }
    
    workflow_interface.register_runner("pipe", ai_pipe_runner)
    workflow_interface.register_runner("memory", memory_runner)
    
    # Create AI workflow steps
    steps = [
        # Memory search
        workflow_interface.create_memory_step(
            step_id="search_memory",
            memory_ref="conversation-memory",
            config={
                "query": "{{user_input}}",
                "k": 3
            }
        ),
        
        # AI processing
        workflow_interface.create_pipe_step(
            step_id="ai_response",
            pipe_ref="ai-pipe",
            config={
                "input": "{{user_input}}",
                "context": "{{search_memory.results}}",
                "model": "gpt-4o-mini",
                "temperature": 0.7
            },
            after=["search_memory"],
            use_secrets=["OPENAI_KEY"],
            timeout=30000,
            retries={"limit": 2, "delay": 2000, "backoff": "exponential"}
        ),
        
        # Thread management
        workflow_interface.create_function_step(
            step_id="update_thread",
            func=lambda ctx: {
                "thread_id": ctx.thread_id,
                "message_added": True,
                "response": ctx.get("ai_response", {}).get("output", "")
            },
            after=["ai_response"]
        )
    ]
    
    # Execute workflow
    result = await workflow_interface.run(
        name="ai_agent",
        inputs={"user_input": "Tell me about artificial intelligence"},
        steps=steps
    )
    
    return result
```

## Best Practices

### 1. Error Handling

```python
# Always wrap workflow execution in try-catch
try:
    result = await workflow_interface.run(name="my_workflow", steps=steps)
except TimeoutError as e:
    # Handle timeout specifically
    logger.error(f"Workflow timed out: {e}")
except RetryExhaustedError as e:
    # Handle retry exhaustion
    logger.error(f"Retries exhausted: {e}")
except WorkflowError as e:
    # Handle other workflow errors
    logger.error(f"Workflow failed: {e}")
```

### 2. Resource Management

```python
# Use appropriate timeouts
step = workflow_interface.create_pipe_step(
    step_id="long_running_task",
    pipe_ref="analysis-pipe",
    timeout=60000,  # 1 minute timeout
    retries={"limit": 2, "delay": 5000, "backoff": "linear"}
)
```

### 3. Secret Management

```python
# Only inject necessary secrets
step = workflow_interface.create_agent_step(
    step_id="ai_agent",
    agent_ref="my-agent",
    use_secrets=["OPENAI_KEY"],  # Only what's needed
    config={"input": "{{user_input}}"}
)
```

### 4. Parallel Execution

```python
# Use parallel execution for independent tasks
parallel_steps = [
    workflow_interface.create_function_step(
        step_id="task_1",
        func=independent_task_1,
        group=["parallel"]
    ),
    workflow_interface.create_function_step(
        step_id="task_2",
        func=independent_task_2,
        group=["parallel"]
    )
]
```

### 5. Debugging

```python
# Enable debug mode for development
workflow_interface = workflow(debug=True)

# Check run history for debugging
history = workflow_interface.list_run_history(
    workflow_name="my_workflow",
    status="failed",
    limit=5
)
```

## Migration from Legacy Workflow

### Old Way (Legacy)

```python
# Legacy workflow creation
from workflow import AsyncWorkflow, WorkflowRegistry

registry = WorkflowRegistry()
workflow_engine = AsyncWorkflow(registry=registry)

# Create workflow
registry.create("my_workflow", steps)

# Run workflow
result = await workflow_engine.run("my_workflow", inputs)
```

### New Way (Enhanced)

```python
# Enhanced workflow creation
from sdk import workflow

workflow_interface = workflow(debug=True)

# Register primitive runners
workflow_interface.register_runner("pipe", my_pipe_runner)

# Use await-able builder
result = await workflow_interface.step(
    id="enhanced_step",
    type="pipe",
    ref="my-pipe",
    config={"input": "Hello"},
    timeout=10000,
    retries={"limit": 3, "delay": 1000, "backoff": "exponential"},
    use_secrets=["OPENAI_KEY"]
)

# Or run full workflow
result = await workflow_interface.run(
    name="my_workflow",
    inputs=inputs,
    steps=steps
)
```

## Conclusion

The enhanced Workflow SDK provides **full Langbase parity** with additional advanced features:

- ✅ **Await-able builder pattern** - `await workflow.step(**config)`
- ✅ **Enhanced error taxonomy** - Comprehensive error handling
- ✅ **Secret scoping** - GitHub Actions-style secret injection
- ✅ **Thread handoff** - Automatic thread ID management
- ✅ **Advanced retry strategies** - Configurable backoff with jitter
- ✅ **Parallel execution** - Group-based concurrent execution
- ✅ **Run history registry** - Persistent run tracking
- ✅ **CLI support** - Command-line workflow management
- ✅ **Rich logging** - Enhanced debug output
- ✅ **Primitive integration** - Seamless runner registration

The SDK is designed to be both powerful and easy to use, providing all the features needed for production-grade workflow orchestration while maintaining backward compatibility with existing code.

---

*For more examples and advanced usage, see the `demos/workflow/` directory and the comprehensive test suite in `tests/workflow/`.* 