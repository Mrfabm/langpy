# LangPy Operators - Complete Reference

## Overview

Operators are **control flow primitives** that act as Lego block connectors, enabling you to compose any agent architecture from simple building blocks.

**Philosophy**: Keep primitives simple (nouns like Agent, Memory, Pipe), use operators as connectors (verbs like loop, branch, map).

---

## All Operators

| Operator | Pattern | Use Case | Composition |
|----------|---------|----------|-------------|
| **`\|`** | Sequential | Chain steps | `memory \| pipe \| agent` |
| **`&`** | Parallel | Run simultaneously | `optimist & pessimist & realist` |
| **`pipeline()`** | Sequential | Explicit chain | `pipeline(step1, step2, step3)` |
| **`parallel()`** | Parallel | Explicit parallel | `parallel(task1, task2, task3)` |
| **`when()`** | Conditional | If/else branching | `when(condition, then_do, else_do)` |
| **`branch()`** | Multi-way | Route selection | `branch(router, routes_dict)` |
| **`loop_while()`** | Iteration | Repeat until condition | `loop_while(condition, body)` |
| **`map_over()`** | For-each | Process list items | `map_over(items, apply)` |
| **`reduce()`** | Aggregation | Combine results | `reduce(inputs, combine)` |
| **`retry()`** | Resilience | Automatic retry | `retry(primitive, max_attempts=3)` |
| **`recover()`** | Error handling | Fallback logic | `recover(primitive, handler)` |

---

## 1. Sequential Composition (`|` and `pipeline()`)

**Pattern**: Execute primitives one after another

### Using `|` Operator

```python
from langpy import Langpy

lb = Langpy(api_key="...")

# Simple chain
rag = lb.memory | lb.pipe

# Multi-step pipeline
process = lb.parser | lb.chunker | lb.memory | lb.agent

# Execute
result = await process.process(Context(query="What is Python?"))
```

### Using `pipeline()` Function

```python
from langpy import pipeline

# Explicit pipeline
rag = pipeline(
    lb.memory,
    lb.pipe,
    lb.agent,
    name="rag-pipeline"
)

result = await rag.process(ctx)
```

**When to use**:
- Chain dependent steps
- Build RAG pipelines
- Create processing flows

---

## 2. Parallel Composition (`&` and `parallel()`)

**Pattern**: Execute primitives simultaneously

### Using `&` Operator

```python
# Run three agents in parallel
multi_perspective = lb.agent & lb.agent & lb.agent

result = await multi_perspective.process(ctx)
```

### Using `parallel()` Function

```python
from langpy import parallel

# Multiple researchers
research = parallel(
    researcher1,
    researcher2,
    researcher3,
    merge_strategy="concat",  # or "first", "list"
    name="parallel-research"
)

result = await research.process(ctx)
```

**Merge Strategies**:
- `"concat"`: Concatenate all outputs
- `"first"`: Return first successful result
- `"list"`: Return list of all results

**When to use**:
- Multiple perspectives
- Parallel research
- Speed optimization

---

## 3. Conditional (`when()`)

**Pattern**: If/else branching based on context

```python
from langpy import when

# Simple conditional
search_or_skip = when(
    condition=lambda ctx: ctx.get("needs_search", False),
    then_do=search_engine,
    else_do=direct_answer
)

# Use in pipeline
agent = lb.memory | search_or_skip | lb.pipe

result = await agent.process(Context(query="What is AI?", needs_search=True))
```

**Real Example**:

```python
# Route based on query complexity
complexity_router = when(
    condition=lambda ctx: len(ctx.get("query", "")) > 100,
    then_do=detailed_research,  # Long query → detailed
    else_do=quick_answer        # Short query → quick
)
```

**When to use**:
- Binary decisions
- Feature flags
- Conditional logic

---

## 4. Multi-way Branch (`branch()`)

**Pattern**: Route to different paths based on key

```python
from langpy import branch

# Create router
task_router = branch(
    router=lambda ctx: ctx.get("task_type"),
    routes={
        "research": research_agent,
        "summarize": summary_agent,
        "translate": translation_agent,
    },
    default=general_agent,  # Fallback
    name="task-router"
)

# Execute
result = await task_router.process(Context(task_type="research", query="AI"))
```

**Real Example**:

```python
# Route by urgency
urgency_router = branch(
    router=lambda ctx: ctx.get("priority", "normal"),
    routes={
        "urgent": fast_llm,      # Use GPT-4o-mini
        "normal": balanced_llm,   # Use GPT-4o
        "thorough": deep_llm,     # Use o1
    },
    default=balanced_llm
)
```

**When to use**:
- Multiple paths
- Dynamic routing
- Strategy selection

---

## 5. Loop (`loop_while()`)

**Pattern**: Repeat until condition is false

```python
from langpy import loop_while

# Iterative refinement
refine = loop_while(
    condition=lambda ctx: ctx.get("quality", 0) < 0.9,
    body=improvement_step,
    max_iterations=5,  # Safety limit
    name="refine-loop"
)

result = await refine.process(Context(content="draft", quality=0.5))
```

**Real Example**:

```python
# Research until confident
research_loop = loop_while(
    condition=lambda ctx: ctx.get("confidence", 0) < 0.8,
    body=lb.agent,  # Keep researching
    max_iterations=3
)

# Use in workflow
wf = lb.workflow(name="iterative-research")
wf.step(id="research", primitive=research_loop)
```

**When to use**:
- Iterative refinement
- Quality thresholds
- Progressive research

---

## 6. Map (`map_over()`)

**Pattern**: Apply primitive to each item in a list

```python
from langpy import map_over

# Process multiple documents in parallel
process_docs = map_over(
    items=lambda ctx: ctx.get("documents"),
    apply=lb.parser | lb.chunker | lb.memory,
    parallel=True,  # Run in parallel
    name="doc-processor"
)

result = await process_docs.process(Context(documents=[doc1, doc2, doc3]))

# Access results
map_results = result.unwrap().get("map_results")
```

**Sequential Processing**:

```python
# Process items one by one
sequential_map = map_over(
    items=lambda ctx: ctx.get("tasks"),
    apply=task_executor,
    parallel=False  # Sequential
)
```

**Real Example**:

```python
# Research multiple topics
research_topics = map_over(
    items=lambda ctx: ctx.get("topics", []),
    apply=lb.agent,
    parallel=True
)

# Execute
result = await research_topics.process(
    Context(topics=["Python", "JavaScript", "Rust"])
)

# Each topic researched in parallel
# Results stored in ctx.get("map_results")
```

**When to use**:
- Process multiple items
- Parallel operations
- Batch processing

---

## 7. Reduce (`reduce()`)

**Pattern**: Aggregate multiple values into one

```python
from langpy import reduce

# Combine research results
synthesize = reduce(
    inputs=["research1", "research2", "research3"],  # Context keys
    combine=lambda results: "\n\n".join(str(r) for r in results),
    name="synthesize"
)

result = await synthesize.process(ctx)
combined = result.unwrap().get("reduce_result")
```

**Dynamic Inputs**:

```python
# Get inputs from context
average_scores = reduce(
    inputs=lambda ctx: ctx.get("all_scores", []),  # Function
    combine=lambda scores: sum(scores) / len(scores)
)
```

**Real Example**:

```python
# Multi-agent synthesis
wf = lb.workflow(name="research-synthesis")

# Step 1: Multiple agents research in parallel
wf.step(id="research1", primitive=agent1)
wf.step(id="research2", primitive=agent2)
wf.step(id="research3", primitive=agent3)

# Step 2: Combine all findings
wf.step(
    id="synthesize",
    primitive=reduce(
        inputs=["research1", "research2", "research3"],
        combine=lambda results: {
            "summary": "\n\n".join(results),
            "count": len(results)
        }
    ),
    after=["research1", "research2", "research3"]
)

result = await wf.run(query="What is LangPy?")
```

**When to use**:
- Combine results
- Aggregate data
- Multi-agent synthesis

---

## 8. Retry (`retry()`)

**Pattern**: Automatically retry on failure

```python
from langpy import retry

# Retry API calls
resilient_api = retry(
    primitive=external_api,
    max_attempts=3,
    delay=0.5,           # Initial delay (seconds)
    backoff_multiplier=2.0,  # Exponential backoff
    name="resilient-api"
)

result = await resilient_api.process(ctx)
```

**Retry Schedule**:
- Attempt 1: Immediate
- Attempt 2: Wait 0.5s
- Attempt 3: Wait 1.0s
- Attempt 4: Wait 2.0s

**Real Example**:

```python
# Retry LLM with backoff
reliable_llm = retry(
    primitive=lb.agent,
    max_attempts=3,
    delay=1.0,
    backoff_multiplier=2.0
)

# Use in pipeline
rag = lb.memory | reliable_llm
```

**When to use**:
- External APIs
- Flaky services
- Network calls

---

## 9. Recover (`recover()`)

**Pattern**: Handle errors with fallback logic

```python
from langpy import recover

# Fallback on error
safe_search = recover(
    primitive=advanced_search,
    handler=lambda err, ctx: ctx.set("result", "Fallback result"),
    name="safe-search"
)

result = await safe_search.process(ctx)
```

**Real Example**:

```python
# Try premium model, fallback to free
smart_agent = recover(
    primitive=premium_llm,  # Try GPT-4o first
    handler=lambda err, ctx: ctx.set("model", "gpt-4o-mini")  # Fallback
)

# With pipeline
rag = lb.memory | smart_agent
```

**When to use**:
- Error handling
- Graceful degradation
- Fallback strategies

---

## Operator Composition Patterns

### Pattern 1: Map-Reduce

Process items in parallel, then aggregate:

```python
from langpy import map_over, reduce, pipeline

# Map: Research multiple topics
research = map_over(
    items=lambda ctx: ctx.get("topics"),
    apply=lb.agent,
    parallel=True
)

# Reduce: Combine findings
synthesize = reduce(
    inputs=lambda ctx: ctx.get("map_results"),
    combine=lambda results: "\n\n".join(results)
)

# Compose
map_reduce = pipeline(research, synthesize)

result = await map_reduce.process(Context(topics=["AI", "ML", "DL"]))
```

### Pattern 2: Retry + Recover

Resilient execution with fallback:

```python
from langpy import retry, recover, pipeline

# Try with retry, recover if still fails
ultra_reliable = pipeline(
    retry(primary_service, max_attempts=3),
    recover(
        primitive=backup_service,
        handler=lambda err, ctx: ctx.set("fallback", True)
    )
)
```

### Pattern 3: Branch + Loop

Dynamic routing with iteration:

```python
from langpy import branch, loop_while, pipeline

# Route to appropriate refinement loop
refiner = branch(
    router=lambda ctx: ctx.get("content_type"),
    routes={
        "code": loop_while(
            condition=lambda ctx: ctx.get("code_quality") < 0.9,
            body=code_refiner
        ),
        "text": loop_while(
            condition=lambda ctx: ctx.get("readability") < 0.9,
            body=text_refiner
        )
    }
)
```

### Pattern 4: Parallel + Reduce

Multiple agents → synthesis:

```python
from langpy import parallel, reduce, pipeline

# Run multiple agents in parallel
multi_agent = parallel(optimist, pessimist, realist)

# Synthesize perspectives
synthesis = reduce(
    inputs=lambda ctx: ctx.get("parallel_results", []),
    combine=lambda views: synthesize_perspectives(views)
)

# Compose
consensus = pipeline(multi_agent, synthesis)
```

---

## Operator + Workflow Integration

All operators work seamlessly with Workflow:

```python
from langpy import Langpy, map_over, reduce

lb = Langpy(api_key="...")
wf = lb.workflow(name="multi-agent-research")

# Step 1: Parallel research with map_over
wf.step(
    id="research",
    primitive=map_over(
        items=lambda ctx: ctx.get("topics"),
        apply=lb.agent,
        parallel=True
    )
)

# Step 2: Aggregate with reduce
wf.step(
    id="synthesize",
    primitive=reduce(
        inputs=lambda ctx: ctx.get("map_results"),
        combine=lambda results: "\n\n".join(results)
    ),
    after=["research"]
)

# Execute
result = await wf.run(topics=["Python", "JavaScript", "Rust"])
```

---

## Best Practices

### 1. **Always Set Max Iterations for Loops**

```python
# ✅ Good - prevents infinite loops
loop_while(condition, body, max_iterations=5)

# ❌ Bad - could run forever
loop_while(condition, body)  # Default is 100, but be explicit
```

### 2. **Use Parallel Map for Independent Tasks**

```python
# ✅ Good - parallel execution
map_over(items=get_topics, apply=researcher, parallel=True)

# ❌ Bad - sequential (slow)
map_over(items=get_topics, apply=researcher, parallel=False)
```

### 3. **Compose with `|` for Readability**

```python
# ✅ Good - clear flow
rag = lb.memory | lb.pipe | lb.agent

# ❌ Less clear
rag = pipeline(lb.memory, lb.pipe, lb.agent)
```

### 4. **Name Complex Operators**

```python
# ✅ Good - debuggable
refiner = loop_while(condition, body, name="quality-refiner")

# ❌ Bad - hard to debug
refiner = loop_while(condition, body)
```

### 5. **Use Retry for External Services**

```python
# ✅ Good - resilient
api = retry(external_api, max_attempts=3)

# ❌ Bad - fails on first error
api = external_api
```

---

## Complete Example: AI Research Agency

Combining all operators:

```python
from langpy import Langpy, map_over, reduce, when, loop_while, pipeline

lb = Langpy(api_key="...")
wf = lb.workflow(name="ai-research-agency")

# Step 1: Break down query
wf.step(id="plan", primitive=lb.agent)

# Step 2: Parallel research with map
wf.step(
    id="research",
    primitive=map_over(
        items=lambda ctx: ctx.get("subtasks", []),
        apply=lb.agent,
        parallel=True
    ),
    after=["plan"]
)

# Step 3: Iterative refinement
wf.step(
    id="refine",
    primitive=loop_while(
        condition=lambda ctx: ctx.get("confidence", 0) < 0.8,
        body=lb.agent,
        max_iterations=3
    ),
    after=["research"]
)

# Step 4: Final synthesis
wf.step(
    id="synthesize",
    primitive=reduce(
        inputs=lambda ctx: ctx.get("map_results", []),
        combine=lambda results: "\n\n".join(str(r) for r in results)
    ),
    after=["refine"]
)

# Execute
result = await wf.run(query="What are the latest trends in AI?")
```

---

## Summary

**Operator Categories**:

1. **Composition**: `|`, `&`, `pipeline()`, `parallel()`
2. **Control Flow**: `when()`, `branch()`, `loop_while()`
3. **Data Flow**: `map_over()`, `reduce()`
4. **Resilience**: `retry()`, `recover()`

**Key Principles**:
- ✅ Operators are Lego block connectors
- ✅ All operators compose with `|` and `&`
- ✅ All operators work in Workflow.step()
- ✅ Keep primitives simple, use operators for logic

**Next Steps**:
- See `examples/operators_guide.py` for working examples
- See `examples/patterns/` for real-world patterns
- See `README.md` for quick start

---

**Ready to build any agent architecture with composable operators!**
