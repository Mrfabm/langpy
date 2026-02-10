# Complete Operator Set - Implementation Plan

## Overview

Complete LangPy's operator set to provide full Lego block composition capabilities for building any agent architecture.

**Goal**: Make all control flow patterns available as composable operators (Lego block connectors)

---

## Current State

### ✅ Working Operators (Exported):
- `|` - Sequential composition
- `&` - Parallel composition
- `pipeline()` - Explicit sequential
- `parallel()` - Explicit parallel
- `when()` - Conditional (if/else)
- `branch()` - Multi-way routing
- `retry()` - Retry with backoff
- `recover()` - Error recovery

### ⚠️ Exists But Not Exported:
- `loop_while()` - Iteration (exists in langpy/core but not in langpy/__init__.py)

### ❌ Missing Operators:
- `map_over()` - For-each pattern (parallel processing of items)
- `reduce()` - Aggregation pattern (combine multiple results)

---

## Phase 1: Export Missing Operator (Immediate)

### Task 1.1: Export `loop_while` in main __init__.py

**File**: `langpy/__init__.py`

**Change**:
```python
# Line 242-248 - Add loop_while to imports
from .core import (
    # ... existing imports ...
    pipeline,
    parallel,
    when,
    recover,
    retry,
    branch,
    loop_while,  # ← ADD THIS
    # ...
)
```

**Update `__all__`**:
```python
# Add to __all__ list around line 280
__all__ = [
    # ... existing ...
    "pipeline",
    "parallel",
    "when",
    "recover",
    "retry",
    "branch",
    "loop_while",  # ← ADD THIS
    # ...
]
```

**Test**:
```python
# Should work after change
from langpy import loop_while
```

---

## Phase 2: Add Missing Operators (High Priority)

### Task 2.1: Create `MapPrimitive` class

**File**: `langpy/core/pipeline.py`

**Location**: After `LoopPrimitive` class (around line 464)

**Implementation**:
```python
class MapPrimitive(BasePrimitive):
    """
    Map a primitive over a list of items (for-each pattern).

    Processes each item through the primitive, optionally in parallel.

    Example:
        process_docs = map_over(
            items=lambda ctx: ctx.get("documents"),
            apply=lb.parser | lb.memory,
            parallel=True
        )
    """

    def __init__(
        self,
        items: Callable[["Context"], List[Any]],
        apply: IPrimitive,
        parallel: bool = True,
        name: Optional[str] = None
    ):
        """
        Create a map primitive.

        Args:
            items: Function (Context) -> List to get items to process
            apply: Primitive to apply to each item
            parallel: Execute in parallel (True) or sequential (False)
            name: Optional name
        """
        super().__init__(name or "Map")
        self._items_fn = items
        self._apply = apply
        self._parallel = parallel

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute primitive for each item."""
        from .result import Success, Failure, PrimitiveError, ErrorCode

        # Get items
        items = self._items_fn(ctx)

        if not items:
            return Success(ctx.set("map_results", []))

        # Process each item
        if self._parallel:
            # Parallel execution
            tasks = []
            for i, item in enumerate(items):
                item_ctx = ctx.set("map_item", item).set("map_index", i)
                tasks.append(self._apply.process(item_ctx))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            map_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    return Failure(PrimitiveError(
                        code=ErrorCode.EXECUTION_ERROR,
                        message=f"Map failed at index {i}: {str(result)}",
                        primitive=self._name
                    ))

                if result.is_failure():
                    return result  # Propagate first failure

                map_results.append(result.unwrap())
        else:
            # Sequential execution
            map_results = []
            for i, item in enumerate(items):
                item_ctx = ctx.set("map_item", item).set("map_index", i)
                result = await self._apply.process(item_ctx)

                if result.is_failure():
                    return result

                map_results.append(result.unwrap())

        # Store results in context
        return Success(ctx.set("map_results", map_results))
```

### Task 2.2: Create `map_over()` helper function

**File**: `langpy/core/pipeline.py`

**Location**: After `loop_while()` function (around line 588)

**Implementation**:
```python
def map_over(
    items: Callable[["Context"], List[Any]],
    apply: IPrimitive,
    parallel: bool = True,
    name: Optional[str] = None
) -> MapPrimitive:
    """
    Apply a primitive to each item in a list (for-each pattern).

    Args:
        items: Function (Context) -> List to get items to process
        apply: Primitive to apply to each item
        parallel: Execute in parallel (True) or sequential (False)
        name: Optional name

    Example:
        # Process multiple documents in parallel
        process_all = map_over(
            items=lambda ctx: ctx.get("documents"),
            apply=lb.parser | lb.chunker | lb.memory,
            parallel=True
        )

        # Research multiple topics
        research_all = map_over(
            items=lambda ctx: ctx.get("topics"),
            apply=lb.agent,
            parallel=True
        )
    """
    return MapPrimitive(items, apply, parallel, name)
```

### Task 2.3: Create `ReducePrimitive` class

**File**: `langpy/core/pipeline.py`

**Location**: After `MapPrimitive` class

**Implementation**:
```python
class ReducePrimitive(BasePrimitive):
    """
    Reduce/aggregate multiple values into one.

    Combines results from multiple context keys or previous steps.

    Example:
        aggregate = reduce(
            inputs=["research1", "research2", "research3"],
            combine=lambda results: "\\n\\n".join(results)
        )
    """

    def __init__(
        self,
        inputs: Union[List[str], Callable[["Context"], List[Any]]],
        combine: Callable[[List[Any]], Any],
        name: Optional[str] = None
    ):
        """
        Create a reduce primitive.

        Args:
            inputs: List of context keys OR function (Context) -> List of values
            combine: Function to combine list into single value
            name: Optional name
        """
        super().__init__(name or "Reduce")
        self._inputs = inputs
        self._combine = combine

    async def _process(self, ctx: "Context") -> "Result[Context]":
        """Execute reduction."""
        from .result import Success, Failure, PrimitiveError, ErrorCode

        # Get input values
        if callable(self._inputs):
            values = self._inputs(ctx)
        else:
            values = []
            for key in self._inputs:
                value = ctx.get(key)
                if value is not None:
                    values.append(value)

        if not values:
            return Failure(PrimitiveError(
                code=ErrorCode.VALIDATION_ERROR,
                message="No values to reduce",
                primitive=self._name
            ))

        # Apply combine function
        try:
            result = self._combine(values)
            return Success(ctx.set("reduce_result", result))
        except Exception as e:
            return Failure(PrimitiveError(
                code=ErrorCode.EXECUTION_ERROR,
                message=f"Reduce failed: {str(e)}",
                primitive=self._name
            ))
```

### Task 2.4: Create `reduce()` helper function

**File**: `langpy/core/pipeline.py`

**Location**: After `map_over()` function

**Implementation**:
```python
def reduce(
    inputs: Union[List[str], Callable[["Context"], List[Any]]],
    combine: Callable[[List[Any]], Any],
    name: Optional[str] = None
) -> ReducePrimitive:
    """
    Reduce/aggregate multiple values into one.

    Args:
        inputs: List of context keys OR function (Context) -> List of values
        combine: Function to combine list into single value
        name: Optional name

    Example:
        # Combine research results
        synthesize = reduce(
            inputs=["research1", "research2", "research3"],
            combine=lambda results: "\\n\\n".join(str(r) for r in results)
        )

        # Aggregate scores
        average_score = reduce(
            inputs=lambda ctx: ctx.get("all_scores", []),
            combine=lambda scores: sum(scores) / len(scores)
        )
    """
    return ReducePrimitive(inputs, combine, name)
```

### Task 2.5: Export new operators

**File**: `langpy/core/__init__.py`

**Add to imports** (around line 70-85):
```python
from .pipeline import (
    # ... existing ...
    LoopPrimitive,
    MapPrimitive,     # ← ADD
    ReducePrimitive,  # ← ADD
    pipeline,
    parallel,
    when,
    recover,
    retry,
    branch,
    loop_while,
    map_over,         # ← ADD
    reduce,           # ← ADD
)
```

**Add to `__all__`** (around line 147-161):
```python
__all__ = [
    # ... existing ...
    "LoopPrimitive",
    "MapPrimitive",     # ← ADD
    "ReducePrimitive",  # ← ADD
    "pipeline",
    "parallel",
    "when",
    "recover",
    "retry",
    "branch",
    "loop_while",
    "map_over",         # ← ADD
    "reduce",           # ← ADD
    # ...
]
```

**File**: `langpy/__init__.py`

**Add to imports** (around line 242-248):
```python
from .core import (
    # ... existing ...
    pipeline,
    parallel,
    when,
    recover,
    retry,
    branch,
    loop_while,  # ADDED IN PHASE 1
    map_over,    # ← ADD
    reduce,      # ← ADD
    # ...
)
```

**Add to `__all__`** (around line 280-288):
```python
__all__ = [
    # ... existing ...
    "pipeline",
    "parallel",
    "when",
    "recover",
    "retry",
    "branch",
    "loop_while",  # ADDED IN PHASE 1
    "map_over",    # ← ADD
    "reduce",      # ← ADD
    # ...
]
```

---

## Phase 3: Documentation (Medium Priority)

### Task 3.1: Create OPERATORS.md

**File**: `docs/OPERATORS.md` (NEW)

**Content**: Complete operator reference with:
- Overview of all operators
- When to use each operator
- Examples for each operator
- Composition patterns
- Best practices

### Task 3.2: Create operator examples

**File**: `examples/operators_guide.py` (NEW)

**Content**: Comprehensive examples showing:
- Each operator individually
- Composing operators together
- Common patterns (map-reduce, iterative refinement, etc.)

### Task 3.3: Update existing documentation

**Files to update**:
- `README.md` - Add operators section
- `docs/CORE_API.md` - Document operator composition
- `docs/GETTING_STARTED.md` - Add operator quick start

---

## Phase 4: Complete Examples (Medium Priority)

### Task 4.1: Create pattern examples

**Files to create**:
1. `examples/patterns/iterative_refinement.py` - Loop until quality
2. `examples/patterns/parallel_research.py` - Map over topics
3. `examples/patterns/map_reduce_analysis.py` - Map-reduce pattern
4. `examples/patterns/conditional_routing.py` - Branch and when
5. `examples/patterns/multi_agent_collaboration.py` - All operators together

### Task 4.2: Update AI Agency example

**File**: `ai_agency_with_workflow.py`

**Enhancement**: Refactor to use new operators instead of Python loops

---

## Implementation Order

### Step 1: Phase 1 - Export `loop_while` (5 min)
1. Edit `langpy/__init__.py`
2. Add `loop_while` to imports
3. Add `loop_while` to `__all__`
4. Test: `from langpy import loop_while`

### Step 2: Phase 2.1 - Create `MapPrimitive` (1 hour)
1. Add `MapPrimitive` class to `langpy/core/pipeline.py`
2. Add `map_over()` function
3. Test with simple example

### Step 3: Phase 2.2 - Create `ReducePrimitive` (1 hour)
1. Add `ReducePrimitive` class to `langpy/core/pipeline.py`
2. Add `reduce()` function
3. Test with simple example

### Step 4: Phase 2.3 - Export new operators (15 min)
1. Update `langpy/core/__init__.py`
2. Update `langpy/__init__.py`
3. Test imports

### Step 5: Phase 2.4 - Integration test (30 min)
1. Test all operators in Workflow.step()
2. Test operator composition
3. Verify with pipeline `|` operator

### Step 6: Phase 3 - Documentation (3 hours)
1. Create `docs/OPERATORS.md`
2. Create `examples/operators_guide.py`
3. Update README.md

### Step 7: Phase 4 - Pattern examples (2-3 hours)
1. Create pattern example files
2. Test each pattern
3. Document learnings

---

## Testing Checklist

### Unit Tests
- [ ] `MapPrimitive` processes items correctly
- [ ] `MapPrimitive` parallel execution works
- [ ] `MapPrimitive` sequential execution works
- [ ] `MapPrimitive` handles failures
- [ ] `ReducePrimitive` combines values
- [ ] `ReducePrimitive` handles empty inputs
- [ ] All operators work with `|` composition
- [ ] All operators work in Workflow.step()

### Integration Tests
- [ ] Map + reduce pattern
- [ ] Loop + map pattern
- [ ] Branch + map pattern
- [ ] All operators in single workflow
- [ ] Nested operator composition

### Example Tests
- [ ] All pattern examples run successfully
- [ ] AI Agency example with operators
- [ ] operators_guide.py examples work

---

## Success Criteria

✅ **Phase 1 Complete When**:
- `loop_while` importable: `from langpy import loop_while`
- Works in code without errors

✅ **Phase 2 Complete When**:
- `map_over` and `reduce` importable
- All operators work in Workflow
- Can build map-reduce pattern
- All operators compose with `|`

✅ **Phase 3 Complete When**:
- `docs/OPERATORS.md` exists with all operators documented
- `examples/operators_guide.py` shows all operators
- README.md updated with operators section

✅ **Phase 4 Complete When**:
- 5 pattern examples created and tested
- AI Agency refactored to use operators
- All examples in repo documentation

---

## Timeline

**Estimated Total**: 1-2 days

- Phase 1: 5 minutes
- Phase 2: 3-4 hours
- Phase 3: 3 hours
- Phase 4: 3-4 hours

**Can be parallelized**:
- Phase 2 (operators) and Phase 3 (docs) can happen simultaneously
- Phase 4 (examples) depends on Phase 2 completion

---

## Files to Modify/Create

### Modify:
1. `langpy/__init__.py` - Add exports
2. `langpy/core/__init__.py` - Add exports
3. `langpy/core/pipeline.py` - Add MapPrimitive, ReducePrimitive
4. `README.md` - Add operators section
5. `docs/CORE_API.md` - Document operators
6. `docs/GETTING_STARTED.md` - Add operator examples

### Create:
1. `docs/OPERATORS.md` - Complete operator reference
2. `examples/operators_guide.py` - Comprehensive examples
3. `examples/patterns/iterative_refinement.py`
4. `examples/patterns/parallel_research.py`
5. `examples/patterns/map_reduce_analysis.py`
6. `examples/patterns/conditional_routing.py`
7. `examples/patterns/multi_agent_collaboration.py`
8. `tests/test_operators.py` - Operator tests

---

## Next Steps After Completion

1. **Blog post** - Announce complete operator set
2. **Video tutorial** - Show operator composition
3. **Update examples** - Refactor existing examples to use operators
4. **Community feedback** - Gather user experiences
5. **Performance optimization** - Profile map/reduce for large datasets

---

**Ready to proceed with implementation!**
