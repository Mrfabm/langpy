# LangPy Extension Guide

> **Status:** This document is a **specification** for extending LangPy. Layer 1 (Primitives) is implemented. Layers 2-4 (Operators, Skills, MCP) are proposed and need to be built.

This document defines LangPy's layered architecture for building AI agents.

## The Four-Layer Model

```
+------------------------------------------------------------------+
|                        LAYER 4: MCP                               |
|     External Connections (GitHub, Slack, Databases, Human)        |
|     Infrastructure requiring persistent connections/state         |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|                       LAYER 3: SKILLS                             |
|     Domain Knowledge (code-review, data-analysis, invoice)        |
|     Portable expertise following Agent Skills standard            |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|                      LAYER 2: OPERATORS                           |
|     Verbs (loop_until, branch, stop, retry, parallel, select)     |
|     Control flow patterns used within Workflows                   |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|                      LAYER 1: PRIMITIVES                          |
|     Nouns (Pipe, Agent, Memory, Thread, Workflow, Parser,         |
|           Chunker, Embed, Tools)                                  |
|     The atoms - minimal, composable, stable                       |
+------------------------------------------------------------------+
```

## Layer Responsibilities

| Layer | Type | Purpose | Examples | Status |
|-------|------|---------|----------|--------|
| **Primitives** | Nouns | Core building blocks | Pipe, Agent, Memory, Workflow | Implemented |
| **Operators** | Verbs | Control flow patterns | loop_until, branch, retry, parallel | To Build |
| **Skills** | Domain knowledge | Portable expertise | code-review, data-analysis, sql-expert | To Build |
| **MCP** | External connections | Infrastructure access | GitHub, Slack, databases, human approval | To Build |

---

# LAYER 1: PRIMITIVES (Implemented)

The 9 core primitives are the foundation. They are minimal, composable, and stable.

| Primitive | Purpose | Key Methods |
|-----------|---------|-------------|
| **Pipe** | Single LLM call | `run()`, `stream()`, `classify()`, `extract()` |
| **Agent** | Autonomous task execution | `run()` with tools |
| **Memory** | Vector storage and retrieval | `add()`, `search()`, `delete()` |
| **Thread** | Conversation history | `create()`, `add_message()`, `get_messages()` |
| **Workflow** | DAG-based orchestration | `step()`, `run()`, nested workflows |
| **Parser** | Document text extraction | `parse()` |
| **Chunker** | Text segmentation | `chunk()` |
| **Embed** | Text to vectors | `embed()`, `embed_many()` |
| **Tools** | Function definitions for agents | `@tool` decorator, `ToolDef` |

### Primitive Composition

Primitives combine naturally:

```python
# Parser -> Chunker -> Embed -> Memory (Document ingestion)
text = await parser.parse("document.pdf")
chunks = await chunker.chunk(text)
embeddings = await embed.embed_many(chunks)
await memory.add_many(chunks, embeddings)

# Memory -> Pipe (RAG query)
results = await memory.search(query)
context = "\n".join([r.text for r in results])
response = await pipe.run(f"Context: {context}\n\nQuestion: {query}")

# Thread -> Pipe (Multi-turn conversation)
history = await thread.get_messages(thread_id)
response = await pipe.run(history)
await thread.add_message(thread_id, "assistant", response.content)
```

---

# LAYER 2: OPERATORS (To Build)

> **Specification:** These operators need to be implemented in `langpy_sdk/operators.py`

Operators are **verbs** - control flow patterns that work within Workflows. They enable iteration, branching, and coordination without requiring external infrastructure.

## Core Operators (Specifications)

The following operators should be implemented. Each example shows the intended API.

### loop_until
Iterate until a condition is met.

```python
# In a Workflow step handler
async def refine_handler(ctx):
    draft = ctx.data["draft"]

    result = await loop_until(
        action=lambda text: pipe.run(f"Improve this: {text}"),
        condition=lambda output: len(output) > 200,
        initial=draft,
        max_iterations=5
    )

    return {"refined": result}
```

### branch
Conditional execution based on classification or rules.

```python
async def route_handler(ctx):
    query = ctx.data["query"]

    result = await branch(
        input=query,
        classifier=lambda q: pipe.classify(q, ["technical", "billing", "general"]),
        handlers={
            "technical": technical_agent.run,
            "billing": billing_agent.run,
            "general": general_pipe.run
        }
    )

    return {"response": result}
```

### retry
Retry with exponential backoff.

```python
async def api_handler(ctx):
    result = await retry(
        action=lambda: external_api.call(ctx.data["params"]),
        max_attempts=3,
        backoff_base=1.0,
        backoff_multiplier=2.0,
        retry_on=[RateLimitError, TimeoutError]
    )

    return {"api_result": result}
```

### parallel
Execute multiple operations concurrently.

```python
async def gather_handler(ctx):
    queries = ctx.data["queries"]

    results = await parallel(
        actions=[lambda q=q: memory.search(q) for q in queries],
        max_concurrency=5
    )

    return {"all_results": results}
```

### select
Pick the best result from multiple options.

```python
async def best_answer_handler(ctx):
    question = ctx.data["question"]

    result = await select(
        candidates=[
            lambda: pipe_a.run(question),
            lambda: pipe_b.run(question),
            lambda: pipe_c.run(question)
        ],
        scorer=lambda response: evaluator.score(response, question),
        strategy="highest"  # or "first_passing", "majority"
    )

    return {"answer": result}
```

### fallback
Try options in order until one succeeds.

```python
async def resilient_handler(ctx):
    query = ctx.data["query"]

    result = await fallback(
        actions=[
            lambda: gpt4_pipe.run(query),      # Try best model first
            lambda: gpt35_pipe.run(query),     # Fall back to faster
            lambda: local_pipe.run(query)       # Last resort
        ],
        catch=[RateLimitError, APIError]
    )

    return {"response": result}
```

### stop
Early termination with condition.

```python
async def process_handler(ctx):
    items = ctx.data["items"]

    results = await stop(
        items=items,
        processor=lambda item: agent.run(f"Process: {item}"),
        stop_when=lambda result: "CRITICAL_ERROR" in result,
        return_partial=True
    )

    return {"processed": results}
```

### wait
Pause execution.

```python
async def rate_limited_handler(ctx):
    items = ctx.data["items"]
    results = []

    for item in items:
        result = await agent.run(f"Process: {item}")
        results.append(result)
        await wait(seconds=1.0)  # Rate limiting

    return {"results": results}
```

## Reference Implementation

Operators should be simple async functions, not classes. Here's the proposed implementation:

```python
# operators.py

async def loop_until(
    action: Callable,
    condition: Callable[[Any], bool],
    initial: Any,
    max_iterations: int = 10
) -> dict:
    """Iterate until condition is met."""
    result = initial

    for i in range(max_iterations):
        result = await action(result)

        if condition(result):
            return {
                "result": result,
                "iterations": i + 1,
                "converged": True
            }

    return {
        "result": result,
        "iterations": max_iterations,
        "converged": False
    }


async def retry(
    action: Callable,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_backoff: float = 60.0,
    retry_on: list = None
) -> Any:
    """Retry with exponential backoff."""
    import asyncio
    import random

    retry_on = retry_on or [Exception]

    for attempt in range(max_attempts):
        try:
            return await action()
        except tuple(retry_on) as e:
            if attempt == max_attempts - 1:
                raise

            wait_time = min(
                backoff_base * (backoff_multiplier ** attempt),
                max_backoff
            )
            wait_time *= random.uniform(0.5, 1.5)  # Jitter
            await asyncio.sleep(wait_time)


async def parallel(
    actions: list[Callable],
    max_concurrency: int = None
) -> list:
    """Execute actions concurrently."""
    import asyncio

    if max_concurrency:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited(action):
            async with semaphore:
                return await action()

        return await asyncio.gather(*[limited(a) for a in actions])

    return await asyncio.gather(*[a() for a in actions])


async def fallback(
    actions: list[Callable],
    catch: list = None
) -> Any:
    """Try actions in order until one succeeds."""
    catch = catch or [Exception]

    for i, action in enumerate(actions):
        try:
            return await action()
        except tuple(catch):
            if i == len(actions) - 1:
                raise
            continue
```

---

# LAYER 3: SKILLS (To Build)

> **Specification:** Skills should be created in `skills/<skill-name>/SKILL.md` following the Agent Skills standard.

Skills provide **domain knowledge** - portable expertise that agents can use. They follow the [Agent Skills standard](https://agentskills.io).

## What Skills Are For

Skills are NOT generic control flow patterns. They are domain-specific knowledge:

| Good Skills (Domain Knowledge) | Bad Skills (Don't Build These) |
|-------------------------------|-------------------------------|
| `code-review` - How to review code | `loop` - Use operators instead |
| `sql-expert` - SQL query patterns | `router` - Use branch operator |
| `invoice-processing` - Extract invoice data | `evaluator` - Use Pipe.classify |
| `data-analysis` - Statistical analysis patterns | `retry` - Use retry operator |
| `legal-contract-review` - Legal document analysis | `aggregator` - Use parallel operator |

## Skill Structure

```
skills/
└── code-review/
    ├── SKILL.md              # Required: frontmatter + instructions
    ├── references/           # Optional: detailed docs
    │   └── patterns.md
    └── assets/               # Optional: templates
        └── checklist.md
```

## SKILL.md Format

```markdown
---
name: code-review
description: Expert code review focusing on security, performance, and maintainability. Use when reviewing pull requests or code changes.
license: MIT
metadata:
  author: langpy
  version: "1.0"
  domain: software-engineering
---

## Overview

This skill provides expertise in code review, covering security vulnerabilities,
performance issues, maintainability concerns, and best practices.

## When to Use

- Pull request reviews
- Pre-commit code checks
- Security audits
- Performance optimization reviews

## Review Process

1. **Security Check**
   - Look for injection vulnerabilities (SQL, command, XSS)
   - Check authentication and authorization
   - Verify input validation
   - Review secrets handling

2. **Performance Check**
   - Identify N+1 queries
   - Check for unnecessary computations
   - Review memory usage patterns
   - Analyze algorithm complexity

3. **Maintainability Check**
   - Assess code clarity
   - Check for proper error handling
   - Review naming conventions
   - Evaluate test coverage

## Output Format

Provide review as:

```
## Summary
[1-2 sentence overview]

## Critical Issues
- [Issue]: [Location] - [Explanation]

## Suggestions
- [Improvement]: [Location] - [Rationale]

## Approval Status
[APPROVE / REQUEST_CHANGES / COMMENT]
```

## Examples

[Include 2-3 example reviews with input code and output review]
```

## Example Skills (Templates)

These are templates showing what domain skills should look like when implemented.

### 1. Data Analysis Skill

```markdown
---
name: data-analysis
description: Statistical data analysis and insights extraction. Use when analyzing datasets, finding patterns, or generating reports.
metadata:
  domain: analytics
  primitives: [Pipe, Memory]
---

## Overview

Expert data analysis covering descriptive statistics, trend identification,
anomaly detection, and insight generation.

## Analysis Framework

1. **Data Understanding**
   - Identify data types and distributions
   - Check for missing values and outliers
   - Understand relationships between variables

2. **Descriptive Statistics**
   - Central tendency (mean, median, mode)
   - Dispersion (variance, standard deviation, range)
   - Distribution shape (skewness, kurtosis)

3. **Pattern Recognition**
   - Trends over time
   - Correlations between variables
   - Clusters and segments

4. **Insight Generation**
   - Key findings
   - Actionable recommendations
   - Confidence levels

## Output Format

```
## Data Summary
[Overview of dataset characteristics]

## Key Findings
1. [Finding with statistical support]
2. [Finding with statistical support]

## Recommendations
- [Action based on finding]

## Limitations
- [Caveats about the analysis]
```
```

### 2. SQL Expert Skill

```markdown
---
name: sql-expert
description: Expert SQL query writing and optimization. Use when constructing database queries or optimizing existing ones.
metadata:
  domain: databases
---

## Overview

SQL expertise covering query construction, optimization, and best practices
for PostgreSQL, MySQL, and SQLite.

## Query Construction

1. **Understand Requirements**
   - What data is needed?
   - What filters apply?
   - What aggregations are required?
   - What ordering is expected?

2. **Build Incrementally**
   - Start with SELECT columns
   - Add FROM and JOINs
   - Apply WHERE conditions
   - Add GROUP BY if aggregating
   - Add ORDER BY last

3. **Optimize**
   - Use indexes effectively
   - Avoid SELECT *
   - Limit result sets
   - Use EXPLAIN to verify

## Anti-Patterns to Avoid

- N+1 queries (use JOINs instead)
- Cartesian products (missing JOIN conditions)
- Functions on indexed columns in WHERE
- Implicit type conversions

## Output Format

```sql
-- Purpose: [What this query does]
-- Expected rows: [Approximate count]
-- Indexes used: [List relevant indexes]

SELECT ...
FROM ...
WHERE ...
```
```

### 3. Invoice Processing Skill

```markdown
---
name: invoice-processing
description: Extract structured data from invoices. Use when processing invoice documents for accounting or data entry.
metadata:
  domain: finance
  primitives: [Parser, Pipe]
---

## Overview

Extract key information from invoice documents including vendor details,
line items, totals, and payment terms.

## Extraction Fields

### Header Information
- Invoice number
- Invoice date
- Due date
- PO number (if present)

### Vendor Information
- Company name
- Address
- Tax ID / VAT number
- Contact information

### Line Items
For each line:
- Description
- Quantity
- Unit price
- Total
- Tax rate

### Totals
- Subtotal
- Tax amount
- Discounts
- Total amount due
- Currency

### Payment Information
- Payment terms
- Bank details
- Payment methods accepted

## Output Format

```json
{
  "invoice_number": "...",
  "date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "vendor": {
    "name": "...",
    "address": "..."
  },
  "line_items": [...],
  "subtotal": 0.00,
  "tax": 0.00,
  "total": 0.00,
  "currency": "USD"
}
```

## Validation Rules

- Invoice number must be present
- Totals must sum correctly
- Dates must be valid
- Currency must be identified
```

---

# LAYER 4: MCP (To Build)

> **Specification:** MCP servers should be created in `mcp_servers/<server-name>/` following the Model Context Protocol.

MCP provides **external connections** - access to infrastructure that requires persistent state, real-time communication, or third-party services.

## When to Use MCP

| Need | Solution |
|------|----------|
| Persistent state across sessions | MCP Server |
| Real-time communication | MCP Server |
| Third-party API access | MCP Server |
| Human approval workflows | MCP Server |
| Database operations | MCP Server |
| File system access | MCP Server |

## MCP vs Tools

- **Tools** = Functions the agent can call (defined in code)
- **MCP** = External services the agent connects to (via protocol)

```
┌─────────────┐     Tool Call      ┌──────────────┐
│    Agent    │ ─────────────────> │   Function   │  (In-process)
└─────────────┘                    └──────────────┘

┌─────────────┐     MCP Protocol   ┌──────────────┐
│    Agent    │ ═════════════════> │  MCP Server  │  (External process)
└─────────────┘                    └──────────────┘
```

## Essential MCP Servers (To Build)

These MCP servers should be implemented to provide common external capabilities.

### 1. GitHub MCP Server

Access repositories, issues, PRs, and actions.

```yaml
tools:
  - github_get_file: Read file from repo
  - github_create_pr: Create pull request
  - github_list_issues: List repository issues
  - github_add_comment: Comment on issue/PR
```

### 2. Database MCP Server

Query and modify databases.

```yaml
tools:
  - db_query: Execute SELECT queries
  - db_execute: Execute INSERT/UPDATE/DELETE
  - db_schema: Get table schemas
  - db_tables: List available tables
```

### 3. Human Approval MCP Server

Human-in-the-loop workflows.

```yaml
tools:
  - human_approve: Request approval (returns when human responds)
  - human_input: Request freeform input
  - human_notify: Send notification (no response needed)
```

### 4. Slack MCP Server

Team communication.

```yaml
tools:
  - slack_send: Send message to channel
  - slack_dm: Direct message a user
  - slack_get_messages: Read channel history
```

### 5. File System MCP Server

Local file operations.

```yaml
tools:
  - fs_read: Read file contents
  - fs_write: Write to file
  - fs_list: List directory contents
  - fs_search: Search for files by pattern
```

### 6. Cache MCP Server

Memoize expensive operations.

```yaml
tools:
  - cache_get: Retrieve cached value
  - cache_set: Store value with TTL
  - cache_delete: Remove cached value
```

## MCP Server Reference Implementation

Example of how an MCP server should be structured:

```python
# mcp_servers/human_server.py
from mcp import MCPServer
import asyncio

class HumanMCPServer(MCPServer):
    def __init__(self, slack_webhook: str):
        self.slack = slack_webhook
        self.pending = {}

    async def human_approve(
        self,
        request_id: str,
        title: str,
        description: str,
        timeout_minutes: int = 60
    ) -> dict:
        """Request human approval via Slack."""
        # Send to Slack with approve/reject buttons
        await self._send_slack_approval(request_id, title, description)

        # Wait for response
        try:
            response = await asyncio.wait_for(
                self._wait_for_response(request_id),
                timeout=timeout_minutes * 60
            )
            return {"approved": response["approved"], "by": response["user"]}
        except asyncio.TimeoutError:
            return {"approved": False, "reason": "timeout"}

    async def human_notify(
        self,
        message: str,
        channel: str,
        priority: str = "normal"
    ) -> dict:
        """Send notification (no response needed)."""
        await self._send_slack_message(channel, message, priority)
        return {"sent": True}
```

---

# Putting It All Together (Vision)

> **Note:** This example shows how the four layers would work together once Operators, Skills, and MCP are implemented.

## Example: Document Processing Pipeline

```python
from langpy_sdk import Workflow, Pipe, Memory, Parser, Chunker, Embed

# Load skills
from skills import code_review_skill, data_analysis_skill

# Connect MCP
from mcp_clients import github_mcp, human_mcp

# Create primitives
pipe = Pipe(model="gpt-4o", system="You are a helpful assistant.")
memory = Memory(name="documents")
parser = Parser()
chunker = Chunker(chunk_size=500)
embed = Embed()

# Build workflow
workflow = Workflow(name="document-qa")

@workflow.step("ingest")
async def ingest(ctx):
    """Parse, chunk, embed, and store document."""
    text = await parser.parse(ctx.data["file_path"])
    chunks = await chunker.chunk(text)
    embeddings = await embed.embed_many(chunks)
    await memory.add_many(chunks, embeddings)
    return {"chunks": len(chunks)}

@workflow.step("query", depends_on=["ingest"])
async def query(ctx):
    """Answer questions using RAG."""
    from operators import loop_until

    question = ctx.data["question"]

    # Search memory
    results = await memory.search(question, limit=5)
    context = "\n".join([r.text for r in results])

    # Generate and refine answer
    answer = await loop_until(
        action=lambda prev: pipe.run(
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            f"Previous attempt: {prev}\n\nProvide a better answer:"
        ),
        condition=lambda ans: len(ans) > 100 and "I don't know" not in ans,
        initial="",
        max_iterations=3
    )

    return {"answer": answer["result"]}

@workflow.step("review", depends_on=["query"])
async def review(ctx):
    """Get human approval for important answers."""
    answer = ctx.results["query"]["answer"]

    if ctx.data.get("requires_approval"):
        approval = await human_mcp.human_approve(
            request_id=ctx.workflow_run_id,
            title="Answer Review",
            description=f"Please review this answer:\n\n{answer}"
        )

        if not approval["approved"]:
            return {"answer": answer, "status": "rejected"}

    return {"answer": answer, "status": "approved"}

# Run
result = await workflow.run({
    "file_path": "report.pdf",
    "question": "What are the key findings?",
    "requires_approval": True
})
```

## Architecture Summary

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                        WORKFLOW                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │  Step 1 │───>│  Step 2 │───>│  Step 3 │───>│  Step 4 │   │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘   │
│       │              │              │              │         │
│  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐   │
│  │ Parser  │    │  Pipe   │    │ Memory  │    │  Agent  │   │
│  │Primitive│    │Primitive│    │Primitive│    │Primitive│   │
│  └─────────┘    └────┬────┘    └─────────┘    └────┬────┘   │
│                      │                             │         │
│                 ┌────▼────┐                   ┌────▼────┐    │
│                 │loop_until│                  │ fallback│    │
│                 │ Operator │                  │ Operator│    │
│                 └─────────┘                   └─────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
     │code-rev │    │ GitHub  │    │ Human   │
     │  Skill  │    │   MCP   │    │   MCP   │
     └─────────┘    └─────────┘    └─────────┘
```

---

# Comparison: LangPy vs Langbase

Both platforms share the "primitives over frameworks" philosophy:

| Aspect | LangPy | Langbase |
|--------|--------|----------|
| Core Primitives | 9 (Pipe, Agent, Memory, Thread, Workflow, Parser, Chunker, Embed, Tools) | 5 (Pipes, Memory, Agents, Workflows, Tools) |
| Control Flow | Operators (explicit layer) | Built into Workflows |
| Domain Knowledge | Skills (Agent Skills standard) | In prompts/memory |
| External Access | MCP (standardized protocol) | Tools |
| Explicit Layers | 4 | 1 |

LangPy's four-layer model provides more explicit separation of concerns, while Langbase keeps everything flatter. Both approaches are valid - choose based on your need for structure vs simplicity.

---

# Implementation Roadmap

## Phase 1: Operators
- [ ] Create `langpy_sdk/operators.py`
- [ ] Implement `loop_until`
- [ ] Implement `retry`
- [ ] Implement `parallel`
- [ ] Implement `fallback`
- [ ] Implement `branch`
- [ ] Implement `select`
- [ ] Implement `stop`
- [ ] Implement `wait`
- [ ] Add tests for all operators
- [ ] Add operator documentation

## Phase 2: Skills Infrastructure
- [ ] Create `skills/` directory structure
- [ ] Build skill loader for LangPy
- [ ] Implement skill discovery
- [ ] Create first domain skill (e.g., `code-review`)

## Phase 3: Example Skills
- [ ] `code-review` - Code review expertise
- [ ] `sql-expert` - SQL query writing
- [ ] `data-analysis` - Statistical analysis
- [ ] `invoice-processing` - Invoice data extraction

## Phase 4: MCP Integration
- [ ] MCP client integration in LangPy
- [ ] Human approval MCP server
- [ ] Database MCP server
- [ ] File system MCP server

## Phase 5: Advanced MCP
- [ ] GitHub MCP server
- [ ] Slack MCP server
- [ ] Cache MCP server

---

# Contributing

## Adding an Operator

1. Implement as async function in `langpy_sdk/operators.py`
2. Follow the pattern: `async def operator_name(action, **params) -> result`
3. Return dict with result and metadata
4. Add tests and docstring
5. Submit PR

## Adding a Skill

1. Create `skills/<skill-name>/SKILL.md`
2. Follow [agentskills.io](https://agentskills.io) format
3. Focus on domain knowledge, not generic patterns
4. Add examples and edge cases
5. Submit PR

## Adding an MCP Server

1. Create `mcp_servers/<server-name>/`
2. Implement using [MCP SDK](https://modelcontextprotocol.io)
3. Define tool schemas
4. Add Docker setup for local dev
5. Document infrastructure requirements
6. Submit PR
