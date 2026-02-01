# LangPy Extension Guide: Skills & MCP Servers

This document defines how to extend LangPy using the **Agent Skills** open standard and **MCP Servers** for external capabilities.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          LangPy                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CORE PRIMITIVES (9)                                      │  │
│  │  Pipe, Agent, Memory, Thread, Workflow                    │  │
│  │  Parser, Chunker, Embed, Tools                            │  │
│  │                                                           │  │
│  │  The atoms - minimal, composable, stable                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  SKILLS (Agent Skills Standard)                           │  │
│  │  Markdown instructions + optional scripts                 │  │
│  │  Loop, Router, Evaluator, Guard, Fallback, etc.           │  │
│  │                                                           │  │
│  │  Portable, cross-platform, progressive disclosure         │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MCP Protocol
┌──────────────────────────▼──────────────────────────────────────┐
│  MCP SERVERS (External Capabilities)                            │
│  Queue, Cache, Scheduler, Human, Graph, Stream, State, Bridge   │
│                                                                 │
│  External infrastructure accessed via Model Context Protocol    │
└─────────────────────────────────────────────────────────────────┘
```

## When to Use Skills vs MCP

| Use Case | Solution | Why |
|----------|----------|-----|
| Control flow patterns | **Skill** | Instructions + scripts, no infrastructure |
| LLM-driven decisions | **Skill** | Agent follows instructions |
| External state/storage | **MCP** | Requires infrastructure |
| Real-time communication | **MCP** | Requires persistent connections |
| Third-party integrations | **MCP** | Requires API access |
| Human-in-the-loop | **MCP** | Requires external UI/notification |

---

# SKILLS (Agent Skills Standard)

Skills follow the [Agent Skills open standard](https://agentskills.io) - a portable format adopted by Claude, Cursor, GitHub Copilot, and other AI tools.

## Skill Structure

```
skill-name/
├── SKILL.md              # Required: frontmatter + instructions
├── scripts/              # Optional: executable code
│   ├── loop.py
│   └── evaluate.py
├── references/           # Optional: detailed documentation
│   └── patterns.md
└── assets/               # Optional: templates, examples
    └── examples.json
```

## SKILL.md Format

```markdown
---
name: skill-name
description: What this skill does and when to use it.
license: MIT
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe, Workflow]
---

## Overview

Brief explanation of the skill.

## When to Use

- Condition 1
- Condition 2

## Instructions

Step-by-step guide for the agent...

## Examples

Input/output examples...
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Lowercase, hyphens only, max 64 chars |
| `description` | Yes | What it does and when to use it, max 1024 chars |
| `license` | No | License name or file reference |
| `compatibility` | No | Environment requirements |
| `metadata` | No | Custom key-value pairs (author, version, primitives used) |
| `allowed-tools` | No | Pre-approved tools the skill may use |

---

## Skill Definitions

### 1. Loop Skill

**Purpose:** Iterate until a condition is met (Workflow is DAG-based, acyclic)

```
skills/
└── loop/
    ├── SKILL.md
    └── scripts/
        └── loop_executor.py
```

**SKILL.md:**
```markdown
---
name: loop
description: Execute a primitive repeatedly until a condition is satisfied. Use when you need iterative refinement, retry-until-success, or convergence patterns.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe, Agent, Workflow]
---

## Overview

The Loop skill enables iterative execution patterns that aren't possible with
Workflow's DAG-based structure. It wraps any primitive and repeats execution
until a condition evaluates to true.

## When to Use

- Iterative refinement (generate → evaluate → improve → repeat)
- Retry until quality threshold met
- Convergence algorithms
- Polling for external state changes

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| primitive | Pipe/Agent | Yes | The primitive to execute each iteration |
| condition | string | Yes | Natural language condition for stopping |
| max_iterations | int | No | Maximum iterations (default: 10) |
| on_iteration | string | No | Action to take between iterations |

## Instructions

1. Initialize the loop with the target primitive and stopping condition
2. Execute the primitive with the current input
3. Evaluate the condition against the output
4. If condition is TRUE → return the result
5. If condition is FALSE and iterations < max → go to step 2
6. If max iterations reached → return with partial result and warning

## Condition Evaluation

The condition should be a natural language description that you evaluate
against the output. Examples:

- "The response contains a valid JSON object"
- "The quality score is above 0.8"
- "The user has confirmed the action"
- "No errors are present in the output"

## Examples

### Example 1: Iterative Refinement

```
Input:
  primitive: Pipe(system="Improve this text")
  initial: "AI is good"
  condition: "The text is at least 100 words and professionally written"
  max_iterations: 5

Iteration 1:
  Output: "Artificial intelligence is a transformative technology..."
  Condition met: No (only 45 words)

Iteration 2:
  Output: "Artificial intelligence represents one of the most significant..."
  Condition met: Yes (120 words, professional tone)

Result: Final refined text after 2 iterations
```

### Example 2: Quality Gate

```
Input:
  primitive: Agent(tools=[generate_report])
  condition: "Report has executive summary, data tables, and conclusions"
  max_iterations: 3

Execution continues until report has all required sections.
```

## Edge Cases

- If max_iterations is reached, return the best result with a warning
- If primitive fails, capture error and retry (counts as iteration)
- If condition is ambiguous, err on the side of continuing

## Script Reference

For deterministic loop execution, use:
```
scripts/loop_executor.py --primitive <name> --condition "<condition>" --max <n>
```
```

**scripts/loop_executor.py:**
```python
#!/usr/bin/env python3
"""
Loop executor script for deterministic iteration control.
"""

import asyncio
import argparse
import json
from typing import Any, Callable

async def execute_loop(
    primitive_runner: Callable,
    condition_checker: Callable,
    initial_input: Any,
    max_iterations: int = 10
) -> dict:
    """Execute a loop until condition is met."""
    result = initial_input

    for i in range(max_iterations):
        # Execute primitive
        result = await primitive_runner(result)

        # Check condition
        if condition_checker(result):
            return {
                "success": True,
                "result": result,
                "iterations": i + 1,
                "converged": True
            }

    return {
        "success": True,
        "result": result,
        "iterations": max_iterations,
        "converged": False,
        "warning": "Max iterations reached"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=10)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    # Script entry point for CLI usage
    print(json.dumps({"status": "ready", "max_iterations": args.max}))
```

---

### 2. Router Skill

**Purpose:** Dynamic dispatch to different primitives based on input classification

```
skills/
└── router/
    ├── SKILL.md
    └── references/
        └── routing-strategies.md
```

**SKILL.md:**
```markdown
---
name: router
description: Route inputs to specialized handlers based on classification. Use when different types of requests need different processing paths.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe, Agent]
---

## Overview

The Router skill classifies incoming requests and dispatches them to
specialized primitives. It uses Pipe.classify() internally and supports
fallback routes.

## When to Use

- Multi-domain assistants (technical vs creative vs factual)
- Intent-based routing
- Load distribution across specialized agents
- A/B testing different approaches

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| routes | dict | Yes | Map of category → primitive |
| default | string | No | Fallback category if no match |
| confidence_threshold | float | No | Minimum confidence to route (0-1) |

## Instructions

1. Receive the input request
2. Use Pipe.classify() with the route categories
3. If classification confidence >= threshold → route to that handler
4. If confidence < threshold → use default route or ask for clarification
5. Execute the selected primitive
6. Return result with routing metadata

## Route Definition

Define routes as a mapping of categories to primitives:

```yaml
routes:
  technical:
    primitive: Agent
    tools: [search_docs, run_code]
    system: "You are a technical expert"

  creative:
    primitive: Pipe
    system: "You are a creative writer"

  factual:
    primitive: Pipe
    system: "Give factual, sourced answers"
```

## Examples

### Example 1: Support Router

```
Input: "My code is throwing a null pointer exception"

Classification:
  - technical: 0.92
  - billing: 0.03
  - general: 0.05

Route: technical (confidence 0.92)
Handler: Agent with debugging tools
```

### Example 2: Low Confidence

```
Input: "Help me with my thing"

Classification:
  - technical: 0.35
  - creative: 0.30
  - general: 0.35

Confidence below threshold (0.7)
Action: Use default route OR ask for clarification
```

## Routing Strategies

See [references/routing-strategies.md](references/routing-strategies.md) for:
- Round-robin routing
- Load-based routing
- Capability-based routing
- Cascade routing (try primary, fall back to secondary)
```

---

### 3. Evaluator Skill

**Purpose:** Score and assess output quality with pass/fail threshold

```
skills/
└── evaluator/
    ├── SKILL.md
    ├── scripts/
    │   └── score_parser.py
    └── references/
        └── evaluation-criteria.md
```

**SKILL.md:**
```markdown
---
name: evaluator
description: Evaluate content quality against criteria and return scores with pass/fail status. Use for quality gates, content moderation, or output validation.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe]
---

## Overview

The Evaluator skill assesses content against specified criteria, returning
a normalized score (0-1) and pass/fail status based on a threshold.

## When to Use

- Quality gates in generation pipelines
- Content moderation
- Self-assessment in agent loops
- A/B comparison of outputs
- Regression testing for prompts

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| criteria | string | Yes | What to evaluate for |
| threshold | float | No | Pass threshold 0-1 (default: 0.7) |
| rubric | dict | No | Detailed scoring rubric |

## Instructions

1. Receive content to evaluate and criteria
2. Construct evaluation prompt with criteria
3. Use Pipe to generate structured assessment
4. Parse score and reasoning from response
5. Compare score against threshold
6. Return evaluation result with metadata

## Evaluation Prompt Template

```
Evaluate this content on a scale of 0-10.

Criteria: {criteria}

Content:
---
{content}
---

Respond with ONLY a JSON object:
{
  "score": <number 0-10>,
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "reasoning": "<brief explanation>"
}
```

## Examples

### Example 1: Writing Quality

```
Criteria: "Clear, concise, and professionally written"
Content: "AI is really really good and stuff..."
Threshold: 0.7

Result:
{
  "score": 0.3,
  "passed": false,
  "strengths": ["On topic"],
  "weaknesses": ["Informal language", "Vague", "Repetitive"],
  "reasoning": "Content lacks professional tone and specificity"
}
```

### Example 2: Code Review

```
Criteria: "Follows best practices, handles errors, is well-documented"
Content: <code snippet>
Threshold: 0.8

Result:
{
  "score": 0.85,
  "passed": true,
  "strengths": ["Good error handling", "Clear naming"],
  "weaknesses": ["Missing docstring on helper function"],
  "reasoning": "Solid code with minor documentation gap"
}
```

## Rubric-Based Evaluation

For complex evaluations, define a rubric:

```yaml
rubric:
  accuracy:
    weight: 0.4
    description: "Factual correctness"
  clarity:
    weight: 0.3
    description: "Easy to understand"
  completeness:
    weight: 0.3
    description: "Covers all aspects"
```

Final score = weighted average of rubric dimensions.
```

---

### 4. Guard Skill

**Purpose:** Validate inputs and outputs against rules

```
skills/
└── guard/
    ├── SKILL.md
    └── scripts/
        └── validators.py
```

**SKILL.md:**
```markdown
---
name: guard
description: Validate inputs and outputs against defined rules. Use for safety checks, format validation, PII detection, or content filtering.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe]
---

## Overview

The Guard skill wraps primitives with input and output validation. It can
reject, sanitize, or warn based on validation results.

## When to Use

- PII detection and removal
- Content policy enforcement
- Format validation (JSON, email, etc.)
- Length limits
- Profanity filtering
- Prompt injection detection

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input_rules | list | No | Validators for input |
| output_rules | list | No | Validators for output |
| on_input_fail | string | No | "reject", "sanitize", or "warn" |
| on_output_fail | string | No | "reject", "sanitize", or "warn" |

## Built-in Validators

| Validator | Description |
|-----------|-------------|
| no_pii | Detects emails, SSNs, phone numbers |
| max_length(n) | Enforces character limit |
| valid_json | Checks JSON parsability |
| no_profanity | Filters inappropriate language |
| no_injection | Detects prompt injection attempts |
| matches_schema(s) | Validates against JSON schema |

## Instructions

1. Receive input and validation rules
2. Run input through each input validator
3. If any fails:
   - reject: Return error immediately
   - sanitize: Attempt to fix and continue
   - warn: Log warning and continue
4. Execute the wrapped primitive
5. Run output through each output validator
6. Apply same fail strategy for output
7. Return result with validation metadata

## Examples

### Example 1: PII Guard

```
Input: "My email is john@example.com and SSN is 123-45-6789"

Input Validators: [no_pii]
On Fail: sanitize

Sanitized Input: "My email is [EMAIL] and SSN is [SSN]"

→ Primitive executes with sanitized input
→ Output checked for PII before returning
```

### Example 2: Format Guard

```
Input: "Generate a JSON config"

Output Validators: [valid_json, matches_schema(config_schema)]
On Fail: reject

Output: "{ invalid json }"

Result: {
  "error": "Output validation failed",
  "validator": "valid_json",
  "message": "Invalid JSON: unexpected token"
}
```

## Custom Validators

Define custom validators in scripts/validators.py:

```python
def no_competitor_mentions(text: str) -> tuple[bool, str]:
    competitors = ["CompetitorA", "CompetitorB"]
    for c in competitors:
        if c.lower() in text.lower():
            return False, f"Contains competitor mention: {c}"
    return True, "OK"
```
```

---

### 5. Fallback Skill

**Purpose:** Try primary handler, fall back to alternatives on failure

```
skills/
└── fallback/
    └── SKILL.md
```

**SKILL.md:**
```markdown
---
name: fallback
description: Execute primitives in order until one succeeds. Use for resilience, model fallbacks, or graceful degradation.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe, Agent]
---

## Overview

The Fallback skill provides resilience by trying multiple primitives in
sequence until one succeeds.

## When to Use

- Model fallbacks (GPT-4 → GPT-3.5 → local model)
- Service resilience
- Graceful degradation
- Cost optimization (try cheap first, fall back to expensive)

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| primitives | list | Yes | Ordered list of primitives to try |
| catch | list | No | Exception types to catch (default: all) |
| on_fallback | string | No | Action when falling back |

## Instructions

1. Receive input and ordered list of primitives
2. Try first primitive
3. If success → return result with index 0
4. If failure → log error, try next primitive
5. Repeat until success or all primitives exhausted
6. Return result with fallback metadata

## Examples

### Example 1: Model Fallback

```
Primitives:
  1. Pipe(model="gpt-4o")
  2. Pipe(model="gpt-4o-mini")
  3. Pipe(model="gpt-3.5-turbo")

Attempt 1: gpt-4o → Rate limit error
Attempt 2: gpt-4o-mini → Success

Result: {
  "result": "...",
  "primitive_index": 1,
  "fallback_used": true,
  "attempts": 2
}
```

### Example 2: Strategy Fallback

```
Primitives:
  1. Agent(tools=[web_search])  # Try real-time data
  2. Agent(tools=[knowledge_base])  # Fall back to cached
  3. Pipe(system="Answer from training data")  # Last resort

Gracefully degrades from live data to cached to parametric knowledge.
```
```

---

### 6. Aggregator Skill

**Purpose:** Combine outputs from parallel executions

```
skills/
└── aggregator/
    └── SKILL.md
```

**SKILL.md:**
```markdown
---
name: aggregator
description: Execute multiple primitives in parallel and combine results. Use for ensemble methods, multi-perspective analysis, or parallel processing.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe, Agent, Workflow]
---

## Overview

The Aggregator skill runs multiple primitives in parallel and combines
their outputs using a specified strategy.

## When to Use

- Ensemble responses (multiple models, take consensus)
- Multi-perspective analysis
- Parallel data processing
- Reducing variance in outputs

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| all | Return all results as a list |
| first | Return first successful result |
| majority | Return most common result (for discrete outputs) |
| concat | Concatenate all outputs |
| synthesize | Use LLM to synthesize results |
| custom | User-defined aggregation function |

## Instructions

1. Receive input and list of primitives
2. Execute all primitives in parallel (asyncio.gather)
3. Collect results, noting any failures
4. Apply aggregation strategy
5. Return combined result with metadata

## Examples

### Example 1: Multi-Perspective

```
Primitives:
  - Pipe(system="You are an optimist")
  - Pipe(system="You are a pessimist")
  - Pipe(system="You are a realist")

Strategy: synthesize

Input: "What's the future of remote work?"

Parallel Outputs:
  - Optimist: "Remote work will become universal..."
  - Pessimist: "Companies will force return to office..."
  - Realist: "Hybrid models will dominate..."

Synthesized Result: "Perspectives vary on remote work's future.
Optimists see universal adoption, pessimists predict office returns,
while realists expect hybrid models to prevail. The likely outcome
is industry-dependent hybrid arrangements."
```

### Example 2: Majority Vote

```
Primitives: [Pipe(...), Pipe(...), Pipe(...)]  # Same prompt, 3x
Strategy: majority

Input: "Is this email spam? Answer: yes or no"

Outputs: ["yes", "yes", "no"]
Result: "yes" (2/3 majority)
```
```

---

### 7. Retry Skill

**Purpose:** Sophisticated retry with backoff and conditions

```
skills/
└── retry/
    ├── SKILL.md
    └── scripts/
        └── backoff.py
```

**SKILL.md:**
```markdown
---
name: retry
description: Retry primitive execution with exponential backoff and custom conditions. Use for handling transient failures and rate limits.
metadata:
  author: langpy
  version: "1.0"
  primitives: [Pipe, Agent]
---

## Overview

The Retry skill provides sophisticated retry logic with exponential backoff,
jitter, and conditional retry based on error types or output content.

## When to Use

- API rate limit handling
- Transient network failures
- Flaky external services
- Output quality retries

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| max_retries | int | No | Maximum retry attempts (default: 3) |
| backoff_base | float | No | Initial wait seconds (default: 1.0) |
| backoff_multiplier | float | No | Multiplier per retry (default: 2.0) |
| max_backoff | float | No | Maximum wait seconds (default: 60) |
| jitter | bool | No | Add randomness to backoff (default: true) |
| retry_on | list | No | Error types to retry on |
| retry_if | string | No | Condition to trigger retry |

## Backoff Calculation

```
wait_time = min(
  backoff_base * (backoff_multiplier ** attempt),
  max_backoff
)

if jitter:
  wait_time = wait_time * random(0.5, 1.5)
```

## Examples

### Example 1: Rate Limit Handling

```
Parameters:
  max_retries: 5
  backoff_base: 1.0
  backoff_multiplier: 2.0
  retry_on: [RateLimitError]

Attempt 1: RateLimitError → wait 1s
Attempt 2: RateLimitError → wait 2s
Attempt 3: RateLimitError → wait 4s
Attempt 4: Success

Result includes retry metadata.
```

### Example 2: Quality Retry

```
Parameters:
  max_retries: 3
  retry_if: "Response is less than 50 words"

Attempt 1: "AI is good." (4 words) → retry
Attempt 2: "Artificial intelligence is..." (48 words) → retry
Attempt 3: "Artificial intelligence represents..." (120 words) → success
```
```

---

# MCP SERVERS (External Capabilities)

MCP Servers provide capabilities that require external infrastructure. These are accessed via [Model Context Protocol](https://modelcontextprotocol.io).

**When Skills aren't enough:**
- Persistent state across sessions
- Real-time communication
- External service integration
- Human interaction

---

## 1. Queue MCP Server

**Purpose:** Async message passing between agents/primitives

**Infrastructure:** Redis, RabbitMQ, SQS, or Kafka

**Why MCP:** Requires persistent message broker infrastructure

### MCP Tools

```yaml
tools:
  - name: queue_publish
    description: Publish a message to a queue
    parameters:
      queue_name: string (required)
      message: string (required)
      priority: integer (default: 0)
      delay_seconds: integer (default: 0)

  - name: queue_consume
    description: Consume messages from a queue
    parameters:
      queue_name: string (required)
      count: integer (default: 1)
      timeout_seconds: integer (default: 30)

  - name: queue_length
    description: Get queue depth
    parameters:
      queue_name: string (required)
```

### Use Cases

- Agent-to-agent task delegation
- Work distribution across agent pool
- Event-driven pipelines
- Decoupled async processing

### Example Implementation

```python
# mcp_servers/queue_server.py
from mcp import MCPServer
import redis

class QueueMCPServer(MCPServer):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def queue_publish(self, queue_name: str, message: str, priority: int = 0):
        self.redis.lpush(queue_name, message)
        return {"status": "published", "queue": queue_name}

    async def queue_consume(self, queue_name: str, timeout_seconds: int = 30):
        result = self.redis.brpop(queue_name, timeout=timeout_seconds)
        if result:
            return {"message": result[1].decode(), "queue": queue_name}
        return {"message": None, "timeout": True}
```

---

## 2. Cache MCP Server

**Purpose:** Memoize expensive LLM calls and embeddings

**Infrastructure:** Redis, Memcached, or DynamoDB

**Why MCP:** Requires persistent cache storage

### MCP Tools

```yaml
tools:
  - name: cache_get
    description: Retrieve cached value
    parameters:
      key: string (required)

  - name: cache_set
    description: Store value with TTL
    parameters:
      key: string (required)
      value: string (required)
      ttl_seconds: integer (default: 3600)

  - name: cache_delete
    description: Remove cached value
    parameters:
      key: string (required)

  - name: cache_exists
    description: Check if key exists
    parameters:
      key: string (required)
```

### Use Cases

- Cache LLM responses for common queries
- Store computed embeddings
- Reduce API costs
- Speed up repeated operations

### Cache Key Strategies

```
# For LLM calls
key = hash(model + system_prompt + user_message + temperature)

# For embeddings
key = hash(model + text)

# For search results
key = hash(query + filters + limit)
```

---

## 3. Human MCP Server

**Purpose:** Human-in-the-loop approval and input

**Infrastructure:** Slack, Email, Web UI, or custom approval system

**Why MCP:** Requires external notification/approval interface

### MCP Tools

```yaml
tools:
  - name: human_approve
    description: Request human approval for an action
    parameters:
      request_id: string (required)
      title: string (required)
      description: string (required)
      options: array of strings (default: ["Approve", "Reject"])
      timeout_minutes: integer (default: 60)
      channel: string (enum: slack, email, ui)
      assignee: string (optional)

  - name: human_input
    description: Request freeform input from human
    parameters:
      request_id: string (required)
      prompt: string (required)
      input_type: string (enum: text, choice, file)
      timeout_minutes: integer (default: 60)

  - name: human_notify
    description: Send notification (no response needed)
    parameters:
      message: string (required)
      channel: string (required)
      priority: string (enum: low, normal, high, urgent)

  - name: human_check
    description: Check status of pending request
    parameters:
      request_id: string (required)
```

### Use Cases

- Approval workflows for sensitive actions
- Content moderation
- Exception handling escalation
- Quality assurance checkpoints

### Example Flow

```
1. Agent encounters high-risk action
2. Calls human_approve with details
3. Human receives Slack message
4. Human clicks Approve/Reject
5. Agent receives response and continues
```

---

## 4. Graph MCP Server

**Purpose:** Relationship-based memory (beyond vector similarity)

**Infrastructure:** Neo4j, Amazon Neptune, TigerGraph, or Memgraph

**Why MCP:** Requires graph database infrastructure

### MCP Tools

```yaml
tools:
  - name: graph_add_node
    description: Create a node
    parameters:
      node_type: string (required)
      properties: object (required)
      id: string (optional, auto-generated if not provided)

  - name: graph_add_edge
    description: Create relationship between nodes
    parameters:
      from_id: string (required)
      to_id: string (required)
      relationship: string (required)
      properties: object (optional)

  - name: graph_query
    description: Query the graph
    parameters:
      query: string (required)
      query_type: string (enum: cypher, natural, pattern)
      parameters: object (optional)

  - name: graph_traverse
    description: Traverse from a starting node
    parameters:
      start_id: string (required)
      relationship_types: array (optional)
      direction: string (enum: outgoing, incoming, both)
      max_depth: integer (default: 3)

  - name: graph_find_path
    description: Find path between nodes
    parameters:
      from_id: string (required)
      to_id: string (required)
      max_hops: integer (default: 5)
```

### Use Cases

- Knowledge graphs
- Entity relationship mapping
- Multi-hop reasoning
- Recommendation systems
- Dependency analysis

### Graph vs Vector Memory

| Use Case | Vector (Memory) | Graph |
|----------|-----------------|-------|
| "Find similar documents" | ✅ | ❌ |
| "What is X related to?" | ❌ | ✅ |
| "How is A connected to B?" | ❌ | ✅ |
| "Find documents about X" | ✅ | ⚠️ |
| "What depends on X?" | ❌ | ✅ |

---

## 5. State MCP Server

**Purpose:** Persistent state machines for complex flows

**Infrastructure:** Redis, PostgreSQL, or DynamoDB

**Why MCP:** Requires persistent state storage across sessions

### MCP Tools

```yaml
tools:
  - name: state_create
    description: Create new state machine instance
    parameters:
      machine_id: string (required)
      definition: string (required, state machine name)
      initial_context: object (optional)

  - name: state_transition
    description: Trigger state transition
    parameters:
      machine_id: string (required)
      event: string (required)
      data: object (optional)

  - name: state_get
    description: Get current state and context
    parameters:
      machine_id: string (required)

  - name: state_history
    description: Get transition history
    parameters:
      machine_id: string (required)
      limit: integer (default: 100)

  - name: state_list
    description: List active state machines
    parameters:
      definition: string (optional, filter by type)
      state: string (optional, filter by current state)
```

### State Machine Definition

```yaml
name: order-fulfillment
states:
  - pending
  - processing
  - shipped
  - delivered
  - cancelled

initial: pending

transitions:
  - from: pending
    to: processing
    event: start_processing

  - from: processing
    to: shipped
    event: ship_order

  - from: shipped
    to: delivered
    event: confirm_delivery

  - from: [pending, processing]
    to: cancelled
    event: cancel_order
```

### Use Cases

- Order processing workflows
- Document approval flows
- Multi-step onboarding
- Long-running agent tasks

---

## 6. Bridge MCP Server

**Purpose:** Inter-agent communication and coordination

**Infrastructure:** WebSocket server, gRPC, or message broker

**Why MCP:** Requires real-time agent-to-agent communication

### MCP Tools

```yaml
tools:
  - name: bridge_register
    description: Register agent with bridge
    parameters:
      agent_id: string (required)
      capabilities: array of strings (required)
      metadata: object (optional)

  - name: bridge_send
    description: Send message to another agent
    parameters:
      to_agent: string (required)
      message_type: string (required)
      payload: object (required)
      wait_response: boolean (default: false)
      timeout_seconds: integer (default: 30)

  - name: bridge_broadcast
    description: Broadcast to agents with capability
    parameters:
      capability: string (required)
      message_type: string (required)
      payload: object (required)

  - name: bridge_discover
    description: Find agents with capability
    parameters:
      capability: string (required)

  - name: bridge_subscribe
    description: Subscribe to message types
    parameters:
      message_types: array of strings (required)
```

### Use Cases

- Multi-agent collaboration
- Agent delegation
- Swarm coordination
- Service mesh for agents

### Example: Agent Delegation

```
Coordinator Agent:
  1. bridge_discover("code_review")
  2. Finds: [agent-1, agent-2, agent-3]
  3. bridge_send(to="agent-1", type="review_request", payload={...})
  4. Waits for response
  5. Aggregates results from delegated agents
```

---

## 7. Scheduler MCP Server

**Purpose:** Time-based and event-based triggers

**Infrastructure:** Celery, APScheduler, or cloud schedulers

**Why MCP:** Requires persistent job scheduling

### MCP Tools

```yaml
tools:
  - name: schedule_once
    description: Schedule one-time job
    parameters:
      job_id: string (required)
      run_at: datetime string (required)
      action: string (required)
      payload: object (optional)

  - name: schedule_recurring
    description: Schedule recurring job
    parameters:
      job_id: string (required)
      cron: string (required, cron expression)
      action: string (required)
      payload: object (optional)

  - name: schedule_cancel
    description: Cancel scheduled job
    parameters:
      job_id: string (required)

  - name: schedule_list
    description: List scheduled jobs
    parameters:
      status: string (optional, enum: pending, running, completed)
```

### Use Cases

- Periodic data ingestion
- Scheduled reports
- Reminder systems
- Delayed actions

---

## 8. Stream MCP Server

**Purpose:** Real-time event streaming

**Infrastructure:** Kafka, Redis Streams, or Pulsar

**Why MCP:** Requires persistent event streaming infrastructure

### MCP Tools

```yaml
tools:
  - name: stream_publish
    description: Publish event to stream
    parameters:
      stream: string (required)
      event_type: string (required)
      data: object (required)

  - name: stream_subscribe
    description: Subscribe to stream
    parameters:
      stream: string (required)
      event_types: array (optional, filter)
      from_position: string (enum: latest, earliest, timestamp)

  - name: stream_read
    description: Read events from stream
    parameters:
      stream: string (required)
      count: integer (default: 10)
      block_ms: integer (default: 0)
```

### Use Cases

- Real-time agent coordination
- Audit logging
- Event-driven architectures
- Live dashboards

---

# Implementation Roadmap

## Phase 1: Core Skills
1. ✅ Define skill format (Agent Skills standard)
2. 🔲 Implement Loop skill
3. 🔲 Implement Router skill
4. 🔲 Implement Evaluator skill
5. 🔲 Skill loader for LangPy

## Phase 2: Safety Skills
6. 🔲 Implement Guard skill
7. 🔲 Implement Fallback skill
8. 🔲 Implement Retry skill

## Phase 3: Composition Skills
9. 🔲 Implement Aggregator skill
10. 🔲 Skill-to-skill composition

## Phase 4: Essential MCP Servers
11. 🔲 Cache MCP Server
12. 🔲 Human MCP Server
13. 🔲 Queue MCP Server

## Phase 5: Advanced MCP Servers
14. 🔲 Graph MCP Server
15. 🔲 State MCP Server
16. 🔲 Bridge MCP Server
17. 🔲 Scheduler MCP Server
18. 🔲 Stream MCP Server

---

# Contributing

## Adding a Skill

1. Create directory: `skills/<skill-name>/`
2. Create `SKILL.md` with required frontmatter
3. Add optional scripts in `scripts/`
4. Add optional references in `references/`
5. Validate with: `skills-ref validate ./skills/<skill-name>`
6. Add tests and examples
7. Submit PR

## Adding an MCP Server

1. Create directory: `mcp_servers/<server-name>/`
2. Define tool schemas in YAML
3. Implement server with MCP SDK
4. Add Docker/docker-compose for local dev
5. Document infrastructure requirements
6. Add health checks and monitoring
7. Submit PR

## Standards

- Skills follow [agentskills.io](https://agentskills.io) specification
- MCP Servers follow [modelcontextprotocol.io](https://modelcontextprotocol.io) specification
- All extensions must include tests
- All extensions must include usage examples
