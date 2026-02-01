# LangPy Extension Guide: Skills & MCP Servers

This document outlines the extension architecture for LangPy, defining patterns that can be built as **Skills** (internal reusable patterns) or **MCP Servers** (external capabilities).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        LangPy                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  CORE PRIMITIVES (9)                                  │  │
│  │  Pipe, Agent, Memory, Thread, Workflow                │  │
│  │  Parser, Chunker, Embed, Tools                        │  │
│  │                                                       │  │
│  │  These are the atoms - minimal, composable, stable    │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  SKILLS (Control Flow Patterns)                       │  │
│  │  Loop, Router, Evaluator, Guard, Fallback, etc.       │  │
│  │                                                       │  │
│  │  Reusable compositions of core primitives             │  │
│  │  No external dependencies, pure Python                │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │ MCP Protocol
┌──────────────────────────▼──────────────────────────────────┐
│  MCP SERVERS (External Capabilities)                        │
│  Queue, Cache, Scheduler, Human, Graph, Stream, etc.        │
│                                                             │
│  External services accessed via Model Context Protocol      │
│  Can be self-hosted or cloud services                       │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Core primitives stay minimal** - Don't add to the 9 primitives unless absolutely necessary
2. **Skills are compositions** - Built entirely from existing primitives
3. **MCP for external state** - Anything requiring external infrastructure goes through MCP
4. **Consistent interfaces** - All extensions follow similar patterns

---

# SKILLS (Internal Patterns)

Skills are reusable compositions of core primitives. They require no external dependencies.

## 1. Loop Skill

**Purpose:** Iterate until a condition is met (not possible in DAG-based Workflow)

**Why needed:** Workflow is acyclic - cannot express "retry until good"

**Implementation approach:**
```python
from langpy_sdk import Pipe, Skill

class LoopSkill(Skill):
    """Execute a primitive repeatedly until condition is satisfied."""

    def __init__(
        self,
        primitive,           # Pipe, Agent, or callable
        condition: callable, # Function that returns True to stop
        max_iterations: int = 10,
        on_iteration: callable = None  # Optional callback
    ):
        self.primitive = primitive
        self.condition = condition
        self.max_iterations = max_iterations
        self.on_iteration = on_iteration

    async def run(self, initial_input):
        result = initial_input
        for i in range(self.max_iterations):
            result = await self.primitive.quick(result)

            if self.on_iteration:
                self.on_iteration(i, result)

            if self.condition(result):
                return {"success": True, "result": result, "iterations": i + 1}

        return {"success": False, "result": result, "iterations": self.max_iterations}
```

**Usage example:**
```python
from langpy_sdk import Pipe
from langpy.skills import LoopSkill

refiner = Pipe(model="gpt-4o-mini", system="Improve this text")
evaluator = lambda text: "excellent" in text.lower()

loop = LoopSkill(
    primitive=refiner,
    condition=evaluator,
    max_iterations=5
)

result = await loop.run("Write about AI")
```

---

## 2. Router Skill

**Purpose:** Dynamic dispatch to different primitives based on input classification

**Why needed:** Common pattern that requires boilerplate with Pipe.classify()

**Implementation approach:**
```python
from langpy_sdk import Pipe, Skill

class RouterSkill(Skill):
    """Route inputs to specialized handlers based on classification."""

    def __init__(
        self,
        routes: dict,           # {"category": primitive}
        classifier: Pipe = None, # Optional custom classifier
        default: str = None      # Default route if no match
    ):
        self.routes = routes
        self.classifier = classifier or Pipe(model="gpt-4o-mini")
        self.default = default

    async def run(self, input_text: str):
        # Classify the input
        categories = list(self.routes.keys())
        category = await self.classifier.classify(input_text, categories=categories)

        # Route to appropriate handler
        if category in self.routes:
            handler = self.routes[category]
            result = await handler.quick(input_text)
            return {"category": category, "result": result}
        elif self.default and self.default in self.routes:
            handler = self.routes[self.default]
            result = await handler.quick(input_text)
            return {"category": self.default, "result": result, "fallback": True}
        else:
            return {"category": None, "error": "No matching route"}
```

**Usage example:**
```python
from langpy_sdk import Pipe, Agent
from langpy.skills import RouterSkill

router = RouterSkill(
    routes={
        "technical": Agent(model="gpt-4o-mini", tools=[search_docs, run_code]),
        "creative": Pipe(model="gpt-4o-mini", system="You are a creative writer"),
        "factual": Pipe(model="gpt-4o-mini", system="Give factual answers only"),
    },
    default="factual"
)

result = await router.run("Write me a poem about robots")
# Routes to "creative" handler
```

---

## 3. Evaluator Skill

**Purpose:** Score/assess output quality with pass/fail threshold

**Why needed:** Quality gates are common in agent pipelines

**Implementation approach:**
```python
from langpy_sdk import Pipe, Skill

class EvaluatorSkill(Skill):
    """Evaluate output quality and return score with pass/fail."""

    def __init__(
        self,
        criteria: str,           # What to evaluate for
        threshold: float = 0.7,  # Pass threshold (0-1)
        model: str = "gpt-4o-mini"
    ):
        self.criteria = criteria
        self.threshold = threshold
        self.pipe = Pipe(model=model)

    async def run(self, content: str):
        prompt = f"""Evaluate this content on a scale of 0-10.

Criteria: {self.criteria}

Content:
{content}

Respond with ONLY a JSON object:
{{"score": <number 0-10>, "reasoning": "<brief explanation>"}}"""

        response = await self.pipe.quick(prompt)
        result = json.loads(response)

        normalized_score = result["score"] / 10
        return {
            "score": normalized_score,
            "passed": normalized_score >= self.threshold,
            "reasoning": result["reasoning"],
            "threshold": self.threshold
        }
```

**Usage example:**
```python
from langpy.skills import EvaluatorSkill, LoopSkill
from langpy_sdk import Pipe

evaluator = EvaluatorSkill(
    criteria="Clear, concise, and technically accurate",
    threshold=0.8
)

generator = Pipe(model="gpt-4o-mini")
content = await generator.quick("Explain neural networks")
result = await evaluator.run(content)

if not result["passed"]:
    # Regenerate or refine
    pass
```

---

## 4. Guard Skill

**Purpose:** Validate input/output against rules before/after primitive execution

**Why needed:** Safety checks, format validation, content filtering

**Implementation approach:**
```python
from langpy_sdk import Skill

class GuardSkill(Skill):
    """Validate inputs and outputs with custom rules."""

    def __init__(
        self,
        primitive,
        input_validators: list = None,   # Functions that validate input
        output_validators: list = None,  # Functions that validate output
        on_input_fail: str = "reject",   # "reject" | "sanitize" | "warn"
        on_output_fail: str = "reject"
    ):
        self.primitive = primitive
        self.input_validators = input_validators or []
        self.output_validators = output_validators or []
        self.on_input_fail = on_input_fail
        self.on_output_fail = on_output_fail

    async def run(self, input_data):
        # Validate input
        for validator in self.input_validators:
            valid, message = validator(input_data)
            if not valid:
                if self.on_input_fail == "reject":
                    return {"error": f"Input validation failed: {message}"}
                elif self.on_input_fail == "warn":
                    print(f"Warning: {message}")

        # Execute primitive
        result = await self.primitive.quick(input_data)

        # Validate output
        for validator in self.output_validators:
            valid, message = validator(result)
            if not valid:
                if self.on_output_fail == "reject":
                    return {"error": f"Output validation failed: {message}"}

        return {"result": result, "validated": True}
```

**Usage example:**
```python
from langpy.skills import GuardSkill
from langpy_sdk import Pipe

def no_pii(text):
    # Check for email patterns, SSN, etc.
    import re
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        return False, "Contains email address"
    return True, "OK"

def max_length(text):
    if len(text) > 1000:
        return False, "Response too long"
    return True, "OK"

guarded_pipe = GuardSkill(
    primitive=Pipe(model="gpt-4o-mini"),
    input_validators=[no_pii],
    output_validators=[max_length, no_pii]
)

result = await guarded_pipe.run("Tell me about AI")
```

---

## 5. Fallback Skill

**Purpose:** Try primary primitive, fall back to alternatives on failure

**Why needed:** Resilience patterns, model fallbacks, graceful degradation

**Implementation approach:**
```python
from langpy_sdk import Skill

class FallbackSkill(Skill):
    """Try primitives in order until one succeeds."""

    def __init__(
        self,
        primitives: list,        # Ordered list of primitives to try
        catch_exceptions: bool = True,
        on_fallback: callable = None  # Callback when falling back
    ):
        self.primitives = primitives
        self.catch_exceptions = catch_exceptions
        self.on_fallback = on_fallback

    async def run(self, input_data):
        errors = []

        for i, primitive in enumerate(self.primitives):
            try:
                result = await primitive.quick(input_data)
                return {
                    "result": result,
                    "primitive_index": i,
                    "fallback_used": i > 0,
                    "errors": errors
                }
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
                if self.on_fallback:
                    self.on_fallback(i, e)
                if not self.catch_exceptions:
                    raise

        return {"error": "All primitives failed", "errors": errors}
```

**Usage example:**
```python
from langpy.skills import FallbackSkill
from langpy_sdk import Pipe

fallback = FallbackSkill([
    Pipe(model="gpt-4o"),        # Try best model first
    Pipe(model="gpt-4o-mini"),   # Fall back to faster model
    Pipe(model="gpt-3.5-turbo"), # Last resort
])

result = await fallback.run("Complex reasoning task")
```

---

## 6. Aggregator Skill

**Purpose:** Combine outputs from parallel branches with configurable strategy

**Why needed:** Workflow runs parallel steps but combining results is manual

**Implementation approach:**
```python
from langpy_sdk import Skill
import asyncio

class AggregatorSkill(Skill):
    """Run primitives in parallel and aggregate results."""

    def __init__(
        self,
        primitives: list,
        strategy: str = "all",  # "all" | "first" | "majority" | "custom"
        custom_aggregator: callable = None
    ):
        self.primitives = primitives
        self.strategy = strategy
        self.custom_aggregator = custom_aggregator

    async def run(self, input_data):
        # Run all primitives in parallel
        tasks = [p.quick(input_data) for p in self.primitives]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        successful = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]

        if self.strategy == "all":
            return {"results": successful, "errors": errors}

        elif self.strategy == "first":
            return {"result": successful[0] if successful else None}

        elif self.strategy == "majority":
            # Count occurrences and return most common
            from collections import Counter
            counts = Counter(successful)
            return {"result": counts.most_common(1)[0][0]}

        elif self.strategy == "custom" and self.custom_aggregator:
            return {"result": self.custom_aggregator(successful)}

        return {"results": successful}
```

**Usage example:**
```python
from langpy.skills import AggregatorSkill
from langpy_sdk import Pipe

# Get multiple perspectives and combine
aggregator = AggregatorSkill(
    primitives=[
        Pipe(model="gpt-4o-mini", system="You are an optimist"),
        Pipe(model="gpt-4o-mini", system="You are a pessimist"),
        Pipe(model="gpt-4o-mini", system="You are a realist"),
    ],
    strategy="custom",
    custom_aggregator=lambda results: "\n\n".join(results)
)

result = await aggregator.run("What's the future of AI?")
```

---

## 7. Retry Skill

**Purpose:** Retry with configurable backoff and conditions

**Why needed:** More sophisticated than Workflow's simple retry count

**Implementation approach:**
```python
from langpy_sdk import Skill
import asyncio

class RetrySkill(Skill):
    """Retry with exponential backoff and custom conditions."""

    def __init__(
        self,
        primitive,
        max_retries: int = 3,
        backoff_base: float = 1.0,      # Seconds
        backoff_multiplier: float = 2.0,
        retry_on: list = None,          # Exception types to retry on
        retry_if: callable = None       # Custom condition
    ):
        self.primitive = primitive
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_multiplier = backoff_multiplier
        self.retry_on = retry_on or [Exception]
        self.retry_if = retry_if

    async def run(self, input_data):
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self.primitive.quick(input_data)

                # Check custom retry condition
                if self.retry_if and self.retry_if(result):
                    raise ValueError("Retry condition met")

                return {
                    "result": result,
                    "attempts": attempt + 1,
                    "retried": attempt > 0
                }

            except tuple(self.retry_on) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.backoff_base * (self.backoff_multiplier ** attempt)
                    await asyncio.sleep(wait_time)

        return {"error": str(last_error), "attempts": self.max_retries + 1}
```

---

# MCP SERVERS (External Capabilities)

MCP Servers provide external capabilities via Model Context Protocol. These require infrastructure.

## 1. Queue MCP Server

**Purpose:** Async message passing between agents/primitives

**Infrastructure:** Redis, RabbitMQ, or SQS

**MCP Interface:**
```python
# MCP Tool definitions
tools = [
    {
        "name": "queue_publish",
        "description": "Publish a message to a queue",
        "parameters": {
            "queue_name": {"type": "string"},
            "message": {"type": "string"},
            "priority": {"type": "integer", "default": 0}
        }
    },
    {
        "name": "queue_consume",
        "description": "Consume a message from a queue",
        "parameters": {
            "queue_name": {"type": "string"},
            "timeout": {"type": "integer", "default": 30}
        }
    },
    {
        "name": "queue_length",
        "description": "Get the number of messages in a queue",
        "parameters": {
            "queue_name": {"type": "string"}
        }
    }
]
```

**Use cases:**
- Agent-to-agent communication
- Work distribution
- Event-driven pipelines

---

## 2. Cache MCP Server

**Purpose:** Memoize expensive LLM calls

**Infrastructure:** Redis, Memcached

**MCP Interface:**
```python
tools = [
    {
        "name": "cache_get",
        "description": "Get a cached value",
        "parameters": {
            "key": {"type": "string"}
        }
    },
    {
        "name": "cache_set",
        "description": "Set a cached value with TTL",
        "parameters": {
            "key": {"type": "string"},
            "value": {"type": "string"},
            "ttl_seconds": {"type": "integer", "default": 3600}
        }
    },
    {
        "name": "cache_delete",
        "description": "Delete a cached value",
        "parameters": {
            "key": {"type": "string"}
        }
    }
]
```

**Use cases:**
- Cache embeddings
- Cache LLM responses for common queries
- Reduce API costs

---

## 3. Scheduler MCP Server

**Purpose:** Time-based and event-based triggers

**Infrastructure:** Cron, Celery, or cloud schedulers

**MCP Interface:**
```python
tools = [
    {
        "name": "schedule_once",
        "description": "Schedule a one-time job",
        "parameters": {
            "job_id": {"type": "string"},
            "run_at": {"type": "string", "format": "datetime"},
            "payload": {"type": "object"}
        }
    },
    {
        "name": "schedule_recurring",
        "description": "Schedule a recurring job",
        "parameters": {
            "job_id": {"type": "string"},
            "cron_expression": {"type": "string"},
            "payload": {"type": "object"}
        }
    },
    {
        "name": "schedule_cancel",
        "description": "Cancel a scheduled job",
        "parameters": {
            "job_id": {"type": "string"}
        }
    }
]
```

**Use cases:**
- Periodic data ingestion
- Scheduled reports
- Delayed actions

---

## 4. Human MCP Server

**Purpose:** Human-in-the-loop approval/input

**Infrastructure:** Slack, Email, Web UI, or custom approval system

**MCP Interface:**
```python
tools = [
    {
        "name": "human_approve",
        "description": "Request human approval for an action",
        "parameters": {
            "request_id": {"type": "string"},
            "description": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "timeout_minutes": {"type": "integer", "default": 60},
            "channel": {"type": "string", "enum": ["slack", "email", "ui"]}
        }
    },
    {
        "name": "human_input",
        "description": "Request freeform input from a human",
        "parameters": {
            "request_id": {"type": "string"},
            "prompt": {"type": "string"},
            "timeout_minutes": {"type": "integer", "default": 60}
        }
    },
    {
        "name": "human_notify",
        "description": "Send a notification to a human (no response needed)",
        "parameters": {
            "message": {"type": "string"},
            "channel": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]}
        }
    }
]
```

**Use cases:**
- Approval workflows
- Content moderation
- Exception handling
- Quality assurance

---

## 5. Graph MCP Server

**Purpose:** Relationship-based memory (beyond vector similarity)

**Infrastructure:** Neo4j, Amazon Neptune, or TigerGraph

**MCP Interface:**
```python
tools = [
    {
        "name": "graph_add_node",
        "description": "Add a node to the graph",
        "parameters": {
            "node_type": {"type": "string"},
            "properties": {"type": "object"}
        }
    },
    {
        "name": "graph_add_edge",
        "description": "Add a relationship between nodes",
        "parameters": {
            "from_node": {"type": "string"},
            "to_node": {"type": "string"},
            "relationship": {"type": "string"},
            "properties": {"type": "object"}
        }
    },
    {
        "name": "graph_query",
        "description": "Query the graph with Cypher or natural language",
        "parameters": {
            "query": {"type": "string"},
            "query_type": {"type": "string", "enum": ["cypher", "natural"]}
        }
    },
    {
        "name": "graph_traverse",
        "description": "Traverse relationships from a starting node",
        "parameters": {
            "start_node": {"type": "string"},
            "relationship_types": {"type": "array"},
            "max_depth": {"type": "integer", "default": 3}
        }
    }
]
```

**Use cases:**
- Knowledge graphs
- Entity relationships
- Reasoning over connections
- Multi-hop queries

---

## 6. Stream MCP Server

**Purpose:** Real-time event streaming between components

**Infrastructure:** Kafka, Redis Streams, or Pulsar

**MCP Interface:**
```python
tools = [
    {
        "name": "stream_publish",
        "description": "Publish an event to a stream",
        "parameters": {
            "stream_name": {"type": "string"},
            "event_type": {"type": "string"},
            "data": {"type": "object"}
        }
    },
    {
        "name": "stream_subscribe",
        "description": "Subscribe to events from a stream",
        "parameters": {
            "stream_name": {"type": "string"},
            "event_types": {"type": "array"},
            "from_position": {"type": "string", "enum": ["latest", "earliest"]}
        }
    },
    {
        "name": "stream_read",
        "description": "Read events from a stream",
        "parameters": {
            "stream_name": {"type": "string"},
            "count": {"type": "integer", "default": 10},
            "block_ms": {"type": "integer", "default": 0}
        }
    }
]
```

**Use cases:**
- Real-time agent coordination
- Event-driven architectures
- Audit logging
- Live dashboards

---

## 7. State MCP Server

**Purpose:** Persistent state machines for complex flows

**Infrastructure:** Redis, DynamoDB, or PostgreSQL

**MCP Interface:**
```python
tools = [
    {
        "name": "state_create",
        "description": "Create a new state machine instance",
        "parameters": {
            "machine_id": {"type": "string"},
            "initial_state": {"type": "string"},
            "context": {"type": "object"}
        }
    },
    {
        "name": "state_transition",
        "description": "Transition to a new state",
        "parameters": {
            "machine_id": {"type": "string"},
            "event": {"type": "string"},
            "data": {"type": "object"}
        }
    },
    {
        "name": "state_get",
        "description": "Get current state and context",
        "parameters": {
            "machine_id": {"type": "string"}
        }
    },
    {
        "name": "state_history",
        "description": "Get state transition history",
        "parameters": {
            "machine_id": {"type": "string"},
            "limit": {"type": "integer", "default": 100}
        }
    }
]
```

**Use cases:**
- Order processing
- Multi-step approvals
- Conversation state
- Long-running workflows

---

## 8. Bridge MCP Server

**Purpose:** Inter-agent communication and coordination

**Infrastructure:** Custom service with WebSocket/gRPC

**MCP Interface:**
```python
tools = [
    {
        "name": "bridge_register",
        "description": "Register an agent with the bridge",
        "parameters": {
            "agent_id": {"type": "string"},
            "capabilities": {"type": "array"},
            "metadata": {"type": "object"}
        }
    },
    {
        "name": "bridge_send",
        "description": "Send a message to another agent",
        "parameters": {
            "to_agent": {"type": "string"},
            "message_type": {"type": "string"},
            "payload": {"type": "object"},
            "wait_response": {"type": "boolean", "default": False}
        }
    },
    {
        "name": "bridge_broadcast",
        "description": "Broadcast a message to all agents with capability",
        "parameters": {
            "capability": {"type": "string"},
            "message_type": {"type": "string"},
            "payload": {"type": "object"}
        }
    },
    {
        "name": "bridge_discover",
        "description": "Discover agents with specific capabilities",
        "parameters": {
            "capability": {"type": "string"}
        }
    }
]
```

**Use cases:**
- Multi-agent systems
- Agent delegation
- Swarm coordination
- Service discovery

---

# Implementation Priority

## Phase 1: Essential Skills
1. **Loop** - Enables iterative refinement patterns
2. **Router** - Enables specialized agent routing
3. **Evaluator** - Enables quality gates

## Phase 2: Resilience Skills
4. **Fallback** - Model/service resilience
5. **Guard** - Input/output validation
6. **Retry** - Sophisticated retry logic

## Phase 3: Composition Skills
7. **Aggregator** - Parallel result combination

## Phase 4: Essential MCP Servers
8. **Cache** - Cost reduction for LLM calls
9. **Human** - Human-in-the-loop workflows
10. **Queue** - Async agent communication

## Phase 5: Advanced MCP Servers
11. **Graph** - Relationship-based memory
12. **State** - Complex state machines
13. **Bridge** - Multi-agent coordination
14. **Stream** - Real-time event processing
15. **Scheduler** - Time-based automation

---

# Contributing

When implementing a new Skill or MCP Server:

1. **Follow the interface patterns** shown in this document
2. **Include comprehensive tests** with edge cases
3. **Document usage examples** with real-world scenarios
4. **Consider error handling** - what happens when things fail?
5. **Think about observability** - logging, metrics, tracing

For Skills:
- Must be built only from existing primitives
- No external dependencies (pure Python)
- Should be stateless where possible

For MCP Servers:
- Define clear tool schemas
- Document infrastructure requirements
- Provide Docker/docker-compose for local development
- Include health checks and monitoring endpoints
