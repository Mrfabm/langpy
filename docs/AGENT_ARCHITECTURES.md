# LangPy Agent Architecture Patterns

A comprehensive guide to the 8 reference agent architectures that can be built by composing LangPy's primitives.

---

## Philosophy: Primitives as Legos

LangPy follows the **composable architecture** philosophy:

> *"Treat AI systems like modular building blocks—similar to Legos—that can be combined into sophisticated pipelines."*

The 9 primitives (Pipe, Memory, Agent, Workflow, Thread, Tools, Parser, Chunker, Embed) are simple building blocks. Complex agent behaviors emerge from **composing** these primitives, not from adding more primitives.

---

## Table of Contents

1. [Pattern 1: Augmented LLM](#pattern-1-augmented-llm)
2. [Pattern 2: Prompt Chaining](#pattern-2-prompt-chaining)
3. [Pattern 3: Agent Routing](#pattern-3-agent-routing)
4. [Pattern 4: Parallelization](#pattern-4-parallelization)
5. [Pattern 5: Orchestration-Workers](#pattern-5-orchestration-workers)
6. [Pattern 6: Evaluator-Optimizer](#pattern-6-evaluator-optimizer)
7. [Pattern 7: Tool Agent](#pattern-7-tool-agent)
8. [Pattern 8: Memory Agent (RAG)](#pattern-8-memory-agent-rag)
9. [Combining Patterns](#combining-patterns)
10. [Quick Reference](#quick-reference)

---

## Pattern 1: Augmented LLM

The foundational pattern - an LLM enhanced with retrieval, tools, and memory.

### Architecture

```
┌─────────────────────────────────────┐
│           Augmented LLM             │
│  ┌─────┐  ┌────────┐  ┌─────────┐  │
│  │Tools│  │ Memory │  │ Thread  │  │
│  └──┬──┘  └───┬────┘  └────┬────┘  │
│     └─────────┼────────────┘       │
│           ┌───┴───┐                │
│           │  LLM  │                │
│           └───────┘                │
└─────────────────────────────────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Agent | LLM with tool execution |
| Memory | Long-term knowledge storage |
| Thread | Conversation history |
| Pipe | LLM backbone |

### Use Cases

- General-purpose AI assistants
- Customer support bots
- Knowledge-based Q&A systems

### Example

```python
from examples.architectures import AugmentedLLM

llm = AugmentedLLM()
await llm.initialize()

# Add knowledge
await llm.learn("LangPy is a Python AI framework.")

# Chat with augmented capabilities
response = await llm.chat("What is LangPy?")
```

### File

`examples/architectures/augmented_llm.py`

---

## Pattern 2: Prompt Chaining

Sequential task decomposition where each LLM call processes the output of the previous call.

### Architecture

```
┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
│ Input │───▶│ Pipe1 │───▶│ Pipe2 │───▶│ Pipe3 │───▶ Output
└───────┘    └───────┘    └───────┘    └───────┘
              Extract      Transform     Format
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Pipe | Each step in the chain |
| Workflow | Optional orchestration |

### Use Cases

- Document processing pipelines
- Multi-step content generation
- Data extraction and transformation
- Translation with quality checks

### Example

```python
from examples.architectures import PromptChain

chain = PromptChain()
chain.add_step("extract", "Extract key facts from: {input}")
chain.add_step("analyze", "Analyze these facts: {input}")
chain.add_step("summarize", "Summarize the analysis: {input}")

result = await chain.run("Long document text...")
print(result.final_output)
```

### Pre-built Chains

- `DocumentProcessor` - Extract → Analyze → Summarize
- `ContentGenerator` - Outline → Draft → Polish
- `TranslationChain` - Translate → Review

### File

`examples/architectures/prompt_chaining.py`

---

## Pattern 3: Agent Routing

Input classification directing queries to specialized agents.

### Architecture

```
                    ┌─────────────┐
                    │   Router    │
                    │   Agent     │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Tech     │   │  Sales    │   │  Support  │
    │  Agent    │   │  Agent    │   │  Agent    │
    └───────────┘   └───────────┘   └───────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Pipe | Classification/routing |
| Agent/Pipe | Specialized handlers |

### Use Cases

- Customer service with departments
- Multi-domain Q&A systems
- Intent-based chatbots
- Task-specific processing

### Example

```python
from examples.architectures import AgentRouter

router = AgentRouter()
router.add_route("technical", "Technical questions", tech_agent)
router.add_route("sales", "Sales inquiries", sales_agent)
router.add_route("support", "Customer support", support_agent)

result = await router.route("How do I reset my password?")
# -> Routes to support_agent
```

### Pre-built Routers

- `CustomerServiceRouter` - Technical, Sales, Support routing

### File

`examples/architectures/agent_routing.py`

---

## Pattern 4: Parallelization

Run multiple agents simultaneously for speed or diverse perspectives.

### Architecture

```
                ┌─────────┐
                │  Input  │
                └────┬────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Agent 1 │    │ Agent 2 │    │ Agent 3 │
└────┬────┘    └────┬────┘    └────┬────┘
     │               │               │
     └───────────────┼───────────────┘
                     ▼
              ┌────────────┐
              │ Aggregator │
              └────────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Pipe/Agent | Parallel execution units |
| Workflow | Parallel step groups |

### Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `CONCAT` | Concatenate all results |
| `VOTE` | Majority voting |
| `BEST` | LLM picks best response |
| `MERGE` | LLM merges into one |
| `FIRST` | First completed |
| `ALL` | Return all as list |

### Example

```python
from examples.architectures import ParallelAgents, AggregationStrategy

parallel = ParallelAgents()
parallel.add_task("positive", "Find positive aspects of: {input}")
parallel.add_task("negative", "Find negative aspects of: {input}")
parallel.add_task("neutral", "Give neutral analysis of: {input}")

result = await parallel.run("Electric vehicles", strategy=AggregationStrategy.MERGE)
```

### Pre-built Patterns

- `MultiPerspectiveAnalyzer` - Optimist, Critic, Pragmatist
- `EnsembleClassifier` - Multiple classification strategies

### File

`examples/architectures/parallelization.py`

---

## Pattern 5: Orchestration-Workers

A supervisor breaks tasks into subtasks and delegates to workers.

### Architecture

```
                ┌─────────────────┐
                │   Orchestrator  │
                │   (Supervisor)  │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │  Decompose Task │
                └────────┬────────┘
                         │
     ┌───────────────────┼───────────────────┐
     ▼                   ▼                   ▼
┌─────────┐        ┌─────────┐        ┌─────────┐
│ Worker1 │        │ Worker2 │        │ Worker3 │
└────┬────┘        └────┬────┘        └────┬────┘
     │                   │                   │
     └───────────────────┼───────────────────┘
                         ▼
                ┌────────────────┐
                │   Synthesize   │
                └────────────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Pipe | Orchestrator and workers |
| Agent | Workers with tools |
| Workflow | Task coordination |

### Example

```python
from examples.architectures import Orchestrator

orch = Orchestrator()
orch.add_worker("researcher", "Research specialist", "Finds information")
orch.add_worker("writer", "Content writer", "Creates content")
orch.add_worker("editor", "Editor", "Reviews content")

result = await orch.execute("Write a blog post about AI trends")
```

### Pre-built Teams

- `ContentTeam` - Researcher, Writer, Editor
- `AnalysisTeam` - Data Analyst, Strategist, Reporter

### File

`examples/architectures/orchestrator.py`

---

## Pattern 6: Evaluator-Optimizer

Iterative refinement with feedback loops until quality threshold is met.

### Architecture

```
┌─────────────────────────────────────────────┐
│                                             │
│    ┌───────────┐        ┌───────────┐      │
│    │ Generator │───────▶│ Evaluator │      │
│    └───────────┘        └─────┬─────┘      │
│          ▲                    │            │
│          │                    ▼            │
│          │              ┌──────────┐       │
│          └──────────────│ Feedback │       │
│                         └──────────┘       │
│                                             │
│         Loop until quality threshold        │
└─────────────────────────────────────────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Pipe | Generator and evaluator |
| Workflow | Loop control (optional) |

### Example

```python
from examples.architectures import EvaluatorOptimizer

eo = EvaluatorOptimizer()
eo.set_criteria([
    "Code is correct",
    "Code is readable",
    "Code is efficient"
])

result = await eo.run(
    task="Write a fibonacci function",
    threshold=0.85,
    max_iterations=3
)

print(f"Final score: {result.final_score}")
print(f"Iterations: {result.iterations_used}")
```

### Pre-built Optimizers

- `CodeOptimizer` - Code quality criteria
- `WritingOptimizer` - Writing quality criteria

### File

`examples/architectures/evaluator_optimizer.py`

---

## Pattern 7: Tool Agent

An agent that can select and execute external tools.

### Architecture

```
┌─────────────────────────────────────────────┐
│                 Tool Agent                   │
│                                              │
│    ┌─────────────────────────────────┐      │
│    │              LLM                 │      │
│    │    (reasoning & tool selection) │      │
│    └───────────────┬─────────────────┘      │
│                    │                         │
│    ┌───────────────┼───────────────┐        │
│    ▼               ▼               ▼        │
│ ┌──────┐      ┌──────┐      ┌──────┐       │
│ │ Tool │      │ Tool │      │ Tool │       │
│ │  A   │      │  B   │      │  C   │       │
│ └──────┘      └──────┘      └──────┘       │
│                                              │
└─────────────────────────────────────────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Agent | Tool execution |
| tool decorator | Define tools |

### Example

```python
from langpy_sdk import Agent, tool
from examples.architectures import ToolAgent

@tool("get_weather", "Get weather", {...})
def get_weather(location: str) -> str:
    return f"Sunny in {location}"

agent = ToolAgent()
agent.add_tool(get_weather)

result = await agent.run("What's the weather in Tokyo?")
```

### Pre-built Agents

- `AssistantAgent` - Weather, calculator, time, search
- `ProductivityAgent` - Reminders, email, time, calculator

### File

`examples/architectures/tool_agent.py`

---

## Pattern 8: Memory Agent (RAG)

Retrieval-Augmented Generation with semantic memory.

### Architecture

```
┌─────────────────────────────────────────────┐
│              Memory Agent                    │
│                                              │
│         ┌──────────────────┐                │
│         │     User Query    │                │
│         └────────┬─────────┘                │
│                  │                           │
│                  ▼                           │
│         ┌──────────────────┐                │
│         │  Semantic Search  │                │
│         │    (Memory)       │                │
│         └────────┬─────────┘                │
│                  │ relevant context          │
│                  ▼                           │
│         ┌──────────────────┐                │
│         │       LLM        │                │
│         │ (query + context)│                │
│         └────────┬─────────┘                │
│                  │                           │
│                  ▼                           │
│         ┌──────────────────┐                │
│         │    Response       │                │
│         └──────────────────┘                │
└─────────────────────────────────────────────┘
```

### Primitives Used

| Primitive | Purpose |
|-----------|---------|
| Memory | Vector storage & retrieval |
| Pipe/Agent | Response generation |
| Chunker | Document ingestion |
| Embed | Vector embeddings |

### Example

```python
from examples.architectures import MemoryAgent

agent = MemoryAgent(name="docs_assistant")
await agent.initialize()

# Add knowledge
await agent.learn("LangPy is a Python AI framework.")
await agent.learn_many(["Fact 1", "Fact 2", "Fact 3"])

# Query with RAG
result = await agent.ask("What is LangPy?")
print(result.answer)
print(f"Retrieved {len(result.retrieved_contexts)} contexts")
```

### Pre-built Agents

- `DocumentAssistant` - Document Q&A
- `KnowledgeBaseAgent` - Knowledge base Q&A

### File

`examples/architectures/memory_agent.py`

---

## Combining Patterns

The real power comes from combining patterns:

### RAG + Tools + Routing

```python
# Customer service bot with knowledge base and tools
class SmartServiceBot:
    def __init__(self):
        self.memory = MemoryAgent("service_kb")
        self.router = AgentRouter()
        self.tool_agent = ToolAgent()

        # Route technical queries to tool agent
        self.router.add_route("technical", "...", self.tool_agent)

        # Route knowledge queries to memory agent
        self.router.add_route("knowledge", "...", self.memory)
```

### Orchestrator + Evaluator

```python
# Content team with quality checks
class QualityContentTeam:
    def __init__(self):
        self.team = ContentTeam()
        self.evaluator = EvaluatorOptimizer()

    async def create_content(self, topic):
        # Generate with team
        draft = await self.team.execute(topic)

        # Refine until quality met
        final = await self.evaluator.run(
            f"Improve: {draft.final_result}",
            threshold=0.9
        )
        return final
```

### Parallel + Chain

```python
# Research from multiple sources, then synthesize
class ResearchPipeline:
    def __init__(self):
        self.parallel = ParallelAgents()
        self.chain = PromptChain()

    async def research(self, topic):
        # Gather from multiple sources in parallel
        sources = await self.parallel.run(topic)

        # Chain: synthesize -> fact-check -> format
        return await self.chain.run(sources.aggregated_result)
```

---

## Quick Reference

| Pattern | When to Use | Key Class |
|---------|-------------|-----------|
| Augmented LLM | General-purpose assistant | `AugmentedLLM` |
| Prompt Chaining | Sequential processing | `PromptChain` |
| Agent Routing | Multi-domain classification | `AgentRouter` |
| Parallelization | Speed or diversity | `ParallelAgents` |
| Orchestrator | Complex task decomposition | `Orchestrator` |
| Evaluator-Optimizer | Quality refinement | `EvaluatorOptimizer` |
| Tool Agent | External capabilities | `ToolAgent` |
| Memory Agent | Knowledge-based Q&A | `MemoryAgent` |

### Primitives → Patterns Mapping

| Primitive | Used In Patterns |
|-----------|------------------|
| Pipe | All patterns |
| Agent | 1, 3, 5, 7 |
| Memory | 1, 8 |
| Thread | 1, 8 |
| Workflow | 2, 4, 5 |
| tool | 1, 7 |
| Chunker | 8 |
| Embed | 8 |

---

## Running the Examples

```bash
# Run individual patterns
python -m examples.architectures.augmented_llm
python -m examples.architectures.prompt_chaining
python -m examples.architectures.agent_routing
python -m examples.architectures.parallelization
python -m examples.architectures.orchestrator
python -m examples.architectures.evaluator_optimizer
python -m examples.architectures.tool_agent
python -m examples.architectures.memory_agent
```

---

*These patterns demonstrate that LangPy's 9 primitives are sufficient building blocks for creating any agent architecture. Complex behaviors emerge from composition, not from adding more primitives.*
