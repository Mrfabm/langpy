# Getting Started with LangPy

A step-by-step guide to building AI agents using LangPy's composable primitives.

---

## Philosophy: Primitives as Lego Blocks

LangPy follows the **composable architecture** philosophy:

> *"Treat AI systems like modular building blocks—similar to Legos—that can be combined into sophisticated pipelines."*

You don't need wrapper classes or complex abstractions. Just combine primitives directly to create powerful AI systems.

---

## New in v2.0: True Lego Blocks Architecture

LangPy 2.0 introduces **pipeline operators** (`|` and `&`) for real Lego-like composition:

```python
from langpy.core import Context, parallel
from langpy_sdk import Memory, Pipe

# Sequential composition with |
rag = Memory(name="docs", k=5) | Pipe(system_prompt="Answer using context.")

# Parallel composition with &
multi_perspective = (
    Pipe(system_prompt="Find positive aspects.") &
    Pipe(system_prompt="Find negative aspects.")
)

# Execute
result = await rag.process(Context(query="What is Python?"))
if result.is_success():
    print(result.unwrap().response)
    print(f"Cost: ${result.unwrap().cost.total_cost:.4f}")
```

**Key benefits:**
- Unified `Context` flows between all primitives
- Built-in cost tracking and observability
- Explicit error handling with `Result` types
- Testing support with mock primitives

See [CORE_API.md](CORE_API.md) for complete documentation of the new architecture.

---

## The 9 Primitives

| Primitive | Purpose | Primary Methods |
|-----------|---------|-----------------|
| **Pipe** | Simple LLM calls | `pipe.run(prompt)` |
| **Agent** | LLM + tool execution | `agent.run(message)` |
| **Memory** | Vector storage (RAG) | `memory.add()`, `memory.search()` |
| **Thread** | Conversation history | `thread.create()`, `thread.add_message()` |
| **Workflow** | Multi-step orchestration | `workflow.step()`, `workflow.run()` |
| **tool** | Define callable tools | `@tool(name, desc, schema)` |
| **Chunker** | Text splitting | `chunker.chunk(text)` |
| **Embed** | Vector embeddings | `embed.create(text)` |
| **Parser** | Output parsing | `parser.parse(text)` |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

### Basic Import

```python
from langpy_sdk import Pipe, Agent, Memory, Thread, Workflow, tool
```

---

## Primitive 1: Pipe

The simplest primitive - make LLM calls.

### Basic Usage

```python
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini")

response = await pipe.run("What is Python?")
print(response.content)
```

### With System Prompt

```python
response = await pipe.run(
    "Explain recursion",
    system="You are a programming tutor. Be concise."
)
```

### Chaining Pipes

```python
pipe = Pipe(model="gpt-4o-mini")

# Step 1: Extract facts
step1 = await pipe.run("Extract key facts from: {document}")

# Step 2: Analyze (using step1 output)
step2 = await pipe.run(f"Analyze these facts:\n{step1.content}")

# Step 3: Summarize (using step2 output)
step3 = await pipe.run(f"Summarize:\n{step2.content}")
```

---

## Primitive 2: Memory

Store and retrieve knowledge using vector search.

### Basic Usage

```python
from langpy_sdk import Memory

memory = Memory(name="my_knowledge")

# Add knowledge
await memory.add("Python was created by Guido van Rossum in 1991.")
await memory.add("Python is known for its readable syntax.")

# Search for relevant knowledge
results = await memory.search("Who created Python?", limit=3)
for r in results:
    print(f"{r.text} (score: {r.score})")
```

### Add Many at Once

```python
facts = [
    "LangPy is a Python framework.",
    "LangPy has 9 primitives.",
    "LangPy is composable like Legos.",
]
await memory.add_many(facts)
```

### Combining with Pipe (RAG)

```python
from langpy_sdk import Memory, Pipe

memory = Memory(name="docs")
pipe = Pipe(model="gpt-4o-mini")

# Add knowledge
await memory.add_many(["fact 1", "fact 2", "fact 3"])

# RAG: Retrieve then Generate
results = await memory.search("query", limit=3)
context = "\n".join([r.text for r in results])

response = await pipe.run(
    f"Context:\n{context}\n\nQuestion: query",
    system="Answer based on context."
)
```

---

## Primitive 3: Thread

Manage conversation history.

### Basic Usage

```python
from langpy_sdk import Thread

thread = Thread()

# Create a conversation
thread_id = await thread.create("My Chat", tags=["demo"])

# Add messages
await thread.add_message(thread_id, "user", "Hello!")
await thread.add_message(thread_id, "assistant", "Hi there!")

# Get history
messages = await thread.get_messages(thread_id)
for m in messages:
    print(f"[{m.role}]: {m.content}")
```

### With Pipe for Conversational AI

```python
from langpy_sdk import Thread, Pipe

thread = Thread()
pipe = Pipe(model="gpt-4o-mini")

thread_id = await thread.create("Chat Session")

async def chat(message: str) -> str:
    # Get history
    history = await thread.get_messages(thread_id)
    history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-4:]])

    # Generate response
    response = await pipe.run(
        f"History:\n{history_text}\n\nUser: {message}",
        system="Continue the conversation naturally."
    )

    # Save to history
    await thread.add_message(thread_id, "user", message)
    await thread.add_message(thread_id, "assistant", response.content)

    return response.content
```

---

## Primitive 4: Agent + tool

LLM with tool execution capabilities.

### Define Tools

```python
from langpy_sdk import tool

@tool(
    "get_weather",
    "Get weather for a location",
    {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)
def get_weather(location: str) -> str:
    return f"Weather in {location}: 72°F, Sunny"

@tool(
    "calculate",
    "Perform math calculations",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
)
def calculate(expression: str) -> str:
    return f"Result: {eval(expression)}"
```

### Use Agent with Tools

```python
from langpy_sdk import Agent

agent = Agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate]
)

response = await agent.run("What's the weather in Tokyo?")
print(response.content)

response = await agent.run("Calculate 15 * 23 + 7")
print(response.content)
```

---

## Primitive 5: Workflow

Orchestrate multi-step processes.

### Basic Usage

```python
from langpy_sdk import Workflow, Pipe

workflow = Workflow()
pipe = Pipe(model="gpt-4o-mini")

@workflow.step("research")
async def research(inputs):
    response = await pipe.run(f"Research: {inputs['topic']}")
    return response.content

@workflow.step("write", depends_on=["research"])
async def write(inputs):
    response = await pipe.run(f"Write about: {inputs['research']}")
    return response.content

result = await workflow.run(initial_input={"topic": "AI agents"})
print(result.outputs["write"])
```

---

## Combining Primitives: 8 Architecture Patterns

### Pattern 1: Augmented LLM

Combine Agent + Memory + Thread for a full-featured assistant.

```python
from langpy_sdk import Agent, Memory, Thread, tool

# Create primitives
agent = Agent(model="gpt-4o-mini", tools=[my_tools])
memory = Memory(name="kb")
thread = Thread()

# Add knowledge
await memory.add_many(["fact 1", "fact 2"])

# Create conversation
thread_id = await thread.create("Session")

# Chat with augmented capabilities
async def chat(message: str) -> str:
    # Retrieve context
    results = await memory.search(message, limit=3)
    context = "\n".join([r.text for r in results])

    # Get history
    history = await thread.get_messages(thread_id)

    # Run agent
    response = await agent.run(
        f"Context:\n{context}\n\nUser: {message}",
        system="Use tools and context to help."
    )

    # Save to thread
    await thread.add_message(thread_id, "user", message)
    await thread.add_message(thread_id, "assistant", response.content)

    return response.content
```

### Pattern 2: Prompt Chaining

Chain multiple Pipe calls sequentially.

```python
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini")

# Extract → Analyze → Summarize
step1 = await pipe.run("Extract key facts from: {document}")
step2 = await pipe.run(f"Analyze: {step1.content}")
step3 = await pipe.run(f"Summarize: {step2.content}")
```

### Pattern 3: Agent Routing

Classify and route to specialized handlers.

```python
from langpy_sdk import Pipe

classifier = Pipe(model="gpt-4o-mini")
tech_handler = Pipe(model="gpt-4o-mini")
sales_handler = Pipe(model="gpt-4o-mini")

# Classify
result = await classifier.run(
    f"Classify as 'tech' or 'sales': {query}",
    system="Respond with only the category."
)

# Route
if "tech" in result.content.lower():
    response = await tech_handler.run(query, system="You are tech support.")
else:
    response = await sales_handler.run(query, system="You are a sales rep.")
```

### Pattern 4: Parallelization

Run multiple Pipes concurrently.

```python
import asyncio
from langpy_sdk import Pipe

pipe = Pipe(model="gpt-4o-mini")

# Run in parallel
results = await asyncio.gather(
    pipe.run("Analyze positive aspects of: {topic}"),
    pipe.run("Analyze negative aspects of: {topic}"),
    pipe.run("Give balanced view of: {topic}"),
)

# Merge results
all_views = "\n".join([r.content for r in results])
merged = await pipe.run(f"Synthesize these views:\n{all_views}")
```

### Pattern 5: Orchestrator-Workers

Decompose tasks and delegate to workers.

```python
import asyncio
from langpy_sdk import Pipe

orchestrator = Pipe(model="gpt-4o-mini")
researcher = Pipe(model="gpt-4o-mini")
writer = Pipe(model="gpt-4o-mini")

# Decompose
plan = await orchestrator.run(
    "Break this into subtasks: {task}",
    system="Create subtasks for researcher and writer."
)

# Execute workers in parallel
results = await asyncio.gather(
    researcher.run("Research: {subtask1}", system="You research."),
    writer.run("Draft: {subtask2}", system="You write."),
)

# Synthesize
final = await orchestrator.run(
    f"Combine this work:\n{results}",
    system="Create cohesive output."
)
```

### Pattern 6: Evaluator-Optimizer

Iterative improvement loop.

```python
from langpy_sdk import Pipe

generator = Pipe(model="gpt-4o-mini")
evaluator = Pipe(model="gpt-4o-mini")

output = None
for _ in range(3):  # Max 3 iterations
    # Generate/improve
    if output:
        prompt = f"Improve based on feedback:\n{output}\n\nFeedback:\n{feedback}"
    else:
        prompt = "Complete: {task}"

    response = await generator.run(prompt)
    output = response.content

    # Evaluate
    eval_response = await evaluator.run(
        f"Score 0-1 and give feedback:\n{output}",
        system="Be critical. Format: SCORE: [0-1]\nFEEDBACK: [text]"
    )

    # Parse score
    if "SCORE:" in eval_response.content:
        score = float(eval_response.content.split("SCORE:")[1].split()[0])
        if score >= 0.85:
            break

    feedback = eval_response.content
```

### Pattern 7: Tool Agent

Agent with multiple tools.

```python
from langpy_sdk import Agent, tool

@tool("search", "Search web", {...})
def search(query: str) -> str: ...

@tool("calculate", "Do math", {...})
def calculate(expr: str) -> str: ...

@tool("get_time", "Get time", {...})
def get_time() -> str: ...

agent = Agent(model="gpt-4o-mini", tools=[search, calculate, get_time])

response = await agent.run("Search for Python, calculate 10*5, and tell me the time")
```

### Pattern 8: Memory Agent (RAG)

Full RAG implementation.

```python
from langpy_sdk import Memory, Pipe, Thread

memory = Memory(name="knowledge")
pipe = Pipe(model="gpt-4o-mini")
thread = Thread()

# Add knowledge
await memory.add_many(["fact 1", "fact 2", "fact 3"])
thread_id = await thread.create("RAG Session")

async def rag_query(question: str) -> str:
    # Retrieve
    results = await memory.search(question, limit=3)
    context = "\n".join([r.text for r in results])

    # Generate
    response = await pipe.run(
        f"Context:\n{context}\n\nQuestion: {question}",
        system="Answer based on context."
    )

    # Save history
    await thread.add_message(thread_id, "user", question)
    await thread.add_message(thread_id, "assistant", response.content)

    return response.content
```

---

## Running the Examples

```bash
# Run individual pattern demos
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

## Key Takeaways

1. **No wrapper classes needed** - Compose primitives directly
2. **Pipe** is the foundation - Simple LLM calls
3. **Memory** enables RAG - Store and retrieve knowledge
4. **Thread** manages history - For conversational AI
5. **Agent + tool** adds capabilities - External tool execution
6. **Workflow** orchestrates - Multi-step processes
7. **asyncio.gather** enables parallelism - Run Pipes concurrently
8. **Patterns emerge from composition** - Complex behaviors from simple parts

---

*LangPy: 9 primitives, infinite possibilities.*
