# LangPy SDK

A clean, simple Python SDK for building AI applications with modular primitives.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Importing the SDK](#importing-the-sdk)
4. [Primitives Overview](#primitives-overview)
5. [Agent](#agent)
6. [Memory](#memory)
7. [Thread](#thread)
8. [Pipe](#pipe)
9. [Workflow](#workflow)
10. [Complete Example](#complete-example)
11. [API Reference](#api-reference)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install openai python-dotenv numpy

# For local vector storage (recommended)
pip install faiss-cpu

# For PostgreSQL vector storage (optional)
pip install asyncpg pgvector
```

### Step 2: Clone or Install the SDK

If using from this repository:

```bash
# Clone the repository
git clone <repository-url>
cd primitives_v5

# Install in development mode
pip install -e .
```

Or copy the `langpy_sdk` folder to your project.

### Step 3: Verify Installation

```python
python -c "from langpy_sdk import Agent, Memory, Thread, Pipe, Workflow; print('Installation successful!')"
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Required - OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# Optional - PostgreSQL connection (only if using pgvector backend)
POSTGRES_DSN=postgresql://username:password@localhost:5432/database_name
```

### Loading Environment Variables

```python
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file
```

---

## Importing the SDK

### Import All Primitives

```python
from langpy_sdk import Agent, Memory, Thread, Pipe, Workflow
```

### Import Specific Components

```python
# Agent and tool decorator
from langpy_sdk import Agent, ToolDef, tool, AgentResponse

# Memory and results
from langpy_sdk import Memory, SearchResult, MemoryStats

# Thread and messages
from langpy_sdk import Thread, Message, ThreadInfo

# Pipe
from langpy_sdk import Pipe, PipeResponse

# Workflow
from langpy_sdk import Workflow, Step, WorkflowResult
```

### Import with Alias

```python
import langpy_sdk as lp

agent = lp.Agent(model="gpt-4o-mini")
memory = lp.Memory(name="docs")
```

---

## Primitives Overview

| Primitive | Purpose | Use Case |
|-----------|---------|----------|
| **Agent** | AI with tool execution | Chatbots, assistants, autonomous tasks |
| **Memory** | Vector storage & search | Knowledge bases, RAG, document search |
| **Thread** | Conversation management | Chat history, multi-turn conversations |
| **Pipe** | Simple LLM calls | Classification, summarization, extraction |
| **Workflow** | Multi-step orchestration | Data pipelines, complex tasks |

---

## Agent

The Agent primitive creates an AI that can use tools to accomplish tasks.

### Basic Usage

```python
import asyncio
from langpy_sdk import Agent

async def main():
    # Create an agent
    agent = Agent(model="gpt-4o-mini")

    # Run the agent
    response = await agent.run("What is the capital of France?")

    # Access the response
    print(response.content)  # "The capital of France is Paris."
    print(response.model)    # "gpt-4o-mini"
    print(response.usage)    # {"prompt_tokens": 10, "completion_tokens": 8, ...}

asyncio.run(main())
```

### Creating Tools

Tools give the agent capabilities to perform actions.

#### Method 1: Using the @tool Decorator

```python
from langpy_sdk import Agent, tool

# Define a tool with the decorator
@tool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["city"]
    }
)
def get_weather(city: str, unit: str = "celsius") -> str:
    # Your implementation here
    return f"The weather in {city} is 22 degrees {unit}"

# Create agent with the tool
agent = Agent(model="gpt-4o-mini", tools=[get_weather])

# The agent can now use this tool
response = await agent.run("What's the weather in Tokyo?")
```

#### Method 2: Using ToolDef Directly

```python
from langpy_sdk import Agent, ToolDef

def calculate(expression: str) -> str:
    return str(eval(expression))

calc_tool = ToolDef(
    name="calculate",
    description="Calculate a math expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    },
    handler=calculate
)

agent = Agent(model="gpt-4o-mini", tools=[calc_tool])
```

### Agent with System Prompt

```python
response = await agent.run(
    "Tell me a joke",
    system="You are a comedian who specializes in dad jokes. Keep jokes family-friendly."
)
```

### Agent with Custom Parameters

```python
agent = Agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate],
    api_key="sk-...",        # Optional: defaults to OPENAI_API_KEY env var
    temperature=0.7,          # Controls randomness (0.0 - 2.0)
    max_tokens=1000,          # Maximum response length
    provider="openai"         # LLM provider
)
```

### Streaming Responses

```python
async def stream_example():
    agent = Agent(model="gpt-4o-mini")

    # Get a streaming response
    stream = await agent.run("Write a short story about a robot", stream=True)

    # Print chunks as they arrive
    async for chunk in stream:
        print(chunk, end="", flush=True)

    print()  # Newline at end

asyncio.run(stream_example())
```

### Agent Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "gpt-4o-mini" | OpenAI model name |
| `tools` | List[ToolDef] | None | List of tools the agent can use |
| `api_key` | str | OPENAI_API_KEY env | Your OpenAI API key |
| `temperature` | float | 0.7 | Randomness (0.0 = deterministic, 2.0 = very random) |
| `max_tokens` | int | 1000 | Maximum tokens in response |
| `provider` | str | "openai" | LLM provider name |

### AgentResponse Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | str | The text response |
| `model` | str | Model used |
| `usage` | dict | Token usage statistics |
| `tool_calls` | list | Tool calls made (if any) |
| `raw` | object | Raw response from API |

---

## Memory

The Memory primitive provides vector-based storage for semantic search.

### Basic Usage

```python
import asyncio
from langpy_sdk import Memory

async def main():
    # Create a memory store
    memory = Memory(name="my_knowledge")

    # Add documents
    doc_id = await memory.add("Python is a programming language created by Guido van Rossum")
    print(f"Added document: {doc_id}")

    # Search for relevant content
    results = await memory.search("Who created Python?")

    for result in results:
        print(f"Score: {result.score:.2f}")
        print(f"Text: {result.text}")
        print(f"Metadata: {result.metadata}")

asyncio.run(main())
```

### Adding Documents

#### Single Document

```python
# Simple add
doc_id = await memory.add("Your text content here")

# With metadata
doc_id = await memory.add(
    "Python is great for data science",
    metadata={
        "category": "programming",
        "author": "John",
        "date": "2024-01-15"
    }
)
```

#### Multiple Documents

```python
# Bulk add (more efficient)
doc_ids = await memory.add_many([
    "First document content",
    "Second document content",
    "Third document content"
])

# With metadata for each
doc_ids = await memory.add_many(
    texts=[
        "React is a JavaScript library",
        "Vue is a progressive framework",
        "Angular is a platform"
    ],
    metadata=[
        {"framework": "react"},
        {"framework": "vue"},
        {"framework": "angular"}
    ]
)
```

### Searching

```python
# Basic search
results = await memory.search("your query")

# With options
results = await memory.search(
    query="web development frameworks",
    limit=10,                    # Maximum results (default: 5)
    min_score=0.7,              # Minimum similarity score (0.0 - 1.0)
    filter={"category": "web"}  # Metadata filter
)

# Process results
for r in results:
    print(f"[{r.score:.2f}] {r.text}")
    print(f"  ID: {r.id}")
    print(f"  Metadata: {r.metadata}")
```

### Deleting Documents

```python
# Delete by ID
deleted_count = await memory.delete(id="doc_abc123")

# Delete by metadata filter
deleted_count = await memory.delete(filter={"category": "outdated"})

# Clear all documents
await memory.clear()
```

### Memory Statistics

```python
stats = await memory.stats()
print(f"Total documents: {stats.total_documents}")
print(f"Backend: {stats.backend}")
```

### Storage Backends

#### FAISS (Local - Default)

Fast, local storage. Good for development and small-medium datasets.

```python
memory = Memory(
    name="local_store",
    backend="faiss"  # Default
)
```

#### pgvector (PostgreSQL)

Persistent storage with PostgreSQL. Good for production and large datasets.

```python
memory = Memory(
    name="production_store",
    backend="pgvector",
    dsn="postgresql://user:pass@localhost:5432/mydb"
)
```

Or use environment variable:

```bash
# In .env
POSTGRES_DSN=postgresql://user:pass@localhost:5432/mydb
```

```python
memory = Memory(name="production_store", backend="pgvector")
```

### Memory Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "default" | Store name (used for persistence) |
| `backend` | str | "faiss" | Storage backend: "faiss" or "pgvector" |
| `embedding_model` | str | "text-embedding-3-small" | OpenAI embedding model |
| `dsn` | str | POSTGRES_DSN env | PostgreSQL connection string |

### SearchResult Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | str | The document text |
| `score` | float | Similarity score (0.0 - 1.0) |
| `metadata` | dict | Document metadata |
| `id` | str | Document ID |

---

## Thread

The Thread primitive manages conversation history with persistence.

### Basic Usage

```python
import asyncio
from langpy_sdk import Thread

async def main():
    # Create thread manager
    thread = Thread()

    # Create a new conversation
    thread_id = await thread.create(name="Customer Support")
    print(f"Created thread: {thread_id}")

    # Add messages
    await thread.add_message(thread_id, "user", "Hello, I need help!")
    await thread.add_message(thread_id, "assistant", "Hi! How can I assist you today?")
    await thread.add_message(thread_id, "user", "I have a billing question")

    # Get messages for LLM
    messages = await thread.get_messages(thread_id)
    print(messages)
    # Output: [
    #   {"role": "user", "content": "Hello, I need help!"},
    #   {"role": "assistant", "content": "Hi! How can I assist you today?"},
    #   {"role": "user", "content": "I have a billing question"}
    # ]

asyncio.run(main())
```

### Creating Threads

```python
# Simple
thread_id = await thread.create()

# With options
thread_id = await thread.create(
    name="Support Chat - John",
    tags=["support", "billing", "priority"],
    metadata={
        "customer_id": "cust_123",
        "department": "billing",
        "started_at": "2024-01-15T10:30:00Z"
    }
)
```

### Adding Messages

```python
# Basic message
await thread.add_message(thread_id, "user", "Hello!")

# With metadata
message = await thread.add_message(
    thread_id,
    role="assistant",
    content="How can I help you?",
    metadata={"confidence": 0.95, "model": "gpt-4o-mini"}
)

print(f"Message ID: {message.id}")
print(f"Created at: {message.created_at}")
```

### Getting Messages

```python
# Get all messages (as dicts for LLM input)
messages = await thread.get_messages(thread_id)

# Get last N messages
recent = await thread.get_messages(thread_id, limit=5)

# Get as Message objects
messages = await thread.get_messages(thread_id, as_dicts=False)
for msg in messages:
    print(f"[{msg.role}] {msg.content}")
    print(f"  ID: {msg.id}, Created: {msg.created_at}")
```

### Thread Information

```python
info = await thread.get(thread_id)

print(f"Thread ID: {info.id}")
print(f"Name: {info.name}")
print(f"Messages: {info.message_count}")
print(f"Tags: {info.tags}")
print(f"Created: {info.created_at}")
print(f"Updated: {info.updated_at}")
```

### Listing Threads

```python
# List all threads
all_threads = await thread.list()

# Filter by tags
support_threads = await thread.list(tags=["support"])

# Limit results
recent_threads = await thread.list(limit=10)

# Combined
threads = await thread.list(tags=["billing"], limit=5)

for t in threads:
    print(f"{t.name} ({t.message_count} messages)")
```

### Updating Threads

```python
await thread.update(
    thread_id,
    name="Resolved - Billing Issue",
    tags=["resolved", "billing"],
    metadata={"resolved_at": "2024-01-15T11:00:00Z"}
)
```

### Deleting

```python
# Delete a thread
deleted = await thread.delete(thread_id)

# Clear messages but keep thread
await thread.clear_messages(thread_id)
```

### Custom Storage Path

```python
# Default: ~/.langpy_sdk/threads/
thread = Thread()

# Custom path
thread = Thread(storage_path="/path/to/your/threads")
```

### Thread Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_path` | str | ~/.langpy_sdk/threads | Directory for thread storage |

### ThreadInfo Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Thread ID |
| `name` | str | Thread name |
| `message_count` | int | Number of messages |
| `created_at` | int | Creation timestamp |
| `updated_at` | int | Last update timestamp |
| `tags` | list | Thread tags |
| `metadata` | dict | Thread metadata |

---

## Pipe

The Pipe primitive provides simple LLM calls without tool execution.

### Basic Usage

```python
import asyncio
from langpy_sdk import Pipe

async def main():
    # Create a pipe
    pipe = Pipe(model="gpt-4o-mini")

    # Simple call
    response = await pipe.run("What is 2 + 2?")
    print(response.content)  # "4" or "2 + 2 equals 4"

asyncio.run(main())
```

### With System Prompt

```python
# Set default system prompt
pipe = Pipe(
    model="gpt-4o-mini",
    system="You are a helpful translator. Translate all input to French."
)

response = await pipe.run("Hello, how are you?")
print(response.content)  # "Bonjour, comment allez-vous?"

# Override system prompt per call
response = await pipe.run(
    "Hello",
    system="Translate to Spanish"
)
```

### Adjusting Parameters

```python
pipe = Pipe(
    model="gpt-4o-mini",
    temperature=0.0,    # Deterministic output
    max_tokens=500      # Shorter responses
)

# Override per call
response = await pipe.run(
    "Write a creative story",
    temperature=1.0,    # More creative
    max_tokens=2000     # Longer response
)
```

### JSON Mode

```python
response = await pipe.run(
    "List 3 programming languages with their main use cases",
    json_mode=True
)

import json
data = json.loads(response.content)
print(data)
# {"languages": [{"name": "Python", "use": "AI/ML"}, ...]}
```

### Streaming

```python
stream = await pipe.run("Write a poem about coding", stream=True)

async for chunk in stream:
    print(chunk, end="", flush=True)
```

### Built-in Helpers

#### Classification

```python
category = await pipe.classify(
    text="I absolutely love this product! Best purchase ever!",
    categories=["positive", "negative", "neutral"]
)
print(category)  # "positive"

# Multiple categories
categories = await pipe.classify(
    text="The food was great but service was slow",
    categories=["food_positive", "food_negative", "service_positive", "service_negative"],
    allow_multiple=True
)
print(categories)  # ["food_positive", "service_negative"]
```

#### Summarization

```python
long_text = "..." # Your long text

# Concise summary
summary = await pipe.summarize(long_text)

# Detailed summary
summary = await pipe.summarize(long_text, style="detailed")

# Bullet points
summary = await pipe.summarize(long_text, style="bullet")

# With length limit
summary = await pipe.summarize(long_text, max_length=50)  # ~50 words
```

#### Data Extraction

```python
text = "Contact John Smith at john@email.com or call 555-1234. He's 30 years old."

data = await pipe.extract(
    text,
    fields=["name", "email", "phone", "age"]
)

print(data)
# {"name": "John Smith", "email": "john@email.com", "phone": "555-1234", "age": "30"}
```

### Pipe Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "gpt-4o-mini" | OpenAI model name |
| `system` | str | None | Default system prompt |
| `temperature` | float | 0.7 | Randomness (0.0 - 2.0) |
| `max_tokens` | int | 1000 | Maximum response length |
| `api_key` | str | OPENAI_API_KEY env | Your OpenAI API key |

---

## Workflow

The Workflow primitive orchestrates multi-step processes with dependencies.

### Basic Usage

```python
import asyncio
from langpy_sdk import Workflow, Step

async def main():
    # Create workflow
    workflow = Workflow(name="data-pipeline")

    # Define handlers
    def fetch_data(inputs):
        url = inputs.get("url")
        return {"data": f"Fetched from {url}"}

    def process_data(inputs):
        data = inputs["fetch"]["data"]  # Access previous step output
        return {"processed": data.upper()}

    def save_data(inputs):
        processed = inputs["process"]["processed"]
        return {"saved": True, "content": processed}

    # Add steps
    workflow.add_step(Step(name="fetch", handler=fetch_data))
    workflow.add_step(Step(name="process", handler=process_data, depends_on=["fetch"]))
    workflow.add_step(Step(name="save", handler=save_data, depends_on=["process"]))

    # Run workflow
    result = await workflow.run({"url": "https://api.example.com"})

    # Check result
    if result.success:
        print("Workflow completed!")
        print(f"Outputs: {result.outputs}")
    else:
        print("Workflow failed!")
        for name, step in result.steps.items():
            if step.status.value == "failed":
                print(f"  {name}: {step.error}")

asyncio.run(main())
```

### Async Handlers

```python
import aiohttp

async def fetch_api(inputs):
    async with aiohttp.ClientSession() as session:
        async with session.get(inputs["url"]) as response:
            return {"status": response.status, "data": await response.json()}

workflow.add_step(Step(name="fetch", handler=fetch_api))
```

### Step Options

```python
workflow.add_step(Step(
    name="risky_operation",
    handler=my_handler,
    depends_on=["step1", "step2"],  # Wait for these steps
    retry=3,                         # Retry up to 3 times on failure
    timeout=30.0,                    # Timeout after 30 seconds
    condition=lambda inputs: inputs.get("should_run", True)  # Conditional execution
))
```

### Decorator Syntax

```python
workflow = Workflow("my-pipeline")

@workflow.step("fetch", retry=3, timeout=30)
async def fetch(inputs):
    return await fetch_data(inputs["url"])

@workflow.step("process", depends_on=["fetch"])
def process(inputs):
    return transform(inputs["fetch"])

@workflow.step("save", depends_on=["process"])
async def save(inputs):
    return await save_to_db(inputs["process"])

result = await workflow.run({"url": "https://api.example.com"})
```

### Parallel Execution

Steps without dependencies run in parallel automatically:

```python
workflow = Workflow("parallel-example")

# These three steps run in parallel
workflow.add_step(Step(name="fetch_users", handler=fetch_users))
workflow.add_step(Step(name="fetch_orders", handler=fetch_orders))
workflow.add_step(Step(name="fetch_products", handler=fetch_products))

# This step waits for all three
workflow.add_step(Step(
    name="combine",
    handler=combine_data,
    depends_on=["fetch_users", "fetch_orders", "fetch_products"]
))
```

### Accessing Results

```python
result = await workflow.run(inputs)

# Overall status
print(result.success)      # True/False
print(result.status)       # "completed" or "failed"
print(result.duration)     # Total time in seconds
print(result.workflow_id)  # Unique run ID

# All outputs
print(result.outputs)  # {"step1": output1, "step2": output2, ...}

# Specific step output
print(result.get("process"))  # Output from "process" step

# Step details
for name, step in result.steps.items():
    print(f"{name}:")
    print(f"  Status: {step.status.value}")
    print(f"  Duration: {step.duration:.2f}s")
    print(f"  Output: {step.output}")
    if step.error:
        print(f"  Error: {step.error}")
```

### Visualize Workflow

```python
print(workflow.visualize())

# Output:
# Workflow: my-pipeline
# ========================================
#   - fetch [retry: 3]
#   - process (depends on: fetch)
#   - save (depends on: process)
```

### Workflow Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Workflow name |
| `max_parallel` | int | 5 | Maximum parallel steps |

### Step Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Unique step name |
| `handler` | callable | required | Function to execute |
| `depends_on` | list | [] | Steps to wait for |
| `retry` | int | 0 | Retry count on failure |
| `timeout` | float | None | Timeout in seconds |
| `condition` | callable | None | Function returning bool |

---

## Complete Example

A travel planning assistant using all primitives:

```python
import asyncio
from langpy_sdk import Agent, Memory, Thread, Pipe, tool

# =============================================================================
# TOOLS
# =============================================================================

@tool(
    "get_destination",
    "Get information about a travel destination",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
)
def get_destination(city: str) -> str:
    destinations = {
        "paris": "Paris, France - City of Light. Best time: Apr-Jun, Sep-Oct. "
                 "Must see: Eiffel Tower, Louvre, Notre-Dame. Budget: $150/day",
        "tokyo": "Tokyo, Japan - Where tradition meets future. Best time: Mar-May, Sep-Nov. "
                 "Must see: Shibuya, Senso-ji, Mt Fuji. Budget: $120/day",
        "rome": "Rome, Italy - The Eternal City. Best time: Apr-Jun, Sep-Oct. "
                "Must see: Colosseum, Vatican, Trevi Fountain. Budget: $130/day",
    }
    return destinations.get(city.lower(), f"No information available for {city}")

@tool(
    "calculate_budget",
    "Calculate trip budget",
    {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "days": {"type": "integer"},
            "travelers": {"type": "integer"}
        },
        "required": ["destination", "days"]
    }
)
def calculate_budget(destination: str, days: int, travelers: int = 1) -> str:
    daily_costs = {"paris": 150, "tokyo": 120, "rome": 130}
    cost = daily_costs.get(destination.lower(), 100)
    total = cost * days * travelers
    return f"Estimated budget for {days} days in {destination} for {travelers} traveler(s): ${total}"

# =============================================================================
# TRAVEL AGENCY
# =============================================================================

class TravelAgency:
    def __init__(self):
        # Initialize all primitives
        self.agent = Agent(
            model="gpt-4o-mini",
            tools=[get_destination, calculate_budget]
        )
        self.memory = Memory(name="travel_knowledge")
        self.thread = Thread()
        self.pipe = Pipe(model="gpt-4o-mini")
        self.current_thread = None

    async def setup(self):
        """Initialize the agency with knowledge and start a conversation."""
        # Add travel tips to memory
        tips = [
            "Book flights 6-8 weeks in advance for the best prices",
            "Always purchase travel insurance for international trips",
            "Learn a few basic phrases in the local language",
            "Pack light - you can buy essentials at your destination",
            "Keep digital copies of important documents in the cloud"
        ]
        await self.memory.add_many(tips)
        print(f"Loaded {len(tips)} travel tips into memory")

        # Create a conversation thread
        self.current_thread = await self.thread.create(
            name="Travel Planning Session",
            tags=["travel", "planning"]
        )
        print(f"Started conversation: {self.current_thread}")

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's response."""
        # Save user message
        await self.thread.add_message(self.current_thread, "user", user_message)

        # Get relevant tips from memory
        tips = await self.memory.search(user_message, limit=2)
        tips_text = "\n".join(f"- {t.text}" for t in tips) if tips else "No specific tips"

        # Get conversation history
        messages = await self.thread.get_messages(self.current_thread, limit=10)

        # Build system prompt with context
        system = f"""You are a helpful travel planning assistant.
Use your tools to provide accurate information about destinations and budgets.
Be friendly and enthusiastic about travel!

Relevant travel tips:
{tips_text}"""

        # Get response from agent
        response = await self.agent.run(messages, system=system)
        content = response.content or "I've processed your request."

        # Save assistant message
        await self.thread.add_message(self.current_thread, "assistant", content)

        return content

    async def summarize_conversation(self) -> str:
        """Get a summary of the current conversation."""
        messages = await self.thread.get_messages(self.current_thread)
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        summary = await self.pipe.summarize(conversation, style="bullet")
        return summary

# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("=" * 60)
    print("   TRAVEL PLANNING AGENCY")
    print("=" * 60)
    print()

    # Initialize
    agency = TravelAgency()
    await agency.setup()
    print()

    # Have a conversation
    queries = [
        "Tell me about Tokyo as a travel destination",
        "What's the weather like in spring?",
        "Calculate a budget for 5 days for 2 people"
    ]

    for query in queries:
        print(f"You: {query}")
        response = await agency.chat(query)
        print(f"Agent: {response}")
        print("-" * 40)

    # Get conversation summary
    print("\nConversation Summary:")
    summary = await agency.summarize_conversation()
    print(summary)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## API Reference

### Quick Reference

```python
# Agent
agent = Agent(model, tools, api_key, temperature, max_tokens, provider)
response = await agent.run(prompt, stream, system, temperature, max_tokens)

# Memory
memory = Memory(name, backend, embedding_model, dsn)
doc_id = await memory.add(text, metadata)
doc_ids = await memory.add_many(texts, metadata)
results = await memory.search(query, limit, min_score, filter)
count = await memory.delete(id, filter)
await memory.clear()
stats = await memory.stats()

# Thread
thread = Thread(storage_path)
thread_id = await thread.create(name, tags, metadata)
message = await thread.add_message(thread_id, role, content, metadata)
messages = await thread.get_messages(thread_id, limit, as_dicts)
info = await thread.get(thread_id)
threads = await thread.list(tags, limit)
info = await thread.update(thread_id, name, tags, metadata)
deleted = await thread.delete(thread_id)
await thread.clear_messages(thread_id)

# Pipe
pipe = Pipe(model, system, temperature, max_tokens, api_key)
response = await pipe.run(prompt, system, stream, temperature, max_tokens, json_mode)
category = await pipe.classify(text, categories, allow_multiple)
summary = await pipe.summarize(text, max_length, style)
data = await pipe.extract(text, fields)

# Workflow
workflow = Workflow(name, max_parallel)
workflow.add_step(Step(name, handler, depends_on, retry, timeout, condition))
@workflow.step(name, depends_on, retry, timeout)
result = await workflow.run(inputs, parallel)
workflow.visualize()
```

---

## Troubleshooting

### "OPENAI_API_KEY not set"

Make sure you have set your API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key"

# Or use .env file
from dotenv import load_dotenv
load_dotenv()
```

### Memory search returns no results

1. Check that documents were added successfully
2. Try lowering `min_score` parameter
3. Verify the embedding model is working

### Thread messages not persisting

Check the storage path is writable:

```python
thread = Thread(storage_path="./my_threads")  # Use local directory
```

---

## License

MIT License
