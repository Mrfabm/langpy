# LangPy

**Langbase for Python** - A comprehensive AI framework implementing all 9 Langbase primitives with full API parity plus composable pipeline architecture.

## Quick Start

```python
from langpy import Langpy

# Initialize (like Langbase SDK)
lb = Langpy(api_key="your-api-key")

# Use any of the 9 primitives
response = await lb.agent.run(
    model="openai:gpt-4",
    input="What is Python?",
    instructions="Be concise"
)
print(response.output)
```

## The 9 Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `lb.agent` | Unified LLM API (100+ models) | `await lb.agent.run(model="openai:gpt-4", input="Hello")` |
| `lb.pipe` | Single LLM call with templates | `await lb.pipe.run(name="summarizer", variables={"text": "..."})` |
| `lb.memory` | Vector storage & RAG | `await lb.memory.retrieve(query="search", top_k=5)` |
| `lb.thread` | Conversation history | `await lb.thread.create()` |
| `lb.workflow` | Multi-step orchestration | `lb.workflow().step(id="s1", primitive=lb.agent)` |
| `lb.parser` | Document text extraction | `await lb.parser.run(document="file.pdf")` |
| `lb.chunker` | Text segmentation | `await lb.chunker.run(content="...", chunk_size=512)` |
| `lb.embed` | Text to vectors | `await lb.embed.run(texts=["hello", "world"])` |
| `lb.tools` | Web search, crawl, custom | `await lb.tools.run(tool="web_search", query="...")` |

## Pipeline Composition

All primitives can be composed with `|` (sequential) and `&` (parallel) operators:

```python
from langpy import Langpy, Context

lb = Langpy()

# RAG pipeline: retrieve → generate
rag = lb.memory | lb.pipe

result = await rag.process(Context(query="What is LangPy?"))
if result.is_success():
    print(result.unwrap().response)
```

## Control Flow Operators

LangPy provides **composable operators** for control flow - these are Lego block connectors that let you build any agent architecture:

| Operator | Pattern | Example |
|----------|---------|---------|
| **`loop_while()`** | Iteration | Refine until quality threshold |
| **`map_over()`** | For-each | Process multiple items in parallel |
| **`reduce()`** | Aggregation | Combine multiple results |
| **`when()`** | Conditional | If/else routing |
| **`branch()`** | Multi-way | Route to different paths |
| **`retry()`** | Resilience | Automatic retry with backoff |
| **`recover()`** | Error handling | Fallback logic |

### Example: Map-Reduce Pattern

```python
from langpy import Langpy, map_over, reduce, pipeline

lb = Langpy(api_key="...")

# Map: Research multiple topics in parallel
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

### Example: Iterative Refinement

```python
from langpy import loop_while

# Keep refining until quality threshold
refine = loop_while(
    condition=lambda ctx: ctx.get("quality") < 0.9,
    body=lb.agent,
    max_iterations=5
)

result = await refine.process(Context(content="draft"))
```

**See [docs/OPERATORS.md](docs/OPERATORS.md) for complete operator reference and patterns.**

### Key Features of v2.0

| Feature | Description |
|---------|-------------|
| **Unified Context** | Single `Context` type flows between all primitives |
| **Pipeline Operators** | `\|` for sequential, `&` for parallel composition |
| **Result Types** | Explicit error handling with `Success`/`Failure` |
| **Built-in Observability** | Token tracking, cost calculation, tracing |
| **Testing Support** | Mock primitives and assertion helpers |
| **Provider Agnostic** | Easy switching between OpenAI, Anthropic, Google, etc. |

### Quick Example: Multi-Perspective Analysis

```python
from langpy.core import Context, parallel
from langpy_sdk import Pipe

# Create multiple perspectives
optimist = Pipe(system_prompt="Find positive aspects.", name="Optimist")
pessimist = Pipe(system_prompt="Find negative aspects.", name="Pessimist")
synthesizer = Pipe(system_prompt="Synthesize both views.", name="Synthesizer")

# Compose: run optimist & pessimist in parallel, then synthesize
pipeline = parallel(optimist, pessimist) | synthesizer

result = await pipeline.process(Context(query="Analyze AI in healthcare"))
```

### Backward Compatibility

The original API continues to work:

```python
# Old API (still works)
from langpy_sdk import Pipe
pipe = Pipe(model="gpt-4o-mini")
response = await pipe.run("Hello!")  # Returns PipeResponse

# New API (recommended for new code)
from langpy.core import Context
result = await pipe.process(Context(query="Hello!"))  # Returns Result[Context]
```

See [docs/CORE_API.md](docs/CORE_API.md) for complete documentation of the new architecture.

---

## 🏗️ Architecture

LangPy follows a modular primitive-based architecture with clean separation of concerns:

### Core Primitives (100% Langbase Parity)

- **Agents** (`agent/`) - Autonomous AI reasoning engines with think-act-observe-reflect loops
- **Pipes** (`pipe/`) - Enhanced LLM-powered components with full Langbase-style integration
- **Memory** (`memory/`) - Vector-based document storage and retrieval with advanced features
- **Threads** (`thread/`) - Conversation state management with advanced archiving and tagging
- **Workflows** (`workflow/`) - **ENHANCED!** Multi-step orchestration engine with **byte-for-byte Langbase parity** including await-able builder pattern, secret scoping, thread handoff, and advanced retry strategies
- **Parser** (`parser/`) - **NEW!** Langbase-compatible document parser with job lifecycle, table extraction, and OCR

### Advanced Modules (Beyond Langbase)

- **Embedders** (`embedders/`) - Pluggable embedding providers (OpenAI, HuggingFace)
- **Chunker** (`chunker/`) - Token-based text chunking with overlap support
- **Parser** (`parser/`) - **ENHANCED!** Advanced document parsing with Docling integration
- **Stores** (`stores/`) - Vector database backends (FAISS, PGVector)
- **Keysets** (`keysets/`) - API key management and rotation
- **Experiments** (`experiments/`) - A/B testing and experiment tracking
- **Versioning** (`versioning/`) - Primitive versioning and forking
- **Analytics** (`analytics/`) - Usage tracking and performance monitoring
- **Auth** (`auth/`) - User and organization management

### Backend Layer

- **API** (`backend/api/`) - REST endpoints for primitives
- **Services** (`backend/services/`) - Thin wrappers that import primitives

### Data Layer

- **Database** (`database/`) - Migrations, schemas, and seed data
- **Data** (`data/`) - JSON, CSV, and database assets

## 🆕 Complete Feature Parity with Langbase

LangPy now provides **100% feature parity** with Langbase plus significant enhancements:

### ✅ **Agent Primitive** - 100% Complete
- **Tool execution loops** with recursive calling
- **Parallel tool execution** support
- **Streaming responses** with real-time output and automatic helper functions
- **Advanced retry logic** and error handling
- **Multi-LLM provider** support

### ✅ **Pipe Primitive** - 100% Complete  
- **Single LLM call** semantics (matches Langbase)
- **Tool definitions** for context (no execution loops)
- **Preset management** with JSON storage
- **Message templating** with variable interpolation
- **Multi-provider** support (OpenAI, Anthropic, Mistral, etc.)

### ✅ **Memory Primitive** - 100% Complete
- **Automatic pipeline orchestration**: Parser → Chunker → Embed → Store
- **Real vector embeddings** using OpenAI text-embedding-3-large (not placeholders!)
- **Vector similarity search** with FAISS, pgvector, and Docling backends
- **Advanced filtering** by metadata, tags, tiers (general, important, critical)
- **Job tracking and progress monitoring** for document processing
- **Standalone primitive access** to parser, chunker, embed, and store individually
- **Multiple storage backends** (FAISS, PostgreSQL+pgvector, Docling)
- **Long-term persistence** with PostgreSQL + pgvector
- **Cross-session memory retention** with ACID compliance
- **100% Langbase vector storage parity** achieved!

### ✅ **Thread Primitive** - 100% Complete
- **Conversation management** with message persistence
- **Advanced archiving** and status management
- **Thread tagging** and organization
- **Message search** across threads
- **Conversation summaries** and analytics

### ✅ **Workflow Primitive** - 100% Complete + Enhanced
- **Multi-primitive step execution** (agent, pipe, memory, thread)
- **Await-able builder pattern** with `await workflow.step(**config)`
- **Enhanced error taxonomy** (TimeoutError, RetryExhaustedError, StepError)
- **Secret scoping** with GitHub Actions-style `use_secrets[]`
- **Thread handoff** with `lb-thread-id` header interception
- **Advanced retry strategies** (fixed, linear, exponential with jitter)
- **Parallel execution** with dependency resolution and groups
- **Run history registry** with SQLite persistence and filtering
- **CLI support** with `python -m workflow run file.py --debug`
- **Rich console logging** with debug telemetry
- **🆕 Jinja2-style template engine** for dynamic config resolution
- **🆕 Streamlit web dashboard** for visual run monitoring and analytics

### ✅ **Parser Primitive** - 100% Complete
- **Job lifecycle management** (queued → processing → ready → failed)
- **Table extraction and reconstruction** with Docling integration
- **OCR processing** for images and scanned PDFs
- **Multiple input types** (file, URL, text)
- **Comprehensive metadata** extraction and statistics
- **Langbase-compatible REST API** with full CRUD operations
- **Background job processing** with async status polling
- **Webhook support** for job completion notifications

### ✅ **SDK Integration** - 100% Complete + Enhanced
- **Unified interface** for all primitives
- **Enhanced workflow integration** with full Langbase parity
- **Advanced module integration** (keysets, experiments, versioning, analytics)
- **Convenience methods** for common workflows
- **Complete primitive integration** with automatic runner registration
- **Enhanced error handling** and retry configurations through SDK

## 🚀 Advanced Features Beyond Langbase

### **Modular Architecture**
- **Pluggable components** for embedders, chunkers, parsers, stores
- **Extensible design** for custom providers and backends
- **Testable architecture** with clear separation of concerns

### **Production Features**
- **Analytics tracking** for usage monitoring
- **Version management** for primitive evolution
- **Experiment framework** for A/B testing
- **User/org management** for multi-tenant applications
- **Key management** with rotation and security

### **Enhanced Workflows**
- **Multi-primitive orchestration** in single workflows
- **Advanced dependency resolution** with circular detection
- **Conditional execution** based on context
- **Parallel step execution** with proper synchronization

### **Simplified Streaming**
- **Automatic streaming helpers** when `stream=True` is set during agent creation
- **No manual chunk parsing** required - just set `stream=True` and call `run()`
- **Clean, readable code** without complex streaming logic
- **Automatic content extraction** and printing from streaming chunks
- **Universal LLM support** - works with OpenAI, Anthropic, Mistral, and all supported providers

### **Advanced Memory Orchestrator**
- **Automatic pipeline execution**: Upload triggers Parser → Chunker → Embed → Store
- **Real embedding generation** with OpenAI text-embedding-3-large
- **Vector similarity search** with FAISS, pgvector, and Docling (cosine similarity)
- **Job tracking and progress monitoring** for document processing
- **Standalone primitive access** to individual components
- **Metadata filtering** and querying with tier-based organization
- **Similarity search** with configurable thresholds
- **Document management** with source tracking and processing statistics
- **True long-term persistence** with PostgreSQL + pgvector
- **Scalable storage** limited only by disk space
- **Production-ready** with connection pooling and indexing
- **Fallback to text search** when embeddings fail gracefully

### **Enhanced Threads**
- **Thread archiving** and status management
- **Message metadata** and search capabilities
- **Conversation analytics** and summaries
- **Tag-based organization** and filtering

## 📦 Installation

### Quick Install (Recommended)

Install directly from GitHub:

```bash
pip install git+https://github.com/Mrfabm/langpy.git
```

**That's it!** Installs in 1-2 minutes with all core features included.

### For Development

If you want to modify the code:

```bash
git clone https://github.com/Mrfabm/langpy.git
cd langpy

# Install in editable mode
pip install -e .
```

### Optional Features

Install additional features as needed:

```bash
# Add more LLM providers (Claude, Mistral)
pip install git+https://github.com/Mrfabm/langpy.git#egg=langpy[providers]

# Add PostgreSQL + pgvector
pip install git+https://github.com/Mrfabm/langpy.git#egg=langpy[postgres]

# Add ML features (PyTorch, Transformers) - Heavy!
pip install git+https://github.com/Mrfabm/langpy.git#egg=langpy[ml]
```

See [INSTALLATION.md](INSTALLATION.md) for detailed installation options.

### Updating LangPy

If you already have LangPy installed, update to the latest version:

```bash
# Check your current version
pip show langpy

# Update to latest version
pip install --upgrade langpy

# Or install specific version
pip install langpy==0.2.0

# Check if you need to update
python examples/package_update_guide.py
```

### Development Installation

```bash
git clone <your-repo-url>
cd primitives

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # (Windows)
# or
source .venv/bin/activate  # (Linux/Mac)

# Install with development dependencies
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

## 🎯 Quick Start

### Basic Usage

```python
from langpy.sdk import parser, memory, agent

# Initialize primitives
parser_instance = parser()
memory_instance = memory()
agent_instance = agent()

# Use primitives
result = await parser_instance.parse_file("document.pdf")
response = await agent_instance.run(model="gpt-4", input="Hello!")

# Simple streaming (NEW!) - works with ALL LLM providers!
from sdk import agent, OpenAI, Anthropic, Mistral

# OpenAI
openai_backend = OpenAI(api_key=openai_key)
agent_interface = agent(async_backend=openai_backend.call_async)
agent_instance = agent_interface.create_agent(
    name="openai_agent",
    instructions="You are a helpful assistant.",
    input="What is AI?",
    model="gpt-4o-mini",
    api_key=openai_key,
    stream=True  # Works with OpenAI!
)

# Anthropic
anthropic_backend = Anthropic(api_key=anthropic_key)
agent_interface = agent(async_backend=anthropic_backend.call_async)
agent_instance = agent_interface.create_agent(
    name="anthropic_agent",
    instructions="You are a helpful assistant.",
    input="What is AI?",
    model="claude-3-haiku-20240307",
    api_key=anthropic_key,
    stream=True  # Works with Anthropic!
)

# Just call run() - no manual streaming logic needed!
full_response = await agent_instance.run()
print(f"Response length: {len(full_response)} characters")
```
from pipe.adapters.openai import openai_async_llm

# Initialize with OpenAI backend
langpy = Primitives(async_backend=openai_async_llm)

# Use any primitive
response = await langpy.agent.run(
    model="gpt-4",
    input="What is artificial intelligence?",
    apiKey="sk-..."
)

# Create a knowledge base
await langpy.create_knowledge_base(
    documents=["AI is transforming industries...", "Machine learning enables..."],
    namespace="ai_knowledge"
)

# Chat with memory integration
result = await langpy.chat_with_memory(
    "Tell me about AI applications",
    memory_query="AI technology",
    k=3
)

# Long-term memory with PostgreSQL (automatic setup)
from memory import AsyncMemory, MemorySettings

settings = MemorySettings(
    backend="pgvector",
    pg_dsn="postgresql://user:pass@localhost/langpy_memory",
    pg_embedding_model="openai:text-embedding-ada-002"  # Real embeddings!
)

# Initialize memory with real vector storage
memory = AsyncMemory(settings)

# Add content with real embeddings
await memory.add(
    "AI is transforming healthcare with early disease detection.",
    tier="important",
    source="healthcare_article.txt",
    tags=["ai", "healthcare", "medicine"]
)

# Vector similarity search (not text search!)
results = await memory.query("AI in medicine", k=5, min_score=0.7)
print(f"Found {len(results)} similar items with vector search")

# Cross-session persistence - data survives restarts!
await memory.close()

## 🧪 Demos & Examples

### Quick Start: Simple User Setup

See exactly what users will do when getting started:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the simple setup guide
python examples/simple_user_setup.py
```

This shows the complete user workflow:
- ✅ Environment setup with .env file
- ✅ Creating your first AI agent
- ✅ Setting up memory system
- ✅ Running simple LLM pipes
- ✅ Memory querying and storage

### Real User Workflow Example

See a complete application example:

```bash
# Run the real workflow demo
python examples/real_user_workflow.py
```

This demonstrates:
- ✅ Complete application architecture
- ✅ PostgreSQL persistence (if configured)
- ✅ Agent with memory integration
- ✅ Workflow system usage
- ✅ Error handling and troubleshooting

### Vector Storage Demo (100% Langbase Parity)

Run the comprehensive vector storage demo to see real embeddings and vector similarity search in action:

```bash
# Run the advanced demo
python demos/memory/vector_storage_demo.py
```

This demo showcases:
- ✅ Real embedding generation with OpenAI text-embedding-ada-002
- ✅ Vector similarity search with pgvector (cosine similarity)
- ✅ PostgreSQL persistence and cross-session memory
- ✅ Metadata filtering and tagging
- ✅ Memory tiers (general, important, critical)
- ✅ Bulk operations and token tracking
- ✅ Fallback to text search when embeddings fail

### Environment Setup

**📖 Complete Setup Guide**: See [setup/ENVIRONMENT_SETUP.md](setup/ENVIRONMENT_SETUP.md) for detailed instructions on configuring all supported LLM providers and database options.

**Quick Start:**
```bash
# Copy the example environment file
cp setup/env.example .env

# Edit .env with your API keys and database settings
# See setup/ENVIRONMENT_SETUP.md for all supported providers
```

**Supported LLM Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google Gemini
- Groq (fast inference)
- Mistral AI
- Perplexity AI
- Ollama (local models)

**Supported Databases:**
- PostgreSQL with pgvector (recommended)
  - Use `python start_pgvector.py` for quick Docker setup
  - See [setup/ENVIRONMENT_SETUP.md](setup/ENVIRONMENT_SETUP.md) for configuration
- In-memory storage (development)
- Local file storage

**Required:**
- At least one LLM API key for text generation
- OpenAI API key for embeddings (recommended)
- PostgreSQL connection string (optional - for persistence)

memory = AsyncMemory(settings)
await memory.add_text(
    "This data persists across application restarts!",
    source="important_note.txt",
    tier="HARD",
    tags=["persistent", "important"]
)
```

### Advanced Workflow

```python
from workflow.async_workflow import AsyncWorkflow, WorkflowRegistry, StepConfig

# Create workflow with multi-primitive steps
registry = WorkflowRegistry()
workflow = AsyncWorkflow(registry=registry)

steps = [
    StepConfig(
        id="extract_data",
        type="function",
        run=lambda ctx: ["data1", "data2"]
    ),
    StepConfig(
        id="process_with_ai",
        type="agent",
        ref="data_processor",
        after=["extract_data"]
    ),
    StepConfig(
        id="store_results",
        type="memory",
        ref="store_processed",
        after=["process_with_ai"]
    )
]

registry.create("data_pipeline", steps)
result = await workflow.run("data_pipeline", inputs={"source": "database"})
```

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Memory backend
MEMORY_BACKEND=docling
MEMORY_NAMESPACE=my_app

# Analytics
ANALYTICS_ENABLED=true
```

### Advanced Settings

```python
from sdk import Primitives
from memory.settings import MemorySettings

# Configure with advanced settings
langpy = Primitives(
    async_backend=openai_async_llm,
    memory_settings={
        "backend": "docling",
        "namespace": "production",
        "api_key": "your-docling-key"
    },
    thread_settings={
        "storage_path": "~/.langpy/threads"
    },
    workflow_settings={
        "storage_path": "~/.langpy/workflows"
    }
)
```

## 📚 Documentation

### New Architecture (v2.0)
- **Core API Reference**: See `docs/CORE_API.md` for the composable primitives architecture
- **Testing Guide**: See `docs/TESTING.md` for mock primitives and testing utilities
- **Getting Started**: See `docs/GETTING_STARTED.md` for quick start guide

### SDK Reference
- **API Reference**: See individual primitive modules for detailed API docs
- **Examples**: Check `demos/` directory for comprehensive examples
- **Architecture**: See `docs/` for design decisions and patterns
- **Memory SDK Options**: See `docs/memory_sdk_options.md` for complete Memory SDK reference
- **Pipe SDK Options**: See `docs/pipe_sdk_options.md` for complete Pipe SDK reference
- **Agent SDK Options**: See `docs/agent_sdk_options.md` for complete Agent SDK reference
- **Workflow SDK Options**: See `docs/workflow_sdk_options.md` for complete Workflow SDK reference
- **Stores SDK Options**: See `docs/stores_sdk_options.md` for complete Stores SDK reference
- **Thread SDK Options**: See `docs/thread_sdk_options.md` for complete Thread SDK reference
- **Tools SDK Options**: See `docs/tools_sdk_options.md` for complete Tools SDK reference
- **Embed SDK Options**: See `docs/embed_sdk_options.md` for complete Embed SDK reference
- **Chunker SDK Options**: See `docs/chunker_sdk_options.md`