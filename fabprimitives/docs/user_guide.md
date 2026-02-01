# LangPy User Guide

A comprehensive guide to building sophisticated AI agents with LangPy primitives.

## 📦 Installation

### Quick Install
```bash
pip install langpy
```

### Development Install
```bash
pip install "langpy[dev]"
```

### Full Install (with docs)
```bash
pip install "langpy[all]"
```

## 🚀 Quick Start

### Basic Setup
```python
import asyncio
from langpy.sdk import Primitives
from langpy.pipe.adapters.openai import openai_async_llm

# Initialize LangPy
langpy = Primitives(async_backend=openai_async_llm)

# Your first AI agent
async def main():
    response = await langpy.agent.run(
        model="gpt-4",
        input="What is artificial intelligence?",
        apiKey="sk-your-openai-key"
    )
    print(response)

asyncio.run(main())
```

## 🏗️ Core Primitives

### 1. Agent Primitive
Autonomous AI reasoning engines with think-act-observe-reflect loops.

```python
from langpy.sdk import agent

# Create an agent
agent_instance = agent()

# Run with tools
response = await agent.run(
    model="gpt-4",
    input="Calculate the weather for New York",
    tools=[weather_tool, calculator_tool],
    apiKey="sk-..."
)

# Stream responses
async for chunk in agent.run_stream(
    model="gpt-4",
    input="Explain quantum computing",
    apiKey="sk-..."
):
    print(chunk, end="")
```

### 2. Pipe Primitive
Enhanced LLM-powered components for single-turn interactions.

```python
from langpy.sdk import pipe

# Create a pipe
pipe_instance = pipe()

# Simple LLM call
response = await pipe.run(
    model="gpt-4",
    input="Translate 'Hello world' to Spanish",
    apiKey="sk-..."
)

# With system prompt
response = await pipe.run(
    model="gpt-4",
    system="You are a helpful coding assistant.",
    input="Write a Python function to sort a list",
    apiKey="sk-..."
)
```

### 3. Memory Primitive
Vector-based document storage and retrieval.

```python
from langpy.sdk import memory

# Initialize memory
memory_instance = memory()

# Add documents
await memory.add_text(
    text="AI is transforming industries worldwide.",
    source="research"
)

# Search for relevant information
results = await memory.query(
    query="artificial intelligence applications",
    k=3
)

# Add with metadata
await memory.add_text(
    text="New AI breakthrough in healthcare",
    source="research_paper",
    tags=["healthcare", "breakthrough"]
)
```

### 4. Thread Primitive
Conversation state management with archiving and tagging.

```python
from langpy.sdk import thread

# Create a thread
thread_instance = thread()

# Create a new thread
thread_obj = await thread.create_thread(
    name="ML Discussion",
    metadata={"topic": "machine learning"}
)

# Add messages
await thread.add_message(
    thread_id=thread_obj.id,
    role="user",
    content="What is machine learning?"
)

await thread.add_message(
    thread_id=thread_obj.id,
    role="assistant", 
    content="Machine learning is a subset of AI..."
)

# Get conversation history
messages = await thread.get_messages(thread_id=thread_obj.id)

# Delete thread
await thread.delete_thread(thread_obj.id)
```

### 5. Workflow Primitive
Multi-step orchestration with dependency resolution.

```python
from langpy.sdk import workflow

# Create workflow interface
workflow_instance = workflow()

# Define workflow steps
steps = [
    workflow.create_function_step(
        step_id="extract_data",
        func=lambda ctx: ["data1", "data2"]
    ),
    workflow.create_agent_step(
        step_id="process_with_ai",
        agent_ref="data_processor"
    ),
    workflow.create_memory_step(
        step_id="store_results",
        memory_ref="store_processed"
    )
]

# Create and run workflow
workflow.create_workflow("data_pipeline", steps)
result = await workflow.run_workflow("data_pipeline", inputs={"source": "database"})
```

## 🔧 Advanced Modules

### Embedders
Pluggable embedding providers for vector operations.

```python
from langpy.embedders import OpenAIEmbedder

# Create embedder
embedder = OpenAIEmbedder(api_key="sk-...")

# Generate embeddings
embeddings = await embedder.embed_texts([
    "Hello world",
    "AI is amazing"
])
```

### Chunker
Docling-powered, structure-aware text chunking with overlap support.

```python
from langpy.sdk import chunker

# Create chunker
chunker_instance = chunker()

# Get chunker information
info = chunker_instance.get_chunker_info()
print(f"Features: {info['features']}")

# Chunk text
chunks = await chunker_instance.chunk_text(
    text="Your long document here...",
    chunk_max_length=2000,
    chunk_overlap=256
)

# Chunk file
chunks = await chunker_instance.chunk_file("document.txt")

# Get statistics
stats = chunker_instance.get_chunking_statistics()
```

### Parser
Standalone document parsing with comprehensive format support.

```python
from langpy.sdk import parser

# Initialize parser
parser_instance = parser()

# Get parser information and capabilities
info = parser_instance.get_parser_info()
formats = parser_instance.get_supported_formats()
file_types = parser_instance.get_supported_file_types()

# Parse different content types
result = await parser_instance.parse_file("document.pdf")
result = await parser_instance.parse_content("text content", filename="document.txt")
result = await parser_instance.parse_url("https://example.com/document.pdf")
result = await parser_instance.parse_text("raw text content")
result = await parser_instance.parse_bytes(b"raw bytes content")

# Access parsed content and metadata
print(f"Content: {result.content}")
print(f"Pages: {result.metadata.page_count}")
print(f"Tables: {result.metadata.table_count}")
print(f"Characters: {result.metadata.char_count:,}")
print(f"Estimated tokens: {result.metadata.token_estimate:,}")

# Create different option configurations
fast_options = parser_instance.create_fast_options()  # Quick text extraction
accurate_options = parser_instance.create_accurate_options()  # Full analysis

# Create custom parser options
options = parser_instance.create_options(
    enable_ocr=True,
    ocr_languages=["eng", "spa"],
    ocr_confidence_threshold=0.8,
    preserve_whitespace=True,
    merge_hyphens=True,
    strip_headers_footers=False,
    table_strategy="docling",
    max_file_size=100 * 1024 * 1024,  # 100MB
    parse_timeout=300,
    language_hint="English",
    speed_mode="Accurate"  # "Fast" or "Accurate"
)

# Validate files before parsing
validation = parser_instance.validate_file("document.pdf")
if validation["supported"] and not validation["issues"]:
    result = await parser_instance.parse_file("document.pdf", options=options)

# Detect MIME types
mime_type = parser_instance.detect_mime_type(content, filename)
description = parser_instance.get_format_description(mime_type)
```

### Stores
Vector database backends for memory storage.

```python
from langpy.stores import FAISSStore, PGVectorStore

# FAISS store (local)
store = FAISSStore(dimension=1536)

# PGVector store (database)
store = PGVectorStore(
    connection_string="postgresql://user:pass@localhost/db",
    table_name="embeddings"
)
```

## 🎯 SDK Integration

### Direct Primitive Imports (Recommended)
```python
# Import only what you need
from langpy.sdk import parser, memory, agent

# Initialize primitives
parser_instance = parser()
memory_instance = memory()
agent_instance = agent()

# Use each primitive directly
result = await parser_instance.parse_file("document.pdf")
memory_results = await memory_instance.query("search query")
response = await agent_instance.run(model="gpt-4", input="Hello")
```

### Unified Interface (Optional)
```python
from langpy.sdk import Primitives

# Initialize with all primitives
langpy = Primitives(
    async_backend=openai_async_llm,
    memory_settings={
        "backend": "docling",
        "namespace": "my_app"
    }
)

# Use any primitive through SDK
response = await langpy.agent.run(...)
memory_results = await langpy.memory.query(...)
thread_messages = await langpy.thread.get_messages(...)
```

### Advanced Configuration
```python
from langpy.sdk import Primitives
from langpy.memory.settings import MemorySettings

langpy = Primitives(
    async_backend=openai_async_llm,
    memory_settings=MemorySettings(
        backend="docling",
        namespace="production",
        api_key="your-docling-key"
    ),
    thread_settings={
        "storage_path": "~/.langpy/threads"
    },
    workflow_settings={
        "storage_path": "~/.langpy/workflows"
    }
)
```

## 🔑 Keyset Management

```python
from langpy.keysets import KeysetManager

# Initialize keyset manager
keysets = KeysetManager()

# Add API keys
await keysets.add_key("openai", "sk-...")
await keysets.add_key("anthropic", "sk-ant-...")

# Use keys in primitives
response = await agent.run(
    model="gpt-4",
    input="Hello",
    keyset="openai"  # Uses stored key
)
```

## 📊 Analytics & Experiments

```python
from langpy.analytics import Logger
from langpy.experiments import ExperimentRunner

# Analytics tracking
logger = Logger()
await logger.log_event("agent_run", {"model": "gpt-4", "tokens": 150})

# A/B testing
experiments = ExperimentRunner()
variant = await experiments.get_variant("new_ui", user_id="123")
```

## 🏢 Production Features

### User & Organization Management
```python
from langpy.auth import UserManager, OrgManager

# User management
user_manager = UserManager()
user = await user_manager.create_user("john@example.com")

# Organization management
org_manager = OrgManager()
org = await org_manager.create_org("Acme Corp")
```

### Version Management
```python
from langpy.versioning import VersionManager

# Version management
version_manager = VersionManager()
version = await version_manager.create_version("agent_v1", config={...})
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Memory backend
MEMORY_BACKEND=docling
MEMORY_NAMESPACE=my_app

# PostgreSQL Vector Storage (for long-term memory)
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/langpy_db
POSTGRES_NAMESPACE=my_app
POSTGRES_TABLE_PREFIX=langpy_

# Analytics
ANALYTICS_ENABLED=true
```

### Vector Storage Configuration

LangPy supports multiple vector storage backends:

#### 1. Docling (Default - In-Memory)
```python
from langpy.memory import AsyncMemory, MemorySettings

# Default in-memory storage
memory = AsyncMemory(MemorySettings(
    backend="docling",
    namespace="my_app"
))
```

#### 2. PostgreSQL with pgvector (Long-term Storage)
```python
# Configure with PostgreSQL for persistent storage
memory = AsyncMemory(MemorySettings(
    backend="postgresql",
    namespace="my_app",
    connection_string="postgresql://user:password@localhost:5432/langpy_db"
))

# Or use environment variables
import os
from dotenv import load_dotenv
load_dotenv()

memory = AsyncMemory(MemorySettings(
    backend="postgresql",
    namespace=os.getenv("POSTGRES_NAMESPACE", "default"),
    connection_string=os.getenv("POSTGRES_CONNECTION_STRING")
))
```

#### 3. Automatic Setup
```bash
# Run the setup script to configure your environment
python setup/setup_env.py

# This will guide you through creating a .env file with:
# - PostgreSQL connection details
# - OpenAI API key for embeddings
# - Memory backend preferences
```

### Advanced Settings
```python
# Custom memory settings
memory_settings = {
    "backend": "docling",
    "namespace": "production",
    "api_key": "your-docling-key",
    "similarity_threshold": 0.8
}

# Custom thread settings
thread_settings = {
    "storage_path": "~/.langpy/threads",
    "max_messages": 1000
}
```

## 🧪 Testing Your Setup

```python
import asyncio
from langpy.sdk import Primitives

async def test_setup():
    # Initialize LangPy
    langpy = Primitives()
    
    # Test basic functionality
    try:
        response = await langpy.pipe.run(
            model="gpt-3.5-turbo",
            input="Say 'Hello from LangPy!'",
            apiKey="sk-..."  # Replace with your key
        )
        print("✅ LangPy is working!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(test_setup())
```

## 📚 Examples & Demos

Check the `demos/` directory for comprehensive examples:
- `demos/agent/` - Agent examples
- `demos/pipe/` - Pipe examples  
- `demos/memory/` - Memory examples
- `demos/workflow/` - Workflow examples
- `demos/sdk/` - SDK integration examples

## 🆘 Troubleshooting

### Common Issues

1. **Import Error: No module named 'jwt'**
   ```bash
   pip uninstall -y jwt  # Remove conflicting package
   pip install PyJWT     # Install correct package
   ```

2. **Memory Backend Issues**
   - Ensure your memory backend (Docling, FAISS, etc.) is properly configured
   - Check API keys and connection strings

3. **LLM Provider Issues**
   - Verify API keys are valid
   - Check rate limits and quotas
   - Ensure model names are correct

### Getting Help

- **Documentation**: Check individual primitive modules for detailed API docs
- **Examples**: Review `demos/` directory for working examples
- **Issues**: Report bugs on GitHub with detailed error messages

## 🎉 Next Steps

1. **Start Simple**: Begin with basic pipe and agent usage
2. **Add Memory**: Integrate vector storage for context
3. **Build Workflows**: Create multi-step AI processes
4. **Scale Up**: Add analytics, experiments, and production features

LangPy provides everything you need to build sophisticated AI applications with 100% feature parity to Langbase plus advanced capabilities! 