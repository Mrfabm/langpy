# LangPy Complete Guide

A comprehensive guide to installing and using all functionalities of LangPy - the complete AI agent framework with 100% Langbase parity plus advanced capabilities.

## 📦 Installation

### Quick Install
```bash
pip install langpy
```

### Development Install
```bash
pip install "langpy[dev]"
```

### Full Install (with all optional dependencies)
```bash
pip install "langpy[all]"
```

## 🚀 Quick Start

```python
import langpy
from langpy.sdk import Primitives
from langpy.pipe.adapters.openai import openai_async_llm

# Initialize LangPy
langpy_instance = Primitives(async_backend=openai_async_llm)

# Use any primitive immediately
response = await langpy_instance.agent.run(
    model="gpt-4",
    input="What is artificial intelligence?",
    apiKey="sk-..."
)
```

## 🏗️ Core Primitives (100% Langbase Parity)

### 1. Agent Primitive

**Purpose**: Autonomous AI reasoning engines with think-act-observe-reflect loops.

```python
from langpy.agent import AsyncAgent

# Create agent
agent = AsyncAgent()

# Basic usage
response = await agent.run(
    model="gpt-4",
    input="Analyze this data and provide insights",
    apiKey="sk-..."
)

# With tools
tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
]

response = await agent.run(
    model="gpt-4",
    input="What's the latest news about AI?",
    tools=tools,
    apiKey="sk-..."
)

# With streaming
async for chunk in agent.run_stream(
    model="gpt-4",
    input="Explain quantum computing",
    apiKey="sk-..."
):
    print(chunk, end="")
```

### 2. Pipe Primitive

**Purpose**: Enhanced LLM-powered components with full Langbase-style integration.

```python
from langpy.pipe import AsyncPipe

# Create pipe
pipe = AsyncPipe()

# Single LLM call
response = await pipe.run(
    model="gpt-4",
    input="Summarize this text: {text}",
    variables={"text": "Long document content..."},
    apiKey="sk-..."
)

# With preset
await pipe.create_preset(
    name="summarizer",
    model="gpt-4",
    input="Summarize the following text in 3 bullet points: {text}",
    temperature=0.3
)

response = await pipe.run_preset(
    name="summarizer",
    variables={"text": "Document to summarize"},
    apiKey="sk-..."
)

# With tools (for context, not execution)
tools = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations"
    }
]

response = await pipe.run(
    model="gpt-4",
    input="Calculate 15 * 23 using the calculator",
    tools=tools,
    apiKey="sk-..."
)
```

### 3. Memory Primitive

**Purpose**: Vector-based document storage and retrieval with advanced features.

```python
from langpy.memory import AsyncMemory

# Create memory instance
memory = AsyncMemory()

# Add documents
documents = [
    "Artificial intelligence is transforming industries worldwide.",
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks with multiple layers."
]

await memory.add_documents(
    documents=documents,
    namespace="ai_knowledge",
    metadata={"source": "research_papers", "date": "2024"}
)

# Search documents
results = await memory.search(
    query="What is machine learning?",
    namespace="ai_knowledge",
    k=3,
    threshold=0.7
)

# Advanced filtering
results = await memory.search(
    query="AI applications",
    namespace="ai_knowledge",
    filters={
        "source": "research_papers",
        "date": {"$gte": "2023"}
    },
    k=5
)

# Bulk operations
await memory.add_documents_bulk(
    documents=large_document_list,
    namespace="bulk_import",
    batch_size=100
)

# Update metadata
await memory.update_metadata(
    document_id="doc_123",
    metadata={"status": "reviewed", "reviewer": "john"}
)

# Delete documents
await memory.delete_documents(
    filters={"source": "old_data"}
)
```

### 4. Thread Primitive

**Purpose**: Conversation state management with advanced archiving and tagging.

```python
from langpy.thread import AsyncThread

# Create thread
thread = AsyncThread()

# Start conversation
thread_id = await thread.create(
    title="AI Discussion",
    tags=["ai", "discussion"],
    metadata={"participants": ["user", "assistant"]}
)

# Add messages
await thread.add_message(
    thread_id=thread_id,
    role="user",
    content="What is artificial intelligence?",
    metadata={"timestamp": "2024-01-01T10:00:00Z"}
)

await thread.add_message(
    thread_id=thread_id,
    role="assistant",
    content="Artificial intelligence is...",
    metadata={"model": "gpt-4"}
)

# Get conversation
messages = await thread.get_messages(thread_id=thread_id)

# Search messages
results = await thread.search_messages(
    query="machine learning",
    thread_ids=[thread_id],
    limit=10
)

# Archive thread
await thread.archive(thread_id=thread_id, reason="completed")

# Get archived threads
archived = await thread.get_archived_threads(
    filters={"tags": ["ai"]}
)

# Update thread metadata
await thread.update_metadata(
    thread_id=thread_id,
    metadata={"status": "resolved", "resolution": "satisfactory"}
)
```

### 5. Workflow Primitive

**Purpose**: Multi-step orchestration engine with multi-primitive step execution.

```python
from langpy.workflow import AsyncWorkflow, WorkflowRegistry, StepConfig

# Create workflow registry
registry = WorkflowRegistry()
workflow = AsyncWorkflow(registry=registry)

# Define workflow steps
steps = [
    StepConfig(
        id="extract_data",
        type="function",
        run=lambda ctx: ["data1", "data2", "data3"]
    ),
    StepConfig(
        id="process_with_ai",
        type="agent",
        ref="data_processor",
        after=["extract_data"],
        inputs={
            "data": "{{extract_data.output}}"
        }
    ),
    StepConfig(
        id="store_results",
        type="memory",
        ref="result_storage",
        after=["process_with_ai"],
        inputs={
            "documents": "{{process_with_ai.output}}"
        }
    ),
    StepConfig(
        id="generate_report",
        type="pipe",
        ref="report_generator",
        after=["store_results"],
        inputs={
            "summary": "{{store_results.output}}"
        }
    )
]

# Register workflow
registry.create("data_processing_pipeline", steps)

# Run workflow
result = await workflow.run(
    "data_processing_pipeline",
    inputs={"source": "database", "filters": {"date": "2024"}}
)

# Parallel execution
steps_parallel = [
    StepConfig(
        id="fetch_data_a",
        type="function",
        run=lambda ctx: "data_a"
    ),
    StepConfig(
        id="fetch_data_b", 
        type="function",
        run=lambda ctx: "data_b"
    ),
    StepConfig(
        id="combine_data",
        type="function",
        after=["fetch_data_a", "fetch_data_b"],
        run=lambda ctx: f"{ctx['fetch_data_a']} + {ctx['fetch_data_b']}"
    )
]

# Conditional execution
steps_conditional = [
    StepConfig(
        id="check_condition",
        type="function",
        run=lambda ctx: ctx.get("should_process", False)
    ),
    StepConfig(
        id="process_if_true",
        type="agent",
        after=["check_condition"],
        condition="{{check_condition.output}} == True"
    )
]
```

## 🔧 Advanced Modules

### Embedders

```python
from langpy.embedders import OpenAIEmbedder, HuggingFaceEmbedder

# OpenAI embeddings
openai_embedder = OpenAIEmbedder(api_key="sk-...")
embeddings = await openai_embedder.embed(["text1", "text2"])

# HuggingFace embeddings
hf_embedder = HuggingFaceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = await hf_embedder.embed(["text1", "text2"])
```

### Chunker

```python
from langpy.sdk import chunker

chunker_instance = chunker()

# Structure-aware chunking (Docling HybridChunker)
chunks = await chunker_instance.chunk_text(
    text="Long document content...",
    chunk_max_length=2000,
    chunk_overlap=256
)

# File chunking
chunks = await chunker_instance.chunk_file("document.txt")

# Session management
stats = chunker_instance.get_chunking_statistics()
```

### Parser

```python
from langpy.parser import AsyncParser

parser = AsyncParser()

# Parse different file types
text = await parser.parse_file("document.pdf")
text = await parser.parse_file("document.docx")
text = await parser.parse_file("document.txt")

# Parse URLs
text = await parser.parse_url("https://example.com/article")
```

### Stores

```python
from langpy.stores import FAISSStore, PGVectorStore

# FAISS store
faiss_store = FAISSStore()
await faiss_store.add_vectors(vectors, ids, metadata)

# PGVector store
pg_store = PGVectorStore(connection_string="postgresql://...")
await pg_store.add_vectors(vectors, ids, metadata)
```

### Keysets

```python
from langpy.keysets import KeysetManager

keyset_manager = KeysetManager()

# Add API keys
await keyset_manager.add_key(
    name="openai_prod",
    provider="openai",
    key="sk-...",
    metadata={"environment": "production"}
)

# Rotate keys
await keyset_manager.rotate_key("openai_prod")

# Get key
key = await keyset_manager.get_key("openai_prod")
```

### Experiments

```python
from langpy.experiments import ExperimentRunner

experiment_runner = ExperimentRunner()

# Create experiment
experiment_id = await experiment_runner.create_experiment(
    name="model_comparison",
    description="Compare GPT-4 vs Claude-3"
)

# Run A/B test
result = await experiment_runner.run_ab_test(
    experiment_id=experiment_id,
    variant_a={"model": "gpt-4"},
    variant_b={"model": "claude-3"},
    input_data=["test1", "test2", "test3"]
)

# Get results
results = await experiment_runner.get_results(experiment_id)
```

### Versioning

```python
from langpy.versioning import VersionManager

version_manager = VersionManager()

# Create version
version_id = await version_manager.create_version(
    primitive_type="agent",
    primitive_id="my_agent",
    version="1.2.0",
    config={"model": "gpt-4", "temperature": 0.7}
)

# Fork primitive
forked_id = await version_manager.fork_primitive(
    primitive_id="my_agent",
    new_name="my_agent_v2"
)

# Get version history
history = await version_manager.get_version_history("my_agent")
```

### Analytics

```python
from langpy.analytics import Logger

logger = Logger()

# Track usage
await logger.track_usage(
    primitive_type="agent",
    primitive_id="my_agent",
    input_tokens=100,
    output_tokens=50,
    duration=1.5,
    metadata={"user_id": "user123"}
)

# Get analytics
analytics = await logger.get_analytics(
    primitive_type="agent",
    date_range={"start": "2024-01-01", "end": "2024-01-31"}
)
```

### Auth

```python
from langpy.auth import UserManager, OrgManager

user_manager = UserManager()
org_manager = OrgManager()

# Create user
user_id = await user_manager.create_user(
    email="user@example.com",
    password="secure_password",
    metadata={"role": "developer"}
)

# Create organization
org_id = await org_manager.create_organization(
    name="My Company",
    owner_id=user_id
)

# Add user to org
await org_manager.add_user_to_org(
    org_id=org_id,
    user_id=user_id,
    role="admin"
)
```

## 🎯 SDK Integration

```python
from langpy.sdk import Primitives
from langpy.pipe.adapters.openai import openai_async_llm

# Initialize with all features
langpy = Primitives(
    async_backend=openai_async_llm,
    memory_settings={
        "backend": "docling",
        "namespace": "my_app"
    },
    thread_settings={
        "storage_path": "~/.langpy/threads"
    },
    workflow_settings={
        "storage_path": "~/.langpy/workflows"
    },
    keyset_settings={
        "storage_path": "~/.langpy/keysets"
    }
)

# Use unified interface
response = await langpy.agent.run(
    model="gpt-4",
    input="What is AI?",
    apiKey="sk-..."
)

# Create knowledge base
await langpy.create_knowledge_base(
    documents=["AI is transforming...", "Machine learning..."],
    namespace="ai_knowledge"
)

# Chat with memory
result = await langpy.chat_with_memory(
    "Tell me about AI applications",
    memory_query="AI technology",
    k=3
)

# Run workflow
workflow_result = await langpy.run_workflow(
    "my_workflow",
    inputs={"data": "..."}
)
```

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Memory backend
MEMORY_BACKEND=docling
MEMORY_NAMESPACE=my_app
MEMORY_API_KEY=your-docling-key

# Analytics
ANALYTICS_ENABLED=true
ANALYTICS_ENDPOINT=https://your-analytics.com

# Database
DATABASE_URL=postgresql://user:pass@localhost/langpy
```

### Advanced Settings

```python
from langpy.sdk import Primitives
from langpy.memory.settings import MemorySettings
from langpy.thread.settings import ThreadSettings

# Configure with advanced settings
langpy = Primitives(
    async_backend=openai_async_llm,
    memory_settings=MemorySettings(
        backend="docling",
        namespace="production",
        api_key="your-docling-key",
        similarity_threshold=0.7
    ),
    thread_settings=ThreadSettings(
        storage_path="~/.langpy/threads",
        max_messages_per_thread=1000
    ),
    workflow_settings={
        "storage_path": "~/.langpy/workflows",
        "max_concurrent_steps": 5
    },
    keyset_settings={
        "storage_path": "~/.langpy/keysets",
        "encryption_key": "your-encryption-key"
    }
)
```

## 🧪 Testing

```python
import pytest
from langpy.agent import AsyncAgent

@pytest.mark.asyncio
async def test_agent():
    agent = AsyncAgent()
    response = await agent.run(
        model="gpt-4",
        input="Hello",
        apiKey="test-key"
    )
    assert response is not None
```

## 🚀 Production Deployment

### Docker Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### FastAPI Integration

```python
from fastapi import FastAPI
from langpy.sdk import Primitives

app = FastAPI()
langpy = Primitives()

@app.post("/chat")
async def chat(message: str):
    response = await langpy.agent.run(
        model="gpt-4",
        input=message,
        apiKey="sk-..."
    )
    return {"response": response}
```

## 📚 Best Practices

1. **Use Environment Variables** for API keys and sensitive data
2. **Implement Proper Error Handling** for all async operations
3. **Use Type Hints** for better code maintainability
4. **Monitor Usage** with analytics for cost optimization
5. **Version Your Primitives** for production stability
6. **Use Memory Efficiently** with proper namespacing
7. **Implement Retry Logic** for network operations
8. **Test Thoroughly** with different scenarios

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Check environment variables and keyset configuration
3. **Memory Issues**: Verify backend configuration and API keys
4. **Workflow Errors**: Check step dependencies and conditions
5. **Performance Issues**: Monitor token usage and implement caching

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for specific modules
logging.getLogger("langpy.agent").setLevel(logging.DEBUG)
logging.getLogger("langpy.memory").setLevel(logging.DEBUG)
```

## 📞 Support

- **Documentation**: Check individual module docstrings
- **Examples**: See `demos/` directory for comprehensive examples
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join discussions and share solutions

---

**LangPy** - Complete AI Agent Framework with 100% Langbase Parity + Advanced Capabilities 