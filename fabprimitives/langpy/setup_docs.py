"""
LangPy Documentation Setup.

Sets up MkDocs documentation structure for LangPy projects.
"""

import os


def setup_docs():
    """Set up documentation structure for a LangPy project."""
    print("Setting up LangPy documentation...")

    # Create docs directory
    os.makedirs("docs", exist_ok=True)
    print("  Created docs/")

    # Create mkdocs.yml
    mkdocs_config = '''site_name: LangPy Documentation
site_description: Documentation for your LangPy project
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo
      accent: indigo

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference: api.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
  - codehilite
'''

    if not os.path.exists("mkdocs.yml"):
        with open("mkdocs.yml", "w") as f:
            f.write(mkdocs_config)
        print("  Created mkdocs.yml")

    # Create index.md
    index_content = '''# Welcome to LangPy

LangPy is a Python implementation of Langbase's AI primitives.

## Quick Start

```python
from langpy import Langpy

lb = Langpy(api_key="your-api-key")

# Use primitives
response = await lb.agent.run(
    model="openai:gpt-4",
    input="Hello!",
    instructions="Be helpful"
)
```

## Features

- **Agent** - Unified LLM API (100+ models)
- **Pipe** - Single LLM call with templates
- **Memory** - Vector storage and RAG
- **Thread** - Conversation history
- **Workflow** - Multi-step orchestration
- **Parser** - Document text extraction
- **Chunker** - Text segmentation
- **Embed** - Text to vectors
- **Tools** - Pre-built and custom tools
'''

    if not os.path.exists("docs/index.md"):
        with open("docs/index.md", "w") as f:
            f.write(index_content)
        print("  Created docs/index.md")

    # Create getting-started.md
    getting_started = '''# Getting Started

## Installation

```bash
pip install langpy
```

## Configuration

Set your API keys:

```bash
export LANGPY_API_KEY=your-key
export OPENAI_API_KEY=your-openai-key
```

Or use a `.env` file:

```
LANGPY_API_KEY=your-key
OPENAI_API_KEY=your-openai-key
```

## First Steps

```python
import asyncio
from langpy import Langpy

async def main():
    lb = Langpy()

    response = await lb.agent.run(
        model="openai:gpt-4",
        input="What is Python?",
    )

    print(response.output)

asyncio.run(main())
```
'''

    if not os.path.exists("docs/getting-started.md"):
        with open("docs/getting-started.md", "w") as f:
            f.write(getting_started)
        print("  Created docs/getting-started.md")

    # Create api.md
    api_content = '''# API Reference

::: langpy
    options:
      show_root_heading: true
      show_source: true
'''

    if not os.path.exists("docs/api.md"):
        with open("docs/api.md", "w") as f:
            f.write(api_content)
        print("  Created docs/api.md")

    print("\nDone! To build documentation:")
    print("  pip install mkdocs mkdocs-material mkdocstrings[python]")
    print("  mkdocs serve  # Development server")
    print("  mkdocs build  # Build static site")


if __name__ == "__main__":
    setup_docs()
