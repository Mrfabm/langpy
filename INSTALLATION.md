# LangPy Installation Guide

## 🚀 Quick Install (Recommended)

LangPy now uses **lightweight dependencies by default** for fast installation!

### Direct Installation (No Clone Required)

```bash
# Install directly from GitHub
pip install git+https://github.com/Mrfabm/langpy.git
```

**That's it!** No cloning, no setup - just one command.

**Install time: 1-2 minutes** ✅
**Packages: ~46 essential packages** ✅
**Size: ~300MB** ✅

### For Development (Editable Install)

If you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/Mrfabm/langpy.git
cd langpy

# Install in editable mode
pip install -e .
```

---

## 📦 What's Included by Default

The default installation includes everything you need for most use cases:

✅ **All 9 LangPy Primitives:**
- Agent (with OpenAI support)
- Pipe
- Memory (with FAISS vector storage)
- Thread
- Workflow
- Parser (PDF, Word, HTML)
- Chunker
- Embed
- Tools

✅ **Core Features:**
- Multi-agent workflows
- RAG (Retrieval-Augmented Generation)
- Vector storage with FAISS
- Conversation tracking
- Document parsing

---

## 🔧 Optional Features

Install additional features only if you need them:

### Add More LLM Providers

```bash
# Add Anthropic Claude
pip install "langpy[providers]"

# Or install manually
pip install anthropic mistralai
```

### Add PostgreSQL + pgvector

```bash
# For production-grade vector storage
pip install "langpy[postgres]"
```

### Add Backend/API Support

```bash
# FastAPI backend for building APIs
pip install "langpy[backend]"
```

### Add Advanced Document Parsing

```bash
# Excel, PowerPoint, images, OCR
pip install "langpy[parsing]"
```

### Add Dashboard/UI

```bash
# Streamlit dashboards
pip install "langpy[ui]"
```

### Add ML Features (HEAVY!)

⚠️ **Warning: This will download 2-3GB of packages!**

```bash
# PyTorch, Transformers, etc.
# Only install if you need local ML models
pip install "langpy[ml]"
```

### Install Everything (SLOW!)

⚠️ **Warning: This takes 30+ minutes and uses 4GB!**

```bash
pip install "langpy[all]"
```

---

## 📋 Installation Options Comparison

| Install Type | Command | Time | Size | Use Case |
|-------------|---------|------|------|----------|
| **Default** ⭐ | `pip install langpy` | 1-2 min | 300MB | Most users |
| **+ Providers** | `pip install langpy[providers]` | 2-3 min | 350MB | Claude, Mistral |
| **+ PostgreSQL** | `pip install langpy[postgres]` | 2-3 min | 400MB | Production DB |
| **+ Backend** | `pip install langpy[backend]` | 2-3 min | 350MB | API services |
| **+ ML** ⚠️ | `pip install langpy[ml]` | 30+ min | 3GB | Local models |
| **All** ⚠️ | `pip install langpy[all]` | 30+ min | 4GB | Everything |

---

## 🎯 Installation Examples

### Example 1: Simple AI Agent

```bash
# Install default (has everything you need)
pip install langpy

# Set API key
export OPENAI_API_KEY="sk-..."

# Use it
python -c "from langpy import Langpy; print('Ready!')"
```

### Example 2: Production RAG System

```bash
# Install with PostgreSQL
pip install "langpy[postgres]"

# Set up database
export POSTGRES_DSN="postgresql://user:pass@localhost/langpy"

# Ready to use with pgvector!
```

### Example 3: Multi-Provider Setup

```bash
# Install with multiple providers
pip install "langpy[providers]"

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MISTRAL_API_KEY="..."

# Use any provider!
```

---

## 🛠️ Development Installation

For contributing to LangPy:

```bash
# Clone repository
git clone https://github.com/Mrfabm/langpy.git
cd langpy

# Install in editable mode with dev tools
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .
```

---

## 📦 Requirements Files

For `pip install -r` installations:

### `requirements.txt` (Default - Lightweight)
```bash
pip install -r requirements.txt
```
**Install time: 2-3 minutes**
**Use for: Most projects**

### `requirements-core.txt` (Minimal)
```bash
pip install -r requirements-core.txt
```
**Install time: 1-2 minutes**
**Use for: Absolute minimum, testing**

### `requirements-full.txt` (Everything)
```bash
pip install -r requirements-full.txt
```
**Install time: 30+ minutes**
**Use for: Only if you need ML models**

---

## 🐛 Troubleshooting

### Issue: Installation is slow

**Solution:** Make sure you're using the default installation, not `[all]`

```bash
# Fast (default)
pip install langpy

# Slow (everything)
pip install langpy[all]  # ← Don't do this unless you need it!
```

### Issue: FAISS installation fails

**Solution:** Try installing FAISS separately first

```bash
# Using conda (recommended)
conda install -c conda-forge faiss-cpu

# Or pip
pip install faiss-cpu

# Then install langpy
pip install langpy --no-deps
pip install -r requirements-core.txt
```

### Issue: Import errors

**Solution:** Check Python version and reinstall

```bash
# Check Python version (need 3.9+)
python --version

# Reinstall
pip uninstall langpy
pip install langpy
```

### Issue: No module named 'langpy'

**Solution:** Install in editable mode if developing

```bash
cd /path/to/langpy
pip install -e .
```

---

## 🌟 What Changed?

### Old Installation (Before)
```bash
pip install langpy
# ⏱️ 30+ minutes
# 💾 4GB download
# ❌ Includes PyTorch, Transformers, etc. (not needed!)
```

### New Installation (Now) ✅
```bash
pip install langpy
# ⏱️ 1-2 minutes
# 💾 300MB download
# ✅ Only essential packages
# ✅ Add optional features as needed
```

---

## 📚 Next Steps

After installation:

1. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Try the quick start:**
   ```python
   from langpy import Langpy

   lb = Langpy()
   response = await lb.agent.run(
       model="openai:gpt-4o-mini",
       input="Hello!",
       instructions="Be helpful"
   )
   print(response.output)
   ```

3. **Run examples:**
   ```bash
   python examples/simple_agent.py
   python demo_agency.py
   ```

4. **Read the docs:**
   - [README.md](README.md) - Overview
   - [INSTALL_GUIDE.md](INSTALL_GUIDE.md) - Detailed guide
   - [docs/](docs/) - Full documentation

---

## ✨ Summary

- ✅ **Default install is now FAST** (1-2 minutes)
- ✅ **Lightweight by default** (only essential packages)
- ✅ **Optional extras** available when needed
- ✅ **No more waiting 30 minutes** for PyTorch!

**Just run:** `pip install langpy` 🚀

---

**Questions?** Open an issue on [GitHub](https://github.com/Mrfabm/langpy/issues)
