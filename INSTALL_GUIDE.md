# LangPy Installation Guide

## 🚀 **Fastest Install (Recommended for Most Users)**

Install directly from GitHub with one command:

```bash
pip install git+https://github.com/Mrfabm/langpy.git
```

**That's it!** This installs the lightweight version (~300MB, 1-2 minutes).

---

## 📚 **Alternative Installation Methods**

For development or specific requirements:

---

## ✅ **Quick Install (Recommended)**

### **Option 1: Core Installation (~1-2 minutes)**

Install only what's essential:

```bash
cd langpyv2
pip install -r requirements-core.txt
```

**This includes:**
- ✅ Agent primitive (OpenAI)
- ✅ Pipe primitive
- ✅ Memory primitive (FAISS vector storage)
- ✅ Thread primitive
- ✅ Workflow primitive
- ✅ Basic document parsing

**Install time: ~1-2 minutes**

---

### **Option 2: Minimal Installation (~2-3 minutes)**

Install with more features but still lightweight:

```bash
cd langpyv2
pip install -r requirements-minimal.txt
```

**Adds:**
- ✅ Anthropic Claude support
- ✅ More document formats (PDF, Word, HTML)
- ✅ JSON validation
- ✅ YAML support

**Install time: ~2-3 minutes**

---

### **Option 3: Full Installation (~30+ minutes)**

⚠️ **Only use if you need ML models, web dashboards, etc.**

```bash
cd langpyv2
pip install -r requirements.txt
```

**Install time: ~30+ minutes (includes PyTorch, Transformers, etc.)**

---

## 🎯 **What Do You Actually Need?**

| Use Case | Install | Time |
|----------|---------|------|
| **Basic AI Agents** | `requirements-core.txt` | 1-2 min |
| **+ Document Parsing** | `requirements-minimal.txt` | 2-3 min |
| **+ ML Models** | `requirements.txt` | 30+ min |

---

## 📦 **Package Comparison**

### **requirements-core.txt** (Recommended)
```
Total packages: ~15
Total size: ~150MB
Install time: 1-2 minutes
```

**Includes:**
- pydantic, python-dotenv
- openai
- httpx, aiohttp
- faiss-cpu, numpy
- aiofiles

### **requirements-minimal.txt**
```
Total packages: ~25
Total size: ~300MB
Install time: 2-3 minutes
```

**Adds:**
- anthropic
- PyPDF2, python-docx
- beautifulsoup4
- jsonschema, PyYAML

### **requirements.txt** (Full)
```
Total packages: 100+
Total size: ~3-4GB
Install time: 30+ minutes
```

**Adds:**
- torch (PyTorch) - 1.5GB
- transformers - 500MB
- sentence-transformers - 300MB
- opencv-python - 200MB
- spacy - 100MB
- langchain - 200MB
- streamlit - 100MB
- chromadb - 100MB

---

## 🚀 **Quick Start After Installation**

1. **Set up environment:**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key:
# OPENAI_API_KEY=sk-your-key-here
```

2. **Test installation:**
```python
from langpy import Langpy

lb = Langpy(api_key="sk-...")

# Test agent
response = await lb.agent.run(
    model="openai:gpt-4o-mini",
    input="Hello!",
    instructions="Be helpful"
)
print(response.output)
```

3. **Run the demo:**
```bash
python demo_agency.py
```

---

## 🔧 **Installing from GitHub (Alternative)**

```bash
# Clone repository
git clone https://github.com/YourRepo/langpy.git
cd langpy

# Install in editable mode with minimal deps
pip install -e . --no-deps
pip install -r requirements-core.txt
```

---

## 📝 **Adding More Features Later**

You can always add more packages later if needed:

```bash
# Add Anthropic Claude support
pip install anthropic>=0.7.0

# Add PDF parsing
pip install PyPDF2>=3.0.0

# Add PostgreSQL + pgvector
pip install pgvector>=0.2.0 psycopg2-binary>=2.9.0 sqlalchemy>=2.0.0

# Add document parsing
pip install python-docx>=0.8.11 beautifulsoup4>=4.12.0
```

---

## ❌ **What You DON'T Need (for most use cases)**

These are ONLY needed for specific advanced features:

- **torch** - Only if training ML models locally (NOT for using GPT/Claude APIs)
- **transformers** - Only if loading HuggingFace models locally
- **sentence-transformers** - Only if using local embedding models
- **opencv-python** - Only for computer vision tasks
- **spacy** - Only for advanced NLP preprocessing
- **langchain** - Duplicate functionality with LangPy
- **streamlit** - Only for web dashboards
- **chromadb** - Alternative to FAISS (don't need both)

---

## 🐛 **Troubleshooting**

### **Problem: Installation still taking forever**

**Solution 1:** Clear pip cache and use core requirements
```bash
pip cache purge
pip install -r requirements-core.txt
```

**Solution 2:** Install without cache
```bash
pip install --no-cache-dir -r requirements-core.txt
```

### **Problem: FAISS installation fails**

**Solution:** Try conda instead:
```bash
conda install -c conda-forge faiss-cpu
pip install -r requirements-core.txt --no-deps
pip install pydantic python-dotenv openai httpx aiohttp aiofiles numpy
```

### **Problem: Import errors**

**Solution:** Make sure you're in the langpyv2 directory:
```bash
cd /c/Users/USER/Desktop/langpyv2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python demo_agency.py
```

---

## ✨ **Summary**

- **Use `requirements-core.txt`** for fast installation (1-2 min)
- **Use `requirements-minimal.txt`** if you need more features (2-3 min)
- **Use `requirements.txt`** only if you specifically need ML models (30+ min)

**Most users should use `requirements-core.txt`!**

---

## 📚 **Next Steps**

After installation:

1. Read the [README.md](README.md) for feature overview
2. Check [examples/](examples/) for code samples
3. Run `python demo_agency.py` to see the full system in action
4. Read [docs/](docs/) for detailed documentation

---

**Built with LangPy** 🚀
