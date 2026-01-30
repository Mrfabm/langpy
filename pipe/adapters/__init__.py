from .openai      import run as openai
from .anthropic   import run as anthropic
from .mistral     import run as mistral
from .groq        import run as groq
from .perplexity  import run as perplexity
from .ollama      import run as ollama
from .gemini      import run as gemini

REGISTRY = {
    "openai"     : openai,
    "anthropic"  : anthropic,
    "mistral"    : mistral,
    "groq"       : groq,
    "perplexity" : perplexity,
    "ollama"     : ollama,
    "gemini"     : gemini,
}

def get_adapter(model: str):
    """Get the appropriate adapter based on model name."""
    if ":" in model:
        provider = model.split(":")[0]
    else:
        provider = "openai"  # default to openai if no provider specified
    if provider not in REGISTRY:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[provider]
