"""
LangPy Providers - Provider configuration and model management.

Supports multiple LLM providers with a unified configuration interface.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    GROQ = "groq"
    AZURE = "azure"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    PERPLEXITY = "perplexity"
    TOGETHER = "together"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """
    Configuration for a specific model.

    Attributes:
        name: Model name/identifier
        provider: LLM provider
        api_key_env: Environment variable for API key
        base_url: Optional custom base URL
        default_temperature: Default temperature
        default_max_tokens: Default max tokens
        context_window: Context window size
        supports_streaming: Whether model supports streaming
        supports_functions: Whether model supports function calling
        supports_vision: Whether model supports vision/images
        extra: Additional provider-specific config
    """
    name: str
    provider: Provider
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    context_window: int = 128000
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_vision: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.getenv(self.api_key_env)


@dataclass
class ProviderConfig:
    """
    Global provider configuration.

    Attributes:
        default_provider: Default provider to use
        default_model: Default model to use
        models: Registered model configurations
        api_keys: API keys by provider
        base_urls: Custom base URLs by provider
        timeout: Default timeout in seconds
        max_retries: Default max retries
    """
    default_provider: Provider = Provider.OPENAI
    default_model: str = "gpt-4o-mini"
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    base_urls: Dict[str, str] = field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 3

    def register_model(self, alias: str, config: ModelConfig) -> None:
        """Register a model with an alias."""
        self.models[alias] = config

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model config by name or alias."""
        return self.models.get(name)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        self.api_keys[provider] = api_key

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        if provider in self.api_keys:
            return self.api_keys[provider]

        # Try environment variable
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "together": "TOGETHER_API_KEY",
        }

        env_var = env_map.get(provider.lower())
        if env_var:
            return os.getenv(env_var)

        return None


# Default model configurations
DEFAULT_MODELS: Dict[str, ModelConfig] = {
    # OpenAI
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_vision=True
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_vision=True
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_vision=True
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=16385
    ),
    "o1-preview": ModelConfig(
        name="o1-preview",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_streaming=False,
        supports_functions=False
    ),
    "o1-mini": ModelConfig(
        name="o1-mini",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_streaming=False,
        supports_functions=False
    ),

    # Anthropic
    "claude-3-5-sonnet": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider=Provider.ANTHROPIC,
        api_key_env="ANTHROPIC_API_KEY",
        context_window=200000,
        supports_vision=True
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus-20240229",
        provider=Provider.ANTHROPIC,
        api_key_env="ANTHROPIC_API_KEY",
        context_window=200000,
        supports_vision=True
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet-20240229",
        provider=Provider.ANTHROPIC,
        api_key_env="ANTHROPIC_API_KEY",
        context_window=200000,
        supports_vision=True
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku-20240307",
        provider=Provider.ANTHROPIC,
        api_key_env="ANTHROPIC_API_KEY",
        context_window=200000,
        supports_vision=True
    ),

    # Google
    "gemini-1.5-pro": ModelConfig(
        name="gemini-1.5-pro",
        provider=Provider.GOOGLE,
        api_key_env="GOOGLE_API_KEY",
        context_window=1000000,
        supports_vision=True
    ),
    "gemini-1.5-flash": ModelConfig(
        name="gemini-1.5-flash",
        provider=Provider.GOOGLE,
        api_key_env="GOOGLE_API_KEY",
        context_window=1000000,
        supports_vision=True
    ),

    # Mistral
    "mistral-large": ModelConfig(
        name="mistral-large-latest",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        context_window=128000
    ),
    "mistral-small": ModelConfig(
        name="mistral-small-latest",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        context_window=128000
    ),

    # Groq
    "llama-3.1-70b": ModelConfig(
        name="llama-3.1-70b-versatile",
        provider=Provider.GROQ,
        api_key_env="GROQ_API_KEY",
        context_window=131072
    ),
    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b-instant",
        provider=Provider.GROQ,
        api_key_env="GROQ_API_KEY",
        context_window=131072
    ),

    # Aliases
    "fast": ModelConfig(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_vision=True
    ),
    "smart": ModelConfig(
        name="gpt-4o",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=128000,
        supports_vision=True
    ),
    "cheap": ModelConfig(
        name="gpt-3.5-turbo",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        context_window=16385
    ),
    "local": ModelConfig(
        name="llama3",
        provider=Provider.OLLAMA,
        api_key_env="",
        base_url="http://localhost:11434",
        context_window=8192
    ),
}


# Global configuration instance
_global_config: Optional[ProviderConfig] = None


def get_config() -> ProviderConfig:
    """Get the global provider configuration."""
    global _global_config
    if _global_config is None:
        _global_config = ProviderConfig()
        # Register default models
        for alias, config in DEFAULT_MODELS.items():
            _global_config.register_model(alias, config)
    return _global_config


def configure(
    default_provider: Optional[str] = None,
    default_model: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    base_urls: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs
) -> ProviderConfig:
    """
    Configure global provider settings.

    Args:
        default_provider: Default provider name
        default_model: Default model name
        api_keys: Dict of provider -> API key
        base_urls: Dict of provider -> base URL
        timeout: Default timeout
        max_retries: Default max retries
        **kwargs: Additional provider-specific settings

    Returns:
        The updated global config

    Example:
        from langpy.core import configure

        configure(
            default_model="gpt-4o",
            api_keys={"openai": "sk-...", "anthropic": "sk-ant-..."}
        )
    """
    config = get_config()

    if default_provider:
        config.default_provider = Provider(default_provider)

    if default_model:
        config.default_model = default_model

    if api_keys:
        for provider, key in api_keys.items():
            config.set_api_key(provider, key)

    if base_urls:
        config.base_urls.update(base_urls)

    if timeout is not None:
        config.timeout = timeout

    if max_retries is not None:
        config.max_retries = max_retries

    return config


def register_model(
    alias: str,
    name: str,
    provider: str,
    **kwargs
) -> ModelConfig:
    """
    Register a custom model configuration.

    Args:
        alias: Short alias for the model
        name: Full model name
        provider: Provider name
        **kwargs: Additional ModelConfig fields

    Returns:
        The created ModelConfig

    Example:
        register_model(
            "my-fine-tuned",
            "ft:gpt-4o-mini:my-org:custom:abc123",
            "openai",
            context_window=128000
        )
    """
    config = get_config()

    model_config = ModelConfig(
        name=name,
        provider=Provider(provider),
        **kwargs
    )

    config.register_model(alias, model_config)
    return model_config


def resolve_model(model_or_alias: str) -> ModelConfig:
    """
    Resolve a model name or alias to its configuration.

    Args:
        model_or_alias: Model name or alias

    Returns:
        ModelConfig for the model

    Raises:
        ValueError: If model not found
    """
    config = get_config()

    # Check registered models
    if model_or_alias in config.models:
        return config.models[model_or_alias]

    # Check if it's a provider:model format
    if ":" in model_or_alias:
        provider_str, model_name = model_or_alias.split(":", 1)
        provider = Provider(provider_str)

        # Create a config on the fly
        api_key_env_map = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GOOGLE: "GOOGLE_API_KEY",
            Provider.MISTRAL: "MISTRAL_API_KEY",
            Provider.GROQ: "GROQ_API_KEY",
        }

        return ModelConfig(
            name=model_name,
            provider=provider,
            api_key_env=api_key_env_map.get(provider, "API_KEY")
        )

    # Unknown model - assume OpenAI
    return ModelConfig(
        name=model_or_alias,
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY"
    )


def get_client(provider: str, api_key: Optional[str] = None):
    """
    Get an LLM client for a provider.

    Args:
        provider: Provider name
        api_key: Optional API key (uses config/env if not provided)

    Returns:
        Provider client instance
    """
    config = get_config()

    if api_key is None:
        api_key = config.get_api_key(provider)

    provider_enum = Provider(provider.lower())

    if provider_enum == Provider.OPENAI:
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_urls.get("openai")
        )

    elif provider_enum == Provider.ANTHROPIC:
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(
            api_key=api_key,
            base_url=config.base_urls.get("anthropic")
        )

    elif provider_enum == Provider.GOOGLE:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai

    elif provider_enum == Provider.GROQ:
        from groq import AsyncGroq
        return AsyncGroq(
            api_key=api_key,
            base_url=config.base_urls.get("groq")
        )

    elif provider_enum == Provider.OLLAMA:
        from openai import AsyncOpenAI
        base_url = config.base_urls.get("ollama", "http://localhost:11434/v1")
        return AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't need a real key
            base_url=base_url
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def list_models(provider: Optional[str] = None) -> List[str]:
    """
    List available model aliases.

    Args:
        provider: Optional provider to filter by

    Returns:
        List of model aliases
    """
    config = get_config()

    models = []
    for alias, model_config in config.models.items():
        if provider is None or model_config.provider.value == provider:
            models.append(alias)

    return sorted(models)
