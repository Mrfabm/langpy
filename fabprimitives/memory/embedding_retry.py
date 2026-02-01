"""
Embedding Retry Helper - Automatic retry logic for embedding generation.

Provides exponential backoff and retry mechanisms for embedding API calls
that may fail due to rate limits, network issues, or temporary errors.
"""

import asyncio
import time
import random
from typing import List, Optional, Callable, Any, Dict
from functools import wraps


class EmbeddingRetryHelper:
    """Helper for retrying embedding generation with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize the retry helper.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if attempt == 0:
            return 0
        
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        
        # Cap at maximum delay
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    async def retry_embedding(
        self,
        embed_func: Callable,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Retry embedding generation with exponential backoff.
        
        Args:
            embed_func: Function to call for embedding generation
            texts: List of texts to embed
            **kwargs: Additional arguments for embed_func
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Call the embedding function
                embeddings = await embed_func(texts, **kwargs)
                
                # Validate embeddings
                if not embeddings or len(embeddings) != len(texts):
                    raise ValueError(f"Invalid embeddings returned: expected {len(texts)}, got {len(embeddings) if embeddings else 0}")
                
                # Check for empty or invalid embeddings
                for i, embedding in enumerate(embeddings):
                    if not embedding or not isinstance(embedding, list):
                        raise ValueError(f"Invalid embedding at index {i}: {embedding}")
                    if not all(isinstance(x, (int, float)) for x in embedding):
                        raise ValueError(f"Non-numeric values in embedding at index {i}")
                
                return embeddings
                
            except Exception as e:
                last_exception = e
                
                # Don't retry on certain types of errors
                if self._should_not_retry(e):
                    raise e
                
                # If this is the last attempt, raise the exception
                if attempt == self.max_retries:
                    break
                
                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise last_exception or Exception("Embedding generation failed after all retries")
    
    def _should_not_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should not trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if the exception should not trigger a retry
        """
        # Don't retry on authentication errors
        if "authentication" in str(exception).lower():
            return True
        
        # Don't retry on invalid input errors
        if "invalid" in str(exception).lower():
            return True
        
        # Don't retry on quota exceeded (permanent)
        if "quota" in str(exception).lower() and "exceeded" in str(exception).lower():
            return True
        
        return False
    
    def retry_decorator(self, max_retries: Optional[int] = None):
        """
        Decorator for retrying embedding functions.
        
        Args:
            max_retries: Override max_retries for this function
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                retry_helper = EmbeddingRetryHelper(
                    max_retries=max_retries or self.max_retries,
                    base_delay=self.base_delay,
                    max_delay=self.max_delay,
                    exponential_base=self.exponential_base,
                    jitter=self.jitter
                )
                
                # Extract texts from args or kwargs
                texts = None
                if args and isinstance(args[0], list):
                    texts = args[0]
                elif 'texts' in kwargs:
                    texts = kwargs['texts']
                
                if not texts:
                    raise ValueError("No texts found in function arguments")
                
                return await retry_helper.retry_embedding(func, texts, **kwargs)
            
            return wrapper
        return decorator


# Global retry helper instance
_default_retry_helper = None


def get_retry_helper() -> EmbeddingRetryHelper:
    """Get the default retry helper instance."""
    global _default_retry_helper
    if _default_retry_helper is None:
        _default_retry_helper = EmbeddingRetryHelper()
    return _default_retry_helper


async def retry_embedding(
    embed_func: Callable,
    texts: List[str],
    max_retries: int = 3,
    **kwargs
) -> List[List[float]]:
    """
    Retry embedding generation with default settings.
    
    Args:
        embed_func: Function to call for embedding generation
        texts: List of texts to embed
        max_retries: Maximum number of retry attempts
        **kwargs: Additional arguments for embed_func
        
    Returns:
        List of embedding vectors
    """
    retry_helper = EmbeddingRetryHelper(max_retries=max_retries)
    return await retry_helper.retry_embedding(embed_func, texts, **kwargs)


def with_retry(max_retries: int = 3):
    """
    Decorator for adding retry logic to embedding functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        Decorated function
    """
    retry_helper = EmbeddingRetryHelper(max_retries=max_retries)
    return retry_helper.retry_decorator(max_retries) 