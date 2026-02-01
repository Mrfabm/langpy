"""
Retry engine with backoff strategies for LangPy workflows.

This module provides comprehensive retry logic with different backoff strategies
matching Langbase's retry configuration.
"""

import asyncio
import random
import time
from typing import Optional, Dict, Any, Literal, Callable, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BackoffStrategy(Enum):
    """Backoff strategies for retry logic."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    limit: int = 3
    delay: int = 1000  # milliseconds
    backoff: Union[BackoffStrategy, str] = BackoffStrategy.EXPONENTIAL
    max_delay: Optional[int] = None  # milliseconds
    jitter: bool = True
    
    def __post_init__(self):
        """Validate and normalize retry configuration."""
        if isinstance(self.backoff, str):
            try:
                self.backoff = BackoffStrategy(self.backoff)
            except ValueError:
                raise ValueError(f"Invalid backoff strategy: {self.backoff}. Must be one of: {list(BackoffStrategy)}")
        
        if self.limit < 0:
            raise ValueError("Retry limit must be non-negative")
        
        if self.delay < 0:
            raise ValueError("Retry delay must be non-negative")
        
        if self.max_delay is not None and self.max_delay < self.delay:
            raise ValueError("Max delay must be greater than or equal to base delay")


class RetryEngine:
    """Engine for executing retry logic with different backoff strategies."""
    
    @staticmethod
    def calculate_delay(
        attempt: int,
        config: RetryConfig
    ) -> int:
        """
        Calculate delay for a retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration
            
        Returns:
            Delay in milliseconds
        """
        base_delay = config.delay
        
        if config.backoff == BackoffStrategy.FIXED:
            delay = base_delay
        elif config.backoff == BackoffStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        elif config.backoff == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** attempt)
        else:
            raise ValueError(f"Unknown backoff strategy: {config.backoff}")
        
        # Apply max delay limit
        if config.max_delay is not None:
            delay = min(delay, config.max_delay)
        
        # Apply jitter to prevent thundering herd
        if config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, int(delay))  # Ensure non-negative
        
        return delay
    
    @staticmethod
    async def execute_with_retry(
        func: Callable[[], T],
        config: RetryConfig,
        step_id: str,
        is_async: bool = True
    ) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            config: Retry configuration
            step_id: ID of the step being retried (for logging)
            is_async: Whether the function is async
            
        Returns:
            Result of the function
            
        Raises:
            RetryExhaustedError: If all retry attempts are exhausted
        """
        from .exceptions import RetryExhaustedError, StepError
        
        last_error = None
        
        for attempt in range(config.limit + 1):
            try:
                logger.debug(f"[{step_id}] Attempt {attempt + 1}/{config.limit + 1}")
                
                if is_async:
                    result = await func()
                else:
                    result = func()
                
                if attempt > 0:
                    logger.info(f"[{step_id}] ✓ Succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_error = e
                
                if attempt < config.limit:
                    delay_ms = RetryEngine.calculate_delay(attempt, config)
                    delay_seconds = delay_ms / 1000.0
                    
                    logger.warning(
                        f"[{step_id}] ✗ Attempt {attempt + 1}/{config.limit + 1} failed: {e}"
                    )
                    logger.info(
                        f"[{step_id}] Retrying in {delay_ms}ms "
                        f"(strategy: {config.backoff.value})"
                    )
                    
                    await asyncio.sleep(delay_seconds)
                else:
                    logger.error(f"[{step_id}] ✗ All retry attempts exhausted")
        
        # All attempts failed
        raise RetryExhaustedError(step_id, config.limit + 1, last_error)
    
    @staticmethod
    def format_retry_info(config: RetryConfig) -> str:
        """
        Format retry configuration for logging.
        
        Args:
            config: Retry configuration
            
        Returns:
            Formatted string describing retry configuration
        """
        info = f"limit={config.limit}, delay={config.delay}ms, backoff={config.backoff.value}"
        
        if config.max_delay:
            info += f", max_delay={config.max_delay}ms"
        
        if config.jitter:
            info += ", jitter=enabled"
        
        return info


def create_retry_config(
    limit: int = 3,
    delay: int = 1000,
    backoff: str = "exponential",
    max_delay: Optional[int] = None,
    jitter: bool = True
) -> RetryConfig:
    """
    Create a retry configuration.
    
    Args:
        limit: Maximum number of retry attempts
        delay: Base delay in milliseconds
        backoff: Backoff strategy ('fixed', 'linear', 'exponential')
        max_delay: Maximum delay in milliseconds
        jitter: Whether to apply jitter to delays
        
    Returns:
        RetryConfig instance
    """
    return RetryConfig(
        limit=limit,
        delay=delay,
        backoff=backoff,
        max_delay=max_delay,
        jitter=jitter
    )


def parse_retry_config(config: Union[Dict[str, Any], RetryConfig, None]) -> Optional[RetryConfig]:
    """
    Parse retry configuration from various formats.
    
    Args:
        config: Retry configuration in various formats
        
    Returns:
        RetryConfig instance or None if no retry configuration
    """
    if config is None:
        return None
    
    if isinstance(config, RetryConfig):
        return config
    
    if isinstance(config, dict):
        return create_retry_config(
            limit=config.get("limit", 3),
            delay=config.get("delay", 1000),
            backoff=config.get("backoff", "exponential"),
            max_delay=config.get("max_delay"),
            jitter=config.get("jitter", True)
        )
    
    raise ValueError(f"Invalid retry configuration type: {type(config)}")


# Example usage and testing
async def test_retry_strategies():
    """Test different retry strategies."""
    import asyncio
    
    # Test function that fails a few times
    class FailingFunction:
        def __init__(self, fail_count: int):
            self.fail_count = fail_count
            self.attempts = 0
        
        async def __call__(self):
            self.attempts += 1
            if self.attempts <= self.fail_count:
                raise Exception(f"Attempt {self.attempts} failed")
            return f"Success on attempt {self.attempts}"
    
    # Test different backoff strategies
    strategies = [
        ("fixed", BackoffStrategy.FIXED),
        ("linear", BackoffStrategy.LINEAR),
        ("exponential", BackoffStrategy.EXPONENTIAL)
    ]
    
    for name, strategy in strategies:
        print(f"\n--- Testing {name} backoff ---")
        
        config = RetryConfig(limit=3, delay=100, backoff=strategy, jitter=False)
        failing_func = FailingFunction(fail_count=2)
        
        try:
            result = await RetryEngine.execute_with_retry(
                failing_func,
                config,
                f"test_{name}",
                is_async=True
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_retry_strategies()) 