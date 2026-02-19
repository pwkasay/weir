"""Error routing — because try/except is not a strategy.

The error system has three layers:

1. **Retry policy**: Transient failures get retried with exponential backoff.
   You configure max attempts, base delay, and which exceptions are retryable.

2. **Error routing**: After retries are exhausted (or for non-retryable errors),
   the error router decides where the failed item goes. Dead letter queue?
   Quarantine? Custom handler? You declare this per exception type.

3. **Poison pill detection**: Items that fail repeatedly get flagged. The framework
   tracks per-item attempt counts so you can identify systemic issues.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Self

logger = logging.getLogger("weir.errors")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Declarative retry configuration for a stage.

    Args:
        max_attempts: Total attempts including the first try. 1 = no retries.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Cap on exponential backoff.
        exponential_base: Multiplier for backoff.
            delay = base_delay * (exponential_base ** attempt).
        retryable_errors: Tuple of exception types that should trigger retries.
                         Everything else is treated as a permanent failure.
    """

    max_attempts: int = 1
    base_delay: float = 0.1
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: tuple[type[Exception], ...] = (Exception,)

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate backoff delay for a given attempt number (0-indexed).

        Applies ±50% uniform jitter to the exponential delay to prevent
        thundering herd when multiple workers retry simultaneously.
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        jitter = delay * random.uniform(-0.5, 0.5)
        return min(delay + jitter, self.max_delay)

    def is_retryable(self, error: Exception) -> bool:
        """Check whether an error should trigger a retry based on its type."""
        return isinstance(error, self.retryable_errors)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Shared retry configuration base class for stage configs.

    Both StageConfig and BatchStageConfig inherit from this to avoid
    duplicating retry-related fields and the retry_policy construction.
    """

    retries: int = 1
    timeout: float | None = None
    retry_base_delay: float = 0.1
    retry_max_delay: float = 30.0
    retryable_errors: tuple[type[Exception], ...] = (Exception,)
    cpu_bound: bool = False
    _retry_policy: RetryPolicy | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_retry_policy",
            RetryPolicy(
                max_attempts=self.retries,
                base_delay=self.retry_base_delay,
                max_delay=self.retry_max_delay,
                retryable_errors=self.retryable_errors,
            ),
        )

    @property
    def retry_policy(self) -> RetryPolicy:
        """Return the cached RetryPolicy for this config's retry settings."""
        assert self._retry_policy is not None
        return self._retry_policy


@dataclass(slots=True)
class FailedItem:
    """An item that has permanently failed processing.

    This is what gets sent to error handlers — not just the exception,
    but the full context: what item, which stage, how many attempts,
    and the complete error chain.
    """

    item: Any
    stage_name: str
    error: Exception
    attempts: int
    error_chain: list[Exception] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a summary showing stage, error type, and attempt count."""
        return (
            f"<FailedItem stage='{self.stage_name}' "
            f"error={type(self.error).__name__} "
            f"attempts={self.attempts}>"
        )


# Type alias for error handlers
type ErrorHandler = Callable[[FailedItem], Awaitable[None]]


async def log_and_discard(failed: FailedItem) -> None:
    """Default error handler: log the failure and move on."""
    logger.error(
        "Item permanently failed in stage '%s' after %d attempts: %s",
        failed.stage_name,
        failed.attempts,
        failed.error,
    )


class DeadLetterCollector:
    """Collects failed items for inspection.

    Use this when you want to accumulate failures and inspect them
    after the pipeline completes (or periodically during a long run).
    """

    def __init__(self, max_size: int = 10_000) -> None:
        """Initialize the collector.

        Args:
            max_size: Maximum number of failed items to store. Overflow is counted
                but discarded.
        """
        self._items: list[FailedItem] = []
        self._max_size = max_size
        self._overflow_count = 0

    async def __call__(self, failed: FailedItem) -> None:
        """Record a failed item. Discards if at capacity but tracks the overflow count."""
        if len(self._items) < self._max_size:
            self._items.append(failed)
        else:
            self._overflow_count += 1
            logger.warning(
                "Dead letter collector full (%d items). Discarding failure from '%s'.",
                self._max_size,
                failed.stage_name,
            )

    @property
    def items(self) -> list[FailedItem]:
        """A copy of the collected failed items."""
        return list(self._items)

    @property
    def count(self) -> int:
        """Total failures seen, including overflowed items that were discarded."""
        return len(self._items) + self._overflow_count

    def clear(self) -> None:
        """Remove all collected items and reset the overflow counter."""
        self._items.clear()
        self._overflow_count = 0


@dataclass(slots=True)
class ErrorRouter:
    """Routes errors to handlers based on exception type.

    The routing logic:
    1. Check if the exception type has a specific handler registered.
    2. Walk the MRO — a handler for ValueError catches ValueError subclasses.
    3. Fall back to the default handler (log_and_discard).

    Usage:
        router = ErrorRouter()
        router.on(ValidationError, dead_letter_collector)
        router.on(RateLimitError, custom_handler)
    """

    _routes: dict[type[Exception], ErrorHandler] = field(default_factory=dict)
    _default: ErrorHandler = field(default=log_and_discard)

    def on(self, error_type: type[Exception], handler: ErrorHandler) -> Self:
        """Register a handler for a specific exception type."""
        self._routes[error_type] = handler
        return self

    def set_default(self, handler: ErrorHandler) -> Self:
        """Set the default handler for unrouted errors."""
        self._default = handler
        return self

    async def handle(self, failed: FailedItem) -> None:
        """Route a failed item to the appropriate handler."""
        error_type = type(failed.error)

        # Exact match first
        if error_type in self._routes:
            await self._routes[error_type](failed)
            return

        # Walk MRO for base class matches
        for cls in error_type.__mro__:
            if cls in self._routes:
                await self._routes[cls](failed)
                return

        # Default
        await self._default(failed)


async def execute_with_retry(
    func: Callable[..., Awaitable[Any]],
    item: Any,
    stage_name: str,
    policy: RetryPolicy,
    on_retry: Callable[[], None] | None = None,
    timeout: float | None = None,
) -> Any:
    """Execute a function with retry logic.

    Returns the result on success.
    Raises StageProcessingError on permanent failure (retries exhausted or non-retryable).

    The on_retry callback is called before each retry (for metrics).
    Timeout is applied per-attempt, not across the entire retry loop.
    TimeoutError is always treated as retryable.
    """
    error_chain: list[Exception] | None = None

    for attempt in range(policy.max_attempts):
        try:
            if timeout is not None:
                async with asyncio.timeout(timeout):
                    return await func(item)
            else:
                return await func(item)
        except Exception as e:
            if error_chain is None:
                error_chain = []
            error_chain.append(e)

            is_last_attempt = attempt == policy.max_attempts - 1
            is_retryable = isinstance(e, TimeoutError) or policy.is_retryable(e)

            if is_last_attempt or not is_retryable:
                raise StageProcessingError(
                    item=item,
                    stage_name=stage_name,
                    error=e,
                    attempts=attempt + 1,
                    error_chain=error_chain,
                ) from e

            delay = policy.delay_for_attempt(attempt)
            logger.debug(
                "Stage '%s' attempt %d/%d failed (%s), retrying in %.2fs",
                stage_name,
                attempt + 1,
                policy.max_attempts,
                type(e).__name__,
                delay,
            )
            if on_retry:
                on_retry()
            await asyncio.sleep(delay)

    # Should be unreachable, but type checker needs it
    raise RuntimeError("Unreachable: retry loop exited without return or raise")


class StageProcessingError(Exception):
    """Raised when a stage permanently fails to process an item."""

    def __init__(
        self,
        item: Any,
        stage_name: str,
        error: Exception,
        attempts: int,
        error_chain: list[Exception],
    ) -> None:
        """Initialize the processing error.

        Args:
            item: The item that failed processing.
            stage_name: Name of the stage where failure occurred.
            error: The final exception that caused the failure.
            attempts: Total number of attempts made.
            error_chain: All exceptions encountered across retry attempts.
        """
        self.item = item
        self.stage_name = stage_name
        self.error = error
        self.attempts = attempts
        self.error_chain = error_chain
        super().__init__(f"Stage '{stage_name}' failed after {attempts} attempts: {error}")

    def to_failed_item(self) -> FailedItem:
        """Convert this exception into a FailedItem for error routing."""
        return FailedItem(
            item=self.item,
            stage_name=self.stage_name,
            error=self.error,
            attempts=self.attempts,
            error_chain=self.error_chain,
        )
