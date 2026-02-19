"""Batch stages — process items in groups for throughput.

Some operations are more efficient in batches: bulk database inserts,
batch API calls, chunked file writes. The @batch_stage decorator lets
you write a function that receives a list of items and processes them
together.

The BatchStageRunner accumulates items until either batch_size is reached
or flush_timeout expires, whichever comes first. On STOP sentinel, any
remaining items are flushed as a partial batch.
"""

import asyncio
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .channel import is_stop_signal
from .errors import (
    RetryConfig,
    StageProcessingError,
    execute_with_retry,
)
from .runner import BaseStageRunner


@dataclass(frozen=True, slots=True)
class BatchStageConfig(RetryConfig):
    """Configuration for a batch stage."""

    batch_size: int = 10
    flush_timeout: float = 5.0
    concurrency: int = 1


class BatchStageFunction:
    """A decorated batch stage function with its configuration.

    Similar to StageFunction but for batch processing. The wrapped function
    receives a list of items rather than a single item.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: BatchStageConfig,
    ) -> None:
        """Initialize a batch stage function wrapper.

        Args:
            func: The function to wrap (async or sync for cpu_bound stages).
            config: Batch stage configuration (batch_size, flush_timeout, etc.).
        """
        self.func = func
        self.config = config
        self.name = func.__name__
        functools.update_wrapper(self, func)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly, bypassing framework machinery."""
        if self.config.cpu_bound:
            return self.func(*args, **kwargs)
        return await self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Return a human-readable representation of the batch stage."""
        return (
            f"<BatchStage '{self.name}' batch_size={self.config.batch_size} "
            f"concurrency={self.config.concurrency}>"
        )


def batch_stage(
    batch_size: int = 10,
    flush_timeout: float = 5.0,
    concurrency: int = 1,
    retries: int = 1,
    timeout: float | None = None,
    retry_base_delay: float = 0.1,
    retry_max_delay: float = 30.0,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
    cpu_bound: bool = False,
) -> Callable[[Callable[..., Any]], BatchStageFunction]:
    """Decorator to declare a function as a batch pipeline stage.

    The decorated function receives a list of items and should process them
    as a batch. If the function returns a list, each element is pushed
    downstream individually.

    Args:
        batch_size: Number of items to accumulate before flushing.
        flush_timeout: Max seconds to wait before flushing a partial batch.
        concurrency: Number of parallel workers for this stage.
        retries: Total attempts (1 = no retries, 3 = up to 2 retries).
        timeout: Per-batch processing timeout in seconds. None = no timeout.
        retry_base_delay: Base delay for exponential backoff.
        retry_max_delay: Maximum backoff delay.
        retryable_errors: Exception types that trigger retries.
        cpu_bound: If True, offload to a ThreadPoolExecutor. Use with sync functions.

    Usage:
        @batch_stage(batch_size=50, flush_timeout=2.0)
        async def bulk_insert(items: list[Record]) -> None:
            await db.insert_many(items)
    """

    def decorator(func: Callable[..., Any]) -> BatchStageFunction:
        config = BatchStageConfig(
            batch_size=batch_size,
            flush_timeout=flush_timeout,
            concurrency=concurrency,
            retries=retries,
            timeout=timeout,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retryable_errors=retryable_errors,
            cpu_bound=cpu_bound,
        )
        return BatchStageFunction(func, config)

    return decorator


class BatchStageRunner(BaseStageRunner):
    """Runs a batch stage function with accumulation and flush logic.

    Accumulates items from the input channel into a buffer. Flushes when:
    - batch_size items have been collected
    - flush_timeout seconds have elapsed since the first item in the buffer
    - STOP sentinel is received (partial flush)
    """

    @property
    def _task_name_prefix(self) -> str:
        return f"{self.name}-batch-worker"

    def _log_start(self) -> None:
        self.logger.info(
            "Starting batch stage '%s' with %d workers (batch_size=%d)",
            self.name,
            self.config.concurrency,
            self.config.batch_size,
        )

    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a batch worker."""
        self.logger.debug("Batch worker %d started", worker_id)
        buffer: list[Any] = []
        flush_deadline: float | None = None

        while True:
            # Calculate remaining time until flush
            if flush_deadline is not None:
                remaining = max(0.0, flush_deadline - time.monotonic())
            else:
                remaining = None

            # Try to get an item, with timeout if we have a partial batch
            try:
                if remaining is not None:
                    async with asyncio.timeout(remaining):
                        item = await self.input_channel.get()
                else:
                    item = await self.input_channel.get()
            except TimeoutError:
                # Flush timeout expired — flush partial batch
                if buffer:
                    await self._flush_batch(buffer)
                    buffer = []
                    flush_deadline = None
                continue

            # Check for shutdown sentinel
            if is_stop_signal(item):
                self.logger.debug("Batch worker %d received STOP", worker_id)
                # Flush remaining items
                if buffer:
                    await self._flush_batch(buffer)
                break

            buffer.append(item)
            if flush_deadline is None:
                flush_deadline = time.monotonic() + self.config.flush_timeout

            # Check if batch is full
            if len(buffer) >= self.config.batch_size:
                await self._flush_batch(buffer)
                buffer = []
                flush_deadline = None

        self.logger.debug("Batch worker %d exiting", worker_id)

    async def _flush_batch(self, batch: list[Any]) -> None:
        """Process a batch of items with retry and error handling."""
        t0 = time.monotonic()
        try:
            result = await execute_with_retry(
                func=self._func,
                item=batch,
                stage_name=self.name,
                policy=self.config.retry_policy,
                on_retry=self._record_retry,
                timeout=self.config.timeout,
            )
            duration = time.monotonic() - t0

            # Record success for each item in the batch
            per_item = duration / len(batch)
            for _ in batch:
                self.metrics.record_success(per_item)

            for hook in self._on_item_hooks:
                await hook.on_item(self.name, batch, result, duration)

            # Push results downstream
            if self.output_channel is not None and result is not None:
                if isinstance(result, list):
                    for item in result:
                        await self.output_channel.put(item)
                else:
                    await self.output_channel.put(result)

        except asyncio.CancelledError:
            self.logger.warning("Batch flush cancelled")
            raise
        except StageProcessingError as e:
            duration = time.monotonic() - t0
            per_item = duration / len(batch)
            for _ in batch:
                self.metrics.record_error(per_item)
            for hook in self._on_error_hooks:
                await hook.on_error(self.name, batch, e.error)
            try:
                await self.error_router.handle(e.to_failed_item())
            except Exception as handler_err:
                self.logger.error(
                    "Error handler failed for batch in stage '%s': %s",
                    self.name,
                    handler_err,
                )
