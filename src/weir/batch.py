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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .channel import Channel, _Sentinel
from .errors import (
    ErrorRouter,
    RetryPolicy,
    StageProcessingError,
    execute_with_retry,
)
from .logging import get_logger
from .metrics import StageMetrics


@dataclass(frozen=True, slots=True)
class BatchStageConfig:
    """Configuration for a batch stage."""

    batch_size: int = 10
    flush_timeout: float = 5.0
    concurrency: int = 1
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
        """Return the cached RetryPolicy for this batch stage's retry configuration."""
        assert self._retry_policy is not None
        return self._retry_policy


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


class BatchStageRunner:
    """Runs a batch stage function with accumulation and flush logic.

    Accumulates items from the input channel into a buffer. Flushes when:
    - batch_size items have been collected
    - flush_timeout seconds have elapsed since the first item in the buffer
    - STOP sentinel is received (partial flush)
    """

    def __init__(
        self,
        stage_func: BatchStageFunction,
        input_channel: Channel[Any],
        output_channel: Channel[Any] | None,
        error_router: ErrorRouter,
        pipeline_name: str,
        hooks: list[Any] | None = None,
    ) -> None:
        """Initialize a batch stage runner.

        Args:
            stage_func: The decorated batch stage function to execute.
            input_channel: Channel to read items from.
            output_channel: Channel to write results to, or None for terminal stages.
            error_router: Router for handling permanently failed items.
            pipeline_name: Name of the owning pipeline (for logging context).
            hooks: Optional lifecycle hooks to invoke during processing.
        """
        self.stage_func = stage_func
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.error_router = error_router
        self.metrics = StageMetrics(stage_name=stage_func.name, _input_channel=input_channel)
        self.logger = get_logger(pipeline_name, stage_func.name)
        self._workers: list[asyncio.Task[None]] = []
        self._executor: ThreadPoolExecutor | None = None
        self._func: Callable[..., Any] = stage_func
        self._record_retry = self.metrics.record_retry
        # Pre-filter hooks by implemented methods
        all_hooks = hooks or []
        self._on_start_hooks = [h for h in all_hooks if hasattr(h, "on_start")]
        self._on_item_hooks = [h for h in all_hooks if hasattr(h, "on_item")]
        self._on_error_hooks = [h for h in all_hooks if hasattr(h, "on_error")]
        self._on_complete_hooks = [h for h in all_hooks if hasattr(h, "on_complete")]
        if stage_func.config.cpu_bound:
            self._executor = ThreadPoolExecutor(
                max_workers=stage_func.config.concurrency,
                thread_name_prefix=f"{stage_func.name}-cpu",
            )

    @property
    def name(self) -> str:
        """The name of the underlying batch stage function."""
        return self.stage_func.name

    @property
    def config(self) -> BatchStageConfig:
        """The batch stage's configuration (batch_size, flush_timeout, concurrency, etc.)."""
        return self.stage_func.config

    async def start(self) -> None:
        """Launch worker tasks."""
        if self._executor is not None:
            loop = asyncio.get_running_loop()
            executor = self._executor
            raw_func = self.stage_func.func

            async def _offload(items: Any) -> Any:
                return await loop.run_in_executor(executor, raw_func, items)

            self._func = _offload

        self.logger.info(
            "Starting batch stage '%s' with %d workers (batch_size=%d)",
            self.name,
            self.config.concurrency,
            self.config.batch_size,
        )
        for hook in self._on_start_hooks:
            await hook.on_start(self.name)
        for i in range(self.config.concurrency):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"{self.name}-batch-worker-{i}",
            )
            self._workers.append(task)

    async def wait(self) -> None:
        """Wait for all workers to complete."""
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        for hook in self._on_complete_hooks:
            await hook.on_complete(self.name)
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    async def cancel(self) -> None:
        """Cancel all worker tasks and wait for them to finish."""
        for task in self._workers:
            if not task.done():
                task.cancel()
        await self.wait()

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
            if isinstance(item, _Sentinel):
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
