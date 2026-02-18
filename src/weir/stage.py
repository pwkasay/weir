"""Stage — the atomic unit of work in a pipeline.

A stage is an async function that processes one item at a time, decorated with
metadata about how it should be run: concurrency, retry policy, timeout.

The @stage decorator doesn't change the function's behavior when called directly.
It just attaches configuration. The Pipeline is what reads that configuration
and orchestrates execution.

Design decision: stages are functions, not classes. Why?
- Functions compose better than classes.
- The configuration is *about* the function, not *part of* the function.
- You can test stage functions by calling them directly — no framework needed.
- The decorator pattern makes the declaration site readable.
"""

import asyncio
import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from .channel import Channel, _Sentinel
from .errors import (
    ErrorRouter,
    RetryPolicy,
    StageProcessingError,
    execute_with_retry,
)
from .logging import get_logger
from .metrics import StageMetrics


@dataclass(frozen=True)
class StageConfig:
    """Configuration attached to a stage function by the @stage decorator."""

    concurrency: int = 1
    retries: int = 1
    timeout: float | None = None
    retry_base_delay: float = 0.1
    retry_max_delay: float = 30.0
    retryable_errors: tuple[type[Exception], ...] = (Exception,)

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            retryable_errors=self.retryable_errors,
        )


class StageFunction:
    """A decorated stage function with its configuration.

    This wraps the original async function and attaches config.
    It's callable — calling it directly just calls the underlying function,
    no framework machinery involved. This is important for testing.
    """

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        config: StageConfig,
    ) -> None:
        self.func = func
        self.config = config
        self.name = func.__name__
        functools.update_wrapper(self, func)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Stage '{self.name}' concurrency={self.config.concurrency}>"


def stage(
    concurrency: int = 1,
    retries: int = 1,
    timeout: float | None = None,
    retry_base_delay: float = 0.1,
    retry_max_delay: float = 30.0,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[Any]]], StageFunction]:
    """Decorator to declare an async function as a pipeline stage.

    Args:
        concurrency: Number of parallel workers for this stage.
        retries: Total attempts (1 = no retries, 3 = up to 2 retries).
        timeout: Per-item processing timeout in seconds. None = no timeout.
        retry_base_delay: Base delay for exponential backoff.
        retry_max_delay: Maximum backoff delay.
        retryable_errors: Exception types that trigger retries.

    Usage:
        @stage(concurrency=5, retries=3, timeout=30)
        async def fetch(url: str) -> dict:
            ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> StageFunction:
        config = StageConfig(
            concurrency=concurrency,
            retries=retries,
            timeout=timeout,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retryable_errors=retryable_errors,
        )
        return StageFunction(func, config)

    return decorator


class StageRunner:
    """Runs a stage function with its full machinery.

    This is the internal runtime representation of a stage within a pipeline.
    It manages the worker tasks, reads from the input channel, writes to the
    output channel, handles retries and errors, and collects metrics.

    Users don't create StageRunners directly — the Pipeline builds them.
    """

    def __init__(
        self,
        stage_func: StageFunction,
        input_channel: Channel[Any],
        output_channel: Channel[Any] | None,
        error_router: ErrorRouter,
        pipeline_name: str,
    ) -> None:
        self.stage_func = stage_func
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.error_router = error_router
        self.metrics = StageMetrics(stage_name=stage_func.name, _input_channel=input_channel)
        self.logger = get_logger(pipeline_name, stage_func.name)
        self._workers: list[asyncio.Task[None]] = []

    @property
    def name(self) -> str:
        return self.stage_func.name

    @property
    def config(self) -> StageConfig:
        return self.stage_func.config

    async def start(self) -> None:
        """Launch worker tasks."""
        self.logger.info(
            "Starting stage '%s' with %d workers",
            self.name,
            self.config.concurrency,
        )
        for i in range(self.config.concurrency):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"{self.name}-worker-{i}",
            )
            self._workers.append(task)

    async def wait(self) -> None:
        """Wait for all workers to complete (called during shutdown drain)."""
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single worker.

        Pull items from input channel, process them, push results to output.
        Exit cleanly when STOP sentinel is received.
        """
        self.logger.debug("Worker %d started", worker_id)

        while True:
            # Pull from input
            item = await self.input_channel.get()

            # Check for shutdown sentinel
            if isinstance(item, _Sentinel):
                self.logger.debug("Worker %d received STOP", worker_id)
                break

            # Process the item
            t0 = time.monotonic()
            try:
                result = await self._process_item(item)
                duration = time.monotonic() - t0
                await self.metrics.record_success(duration)

                # Push to output channel (backpressure point)
                if self.output_channel is not None and result is not None:
                    await self.output_channel.put(result)

            except asyncio.CancelledError:
                self.logger.warning("Worker %d cancelled", worker_id)
                raise
            except StageProcessingError as e:
                duration = time.monotonic() - t0
                await self.metrics.record_error(duration)
                try:
                    await self.error_router.handle(e.to_failed_item())
                except Exception as handler_err:
                    self.logger.error(
                        "Error handler failed for item in stage '%s': %s",
                        self.name,
                        handler_err,
                    )

        # Worker is done — if this is the last worker, signal downstream
        self.logger.debug("Worker %d exiting", worker_id)

    async def _process_item(self, item: Any) -> Any:
        """Process a single item with retry and per-attempt timeout."""
        return await execute_with_retry(
            func=self.stage_func,
            item=item,
            stage_name=self.name,
            policy=self.config.retry_policy,
            on_retry=self.metrics.record_retry,
            timeout=self.config.timeout,
        )
