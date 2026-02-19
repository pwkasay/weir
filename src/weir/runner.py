"""Base stage runner — shared lifecycle for stage and batch runners.

Both StageRunner and BatchStageRunner share identical start/wait/cancel logic,
worker task management, hook pre-filtering, and executor setup. This ABC
factors that out so each subclass only implements the processing loop.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .channel import Channel
from .errors import ErrorRouter
from .hooks import Hook
from .logging import PipelineLoggerAdapter, get_logger
from .metrics import StageMetrics


class BaseStageRunner(ABC):
    """Abstract base class for pipeline stage runners.

    Subclasses implement _worker_loop() for their specific processing model
    (item-at-a-time or batch). Everything else — lifecycle, hooks, metrics,
    executor setup — is shared.
    """

    def __init__(
        self,
        stage_func: Any,
        input_channel: Channel[Any],
        output_channel: Channel[Any] | None,
        error_router: ErrorRouter,
        pipeline_name: str,
        hooks: list[Hook] | None = None,
    ) -> None:
        self.stage_func = stage_func
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.error_router = error_router
        self.metrics = StageMetrics(stage_name=stage_func.name, _input_channel=input_channel)
        self.logger: PipelineLoggerAdapter = get_logger(pipeline_name, stage_func.name)
        self._workers: list[asyncio.Task[None]] = []
        self._executor: ThreadPoolExecutor | None = None
        self._func: Callable[..., Any] = stage_func
        self._record_retry = self.metrics.record_retry
        # Pre-filter hooks by implemented methods for zero overhead when unused
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
        """The name of the underlying stage function."""
        return str(self.stage_func.name)

    @property
    def config(self) -> Any:
        """The stage's configuration."""
        return self.stage_func.config

    @property
    def _task_name_prefix(self) -> str:
        """Prefix for asyncio task names. Override for custom naming."""
        return f"{self.name}-worker"

    def _log_start(self) -> None:
        """Log the stage start message. Override for custom log content."""
        self.logger.info(
            "Starting stage '%s' with %d workers",
            self.name,
            self.config.concurrency,
        )

    async def start(self) -> None:
        """Launch worker tasks."""
        if self._executor is not None:
            loop = asyncio.get_running_loop()
            executor = self._executor
            raw_func = self.stage_func.func

            async def _offload(item: Any) -> Any:
                return await loop.run_in_executor(executor, raw_func, item)

            self._func = _offload

        self._log_start()
        for hook in self._on_start_hooks:
            await hook.on_start(self.name)
        for i in range(self.config.concurrency):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"{self._task_name_prefix}-{i}",
            )
            self._workers.append(task)

    async def wait(self) -> None:
        """Wait for all workers to complete (called during shutdown drain)."""
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

    @abstractmethod
    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single worker. Implemented by subclasses."""
