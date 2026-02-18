"""Pipeline — the composed whole.

A Pipeline is a builder pattern that wires stages together with channels,
then runs the whole thing with proper lifecycle management.

The builder API is designed to read like a sentence:

    Pipeline("ingest")
        .source(items)
        .then(fetch)
        .then(transform)
        .then(save)
        .on_error(ValidationError, dead_letters)
        .build()

Once built, calling `await pipeline.run()` starts the machinery:
1. Install signal handlers for graceful shutdown.
2. Launch all stage workers.
3. Feed source items into the first channel.
4. Wait for all stages to drain.
5. Collect metrics and return a summary.

Design decision: the pipeline is built in two phases (configure, then run)
rather than being "live" from construction. This makes it testable —
you can inspect the built pipeline's topology without running it.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Self

from .batch import BatchStageFunction, BatchStageRunner
from .channel import Channel
from .errors import DeadLetterCollector, ErrorHandler, ErrorRouter
from .logging import PipelineLoggerAdapter, configure_logging, get_logger
from .metrics import StageMetricsSnapshot
from .shutdown import ShutdownCoordinator
from .stage import StageFunction, StageRunner

logger = logging.getLogger("weir.pipeline")

# Sources can be sync iterables or async iterables
type SourceType = (
    Iterable[Any] | AsyncIterable[Any] | Callable[[], Iterable[Any] | AsyncIterable[Any]]
)


@dataclass(slots=True)
class PipelineResult:
    """Summary of a pipeline run."""

    pipeline_name: str
    duration_seconds: float
    stage_metrics: list[StageMetricsSnapshot]
    completed: bool  # False if shutdown was forced / timed out
    dead_letters: int

    def __repr__(self) -> str:
        """Return a compact summary showing status, duration, and stage count."""
        status = "completed" if self.completed else "interrupted"
        return (
            f"<PipelineResult '{self.pipeline_name}' {status} "
            f"in {self.duration_seconds:.2f}s, "
            f"{len(self.stage_metrics)} stages>"
        )

    def summary(self) -> str:
        """Human-readable summary of the run."""
        lines = [
            f"Pipeline '{self.pipeline_name}' — {'completed' if self.completed else 'interrupted'}",
            f"  Duration: {self.duration_seconds:.2f}s",
            f"  Dead letters: {self.dead_letters}",
            "",
        ]
        for sm in self.stage_metrics:
            p50 = sm.get("latency_p50")
            lat = f"{p50 * 1000:.1f}ms" if p50 is not None else "n/a"
            lines.append(
                f"  {sm['stage']}: "
                f"{sm['items_out']}/{sm['items_in']} ok, "
                f"{sm['items_errored']} errors, "
                f"{sm['items_retried']} retries, "
                f"p50={lat}, "
                f"{sm['throughput_per_sec']:.1f}/s"
            )
        return "\n".join(lines)


class Pipeline:
    """Builder and runtime for a data pipeline.

    Usage:
        pipe = (
            Pipeline("my-pipeline")
            .source(items)
            .then(fetch_stage)
            .then(transform_stage)
            .then(save_stage)
            .on_error(ValidationError, dead_letter_collector)
            .build()
        )
        result = await pipe.run()
        print(result.summary())
    """

    def __init__(
        self,
        name: str,
        channel_capacity: int = 64,
        drain_timeout: float = 30.0,
        log_level: int = logging.INFO,
        structured_logging: bool = False,
    ) -> None:
        """Initialize a pipeline builder.

        Args:
            name: Human-readable name for logging and metrics.
            channel_capacity: Bounded queue size between stages.
            drain_timeout: Seconds to wait for graceful shutdown drain.
            log_level: Python logging level for pipeline logs.
            structured_logging: If True, emit JSON lines instead of human-readable logs.
        """
        self._name = name
        self._channel_capacity = channel_capacity
        self._drain_timeout = drain_timeout
        self._log_level = log_level
        self._structured_logging = structured_logging

        # Builder state
        self._source: SourceType | None = None
        self._stages: list[StageFunction | BatchStageFunction] = []
        self._error_router = ErrorRouter()
        self._dead_letters = DeadLetterCollector()
        self._hooks: list[Any] = []
        self._metrics_callback: Callable[..., Any] | None = None
        self._metrics_interval: float = 5.0
        self._built = False

        # Runtime state (populated by build)
        self._runners: list[StageRunner | BatchStageRunner] = []
        self._channels: list[Channel[Any]] = []
        self._coordinator: ShutdownCoordinator | None = None

    @property
    def name(self) -> str:
        """The pipeline's name."""
        return self._name

    def source(self, src: SourceType) -> Self:
        """Set the data source.

        Accepts:
        - A sync iterable (list, generator, etc.)
        - An async iterable (async generator, etc.)
        - A callable that returns either of the above
        """
        if self._built:
            raise RuntimeError("Cannot modify a built pipeline")
        self._source = src
        return self

    def then(self, stage_func: StageFunction | BatchStageFunction) -> Self:
        """Append a stage to the pipeline."""
        if self._built:
            raise RuntimeError("Cannot modify a built pipeline")
        if not isinstance(stage_func, (StageFunction, BatchStageFunction)):
            raise TypeError(
                f"Expected a @stage or @batch_stage decorated function, "
                f"got {type(stage_func).__name__}. "
                f"Did you forget the decorator?"
            )
        self._stages.append(stage_func)
        return self

    def on_error(
        self,
        error_type: type[Exception],
        handler: ErrorHandler | None = None,
    ) -> Self:
        """Register an error handler for a specific exception type.

        If no handler is provided, uses the built-in dead letter collector.
        """
        if self._built:
            raise RuntimeError("Cannot modify a built pipeline")
        self._error_router.on(error_type, handler or self._dead_letters)
        return self

    def with_hook(self, hook: Any) -> Self:
        """Register a lifecycle hook for all stages.

        Hooks receive callbacks at stage lifecycle points (start, item, error, complete).
        Only methods that exist on the hook object are called — implement only what you need.
        """
        if self._built:
            raise RuntimeError("Cannot modify a built pipeline")
        self._hooks.append(hook)
        return self

    def on_metrics(
        self,
        callback: Callable[..., Any],
        interval: float = 5.0,
    ) -> Self:
        """Register a callback for periodic metrics snapshots during pipeline.run().

        The callback receives a list of StageMetricsSnapshot dicts, one per stage.
        Both sync and async callables are supported.

        Args:
            callback: Function called with [StageMetricsSnapshot, ...] each interval.
            interval: Seconds between callback invocations.
        """
        if self._built:
            raise RuntimeError("Cannot modify a built pipeline")
        self._metrics_callback = callback
        self._metrics_interval = interval
        return self

    def build(self) -> Self:
        """Finalize the pipeline topology. Must be called before run()."""
        if self._built:
            raise RuntimeError("Pipeline already built")
        if self._source is None:
            raise ValueError("Pipeline has no source. Call .source() first.")
        if not self._stages:
            raise ValueError("Pipeline has no stages. Call .then() at least once.")

        # Set default error handler to dead letter collector
        self._error_router.set_default(self._dead_letters)

        self._wire_topology()
        self._built = True
        return self

    def reset(self, new_source: SourceType | None = None) -> Self:
        """Reset the pipeline for re-running with fresh state.

        Rebuilds channels, runners, and coordinator. Clears dead letters.
        Pipeline stays "built" and ready for another run().

        Args:
            new_source: Optional new data source. If None, reuses the original source.
        """
        if not self._built:
            raise RuntimeError("Cannot reset a pipeline that hasn't been built yet.")
        if new_source is not None:
            self._source = new_source
        self._dead_letters.clear()
        self._wire_topology()
        return self

    def _wire_topology(self) -> None:
        """Create channels and runners from the stage list. Used by build() and reset()."""
        # Create channels between stages (n stages -> n channels, including source->first)
        self._channels = []
        for i in range(len(self._stages)):
            ch: Channel[Any] = Channel(
                capacity=self._channel_capacity,
                name=f"{self._name}:{self._stages[i].name}:in",
            )
            self._channels.append(ch)

        # Create stage runners
        self._runners = []
        for i, stage_func in enumerate(self._stages):
            input_ch = self._channels[i]
            # Output channel is the next stage's input, or None for the last stage
            output_ch = self._channels[i + 1] if i + 1 < len(self._channels) else None

            hooks = self._hooks if self._hooks else None
            if isinstance(stage_func, BatchStageFunction):
                runner: StageRunner | BatchStageRunner = BatchStageRunner(
                    stage_func=stage_func,
                    input_channel=input_ch,
                    output_channel=output_ch,
                    error_router=self._error_router,
                    pipeline_name=self._name,
                    hooks=hooks,
                )
            else:
                runner = StageRunner(
                    stage_func=stage_func,
                    input_channel=input_ch,
                    output_channel=output_ch,
                    error_router=self._error_router,
                    pipeline_name=self._name,
                    hooks=hooks,
                )
            self._runners.append(runner)

        self._coordinator = ShutdownCoordinator(drain_timeout=self._drain_timeout)

    async def run(self) -> PipelineResult:
        """Run the pipeline to completion (or until interrupted).

        Returns a PipelineResult with metrics and status.
        """
        if not self._built:
            raise RuntimeError("Pipeline not built. Call .build() first.")

        assert self._coordinator is not None

        # Configure logging
        configure_logging(level=self._log_level, structured=self._structured_logging)
        log = get_logger(self._name)

        log.info("Pipeline '%s' starting (%d stages)", self._name, len(self._runners))
        t0 = time.monotonic()

        # Install signal handlers
        self._coordinator.install_signal_handlers()

        metrics_task: asyncio.Task[None] | None = None
        try:
            # Start all stage workers
            for runner in self._runners:
                await runner.start()

            # Launch periodic metrics callback if configured
            if self._metrics_callback is not None:
                metrics_task = asyncio.create_task(
                    self._metrics_loop(),
                    name=f"{self._name}-metrics",
                )

            # Feed source into first channel
            first_channel = self._channels[0]
            first_stage = self._runners[0]
            await self._feed_source(first_channel, first_stage.config.concurrency, log)

            # Drain stages — race against force-kill and optional timeout
            drain_task = asyncio.create_task(self._drain_stages())
            force_task = asyncio.create_task(self._coordinator.wait_for_force())
            timeout = self._drain_timeout if self._coordinator.should_stop else None
            try:
                async with asyncio.timeout(timeout):
                    done, pending = await asyncio.wait(
                        [drain_task, force_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
                    if force_task in done:
                        log.warning("Force shutdown — cancelling all workers")
                        for runner in self._runners:
                            await runner.cancel()
            except TimeoutError:
                drain_task.cancel()
                force_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain_task
                with contextlib.suppress(asyncio.CancelledError):
                    await force_task
                log.warning(
                    "Drain timed out after %.1fs, cancelling remaining workers",
                    self._drain_timeout,
                )
                for runner in self._runners:
                    await runner.cancel()

            completed = True

        except asyncio.CancelledError:
            log.warning("Pipeline cancelled")
            completed = False
        except Exception as e:
            log.error("Pipeline failed with unexpected error: %s", e, exc_info=True)
            completed = False
        finally:
            if metrics_task is not None:
                metrics_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await metrics_task
            self._coordinator.restore_signal_handlers()

        duration = time.monotonic() - t0

        result = PipelineResult(
            pipeline_name=self._name,
            duration_seconds=round(duration, 3),
            stage_metrics=[r.metrics.snapshot() for r in self._runners],
            completed=completed,
            dead_letters=self._dead_letters.count,
        )

        log.info("\n%s", result.summary())
        return result

    async def _drain_stages(self) -> None:
        """Wait for all stages to drain sequentially, sending STOP sentinels downstream."""
        for idx, runner in enumerate(self._runners):
            await runner.wait()

            # After this stage is done, send STOP to the next stage's workers
            if idx + 1 < len(self._runners):
                next_runner = self._runners[idx + 1]
                next_channel = self._channels[idx + 1]
                await next_channel.send_stop(next_runner.config.concurrency)

    async def _metrics_loop(self) -> None:
        """Periodically snapshot metrics and invoke the callback."""
        assert self._metrics_callback is not None
        callback = self._metrics_callback
        interval = self._metrics_interval
        while True:
            await asyncio.sleep(interval)
            snapshots = [r.metrics.snapshot() for r in self._runners]
            result = callback(snapshots)
            if asyncio.iscoroutine(result):
                await result

    async def _feed_source(
        self,
        channel: Channel[Any],
        num_consumers: int,
        log: PipelineLoggerAdapter,
    ) -> None:
        """Feed items from the source into the first channel.

        Handles both sync and async iterables. Respects shutdown signals.
        """
        assert self._coordinator is not None
        source = self._source

        # If source is callable, call it to get the iterable
        if (
            callable(source)
            and not hasattr(source, "__aiter__")
            and not hasattr(source, "__iter__")
        ):
            source = source()

        item_count = 0

        if hasattr(source, "__aiter__"):
            # Async iterable
            async for item in source:  # type: ignore[union-attr]
                if self._coordinator.should_stop:
                    log.info("Source interrupted by shutdown after %d items", item_count)
                    break
                await channel.put(item)
                item_count += 1
        else:
            # Sync iterable
            for item in source:  # type: ignore[union-attr]
                if self._coordinator.should_stop:
                    log.info("Source interrupted by shutdown after %d items", item_count)
                    break
                await channel.put(item)
                item_count += 1

        log.info("Source exhausted after %d items. Sending stop signals.", item_count)
        await channel.send_stop(num_consumers)

    # ── Introspection ──

    @property
    def stage_names(self) -> list[str]:
        """Ordered list of stage names in the pipeline."""
        return [s.name for s in self._stages]

    @property
    def topology(self) -> str:
        """Return a human-readable description of the pipeline topology."""
        if not self._built:
            return f"Pipeline '{self._name}' (not built)"

        parts = [f"Pipeline '{self._name}':"]
        parts.append(f"  source → [{self._channels[0].name}]")
        for i, runner in enumerate(self._runners):
            concurrency = runner.config.concurrency
            retries = runner.config.retries
            if i + 1 < len(self._channels):
                out = f" → [{self._channels[i + 1].name}]"
            else:
                out = " → (end)"
            parts.append(f"  → {runner.name}(n={concurrency}, retries={retries}){out}")
        return "\n".join(parts)

    @property
    def dead_letter_items(self) -> list[Any]:
        """Access dead letter items for inspection."""
        return self._dead_letters.items
