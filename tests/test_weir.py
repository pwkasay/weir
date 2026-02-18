"""Tests for weir.

The test philosophy mirrors the framework philosophy: test *behaviors*, not internals.

What we care about:
1. Items flow through stages in order.
2. Backpressure works (slow downstream doesn't cause unbounded memory growth).
3. Retries happen on transient failures.
4. Permanent failures route to error handlers.
5. Graceful shutdown drains in-flight items.
6. Metrics are accurate.

What we don't care about:
- Internal queue implementation details.
- Exact timing (use tolerances).
- Logging output (test behavior, not side effects).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from weir import (
    DeadLetterCollector,
    Pipeline,
    StageMetrics,
    batch_stage,
    stage,
)
from weir.channel import STOP, Channel
from weir.errors import (
    ErrorRouter,
    FailedItem,
    RetryPolicy,
    StageProcessingError,
    execute_with_retry,
)
from weir.metrics import LatencyHistogram

# ── Channel Tests ──


class TestChannel:
    """Bounded channel operations: put, get, backpressure, stop sentinels, and stats."""

    async def test_basic_put_get(self):
        ch = Channel(capacity=10, name="test")
        await ch.put("hello")
        item = await ch.get()
        assert item == "hello"

    async def test_bounded_capacity(self):
        ch = Channel(capacity=2, name="tiny")
        await ch.put("a")
        await ch.put("b")
        assert ch.is_full

        # This should block — verify with a timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ch.put("c"), timeout=0.1)

    async def test_stop_sentinel(self):
        ch = Channel(capacity=10, name="test")
        await ch.put("item")
        await ch.send_stop(num_consumers=1)

        item = await ch.get()
        assert item == "item"

        sentinel = await ch.get()
        assert sentinel is STOP

    async def test_multiple_stop_sentinels(self):
        ch = Channel(capacity=10, name="test")
        await ch.send_stop(num_consumers=3)

        for _ in range(3):
            item = await ch.get()
            assert item is STOP

    async def test_stats(self):
        ch = Channel(capacity=5, name="test")
        await ch.put("a")
        await ch.put("b")
        _ = await ch.get()

        stats = ch.stats()
        assert stats.capacity == 5
        assert stats.current_depth == 1
        assert stats.total_put == 2
        assert stats.total_get == 1

    async def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            Channel(capacity=0)


# ── Retry Tests ──


class TestRetryLogic:
    """Retry policy execution: success, transient failure, permanent failure, and callbacks."""

    async def test_no_retry_on_success(self):
        call_count = 0

        async def fn(item):
            nonlocal call_count
            call_count += 1
            return item * 2

        policy = RetryPolicy(max_attempts=3)
        result = await execute_with_retry(fn, 5, "test-stage", policy)
        assert result == 10
        assert call_count == 1

    async def test_retry_on_transient_failure(self):
        call_count = 0

        async def fn(item):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return item

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),
        )
        result = await execute_with_retry(fn, "ok", "test-stage", policy)
        assert result == "ok"
        assert call_count == 3

    async def test_permanent_failure_after_retries(self):
        async def fn(item):
            raise ConnectionError("always fails")

        policy = RetryPolicy(max_attempts=3, base_delay=0.01)

        with pytest.raises(StageProcessingError) as exc_info:
            await execute_with_retry(fn, "bad", "test-stage", policy)

        assert exc_info.value.attempts == 3
        assert len(exc_info.value.error_chain) == 3

    async def test_non_retryable_error_fails_immediately(self):
        call_count = 0

        async def fn(item):
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),  # ValueError is NOT retryable
        )

        with pytest.raises(StageProcessingError):
            await execute_with_retry(fn, "bad", "test-stage", policy)

        assert call_count == 1  # No retries

    async def test_retry_callback(self):
        retry_count = 0

        async def fn(item):
            raise ConnectionError("fail")

        def on_retry():
            nonlocal retry_count
            retry_count += 1

        policy = RetryPolicy(max_attempts=3, base_delay=0.01)

        with pytest.raises(StageProcessingError):
            await execute_with_retry(fn, "x", "test", policy, on_retry=on_retry)

        assert retry_count == 2  # 3 attempts = 2 retries


# ── Error Router Tests ──


class TestErrorRouter:
    """Error routing by exception type, MRO-based subclass matching, and default handler."""

    async def test_routes_to_specific_handler(self):
        handled = []

        async def handler(failed):
            handled.append(failed)

        router = ErrorRouter()
        router.on(ValueError, handler)

        failed = FailedItem(item="test", stage_name="s", error=ValueError("bad"), attempts=1)
        await router.handle(failed)
        assert len(handled) == 1

    async def test_routes_subclass_to_parent_handler(self):
        handled = []

        async def handler(failed):
            handled.append(failed)

        router = ErrorRouter()
        router.on(OSError, handler)

        # ConnectionError is a subclass of OSError
        failed = FailedItem(item="test", stage_name="s", error=ConnectionError("net"), attempts=1)
        await router.handle(failed)
        assert len(handled) == 1

    async def test_default_handler(self):
        handled = []

        async def default(failed):
            handled.append(failed)

        router = ErrorRouter()
        router.set_default(default)

        failed = FailedItem(
            item="test", stage_name="s", error=RuntimeError("unexpected"), attempts=1
        )
        await router.handle(failed)
        assert len(handled) == 1


# ── Dead Letter Collector Tests ──


class TestDeadLetterCollector:
    """Dead letter collection with capacity limits and overflow counting."""

    async def test_collects_failures(self):
        dlc = DeadLetterCollector()

        for i in range(5):
            await dlc(FailedItem(item=i, stage_name="s", error=ValueError(), attempts=1))

        assert dlc.count == 5
        assert len(dlc.items) == 5

    async def test_respects_max_size(self):
        dlc = DeadLetterCollector(max_size=3)

        for i in range(10):
            await dlc(FailedItem(item=i, stage_name="s", error=ValueError(), attempts=1))

        assert len(dlc.items) == 3
        assert dlc.count == 10  # Includes overflow


# ── Stage Decorator Tests ──


class TestStageDecorator:
    """The @stage decorator: function callability, config attachment, name preservation."""

    async def test_decorated_function_is_callable(self):
        @stage(concurrency=3)
        async def double(x: int) -> int:
            return x * 2

        result = await double(5)
        assert result == 10

    def test_config_attached(self):
        @stage(concurrency=5, retries=3, timeout=30)
        async def fetch(url: str) -> dict:
            return {}

        assert fetch.config.concurrency == 5
        assert fetch.config.retries == 3
        assert fetch.config.timeout == 30

    def test_name_preserved(self):
        @stage()
        async def my_stage(x):
            return x

        assert my_stage.name == "my_stage"


# ── Pipeline Integration Tests ──


class TestPipeline:
    """End-to-end pipeline: linear flow, concurrency, errors, retries, backpressure, metrics."""

    async def test_simple_linear_pipeline(self):
        """The most basic test: items flow through stages in order."""
        results = []

        @stage(concurrency=1)
        async def double(x: int) -> int:
            return x * 2

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-simple", channel_capacity=10)
            .source([1, 2, 3, 4, 5])
            .then(double)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [2, 4, 6, 8, 10]

    async def test_concurrent_processing(self):
        """Multiple workers process items concurrently."""
        active = 0
        max_active = 0

        @stage(concurrency=3)
        async def slow_work(x: int) -> int:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-concurrent", channel_capacity=20)
            .source(range(10))
            .then(slow_work)
            .then(sink)
            .build()
        )

        await pipe.run()
        # With 3 workers and enough items, we should see concurrency > 1
        assert max_active > 1

    async def test_error_routing(self):
        """Failed items are routed to the error handler."""
        dead_letters = DeadLetterCollector()

        @stage(concurrency=1, retries=1)
        async def failing(x: int) -> int:
            if x % 2 == 0:
                raise ValueError(f"even number: {x}")
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-errors", channel_capacity=10)
            .source([1, 2, 3, 4, 5])
            .then(failing)
            .then(sink)
            .on_error(ValueError, dead_letters)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        # Items 2 and 4 should be dead-lettered
        assert dead_letters.count == 2
        failed_items = [dl.item for dl in dead_letters.items]
        assert sorted(failed_items) == [2, 4]

    async def test_retry_then_succeed(self):
        """Items that fail transiently succeed after retries."""
        attempt_counts: dict[int, int] = {}

        @stage(concurrency=1, retries=3, retry_base_delay=0.01)
        async def flaky(x: int) -> int:
            attempt_counts[x] = attempt_counts.get(x, 0) + 1
            if attempt_counts[x] < 2:
                raise ConnectionError("transient")
            return x

        results = []

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-retry", channel_capacity=10)
            .source([1, 2, 3])
            .then(flaky)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [1, 2, 3]

    async def test_backpressure(self):
        """Slow downstream should naturally throttle upstream."""
        upstream_items_emitted = 0

        @stage(concurrency=1)
        async def fast_producer(x: int) -> int:
            nonlocal upstream_items_emitted
            upstream_items_emitted += 1
            return x

        @stage(concurrency=1)
        async def slow_consumer(x: int) -> None:
            await asyncio.sleep(0.05)

        pipe = (
            Pipeline("test-backpressure", channel_capacity=2)  # Tiny channels!
            .source(range(20))
            .then(fast_producer)
            .then(slow_consumer)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert upstream_items_emitted == 20  # All items processed eventually

    async def test_metrics_collected(self):
        """Stage metrics are populated after a run."""

        @stage(concurrency=1)
        async def double(x: int) -> int:
            return x * 2

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-metrics", channel_capacity=10)
            .source(range(100))
            .then(double)
            .then(sink)
            .build()
        )

        result = await pipe.run()

        assert len(result.stage_metrics) == 2

        double_metrics = result.stage_metrics[0]
        assert double_metrics["stage"] == "double"
        assert double_metrics["items_in"] == 100
        assert double_metrics["items_out"] == 100
        assert double_metrics["items_errored"] == 0
        assert double_metrics["latency_p50"] is not None

    async def test_pipeline_topology(self):
        """The topology string is human-readable."""

        @stage(concurrency=5, retries=3)
        async def fetch(x):
            return x

        @stage(concurrency=2)
        async def save(x):
            return None

        pipe = Pipeline("topo-test").source([]).then(fetch).then(save).build()
        topo = pipe.topology
        assert "fetch" in topo
        assert "save" in topo
        assert "n=5" in topo

    async def test_build_validation(self):
        """Pipeline validates configuration at build time."""
        with pytest.raises(ValueError, match="no source"):
            Pipeline("bad").then(stage()(AsyncMock())).build()

        with pytest.raises(ValueError, match="no stages"):
            Pipeline("bad").source([1, 2, 3]).build()

    async def test_builder_returns_same_instance(self):
        """Builder chaining returns the same Pipeline instance (self)."""

        @stage(concurrency=1)
        async def noop(x: int) -> int:
            return x

        pipe = Pipeline("test-builder")
        assert pipe.source([1]) is pipe
        assert pipe.then(noop) is pipe
        assert pipe.on_error(ValueError) is pipe
        assert pipe.build() is pipe

    async def test_async_source(self):
        """Pipeline accepts async iterables as sources."""
        results = []

        async def async_source():
            for i in range(5):
                yield i

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-async-source", channel_capacity=10)
            .source(async_source())
            .then(collect)
            .build()
        )

        await pipe.run()
        assert sorted(results) == [0, 1, 2, 3, 4]


# ── Metrics Unit Tests (updated for async record methods) ──


class TestMetrics:
    """StageMetrics snapshot accuracy and error rate calculation."""

    async def test_snapshot(self):
        m = StageMetrics(stage_name="test")
        m.record_success(0.1)
        m.record_success(0.2)
        m.record_error(0.3)

        snap = m.snapshot()
        assert snap["items_in"] == 3
        assert snap["items_out"] == 2
        assert snap["items_errored"] == 1
        assert snap["latency_p50"] is not None

    async def test_error_rate(self):
        m = StageMetrics(stage_name="test")
        for _ in range(7):
            m.record_success(0.1)
        for _ in range(3):
            m.record_error(0.1)

        assert abs(m.error_rate - 0.3) < 0.01

    async def test_snapshot_type_completeness(self):
        """Snapshot includes queue fields when an input channel is provided."""
        ch = Channel(capacity=8, name="test-ch")
        m = StageMetrics(stage_name="test", _input_channel=ch)
        m.record_success(0.05)

        snap = m.snapshot()
        assert "queue_depth" in snap
        assert "queue_capacity" in snap
        assert snap["queue_capacity"] == 8
        assert "queue_utilization" in snap


# ── Timeout Behavior Tests ──


class TestTimeoutBehavior:
    """Per-attempt timeout with retry, exhaustion, and end-to-end pipeline timeout."""

    async def test_per_attempt_timeout_retries(self):
        """Timeout fires per-attempt and retries on timeout."""
        call_count = 0

        async def slow_then_fast(item):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                await asyncio.sleep(5)  # Will timeout
            return item

        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        result = await execute_with_retry(slow_then_fast, "ok", "test", policy, timeout=0.05)
        assert result == "ok"
        assert call_count == 3

    async def test_timeout_exhaustion_routes_to_dead_letters(self):
        """After all retries timeout, item routes to dead letters."""

        async def always_slow(item):
            await asyncio.sleep(5)
            return item

        policy = RetryPolicy(max_attempts=2, base_delay=0.01)
        with pytest.raises(StageProcessingError) as exc_info:
            await execute_with_retry(always_slow, "slow", "test", policy, timeout=0.05)
        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.error, asyncio.TimeoutError)

    async def test_timeout_in_pipeline(self):
        """Per-attempt timeout works end-to-end in a pipeline."""
        dead_letters = DeadLetterCollector()
        call_count = 0

        @stage(concurrency=1, retries=2, timeout=0.05, retry_base_delay=0.01)
        async def slow_stage(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(5)  # Always times out
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-timeout", channel_capacity=10)
            .source([1])
            .then(slow_stage)
            .then(sink)
            .on_error(Exception, dead_letters)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert dead_letters.count == 1
        assert call_count == 2  # 2 attempts, both timed out


# ── Error Handler Recovery Tests ──


class TestErrorHandlerRecovery:
    """Worker resilience when error handlers themselves throw exceptions."""

    async def test_worker_survives_broken_error_handler(self):
        """Worker continues processing after error handler throws."""
        results = []

        async def broken_handler(failed):
            raise RuntimeError("handler crash!")

        @stage(concurrency=1, retries=1)
        async def maybe_fail(x: int) -> int:
            if x == 2:
                raise ValueError("bad item")
            return x

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-handler-recovery", channel_capacity=10)
            .source([1, 2, 3, 4, 5])
            .then(maybe_fail)
            .then(collect)
            .on_error(ValueError, broken_handler)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        # Item 2 fails, but worker continues — items 1, 3, 4, 5 should succeed
        assert sorted(results) == [1, 3, 4, 5]


# ── Latency Histogram Tests ──


class TestLatencyHistogram:
    """Reservoir sampling histogram: recording, counting, reset, and Algorithm R fidelity."""

    def test_basic_recording(self):
        h = LatencyHistogram()
        for i in range(100):
            h.record(float(i))
        assert h.count == 100
        assert h.percentiles(50)[0] is not None

    def test_count_reflects_total_seen(self):
        """count returns total observations, not reservoir size."""
        h = LatencyHistogram(_max_size=10)
        for i in range(1000):
            h.record(float(i))
        assert h.count == 1000
        assert len(h._values) == 10  # Reservoir capped

    def test_reset_clears_everything(self):
        h = LatencyHistogram()
        for i in range(50):
            h.record(float(i))
        h.reset()
        assert h.count == 0
        assert h.percentiles(50)[0] is None

    def test_algorithm_r_sampling(self):
        """Algorithm R produces a representative sample."""
        import random

        random.seed(42)
        h = LatencyHistogram(_max_size=100)
        # Record 10000 values from 0-999
        for i in range(10000):
            h.record(float(i % 1000))

        # The median should be close to 500 (within tolerance)
        p50 = h.percentiles(50)[0]
        assert p50 is not None
        assert 300 < p50 < 700  # Wide tolerance for statistical test

    def test_batch_percentiles(self):
        """percentiles() returns ordered values from a populated histogram."""
        h = LatencyHistogram()
        for i in range(100):
            h.record(float(i))
        p50, p95, p99 = h.percentiles(50, 95, 99)
        assert p50 is not None
        assert p95 is not None
        assert p99 is not None
        assert p50 <= p95 <= p99

    def test_batch_percentiles_empty(self):
        """percentiles() returns [None, None, None] when histogram is empty."""
        h = LatencyHistogram()
        result = h.percentiles(50, 95, 99)
        assert result == [None, None, None]


# ── Graceful Shutdown Tests ──


class TestGracefulShutdown:
    """STOP sentinel propagation through multi-stage pipelines."""

    async def test_stop_propagation_multi_stage(self):
        """STOP sentinels cascade through a multi-stage pipeline."""
        results = []

        @stage(concurrency=1)
        async def add_one(x: int) -> int:
            return x + 1

        @stage(concurrency=1)
        async def multiply(x: int) -> int:
            return x * 2

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-shutdown-propagation", channel_capacity=10)
            .source(range(10))
            .then(add_one)
            .then(multiply)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        # (0+1)*2=2, (1+1)*2=4, ..., (9+1)*2=20
        assert sorted(results) == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


# ── Edge Case Tests ──


class TestEdgeCases:
    """Edge cases: empty source, single stage, callable source, queue depth in metrics."""

    async def test_empty_source(self):
        """Pipeline handles empty source gracefully."""

        @stage(concurrency=1)
        async def identity(x: int) -> int:
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-empty", channel_capacity=10).source([]).then(identity).then(sink).build()
        )

        result = await pipe.run()
        assert result.completed
        assert result.stage_metrics[0]["items_in"] == 0

    async def test_single_stage_pipeline(self):
        """Pipeline works with just one stage."""
        results = []

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = Pipeline("test-single", channel_capacity=10).source([1, 2, 3]).then(collect).build()

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [1, 2, 3]

    async def test_callable_source(self):
        """Pipeline accepts a callable that returns an iterable."""

        results = []

        def make_source():
            return [10, 20, 30]

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-callable", channel_capacity=10).source(make_source).then(collect).build()
        )

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [10, 20, 30]

    async def test_queue_depth_in_metrics(self):
        """Metrics snapshot includes queue depth info."""

        @stage(concurrency=1)
        async def identity(x: int) -> int:
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-queue-depth", channel_capacity=10)
            .source([1, 2, 3])
            .then(identity)
            .then(sink)
            .build()
        )

        result = await pipe.run()
        snap = result.stage_metrics[0]
        # After run, queue should be drained
        assert "queue_depth" in snap
        assert "queue_capacity" in snap
        assert snap["queue_capacity"] == 10
        assert "queue_utilization" in snap


# ── Batch Stage Tests ──


class TestBatchStage:
    """Batch stage: accumulation, partial flush, timeout flush, downstream flow, decorator."""

    async def test_batch_accumulation(self):
        """Batch stage accumulates items up to batch_size."""
        received_batches = []

        @batch_stage(batch_size=3, flush_timeout=5.0)
        async def batch_collect(items: list[int]) -> None:
            received_batches.append(list(items))

        pipe = (
            Pipeline("test-batch", channel_capacity=10)
            .source([1, 2, 3, 4, 5, 6])
            .then(batch_collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        # 6 items / batch_size 3 = 2 full batches
        assert len(received_batches) == 2
        all_items = sorted([x for batch in received_batches for x in batch])
        assert all_items == [1, 2, 3, 4, 5, 6]

    async def test_partial_flush_on_stop(self):
        """Remaining items flush as partial batch when STOP arrives."""
        received_batches = []

        @batch_stage(batch_size=4, flush_timeout=5.0)
        async def batch_collect(items: list[int]) -> None:
            received_batches.append(list(items))

        pipe = (
            Pipeline("test-batch-partial", channel_capacity=10)
            .source([1, 2, 3, 4, 5])  # 5 items, batch_size 4
            .then(batch_collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        # Should get one full batch of 4 and one partial batch of 1
        all_items = sorted([x for batch in received_batches for x in batch])
        assert all_items == [1, 2, 3, 4, 5]
        batch_sizes = sorted([len(b) for b in received_batches])
        assert batch_sizes == [1, 4]

    async def test_flush_timeout(self):
        """Partial batch flushes after timeout even if batch_size not reached."""
        received_batches = []

        @batch_stage(batch_size=100, flush_timeout=0.1)
        async def batch_collect(items: list[int]) -> None:
            received_batches.append(list(items))

        pipe = (
            Pipeline("test-batch-timeout", channel_capacity=10)
            .source([1, 2, 3])  # Only 3 items, batch_size 100
            .then(batch_collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        all_items = sorted([x for batch in received_batches for x in batch])
        assert all_items == [1, 2, 3]

    async def test_batch_downstream_flow(self):
        """Batch stage can push results downstream."""
        results = []

        @batch_stage(batch_size=2, flush_timeout=5.0)
        async def batch_double(items: list[int]) -> list[int]:
            return [x * 2 for x in items]

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-batch-downstream", channel_capacity=10)
            .source([1, 2, 3, 4])
            .then(batch_double)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [2, 4, 6, 8]

    async def test_batch_stage_decorator(self):
        """@batch_stage decorator preserves function and attaches config."""

        @batch_stage(batch_size=50, flush_timeout=2.0, concurrency=3)
        async def bulk_insert(items: list) -> None:
            pass

        assert bulk_insert.name == "bulk_insert"
        assert bulk_insert.config.batch_size == 50
        assert bulk_insert.config.flush_timeout == 2.0
        assert bulk_insert.config.concurrency == 3
        # Should be directly callable
        await bulk_insert([1, 2, 3])


# ── CPU-Bound Stage Tests ──


class TestCpuBoundStage:
    """cpu_bound=True: thread pool offloading, direct call, retry, pipeline, and timeout."""

    async def test_cpu_bound_runs_off_event_loop(self):
        """cpu_bound stage executes in a different thread than the event loop."""
        import threading

        event_loop_thread = threading.current_thread().ident
        stage_thread_ids: list[int | None] = []

        @stage(concurrency=2, cpu_bound=True)
        def record_thread(x: int) -> int:
            stage_thread_ids.append(threading.current_thread().ident)
            return x * 2

        results: list[int] = []

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-cpu-bound", channel_capacity=10)
            .source(range(5))
            .then(record_thread)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [0, 2, 4, 6, 8]
        # Every invocation should have run on a thread other than the event loop
        assert all(tid != event_loop_thread for tid in stage_thread_ids)

    async def test_cpu_bound_direct_call(self):
        """StageFunction.__call__ works for sync cpu_bound functions (testing convenience)."""

        @stage(concurrency=1, cpu_bound=True)
        def double(x: int) -> int:
            return x * 2

        result = await double(5)
        assert result == 10

    async def test_cpu_bound_with_retry(self):
        """Retries work with executor-offloaded stages."""
        call_count = 0

        @stage(concurrency=1, retries=3, retry_base_delay=0.01, cpu_bound=True)
        def flaky(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return x

        results: list[int] = []

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-cpu-retry", channel_capacity=10)
            .source([42])
            .then(flaky)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert results == [42]
        assert call_count == 3

    async def test_cpu_bound_in_pipeline(self):
        """End-to-end pipeline mixing async and cpu_bound stages."""
        results: list[int] = []

        @stage(concurrency=2, cpu_bound=True)
        def parse(x: int) -> int:
            return x * 10

        @stage(concurrency=1)
        async def enrich(x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-cpu-mixed", channel_capacity=10)
            .source([1, 2, 3])
            .then(parse)
            .then(enrich)
            .then(collect)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert sorted(results) == [11, 21, 31]

    async def test_cpu_bound_with_timeout(self):
        """Per-attempt timeout works with executor stages."""
        import time as time_mod

        dead_letters = DeadLetterCollector()

        @stage(concurrency=1, retries=1, timeout=0.05, cpu_bound=True)
        def slow_sync(x: int) -> int:
            time_mod.sleep(5)  # Will exceed timeout
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-cpu-timeout", channel_capacity=10)
            .source([1])
            .then(slow_sync)
            .then(sink)
            .on_error(Exception, dead_letters)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert dead_letters.count == 1


# ── Lifecycle Hook Tests ──


class TestLifecycleHooks:
    """Lifecycle hooks: on_item, on_error, on_start/on_complete, partial implementation."""

    async def test_on_item_hook_called(self):
        """on_item fires for every successfully processed item."""
        hook_calls: list[tuple[str, Any, Any, float]] = []

        class ItemHook:
            async def on_item(self, stage_name: str, item: Any, result: Any, duration: float):
                hook_calls.append((stage_name, item, result, duration))

        @stage(concurrency=1)
        async def double(x: int) -> int:
            return x * 2

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-hook-item", channel_capacity=10)
            .source([1, 2, 3])
            .then(double)
            .then(sink)
            .with_hook(ItemHook())
            .build()
        )

        await pipe.run()
        # on_item called for each item in each stage
        double_calls = [(s, i, r) for s, i, r, _ in hook_calls if s == "double"]
        assert sorted(double_calls, key=lambda t: t[1]) == [
            ("double", 1, 2),
            ("double", 2, 4),
            ("double", 3, 6),
        ]

    async def test_on_error_hook_called(self):
        """on_error fires when a stage permanently fails to process an item."""
        error_calls: list[tuple[str, Any, Exception]] = []

        class ErrorHook:
            async def on_error(self, stage_name: str, item: Any, error: Exception):
                error_calls.append((stage_name, item, error))

        @stage(concurrency=1, retries=1)
        async def failing(x: int) -> int:
            if x == 2:
                raise ValueError("bad")
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-hook-error", channel_capacity=10)
            .source([1, 2, 3])
            .then(failing)
            .then(sink)
            .with_hook(ErrorHook())
            .build()
        )

        await pipe.run()
        assert len(error_calls) == 1
        assert error_calls[0][0] == "failing"
        assert error_calls[0][1] == 2

    async def test_on_start_on_complete_hooks(self):
        """on_start and on_complete fire at stage lifecycle boundaries."""
        events: list[tuple[str, str]] = []

        class LifecycleHook:
            async def on_start(self, stage_name: str):
                events.append(("start", stage_name))

            async def on_complete(self, stage_name: str):
                events.append(("complete", stage_name))

        @stage(concurrency=1)
        async def identity(x: int) -> int:
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-hook-lifecycle", channel_capacity=10)
            .source([1])
            .then(identity)
            .then(sink)
            .with_hook(LifecycleHook())
            .build()
        )

        await pipe.run()
        start_events = [e for e in events if e[0] == "start"]
        complete_events = [e for e in events if e[0] == "complete"]
        assert ("start", "identity") in start_events
        assert ("start", "sink") in start_events
        assert ("complete", "identity") in complete_events
        assert ("complete", "sink") in complete_events

    async def test_partial_hook_implementation(self):
        """A hook that only implements on_start doesn't crash the pipeline."""

        class StartOnlyHook:
            async def on_start(self, stage_name: str):
                pass  # Just verifying no crash

        @stage(concurrency=1)
        async def identity(x: int) -> int:
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-hook-partial", channel_capacity=10)
            .source([1, 2, 3])
            .then(identity)
            .then(sink)
            .with_hook(StartOnlyHook())
            .build()
        )

        result = await pipe.run()
        assert result.completed


# ── Real-Time Metrics Callback Tests ──


class TestMetricsCallback:
    """Periodic metrics callback during pipeline execution."""

    async def test_metrics_callback_fires(self):
        """Async callback receives snapshots during a slow pipeline."""
        snapshots_received: list[list[dict[str, Any]]] = []

        async def on_metrics(snapshots: list[dict[str, Any]]) -> None:
            snapshots_received.append(snapshots)

        @stage(concurrency=1)
        async def slow_stage(x: int) -> int:
            await asyncio.sleep(0.15)
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-metrics-cb", channel_capacity=10)
            .source(range(5))
            .then(slow_stage)
            .then(sink)
            .on_metrics(on_metrics, interval=0.1)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        # With 5 items at 0.15s each (~0.75s total), at 0.1s intervals we expect multiple calls
        assert len(snapshots_received) >= 1
        # Each snapshot list has one entry per stage
        assert len(snapshots_received[0]) == 2

    async def test_sync_metrics_callback(self):
        """Plain sync function works as a metrics callback."""
        call_count = 0

        def on_metrics(snapshots: list[dict[str, Any]]) -> None:
            nonlocal call_count
            call_count += 1

        @stage(concurrency=1)
        async def slow_stage(x: int) -> int:
            await asyncio.sleep(0.15)
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-sync-metrics-cb", channel_capacity=10)
            .source(range(5))
            .then(slow_stage)
            .then(sink)
            .on_metrics(on_metrics, interval=0.1)
            .build()
        )

        result = await pipe.run()
        assert result.completed
        assert call_count >= 1


# ── Force-Kill Escalation Tests ──


class TestForceKillEscalation:
    """Second SIGINT/SIGTERM forces immediate cancellation of all workers."""

    async def test_second_signal_sets_force(self):
        """ShutdownCoordinator: second request_shutdown() sets force event."""
        from weir.shutdown import ShutdownCoordinator

        coord = ShutdownCoordinator()
        assert not coord.should_stop
        assert not coord.should_force

        coord.request_shutdown()
        assert coord.should_stop
        assert not coord.should_force

        coord.request_shutdown()
        assert coord.should_force

    async def test_force_kill_cancels_workers(self):
        """Pipeline exits quickly on double-signal (force-kill)."""

        @stage(concurrency=1)
        async def hang(x: int) -> int:
            await asyncio.sleep(100)  # Block forever
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-force-kill", channel_capacity=10)
            .source(range(5))
            .then(hang)
            .then(sink)
            .build()
        )

        assert pipe._coordinator is not None

        async def trigger_force_kill():
            await asyncio.sleep(0.1)
            pipe._coordinator.request_shutdown()  # First signal: graceful
            await asyncio.sleep(0.1)
            pipe._coordinator.request_shutdown()  # Second signal: force

        trigger_task = asyncio.create_task(trigger_force_kill())
        result = await pipe.run()
        await trigger_task
        # Pipeline should have exited (not hung for 100s)
        assert result is not None


# ── Pipeline Reset / Re-runnability Tests ──


class TestPipelineReset:
    """Pipeline reset: re-run, new source, dead letter clearing, guard rails, fresh metrics."""

    async def test_reset_and_rerun(self):
        """Run, reset, run again with the same source."""
        results: list[int] = []

        @stage(concurrency=1)
        async def double(x: int) -> int:
            return x * 2

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-reset", channel_capacity=10)
            .source([1, 2, 3])
            .then(double)
            .then(collect)
            .build()
        )

        r1 = await pipe.run()
        assert r1.completed
        assert sorted(results) == [2, 4, 6]

        results.clear()
        pipe.reset()
        r2 = await pipe.run()
        assert r2.completed
        assert sorted(results) == [2, 4, 6]

    async def test_reset_with_new_source(self):
        """Reset with a different source replaces the data."""
        results: list[int] = []

        @stage(concurrency=1)
        async def collect(x: int) -> None:
            results.append(x)

        pipe = (
            Pipeline("test-reset-source", channel_capacity=10).source([1, 2]).then(collect).build()
        )

        await pipe.run()
        assert sorted(results) == [1, 2]

        results.clear()
        pipe.reset(new_source=[10, 20, 30])
        await pipe.run()
        assert sorted(results) == [10, 20, 30]

    async def test_reset_clears_dead_letters(self):
        """Dead letters are zeroed on reset."""

        @stage(concurrency=1, retries=1)
        async def fail_evens(x: int) -> int:
            if x % 2 == 0:
                raise ValueError("even")
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-reset-dl", channel_capacity=10)
            .source([1, 2, 3, 4])
            .then(fail_evens)
            .then(sink)
            .build()
        )

        r1 = await pipe.run()
        assert r1.dead_letters == 2

        pipe.reset()
        r2 = await pipe.run()
        # Dead letters from run 1 were cleared; only run 2's count
        assert r2.dead_letters == 2

    async def test_reset_before_build_raises(self):
        """RuntimeError on unbuilt pipeline."""

        @stage(concurrency=1)
        async def noop(x: int) -> int:
            return x

        pipe = Pipeline("test-reset-guard").source([1]).then(noop)
        with pytest.raises(RuntimeError, match="hasn't been built"):
            pipe.reset()

    async def test_metrics_fresh_after_reset(self):
        """Metrics don't accumulate across runs."""

        @stage(concurrency=1)
        async def identity(x: int) -> int:
            return x

        @stage(concurrency=1)
        async def sink(x: int) -> None:
            pass

        pipe = (
            Pipeline("test-reset-metrics", channel_capacity=10)
            .source(range(10))
            .then(identity)
            .then(sink)
            .build()
        )

        r1 = await pipe.run()
        assert r1.stage_metrics[0]["items_in"] == 10

        pipe.reset(new_source=range(5))
        r2 = await pipe.run()
        # Should be 5, not 15
        assert r2.stage_metrics[0]["items_in"] == 5
