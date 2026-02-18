"""Per-stage metrics collection.

Every stage automatically records throughput, latency, error rates, and queue depth.
Metrics are the *consequence* of the architecture, not an afterthought bolted on.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from .channel import Channel


class StageMetricsSnapshot(TypedDict):
    """Typed dictionary for stage metrics snapshots."""

    stage: str
    items_in: int
    items_out: int
    items_errored: int
    items_retried: int
    error_rate: float
    throughput_per_sec: float
    latency_p50: float | None
    latency_p95: float | None
    latency_p99: float | None
    queue_depth: NotRequired[int]
    queue_capacity: NotRequired[int]
    queue_utilization: NotRequired[float]


@dataclass(slots=True)
class LatencyHistogram:
    """Simple streaming histogram using reservoir sampling (Algorithm R).

    Not trying to be HdrHistogram. Just enough to report p50/p95/p99
    without keeping every observation in memory.
    """

    _values: list[float] = field(default_factory=list)
    _max_size: int = 1000
    _total_seen: int = 0

    def record(self, value: float) -> None:
        """Record a latency observation using reservoir sampling (Algorithm R)."""
        self._total_seen += 1
        if len(self._values) < self._max_size:
            self._values.append(value)
        else:
            # Algorithm R: include Nth item with probability k/N
            idx = int(random.random() * self._total_seen)
            if idx < self._max_size:
                self._values[idx] = value

    def percentiles(self, *ps: float) -> list[float | None]:
        """Return multiple percentiles (0-100) with a single sort. None if no observations."""
        if not self._values:
            return [None] * len(ps)
        sorted_vals = sorted(self._values)
        n = len(sorted_vals)
        return [sorted_vals[min(int(n * p / 100), n - 1)] for p in ps]

    @property
    def count(self) -> int:
        """Total number of observations recorded, including evicted samples."""
        return self._total_seen

    def reset(self) -> None:
        """Clear all recorded observations and reset the counter."""
        self._values.clear()
        self._total_seen = 0


@dataclass(slots=True)
class StageMetrics:
    """Metrics for a single pipeline stage.

    These are collected automatically by the stage runner. You never
    need to touch them directly unless you're building a dashboard.
    """

    stage_name: str
    items_in: int = 0
    items_out: int = 0
    items_errored: int = 0
    items_retried: int = 0
    latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    _started_at: float = field(default_factory=time.monotonic)
    _input_channel: Channel[Any] | None = None

    def record_success(self, duration: float) -> None:
        """Record a successful item processing with its duration in seconds."""
        self.items_in += 1
        self.items_out += 1
        self.latency.record(duration)

    def record_error(self, duration: float) -> None:
        """Record a failed item processing with its duration in seconds."""
        self.items_in += 1
        self.items_errored += 1
        self.latency.record(duration)

    def record_retry(self) -> None:
        """Increment the retry counter."""
        self.items_retried += 1

    @property
    def error_rate(self) -> float:
        """Fraction of input items that errored, from 0.0 to 1.0."""
        if self.items_in == 0:
            return 0.0
        return self.items_errored / self.items_in

    @property
    def throughput(self) -> float:
        """Items per second since metrics started."""
        elapsed = time.monotonic() - self._started_at
        if elapsed == 0:
            return 0.0
        return self.items_out / elapsed

    def snapshot(self) -> StageMetricsSnapshot:
        """Return a JSON-serializable snapshot of current metrics."""
        p50, p95, p99 = self.latency.percentiles(50, 95, 99)
        snap: StageMetricsSnapshot = {
            "stage": self.stage_name,
            "items_in": self.items_in,
            "items_out": self.items_out,
            "items_errored": self.items_errored,
            "items_retried": self.items_retried,
            "error_rate": round(self.error_rate, 4),
            "throughput_per_sec": round(self.throughput, 2),
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
        }
        if self._input_channel is not None:
            snap["queue_depth"] = self._input_channel.depth
            snap["queue_capacity"] = self._input_channel.capacity
            cap = self._input_channel.capacity
            snap["queue_utilization"] = round(self._input_channel.depth / cap, 4) if cap else 0.0
        return snap

    def __repr__(self) -> str:
        """Return a compact summary of throughput, errors, and latency."""
        p50 = self.latency.percentiles(50)[0]
        lat_str = f"{p50 * 1000:.1f}ms" if p50 is not None else "n/a"
        return (
            f"<StageMetrics {self.stage_name}: "
            f"{self.items_out}/{self.items_in} ok, "
            f"{self.items_errored} err, "
            f"p50={lat_str}, "
            f"{self.throughput:.1f}/s>"
        )
