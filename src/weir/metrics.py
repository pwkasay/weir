"""Per-stage metrics collection.

Every stage automatically records throughput, latency, error rates, and queue depth.
Metrics are the *consequence* of the architecture, not an afterthought bolted on.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .channel import Channel


@dataclass
class LatencyHistogram:
    """Simple streaming histogram using reservoir sampling (Algorithm R).

    Not trying to be HdrHistogram. Just enough to report p50/p95/p99
    without keeping every observation in memory.
    """

    _values: list[float] = field(default_factory=list)
    _max_size: int = 1000
    _total_seen: int = 0

    def record(self, value: float) -> None:
        self._total_seen += 1
        if len(self._values) < self._max_size:
            self._values.append(value)
        else:
            # Algorithm R: include Nth item with probability k/N
            idx = random.randint(0, self._total_seen - 1)
            if idx < self._max_size:
                self._values[idx] = value

    def percentile(self, p: float) -> float | None:
        """Return the p-th percentile (0-100). None if no observations."""
        if not self._values:
            return None
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def count(self) -> int:
        return self._total_seen

    def reset(self) -> None:
        self._values.clear()
        self._total_seen = 0


@dataclass
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
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _input_channel: Channel[Any] | None = None

    async def record_success(self, duration: float) -> None:
        async with self._lock:
            self.items_in += 1
            self.items_out += 1
            self.latency.record(duration)

    async def record_error(self, duration: float) -> None:
        async with self._lock:
            self.items_in += 1
            self.items_errored += 1
            self.latency.record(duration)

    def record_retry(self) -> None:
        self.items_retried += 1

    @property
    def error_rate(self) -> float:
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

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of current metrics."""
        snap: dict[str, Any] = {
            "stage": self.stage_name,
            "items_in": self.items_in,
            "items_out": self.items_out,
            "items_errored": self.items_errored,
            "items_retried": self.items_retried,
            "error_rate": round(self.error_rate, 4),
            "throughput_per_sec": round(self.throughput, 2),
            "latency_p50": self.latency.percentile(50),
            "latency_p95": self.latency.percentile(95),
            "latency_p99": self.latency.percentile(99),
        }
        if self._input_channel is not None:
            snap["queue_depth"] = self._input_channel.depth
            snap["queue_capacity"] = self._input_channel.capacity
            cap = self._input_channel.capacity
            snap["queue_utilization"] = round(self._input_channel.depth / cap, 4) if cap else 0.0
        return snap

    def __repr__(self) -> str:
        snap = self.snapshot()
        p50 = snap["latency_p50"]
        lat_str = f"{p50 * 1000:.1f}ms" if p50 is not None else "n/a"
        return (
            f"<StageMetrics {self.stage_name}: "
            f"{self.items_out}/{self.items_in} ok, "
            f"{self.items_errored} err, "
            f"p50={lat_str}, "
            f"{self.throughput:.1f}/s>"
        )
