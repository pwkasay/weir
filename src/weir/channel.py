"""Bounded async channels — the backpressure mechanism.

A Channel is just an asyncio.Queue with opinions:
- Always bounded (no infinite queues, ever)
- Supports a sentinel value for clean shutdown signaling
- Exposes depth for observability
- Configurable overflow strategies (for future use)

The key insight: backpressure in weir is *implicit*. When a channel is full,
the upstream stage's `put()` call awaits. The upstream stage's worker is blocked,
so it can't pull from *its* input channel. This propagates all the way back to
the source. No explicit signaling needed. Bounded queues do it for free.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum, auto


class _Sentinel(Enum):
    """Shutdown sentinel. When a stage sees this, it knows to stop."""

    STOP = auto()


STOP = _Sentinel.STOP


def is_stop_signal(item: object) -> bool:
    """Check whether an item is the shutdown sentinel."""
    return isinstance(item, _Sentinel)


@dataclass(slots=True)
class ChannelStats:
    """Observable state of a channel."""

    capacity: int
    current_depth: int
    total_put: int
    total_get: int

    @property
    def utilization(self) -> float:
        """Channel utilization as a ratio from 0.0 (empty) to 1.0 (full)."""
        if self.capacity == 0:
            return 0.0
        return self.current_depth / self.capacity


class Channel[T]:
    """A bounded async channel connecting two pipeline stages.

    This is the core backpressure primitive. When full, puts block.
    When empty, gets block. Shutdown is signaled by sending STOP sentinels.

    Args:
        capacity: Maximum items buffered. Must be > 0.
                  Lower = more responsive backpressure, higher = more throughput smoothing.
        name: Optional name for logging/debugging.
    """

    def __init__(self, capacity: int = 64, name: str | None = None) -> None:
        """Initialize a bounded channel.

        Args:
            capacity: Maximum items buffered. Must be >= 1.
            name: Optional name for logging and debugging.

        Raises:
            ValueError: If capacity < 1.
        """
        if capacity < 1:
            raise ValueError(f"Channel capacity must be >= 1, got {capacity}")
        self._queue: asyncio.Queue[T | _Sentinel] = asyncio.Queue(maxsize=capacity)
        self._capacity = capacity
        self._name = name or "channel"
        self._total_put = 0
        self._total_get = 0
        self._closed = False

    @property
    def name(self) -> str:
        """The channel's display name."""
        return self._name

    @property
    def depth(self) -> int:
        """Current number of items in the channel."""
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        """Maximum number of items the channel can hold."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Whether the channel is at capacity."""
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        """Whether the channel has no items."""
        return self._queue.empty()

    async def put(self, item: T) -> None:
        """Put an item into the channel. Blocks if full (backpressure)."""
        if self._closed:
            raise ChannelClosedError(f"Cannot put into closed channel '{self._name}'")
        await self._queue.put(item)
        self._total_put += 1

    async def get(self) -> T | _Sentinel:
        """Get an item from the channel. Blocks if empty.

        Returns STOP sentinel when the channel is being drained for shutdown.
        Callers must check for this.
        """
        item = await self._queue.get()
        self._total_get += 1
        return item

    async def send_stop(self, num_consumers: int = 1) -> None:
        """Send stop sentinels for each consumer worker.

        Call this when the upstream stage is done. Each consumer worker
        will receive one STOP and know to exit.
        """
        for _ in range(num_consumers):
            await self._queue.put(STOP)
        self._closed = True

    def stats(self) -> ChannelStats:
        """Return a snapshot of the channel's observable state."""
        return ChannelStats(
            capacity=self._capacity,
            current_depth=self.depth,
            total_put=self._total_put,
            total_get=self._total_get,
        )

    def __repr__(self) -> str:
        """Return a human-readable representation showing depth and capacity."""
        return f"<Channel '{self._name}' {self.depth}/{self._capacity}>"


class ChannelClosedError(Exception):
    """Raised when trying to put into a closed channel."""
