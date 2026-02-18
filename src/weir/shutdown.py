"""Graceful shutdown coordination.

Shutdown is the hardest part of any pipeline to get right. The failure mode
everyone hits: SIGINT fires, tasks get cancelled mid-flight, data is lost,
partial writes corrupt state. Or worse: the shutdown hangs forever because
of a deadlock between a producer waiting to put and a consumer that's already gone.

weir's shutdown protocol:

1. Signal received (SIGINT/SIGTERM) → set the shutdown event.
2. Source stops emitting new items.
3. Source sends STOP sentinels into the first channel (one per consumer worker).
4. Each stage worker, upon receiving STOP, finishes its current item,
   sends STOP sentinels to the *next* channel, and exits.
5. This cascades through the pipeline. Last stage drains, pipeline exits.

The key invariant: every item that entered a channel before shutdown
will be fully processed. No data loss.
"""

import asyncio
import contextlib
import logging
import signal
from typing import Any

logger = logging.getLogger("weir.shutdown")


class ShutdownCoordinator:
    """Coordinates graceful shutdown across all pipeline components.

    Usage:
        coordinator = ShutdownCoordinator()
        coordinator.install_signal_handlers()

        # In your run loop:
        if coordinator.should_stop:
            break

        # Wait for full drain:
        await coordinator.wait_for_completion()
    """

    def __init__(self, drain_timeout: float = 30.0) -> None:
        self._shutdown_event = asyncio.Event()
        self._completion_event = asyncio.Event()
        self._drain_timeout = drain_timeout
        self._original_handlers: dict[signal.Signals, Any] = {}

    @property
    def should_stop(self) -> bool:
        return self._shutdown_event.is_set()

    def request_shutdown(self) -> None:
        """Trigger shutdown. Idempotent."""
        if not self._shutdown_event.is_set():
            logger.info("Shutdown requested. Draining pipeline...")
            self._shutdown_event.set()

    def mark_complete(self) -> None:
        """Signal that the pipeline has fully drained."""
        self._completion_event.set()

    async def wait_for_shutdown(self) -> None:
        """Block until shutdown is requested."""
        await self._shutdown_event.wait()

    async def wait_for_completion(self) -> bool:
        """Wait for pipeline to drain, with timeout.

        Returns True if drain completed, False if timed out.
        """
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=self._drain_timeout)
            logger.info("Pipeline drained successfully.")
            return True
        except TimeoutError:
            logger.warning(
                "Pipeline drain timed out after %.1fs. Some items may be lost.",
                self._drain_timeout,
            )
            return False

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install SIGINT/SIGTERM handlers that trigger graceful shutdown.

        Only works on Unix. On Windows, falls back to less graceful handling.
        """
        _loop = loop or asyncio.get_running_loop()

        def _handler(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            logger.info("Received %s", sig_name)
            self.request_shutdown()

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._original_handlers[sig] = signal.getsignal(sig)
                _loop.add_signal_handler(sig, _handler, sig)
            logger.debug("Signal handlers installed for SIGINT/SIGTERM")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.debug("Signal handlers not available on this platform")

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, OSError):
                loop.remove_signal_handler(sig)
            if sig in self._original_handlers:
                signal.signal(sig, self._original_handlers[sig])
        self._original_handlers.clear()
