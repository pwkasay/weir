"""Lifecycle hooks — extensibility without modification.

Hooks let you observe pipeline behavior without changing stage functions.
Use cases: tracing, validation, logging, metrics dashboards.

All methods are optional — implement only what you need. The framework
checks at build time which methods exist and pre-filters, so there's
zero overhead for unimplemented methods.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Hook(Protocol):
    """Protocol for pipeline lifecycle hooks.

    All methods are optional. Implement only the ones you need.

    Methods:
        on_start: Called when a stage begins processing (workers launched).
        on_item: Called after each successfully processed item.
        on_error: Called when a stage permanently fails to process an item.
        on_complete: Called when a stage finishes (all workers done).
    """

    async def on_start(self, stage_name: str) -> None: ...
    async def on_item(self, stage_name: str, item: Any, result: Any, duration: float) -> None: ...
    async def on_error(self, stage_name: str, item: Any, error: Exception) -> None: ...
    async def on_complete(self, stage_name: str) -> None: ...
