"""weir — Lightweight async data pipeline framework.

The public API is intentionally small:

    from weir import Pipeline, stage

That's all you need for most pipelines. Everything else is available
if you need it, but you probably don't.
"""

from .batch import BatchStageConfig, BatchStageFunction, batch_stage
from .channel import Channel, ChannelClosedError
from .errors import (
    DeadLetterCollector,
    ErrorRouter,
    FailedItem,
    RetryPolicy,
    StageProcessingError,
)
from .logging import configure_logging
from .metrics import StageMetrics
from .pipeline import Pipeline, PipelineResult
from .stage import StageConfig, StageFunction, stage

__all__ = [
    "BatchStageConfig",
    "BatchStageFunction",
    "Channel",
    "ChannelClosedError",
    "DeadLetterCollector",
    "ErrorRouter",
    "FailedItem",
    "Pipeline",
    "PipelineResult",
    "RetryPolicy",
    "StageConfig",
    "StageFunction",
    "StageMetrics",
    "StageProcessingError",
    "batch_stage",
    "configure_logging",
    "stage",
]

__version__ = "0.1.0"
