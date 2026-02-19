"""weir — Lightweight async data pipeline framework.

The public API is intentionally small:

    from weir import Pipeline, stage

That's all you need for most pipelines. Everything else is available
if you need it, but you probably don't.
"""

from .batch import BatchStageConfig, BatchStageFunction, batch_stage
from .channel import Channel, ChannelClosedError, is_stop_signal
from .errors import (
    DeadLetterCollector,
    ErrorRouter,
    FailedItem,
    RetryConfig,
    RetryPolicy,
    StageProcessingError,
)
from .hooks import Hook
from .logging import configure_logging
from .metrics import StageMetrics, StageMetricsSnapshot
from .pipeline import Pipeline, PipelineResult
from .runner import BaseStageRunner
from .stage import StageConfig, StageFunction, stage

__all__ = [
    "BaseStageRunner",
    "BatchStageConfig",
    "BatchStageFunction",
    "Channel",
    "ChannelClosedError",
    "DeadLetterCollector",
    "ErrorRouter",
    "FailedItem",
    "Hook",
    "Pipeline",
    "PipelineResult",
    "RetryConfig",
    "RetryPolicy",
    "StageConfig",
    "StageFunction",
    "StageMetrics",
    "StageMetricsSnapshot",
    "StageProcessingError",
    "batch_stage",
    "configure_logging",
    "is_stop_signal",
    "stage",
]

__version__ = "0.4.0"
