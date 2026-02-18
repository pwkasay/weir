"""Structured logging for weir.

Opinionated defaults:
- JSON lines in production (structured, parseable, grep-friendly)
- Human-readable in development (because JSON is miserable to read in a terminal)
- Stage name and pipeline name automatically injected into every log line

The logging module is intentionally minimal. We're not building a logging framework.
We're setting up sensible defaults so every stage gets consistent, useful logs
without anyone thinking about it.
"""

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class StructuredFormatter(logging.Formatter):
    """JSON Lines formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Inject pipeline/stage context if present
        pipeline = getattr(record, "pipeline", None)
        if pipeline is not None:
            log_entry["pipeline"] = pipeline
        stage = getattr(record, "stage", None)
        if stage is not None:
            log_entry["stage"] = stage

        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry)


class HumanFormatter(logging.Formatter):
    """Readable formatter for development."""

    COLORS: dict[str, str] = {  # noqa: RUF012
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        parts = [
            f"{color}{record.levelname:<8}{reset}",
            f"{record.name}",
        ]

        # Add context
        pipeline = getattr(record, "pipeline", None)
        if pipeline is not None:
            parts.append(f"[{pipeline}]")
        stage = getattr(record, "stage", None)
        if stage is not None:
            parts.append(f"({stage})")

        parts.append(record.getMessage())

        line = " ".join(parts)

        if record.exc_info and record.exc_info[1]:
            line += f"\n  {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"

        return line


class PipelineLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that injects pipeline and stage context."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(
    level: int = logging.INFO,
    structured: bool = False,
) -> None:
    """Configure weir logging.

    Args:
        level: Log level (default INFO).
        structured: If True, use JSON lines format. If False, use human-readable.
    """
    root_logger = logging.getLogger("weir")
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter() if structured else HumanFormatter())
    root_logger.addHandler(handler)

    # Don't propagate to root logger
    root_logger.propagate = False


def get_logger(pipeline_name: str, stage_name: str | None = None) -> PipelineLoggerAdapter:
    """Get a logger with pipeline/stage context baked in."""
    base_logger = logging.getLogger(f"weir.pipeline.{pipeline_name}")
    extra: dict[str, str] = {"pipeline": pipeline_name}
    if stage_name:
        extra["stage"] = stage_name
    return PipelineLoggerAdapter(base_logger, extra)
