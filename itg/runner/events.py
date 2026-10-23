"""Event logging: a human console stream plus per-case/per-stage JSONL files."""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("itg")
# Set here, not in `configure_console`: the per-case JSONL handlers must work
# whether or not a caller wants console output.
LOGGER.setLevel(logging.INFO)


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(payload, sort_keys=True, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        rendered = " ".join(
            f"{key}={value!r}" for key, value in fields.items() if key != "traceback"
        )
        return f"[itg] {record.getMessage()} {rendered}".rstrip()


def configure_console() -> None:
    if any(type(handler) is logging.StreamHandler for handler in LOGGER.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(ConsoleFormatter())
    LOGGER.addHandler(handler)


def log_event(event: str, **fields: Any) -> None:
    LOGGER.info(event, extra={"fields": fields})


@contextmanager
def log_to(path: Path) -> Iterator[None]:
    """Append this block's events to `path` as JSONL, keeping earlier attempts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(JsonlFormatter())
    LOGGER.addHandler(handler)
    try:
        yield
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


@contextmanager
def stage(case_dir: Path, name: str) -> Iterator[None]:
    started = time.perf_counter()
    with ExitStack() as scope:
        scope.enter_context(log_to(case_dir / "logs" / f"{name}.log"))
        log_event("stage:begin", stage=name)
        try:
            yield
        except BaseException as error:
            log_event(
                "stage:error",
                stage=name,
                duration_s=round(time.perf_counter() - started, 3),
                exception=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
            raise
        log_event(
            "stage:finish", stage=name, duration_s=round(time.perf_counter() - started, 3)
        )
