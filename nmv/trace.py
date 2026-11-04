"""One record per user action: numbers, IDs and enum values, never content.

Logging, the System panel, the evals and Inspect all read the same ``RunRecord``, so
none of them keeps its own timings. A line in the log, the row in the panel and the
figure in an eval result are one measurement.

Two types, because a frozen record cannot be filled in from the worker.
``runtime.submit`` hands the worker a *copy* of the context, so a ``ContextVar.set``
there is never seen page-side. Both sides share the same ``RunBuilder`` object though,
so mutating it in place does flow back. Never set ``_current`` on the worker.

A run that Streamlit interrupts mid-stream never reaches ``finish``, so it leaves no
record. No MLX and no Streamlit here: the page passes in the session prefix.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

TRACE_FILE_ENV = "NMV_TRACE_FILE"
RING_SIZE = 10
NO_ID = "-"

Kind = Literal["chat", "ground", "transcribe", "ask", "locate"]
SpanName = Literal["prep", "prefill", "decode", "parse", "store"]
# Every classify() kind, plus the one failure that is not an exception worth classifying.
OUTCOMES = (
    "ok",
    "context_exceeded",
    "out_of_memory",
    "download",
    "disk_full",
    "bad_image",
    "internal",
)
STORE_STATES = ("hit", "fallback-budget", "miss", "not-saved", "n/a")

clock: Callable[[], float] = time.perf_counter


class Stats(Protocol):
    """What ``finish`` reads from ``model.RunStats``, without importing Streamlit."""

    @property
    def prompt_tokens(self) -> int: ...
    @property
    def generation_tokens(self) -> int: ...
    @property
    def prompt_tps(self) -> float: ...
    @property
    def generation_tps(self) -> float: ...
    @property
    def peak_memory(self) -> float: ...
    @property
    def finish_reason(self) -> str | None: ...


@dataclass(frozen=True)
class RunRecord:
    """Adding a free-text field here fails ``tests/test_trace.py``, on purpose.

    ``duration`` keeps OpenTelemetry's name, so moving to OTel later is a rename.
    """

    run_id: str
    session: str
    kind: str
    outcome: str = "ok"
    duration: float = 0.0
    queue_wait_s: float = 0.0
    first_token_s: float | None = None
    spans: dict[str, float] = field(default_factory=dict)
    image_tokens: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    thinking_tokens: int = 0
    budget_px: int = 0
    resolved_px: int = 0
    pages_used: int = 0
    pages_dropped: int = 0
    store: str = "n/a"
    prefix: str = "n/a"  # always n/a until prefix reuse is built
    peak_gb: float = 0.0
    finish_reason: str | None = None


# What a page may fill in. The rest comes from the clock, or from finish().
_SETTABLE = frozenset(
    {
        "image_tokens",
        "thinking_tokens",
        "budget_px",
        "resolved_px",
        "pages_used",
        "pages_dropped",
        "store",
    }
)


class RunBuilder:
    """The mutable half, shared by the page and the worker. Every write takes the lock."""

    def __init__(self, run_id: str, session: str, kind: Kind) -> None:
        self.run_id = run_id
        self.session = session
        self.kind = kind
        self._lock = threading.Lock()
        self._started = clock()
        self._queue_wait = 0.0
        self._first_token: float | None = None
        self._spans: dict[str, float] = {}
        self._values: dict[str, int | str] = {}

    def set(self, **values: int | str) -> None:
        unknown = values.keys() - _SETTABLE
        if unknown:
            raise TypeError(f"not a settable run field: {sorted(unknown)}")
        with self._lock:
            self._values.update(values)

    def add_span(self, name: SpanName, seconds: float) -> None:
        """Repeats add up, so a span names where the time went, not one event."""
        with self._lock:
            self._spans[name] = self._spans.get(name, 0.0) + seconds

    def add_queue_wait(self, seconds: float) -> None:
        """Summed, since one action can queue more than once: load, then generate."""
        with self._lock:
            self._queue_wait += seconds

    def mark_first_token(self) -> None:
        with self._lock:
            if self._first_token is None:
                self._first_token = clock() - self._started

    def freeze(self, outcome: str, stats: Stats | None) -> RunRecord:
        with self._lock:
            spans = dict(self._spans)
            measured: dict = {}
            if stats is not None:
                # MLX's own statistics, so nothing inside mlx-vlm or mlx-lm changes.
                if stats.prompt_tps:
                    spans["prefill"] = stats.prompt_tokens / stats.prompt_tps
                if stats.generation_tps:
                    spans["decode"] = stats.generation_tokens / stats.generation_tps
                measured = {
                    "prompt_tokens": stats.prompt_tokens,
                    "generated_tokens": stats.generation_tokens,
                    "peak_gb": stats.peak_memory,
                    "finish_reason": stats.finish_reason,
                }
            return RunRecord(
                run_id=self.run_id,
                session=self.session,
                kind=self.kind,
                outcome=outcome,
                duration=clock() - self._started,
                queue_wait_s=self._queue_wait,
                first_token_s=self._first_token,
                spans=spans,
                **measured,
                **self._values,
            )


_current: ContextVar[RunBuilder | None] = ContextVar("nmv_run", default=None)
_ring: deque[RunRecord] = deque(maxlen=RING_SIZE)
_ring_lock = threading.Lock()


def start(kind: Kind, session: str = NO_ID) -> RunBuilder:
    """Begin an action, page-side. Close it with ``finish``."""
    run = RunBuilder(secrets.token_hex(4), session, kind)
    _current.set(run)
    return run


def active() -> RunBuilder | None:
    return _current.get()


@contextmanager
def span(name: SpanName) -> Generator[None, None, None]:
    """Time a block into the current run. Does nothing when there is none."""
    run = _current.get()
    began = clock()
    try:
        yield
    finally:
        if run is not None:
            run.add_span(name, clock() - began)


def job_started(submitted: float) -> None:
    """Runs on the worker as a job begins, with the time ``submit`` took it."""
    run = _current.get()
    if run is not None:
        run.add_queue_wait(clock() - submitted)


def first_token() -> None:
    """Runs page-side as each chunk arrives; only the first one counts."""
    run = _current.get()
    if run is not None:
        run.mark_first_token()


def summary(record: RunRecord) -> str:
    """``key=value`` pairs for the log line. The filter adds the run and session."""
    parts = []
    for item in fields(record):
        value = getattr(record, item.name)
        if item.name in ("run_id", "session"):
            continue
        if item.name == "spans":
            parts += [f"{name}={seconds:.3f}" for name, seconds in value.items()]
        elif isinstance(value, float):
            parts.append(f"{item.name}={value:.3f}")
        else:
            parts.append(f"{item.name}={value}")
    return " ".join(parts)


def _append_to_file(path: str, record: RunRecord) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")
    except OSError as error:
        logger.warning("could not write the trace file: %s", type(error).__name__)


def finish(run: RunBuilder, outcome: str = "ok", stats: Stats | None = None) -> RunRecord:
    """Freeze the run and hand it to its consumers. Runs while reporting a failure,
    so a consumer that fails must not raise out of here."""
    record = run.freeze(outcome, stats)
    level = logging.INFO if outcome == "ok" else logging.WARNING
    logger.log(level, "%s", summary(record))
    with _ring_lock:
        _ring.append(record)
        path = os.environ.get(TRACE_FILE_ENV)
        if path:
            _append_to_file(path, record)
    if _current.get() is run:
        _current.set(None)
    return record


def recent() -> list[RunRecord]:
    """The last ``RING_SIZE`` records, oldest first, across every tab."""
    with _ring_lock:
        return list(_ring)


class RunFilter(logging.Filter):
    """Stamps ``run`` and ``session`` on every record, on either thread."""

    def filter(self, record: logging.LogRecord) -> bool:
        run = _current.get()
        record.run = run.run_id if run is not None else NO_ID
        record.session = run.session if run is not None else NO_ID
        return True
