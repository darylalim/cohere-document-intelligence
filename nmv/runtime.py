"""Every MLX call runs on one dedicated, long-lived thread.

MLX binds each thread's default stream to that thread, and an array that has
not been evaluated yet is work queued on one of those streams: only that
thread can evaluate it. Anywhere else raises

    RuntimeError: There is no Stream(gpu, 1) in current thread.

mlx-vlm's ``load()`` leaves arrays like that behind. It evaluates
``model.parameters()``, which skips underscore-named attributes, so the text
model's shared ``rotary_embeddings["sliding_attention"]._inv_freq`` stays
pending until the first forward pass reads it. Streamlit starts a fresh
ScriptRunner thread for each rerun the browser requests, and
``st.cache_resource`` loads on whichever rerun asks first, so page code that
drives mlx-vlm directly fails on its first reply. Running the import, the load
and every later call on a single worker means nothing pending ever changes
threads.

The arrangement pays for itself twice over: MLX generation is not reentrant,
so funnelling every session through one worker also stops two browser tabs
from trampling each other's decode.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Literal, TypedDict, TypeVar

from nmv import trace

T = TypeVar("T")

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")
_BOOT_LOCK = threading.Lock()
_booted = False

# How long a stalled producer waits between checks that the consumer is gone.
_PUT_TIMEOUT = 0.1

# MLX keeps freed buffers for reuse. With both models loaded on 16 GB that cache
# reached 3.89 GB on top of 9.43 GB live, past the working-set limit; at this size
# the worst case is about 11 GB, at no measurable cost in speed (docs/performance.md).
MLX_CACHE_LIMIT_BYTES = 1024**3

GB = 1e9
NO_RUN = "other"  # a job outside any run: a load at page render, or housekeeping

ModelName = Literal["nmv", "reader"]


class Snapshot(TypedDict):
    """What the worker last knew, so a page can show it without queueing behind a job.

    Only as fresh as the last job to end: the figures hold still during an answer.
    ``peak_gb`` is the peak since the last run's job began, and ``updated`` is
    ``None`` until a job has finished.
    """

    active_gb: float
    cache_gb: float
    peak_gb: float
    limit_gb: float
    models: dict[ModelName, bool]
    loads: int  # model loads since the server started, reloads included
    last_kind: str | None
    last_seconds: float
    updated: float | None


_state_lock = threading.Lock()
_snapshot: Snapshot = {
    "active_gb": 0.0,
    "cache_gb": 0.0,
    "peak_gb": 0.0,
    "limit_gb": 0.0,
    "models": {"nmv": False, "reader": False},
    "loads": 0,
    "last_kind": None,
    "last_seconds": 0.0,
    "updated": None,
}
_pending = 0


def _import_mlx() -> None:
    """First touch of both MLX libraries, on the worker, so any import-time MLX
    work is too."""
    import mlx_lm  # noqa: F401
    import mlx_vlm  # noqa: F401


def limit_cache() -> None:
    """Cap MLX's buffer cache. Call it on the worker, like all MLX work."""
    import mlx.core as mx

    mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)


def active_gb() -> float:
    """What MLX holds right now. Call it on the worker, like all MLX work."""
    import mlx.core as mx

    return mx.get_active_memory() / 1e9


def clear_cache() -> None:
    """Free MLX's cached buffers, not the models. Call it from page code."""

    def clear() -> None:
        import mlx.core as mx

        mx.clear_cache()

    call(clear)


def _memory() -> tuple[float, float, float, float]:
    """Active, cache, peak and limit. Runs on the worker: MLX's queries work from any
    thread, but monitoring must not be the precedent for an MLX call off it."""
    import mlx.core as mx

    # Fixed for the life of the process, so read once.
    limit = _snapshot["limit_gb"] or (
        int(mx.device_info()["max_recommended_working_set_size"]) / GB
    )
    return (
        mx.get_active_memory() / GB,
        mx.get_cache_memory() / GB,
        mx.get_peak_memory() / GB,
        limit,
    )


def _reset_peak() -> None:
    """Runs on the worker. MLX's peak otherwise only ever rises, so every run would
    report the highest figure since the server started."""
    import mlx.core as mx

    mx.reset_peak_memory()


def _publish(kind: str, seconds: float) -> None:
    """Runs on the worker as a job ends. It sits in a ``finally``, so a failure here
    must not replace the job's own result or exception."""
    try:
        memory = _memory()
    except Exception as error:
        logger.warning("could not read MLX memory: %s", type(error).__name__)
        memory = None
    with _state_lock:
        if memory is not None:
            active, cache, peak, limit = memory
            _snapshot["active_gb"] = active
            _snapshot["cache_gb"] = cache
            _snapshot["peak_gb"] = peak
            _snapshot["limit_gb"] = limit
        _snapshot["last_kind"] = kind
        _snapshot["last_seconds"] = seconds
        _snapshot["updated"] = time.monotonic()


def mark_loaded(name: ModelName, loaded: bool = True) -> None:
    """Record that a model is resident. The job's end publishes the memory it took."""
    with _state_lock:
        _snapshot["models"][name] = loaded
        if loaded:
            _snapshot["loads"] += 1


def snapshot() -> Snapshot:
    """A copy of the latest snapshot. Never touches the worker, so it never blocks."""
    with _state_lock:
        return {**_snapshot, "models": dict(_snapshot["models"])}


def pending() -> int:
    """Jobs submitted and not yet finished, counting the one that is running."""
    with _state_lock:
        return _pending


def _job_done(_: Future) -> None:
    global _pending
    with _state_lock:
        _pending -= 1


def _ensure_booted() -> None:
    global _booted
    if _booted:
        return
    with _BOOT_LOCK:
        if not _booted:
            _EXECUTOR.submit(_import_mlx).result()
            _booted = True


def submit(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
    """Queue work on the MLX worker. Never call this from the worker itself.

    The job runs in a copy of the caller's context, which is how the worker finds
    the current run: a bare executor submit arrives with every default instead.
    """
    global _pending
    _ensure_booted()
    context = copy_context()
    submitted = trace.clock()

    def job() -> T:
        context.run(trace.job_started, submitted)
        run = context.run(trace.active)
        # Only for a run: scripts/memory_check.py reads the peak across several jobs,
        # from a job of its own.
        if run is not None:
            _reset_peak()
        began = trace.clock()
        try:
            return context.run(fn, *args, **kwargs)
        finally:
            _publish(run.kind if run is not None else NO_RUN, trace.clock() - began)

    with _state_lock:
        _pending += 1
    future = _EXECUTOR.submit(job)
    # On the future and not in job(), so it counts jobs that raise as well.
    future.add_done_callback(_job_done)
    return future


def call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on the MLX worker and wait for its result."""
    return submit(fn, *args, **kwargs).result()


def iterate(
    produce: Callable[[Callable[[Any], bool]], object],
) -> Generator[Any, None, None]:
    """Stream items out of a worker-side producer.

    ``produce`` is handed an ``emit(item) -> bool`` callback and runs on the
    worker; this generator yields each emitted item on the calling thread.
    ``emit`` returns False once the consumer has walked away, which lets a
    generation loop bail out instead of wedging the single worker on a full
    queue — the case that matters when a user navigates away mid-response.

    The return type is ``Generator`` rather than ``Iterator`` because closing
    it is part of the contract: that is how Streamlit signals abandonment when
    it interrupts a script mid-stream.
    """
    channel: queue.Queue = queue.Queue(maxsize=128)
    finished = object()
    abandoned = threading.Event()

    def emit(item: Any) -> bool:
        while not abandoned.is_set():
            try:
                channel.put(item, timeout=_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

    def run() -> None:
        try:
            produce(emit)
        except BaseException as error:
            # Hand the failure to the consumer. If nobody is listening any more
            # the traceback would otherwise vanish silently, so log it instead.
            if not emit(error):
                logger.exception("MLX worker failed after the consumer left")
        finally:
            # Bounded, exactly like every other put. A plain `channel.put()`
            # here blocks forever when the consumer abandons a full channel —
            # and because the executor has a single worker, that wedges MLX for
            # every session in the process until the server restarts.
            emit(finished)

    future = submit(run)
    try:
        while True:
            item = channel.get()
            if item is finished:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        abandoned.set()

    future.result()
