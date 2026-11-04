# tier: mlx
"""Regression test: abandoning a stream must not wedge the MLX worker.

`nmv.runtime` funnels every MLX call through one process-wide worker thread.
`iterate` hands its producer a bounded `emit`, so a consumer that walks away
cannot block it — but the sentinel put in the producer's `finally` was
originally a plain `channel.put()`. Whenever a consumer abandoned a *full*
channel, that call blocked forever, and with a single worker it took down model
loading and generation for every browser session until the server restarted.

Everything here is deadlock-prone by nature, so each check runs on its own
thread with a join timeout: a regression fails the run instead of hanging it.

Run:  uv run python tests/test_runtime_abandon.py
"""

import sys
import threading
import time
from contextvars import ContextVar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import runtime, trace  # noqa: E402

CHANNEL_LIMIT = 128  # nmv.runtime.iterate's queue maxsize
PROBE_TIMEOUT = 15.0


def worker_responds(timeout: float = PROBE_TIMEOUT) -> bool:
    """True if the shared MLX worker can still accept and finish a task."""
    answered = threading.Event()

    def probe() -> None:
        runtime.call(lambda: None)
        answered.set()

    thread = threading.Thread(target=probe, daemon=True)
    thread.start()
    thread.join(timeout)
    return answered.is_set()


print("abandoning a full channel")


def flood(emit) -> None:
    for i in range(CHANNEL_LIMIT * 20):
        if not emit(i):
            return


gen = runtime.iterate(flood)
next(gen)
time.sleep(1.0)  # let the producer saturate the channel
gen.close()  # what Streamlit does when it interrupts a script
check(
    "worker survives an abandoned full stream",
    worker_responds(),
    f"probe answered within {PROBE_TIMEOUT:.0f}s",
)

print("normal completion")
drained: list = []


def drain() -> None:
    def counter(emit) -> None:
        for i in range(500):
            emit(i)

    drained.extend(runtime.iterate(counter))


t = threading.Thread(target=drain, daemon=True)
t.start()
t.join(PROBE_TIMEOUT)
check(
    "a fully consumed stream yields every item",
    drained == list(range(500)),
    f"{len(drained)} items",
)

print("producer failure")
raised: list = []


def expect_raise() -> None:
    def boom(emit):
        emit("first")
        raise RuntimeError("producer exploded")

    try:
        list(runtime.iterate(boom))
    except RuntimeError as error:
        raised.append(str(error))


t = threading.Thread(target=expect_raise, daemon=True)
t.start()
t.join(PROBE_TIMEOUT)
check(
    "a producer exception reaches the consumer",
    raised == ["producer exploded"],
    str(raised),
)
check("worker still usable after a failure", worker_responds())

print("run context")
marker: ContextVar[str] = ContextVar("marker", default="unset")
marker.set("from the page")
run = trace.start("chat", "abc123")


def on_worker() -> tuple[str, str | None]:
    with trace.span("parse"):
        time.sleep(0.01)
    current = trace.active()
    return marker.get(), current.run_id if current else None


seen_marker, seen_run = runtime.submit(on_worker).result(PROBE_TIMEOUT)
record = trace.finish(run)
check("a context variable reaches the job", seen_marker == "from the page", seen_marker)
check("so does the run", seen_run == run.run_id, str(seen_run))
check(
    "a span added on the worker is visible page-side",
    record.spans.get("parse", 0.0) >= 0.01,
    str(record.spans),
)
check("queue wait is measured", record.queue_wait_s >= 0.0, str(record.queue_wait_s))

print("waiting behind another job")
release = threading.Event()
blocker = runtime.submit(release.wait, PROBE_TIMEOUT)
run = trace.start("chat")
waiting = runtime.submit(lambda: None)
time.sleep(0.05)
release.set()
waiting.result(PROBE_TIMEOUT)
blocker.result(PROBE_TIMEOUT)
record = trace.finish(run)
check("the wait is the queue's", record.queue_wait_s >= 0.05, str(record.queue_wait_s))

finish()
