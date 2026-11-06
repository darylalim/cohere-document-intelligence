"""The ``nmv`` handler: one of it, the right level, and the fields that diagnose.

Run:  uv run python tests/test_logging.py
"""

import io
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from nmv import log, trace  # noqa: E402

root = logging.getLogger("nmv")


def fresh(level: str | None = None) -> io.StringIO:
    """A first-time setup, with the handler writing to a buffer."""
    for handler in list(root.handlers):
        root.removeHandler(handler)
    os.environ.pop(log.LEVEL_ENV, None)
    if level is not None:
        os.environ[log.LEVEL_ENV] = level
    buffer = io.StringIO()
    real = sys.stderr
    sys.stderr = buffer  # StreamHandler binds stderr when it is made
    log.setup()
    sys.stderr = real
    return buffer


print("setup")
fresh()
log.setup()
check("twice leaves one handler", len(root.handlers) == 1, str(root.handlers))
check("it does not propagate", root.propagate is False)
check("INFO by default", root.level == logging.INFO)

print("the level")
fresh("DEBUG")
check("the variable sets it", root.level == logging.DEBUG)
fresh("debug")
check("in any case", root.level == logging.DEBUG)
output = fresh("LOUD")
check("an unknown value falls back", root.level == logging.INFO)
check("with one warning", output.getvalue().count("unknown NMV_LOG_LEVEL") == 1)

print("the format")
check("it has the thread", "%(threadName)" in log.FORMAT)
output = fresh()
child = logging.getLogger("nmv.model")
child.info("from a page")
check("a child's line has no run yet", "run=- session=- ·" in output.getvalue())

run = trace.start("chat", "abc123")
with ThreadPoolExecutor(thread_name_prefix="mlx-worker") as pool:
    pool.submit(copy_context().run, child.info, "from the worker").result()
last = output.getvalue().splitlines()[-1]
check("the worker's thread is named", "mlx-worker_0" in last, last)
check("its line carries the run", f"run={run.run_id} session=abc123" in last, last)

print("the run record's line")
trace.finish(run)
last = output.getvalue().splitlines()[-1]
check("it comes through nmv", "nmv.trace" in last and "kind=chat" in last, last)
check("it is stamped with its own run", f"run={run.run_id}" in last, last)

print("tracebacks")
SECRET = "Invoice for Jane Roe.png"


def fail() -> None:
    raise ValueError(SECRET)


output = fresh()
try:
    fail()
except ValueError as error:
    logging.getLogger("nmv.ui").error("internal", exc_info=error)
    logging.getLogger("nmv.runtime").exception("worker failed")
written = output.getvalue()
check("exc_info leaves the message out", SECRET not in written, written)
check("the frames stay", written.count("in fail") == 2, written)
check("the class stays", written.count("ValueError, message left out") == 2, written)

finish()
