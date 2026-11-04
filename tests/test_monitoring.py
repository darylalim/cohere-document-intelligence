"""The monitoring data a page reads without asking the worker.

Run:  uv run python tests/test_monitoring.py

Importing ``nmv.runtime`` loads no MLX; that starts with the first ``submit``. So
nothing here submits a job. What needs a real job is in ``test_runtime_abandon.py``.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import runtime  # noqa: E402

print("snapshot")
first = runtime.snapshot()
check("starts with nothing loaded", not any(first["models"].values()))
check("and no job seen", first["updated"] is None and first["last_kind"] is None)
check("nothing pending", runtime.pending() == 0)

first["models"]["nmv"] = True
first["active_gb"] = 99.0
second = runtime.snapshot()
check("a caller's edit stays in its copy", second["models"]["nmv"] is False)
check("at the top level too", second["active_gb"] == 0.0)

runtime.mark_loaded("reader")
check("a load shows", runtime.snapshot()["models"]["reader"] is True)
check("an earlier copy does not change", second["models"]["reader"] is False)
runtime.mark_loaded("reader", False)
check("so does an unload", runtime.snapshot()["models"]["reader"] is False)

print("several threads")
errors: list[BaseException] = []
stop = threading.Event()


def read() -> None:
    try:
        while not stop.is_set():
            latest = runtime.snapshot()
            assert set(latest["models"]) == {"nmv", "reader"}
    except BaseException as error:
        errors.append(error)


readers = [threading.Thread(target=read, daemon=True) for _ in range(4)]
for thread in readers:
    thread.start()
for number in range(2000):
    runtime.mark_loaded("nmv", number % 2 == 0)
stop.set()
for thread in readers:
    thread.join(5)
check("reads race writes safely", not errors, repr(errors[:1]))
check("every reader finished", not any(thread.is_alive() for thread in readers))

finish()
