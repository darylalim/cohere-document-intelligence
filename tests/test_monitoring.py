"""The monitoring data a page reads without asking the worker.

Run:  uv run python tests/test_monitoring.py

Importing ``nmv.runtime`` loads no MLX; that starts with the first ``submit``. So
nothing here submits a job. What needs a real job is in ``test_runtime_abandon.py``.
"""

import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import runtime, system  # noqa: E402

print("sysctl output")
SWAP = "total = 2048.00M  used = 1340.50M  free = 707.50M  (encrypted)\n"
check("pressure parses", system.parse_pressure("1\n") == 1)
check("so does critical", system.parse_pressure("4") == 4)
check("swap parses", system.parse_swap_used(SWAP) == 1340.5)
check("a zero is a reading", system.parse_swap_used("used = 0.00M") == 0.0)
check("gigabytes convert", system.parse_swap_used("used = 1.50G") == 1536.0)
for bad in ("", "\n", "banana", "3", "-1", "1.5", "٢", None):
    check(f"pressure {bad!r} is None", system.parse_pressure(bad) is None)
for bad in ("", "banana", "total = 2048.00M", "used = M", None):
    check(f"swap {bad!r} is None", system.parse_swap_used(bad) is None)


def ran(stdout: str = "", returncode: int = 0):
    return lambda *args, **kwargs: subprocess.CompletedProcess(
        args, returncode, stdout, ""
    )


def raises(error: Exception):
    def fail(*args, **kwargs):
        raise error

    return fail


with patch("subprocess.run", ran("2\n")):
    check("a reading comes through", system.pressure_level() == 2)
with patch("subprocess.run", ran(SWAP)):
    check("so does swap", system.swap_used_mb() == 1340.5)
with patch("subprocess.run", ran("2\n", returncode=1)):
    check("a failed command is None", system.pressure_level() is None)
for error in (subprocess.TimeoutExpired("sysctl", 1), FileNotFoundError("sysctl")):
    with patch("subprocess.run", raises(error)):
        name = type(error).__name__
        check(f"{name} is None", system.pressure_level() is None)
        check(f"{name} is None for swap", system.swap_used_mb() is None)

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
