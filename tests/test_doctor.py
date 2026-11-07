"""The preflight check: every boundary, and the facts it reads rather than copies.

Run:  uv run python tests/test_doctor.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from nmv.model import MODEL_REPO, MODEL_REVISION  # noqa: E402
from nmv.reader import READER_REPO, READER_REVISION  # noqa: E402
from scripts.doctor import (  # noqa: E402
    FAIL,
    GB,
    GIB,
    OK,
    WARN,
    Result,
    check_disk,
    check_machine,
    check_macos,
    check_memory,
    check_uv,
    check_weights,
    macos_floor,
    prefetch_command,
    read_pins,
    report,
)

FLOOR = (14, 0)
HERE = Path("/Volumes/Data")

print("machine")
check("arm64 passes", check_machine("arm64").status == OK)
check("x86_64 fails", check_machine("x86_64").status == FAIL)
check("the failure says why", "Apple Silicon" in check_machine("x86_64").fix)

print("macOS")
check("13.6 fails", check_macos("13.6", FLOOR).status == FAIL)
check("14.0 passes", check_macos("14.0", FLOOR).status == OK)
check("14 alone passes", check_macos("14", FLOOR).status == OK)
check("a two-digit major passes", check_macos("27.0.1", FLOOR).status == OK)
check("no version fails", check_macos("", FLOOR).status == FAIL)
check("the failure has a fix", check_macos("13.6", FLOOR).fix != "")

print("memory")
check("15.99 GiB warns", check_memory(int(15.99 * GIB)).status == WARN)
check("16 GiB passes", check_memory(16 * GIB).status == OK)
check("unreadable memory warns", check_memory(None).status == WARN)
check("the warning names Ask documents", "Ask" in check_memory(8 * GIB).detail)

print("disk")
check("9.9 GB fails", check_disk(int(9.9 * GB), HERE).status == FAIL)
check("10 GB passes", check_disk(int(10 * GB), HERE).status == OK)
check("the failure names HF_HOME", "HF_HOME" in check_disk(0, HERE).fix)

print("uv")
check("found passes", check_uv("/opt/uv").status == OK)
check("missing fails", check_uv(None).status == FAIL)
check("the failure links the installer", "https://" in check_uv(None).fix)

print("weights")
PINS = {"org/a": "1" * 40, "org/b": "2" * 40}
check("all cached passes", check_weights(PINS, []).status == OK)
one = check_weights(PINS, ["org/b"])
check("one missing warns", one.status == WARN)
check("only the missing repo is listed", prefetch_command("org/b", "2" * 40) in one.fix)
check("the cached repo is not listed", "org/a" not in one.fix)
before = check_weights(PINS, None)
check("before uv sync warns", before.status == WARN)
check("before uv sync lists both", "org/a" in before.fix and "org/b" in before.fix)

print("report")
check("a warning still exits 0", report([Result("a", WARN, "b")]) == 0)
check("a failure exits 1", report([Result("a", OK, "b"), Result("c", FAIL, "d")]) == 1)

print("read, not copied")
pins = read_pins()
check("the NMV pin matches the code", pins.get(MODEL_REPO) == MODEL_REVISION)
check("the reader pin matches the code", pins.get(READER_REPO) == READER_REVISION)
check("both checkpoints are read", len(pins) == 2, str(sorted(pins)))
check("the floor from uv.lock is 14.0", macos_floor() == FLOOR, str(macos_floor()))

print("runs before uv sync")
tree = ast.parse((ROOT / "scripts" / "doctor.py").read_text(encoding="utf-8"))
top = [
    alias.name.split(".")[0] if isinstance(node, ast.Import) else str(node.module)
    for node in tree.body
    if isinstance(node, ast.Import | ast.ImportFrom)
    for alias in node.names
]
third_party = [name for name in top if name.split(".")[0] not in sys.stdlib_module_names]
check("only the standard library at module level", not third_party, str(third_party))

finish()
