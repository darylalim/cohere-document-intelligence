"""``errors.classify``: each failure gets its own remedy, and the order holds.

Run:  uv run python tests/test_errors.py
"""

import errno
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402
from huggingface_hub import errors as hf  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from nmv.errors import classify  # noqa: E402


def kind(exc: BaseException) -> str:
    return classify(exc).kind


print("out of memory")
for message in (
    "[metal::malloc] Attempting to allocate 21474836480 bytes which is greater",
    "[metal::malloc] Resource limit (499000) exceeded.",
    "Attempting to allocate 9 bytes, which exceeds the maximum Metal buffer size of 8",
    "[METAL] Command buffer execution failed: Insufficient Memory.",
):
    check(message[:40], kind(RuntimeError(message)) == "out_of_memory")
check(
    "the thread-affinity bug stays loud",
    kind(RuntimeError("There is no Stream(gpu, 1) in current thread.")) == "internal",
)
check(
    "only a RuntimeError", kind(KeyError("[metal::malloc] Resource limit")) == "internal"
)

print("disk and images")
check("ENOSPC", kind(OSError(errno.ENOSPC, "No space left on device")) == "disk_full")
check("a plain OSError", kind(OSError("cannot read")) == "bad_image")
check("an unidentified image", kind(UnidentifiedImageError("no")) == "bad_image")
check("a bomb", kind(Image.DecompressionBombError("big")) == "bad_image")

print("download")
response = Mock(headers={}, request=Mock(), status_code=404)
for exc in (
    hf.LocalEntryNotFoundError("missing"),
    hf.RevisionNotFoundError("missing", response=response),
    hf.RepositoryNotFoundError("missing", response=response),
    hf.HfHubHTTPError("failed", response=response),
    ConnectionError("down"),
    ConnectionResetError("reset"),
    TimeoutError("slow"),
):
    check(type(exc).__name__, kind(exc) == "download")

print("anything else")
check("an unknown KeyError", kind(KeyError("x")) == "internal")


class Hostile(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("no message for you")


check("classify never raises", kind(Hostile()) == "internal")

finish()
