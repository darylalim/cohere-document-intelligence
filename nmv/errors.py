"""Sort exceptions into what the user can do about them. No MLX, no Streamlit.

The order is the point: ``IMAGE_ERRORS`` includes ``OSError``, and so are a full disk
and every network failure, so ``bad_image`` has to come after both of them.
``tests/test_errors.py`` pins it.
"""

from __future__ import annotations

import errno
from dataclasses import dataclass

from nmv.imaging import IMAGE_ERRORS

# The strings in libmlx.dylib. A raised RuntimeError is all MLX gives us.
# "[METAL] Command buffer execution failed" comes from a Metal completion handler
# and may end the process instead of reaching Python at all.
_OUT_OF_MEMORY = (
    "[metal::malloc] Attempting to allocate",
    "[metal::malloc] Resource limit",
    "exceeds the maximum Metal buffer size",
    "[METAL] Command buffer execution failed",
)


@dataclass(frozen=True)
class Failure:
    kind: str
    title: str
    remedy: str


OUT_OF_MEMORY = Failure(
    "out_of_memory",
    "Ran out of GPU memory",
    "Lower the image budget or max tokens. On Ask documents, ask about fewer pages.",
)
DOWNLOAD = Failure(
    "download",
    "Could not reach Hugging Face",
    "Check the connection. The pinned weights download once, then work offline.",
)
DISK_FULL = Failure(
    "disk_full",
    "The disk is full",
    "Free some disk space. Model weights live in the Hugging Face cache, "
    "`~/.cache/huggingface`.",
)
BAD_IMAGE = Failure("bad_image", "That image could not be read", "Try another file.")
INTERNAL = Failure(
    "internal", "Something went wrong", "The details below are worth a bug report."
)


def _is_download_error(exc: BaseException) -> bool:
    # Here rather than at module scope, to keep this module cheap to import.
    from huggingface_hub import errors

    return isinstance(
        exc,
        (
            errors.LocalEntryNotFoundError,
            errors.RevisionNotFoundError,
            errors.RepositoryNotFoundError,
            errors.HfHubHTTPError,
            ConnectionError,
            TimeoutError,
        ),
    )


def classify(exc: BaseException) -> Failure:
    """Never raises, because it runs while reporting a failure."""
    try:
        if isinstance(exc, RuntimeError) and any(s in str(exc) for s in _OUT_OF_MEMORY):
            return OUT_OF_MEMORY
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            return DISK_FULL
        if _is_download_error(exc):
            return DOWNLOAD
        if isinstance(exc, IMAGE_ERRORS):
            return BAD_IMAGE
    except Exception:
        return INTERNAL
    return INTERNAL
