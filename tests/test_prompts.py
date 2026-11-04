"""A prompt's text cannot change without its version.

Run:  uv run python tests/test_prompts.py

When this fails because you reworded a prompt: bump its ``_VERSION``, then
update the version and hash here together.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import prompts  # noqa: E402

PINNED = {
    "TRANSCRIBE": (1, "1dbdc22cd3c61c6173ece7642714ca14b1b1768d59553a4128dfe1078d6c500a"),
    "LOCATE": (1, "a9db34d44a92de07c9fe49057c7ac15c1d27f970889cf22e2a1804c440c56eb3"),
}

print("prompt versions")
for name, (version, digest) in PINNED.items():
    text = getattr(prompts, f"{name}_PROMPT")
    actual = getattr(prompts, f"{name}_PROMPT_VERSION")
    check(f"{name} version is {version}", actual == version, str(actual))
    check(
        f"{name} text matches its version",
        hashlib.sha256(text.encode()).hexdigest() == digest,
    )

print("every prompt is pinned")
declared = {n.removesuffix("_PROMPT") for n in vars(prompts) if n.endswith("_PROMPT")}
check("no unpinned prompt", declared == set(PINNED), str(declared ^ set(PINNED)))

print("LOCATE_PROMPT")
filled = prompts.LOCATE_PROMPT.format(text="the PAID stamp")
check(
    "fills to the literal JSON shape",
    filled == 'Locate "the PAID stamp". Return a JSON list of '
    '{"bbox_2d": [x1, y1, x2, y2]}.',
    filled,
)

finish()
