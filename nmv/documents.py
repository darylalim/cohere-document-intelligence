"""The Ask documents page's decisions, kept out of the script so they can be tested.

Pure: no MLX, no Streamlit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from nmv.domain import Derivation, Transcript

MATCH_THRESHOLD = 0.5
_DECORATION = " \t|#*>-"


def natural_key(name: str) -> list[tuple[int, int, str]]:
    """Sort key that puts ``page-2.png`` before ``page-10.png``."""
    return [
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.split(r"(\d+)", name)
        if part
    ]


def match_line(span: str, transcript: str) -> str:
    """The transcript line closest to a cited span, or the span itself.

    A citation wraps the *answer's* wording, which is often not on the page, so
    grounding it directly asks the model to find text that is not there.
    """
    best, best_ratio = span, MATCH_THRESHOLD
    wanted = span.lower()
    for raw_line in transcript.splitlines():
        line = raw_line.strip(_DECORATION)
        if not line:
            continue
        ratio = SequenceMatcher(None, wanted, line.lower(), autojunk=False).ratio()
        if ratio > best_ratio:
            best, best_ratio = line, ratio
    return best


def best_transcript(
    candidates: Iterable[Transcript], wanted: Derivation
) -> Transcript | None:
    """The highest-budget transcript of a page from the same model and prompt.

    The budget slider is session-wide, so lowering it on Grounding must not turn
    every transcript into a miss. Another model or prompt is a different result,
    and that is never served.
    """
    usable = [
        transcript
        for transcript in candidates
        if (made := transcript.derivation).page == wanted.page
        and made.model_revision == wanted.model_revision
        and made.prompt_version == wanted.prompt_version
        and made.max_tokens == wanted.max_tokens
    ]
    if not usable:
        return None
    return max(usable, key=lambda t: t.derivation.resolved_px)
