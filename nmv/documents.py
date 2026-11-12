"""The Ask documents page's decisions, kept out of the script so they can be tested.

Pure: no MLX, no Streamlit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Protocol, TypeVar

from nmv.domain import Derivation, PageId, Transcript
from nmv.transcripts import Hit

MATCH_THRESHOLD = 0.5
_DECORATION = " \t|#*>-"
_SPAN_EDGES = " \t\n.,;:!?"


class _File(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def size(self) -> int: ...


F = TypeVar("F", bound=_File)


def carry_uploads(
    widget: Sequence[F], chosen: Sequence[F], kept: Sequence[F]
) -> tuple[list[F], list[F]]:
    """The files the user chose and the files kept from before, after this run.

    The uploader forgets its files when the user visits another page, and it has no
    ``persist_state``. ``chosen`` is what the user last picked, recorded by the
    uploader's ``on_change``, which a page switch does not fire. So an empty widget
    with files still chosen means the page switch emptied it: those files are kept.
    A kept file the user uploads again is dropped from ``kept``, so it is not listed
    twice. Same name and size is taken as the same file; a near miss only means the
    page ids match and the page list names it a duplicate.
    """
    carried = list(kept)
    current = list(chosen)
    if not widget and current:
        carried += current
        current = []
    uploaded = {(file.name, file.size) for file in widget}
    seen: set[tuple[str, int]] = set()
    unique = []
    for file in carried:
        ident = (file.name, file.size)
        if ident not in uploaded and ident not in seen:
            seen.add(ident)
            unique.append(file)
    return current, unique


def natural_key(name: str) -> list[tuple[int, int, str]]:
    """Sort key that puts ``page-2.png`` before ``page-10.png``."""
    return [
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.split(r"(\d+)", name)
        if part
    ]


def match_line(span: str, transcript: str) -> str:
    """The document's own text for a cited span.

    A citation wraps the *answer's* wording, which is often not on the page, so
    grounding it directly can ask the model to find text that is not there. A span
    that is on the page is sent as it stands, though, and never widened to its line:
    on the sample invoice ``$4,182.55`` gets a box, while ``Total due: $4,182.55``
    gets a meaningless point. Otherwise this is the closest line, or the span itself.
    """
    wanted = span.strip(_SPAN_EDGES).lower()
    at = transcript.lower().find(wanted) if wanted else -1
    if at != -1:
        return transcript[at : at + len(wanted)]

    best, best_ratio = span, MATCH_THRESHOLD
    for raw_line in transcript.splitlines():
        line = raw_line.strip(_DECORATION)
        if not line:
            continue
        ratio = SequenceMatcher(None, wanted, line.lower(), autojunk=False).ratio()
        if ratio > best_ratio:
            best, best_ratio = line, ratio
    return best


def first_per_page(hits: Iterable[Hit]) -> list[Hit]:
    """One hit per page, its best-ranked. A page saved at several budgets has a row
    for each, and listing them all would offer the same page several times."""
    seen: set[PageId] = set()
    unique = []
    for hit in hits:
        if hit.page not in seen:
            seen.add(hit.page)
            unique.append(hit)
    return unique


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
