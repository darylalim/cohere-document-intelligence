"""Scoring for the evals. Pure functions: no MLX, no Streamlit.

Metric bugs mislead silently, so ``tests/test_eval_metrics.py`` covers these
like any other code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

IOU_HIT = 0.5

_HTML_TAG = re.compile(r"<[^>]+>")
_RULE_ROW = re.compile(r"^[ \t|:-]*-[ \t|:-]*$", re.MULTILINE)
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.MULTILINE)
_EMPHASIS = re.compile(r"\*\*|__|`")

BBox = Sequence[float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def best_iou(truth: BBox, boxes: Iterable[BBox]) -> float | None:
    """The IoU of the best-matching box, or None if nothing parsed.

    None is a parse failure, reported apart from a miss (0.0) so a format
    problem is not mistaken for a localisation problem.
    """
    scores = [iou(truth, box) for box in boxes]
    return max(scores) if scores else None


def normalise(text: str) -> str:
    """Casefold and collapse whitespace, so CER measures reading and not layout."""
    return " ".join(text.casefold().split())


def strip_markup(text: str) -> str:
    """Remove the Markdown and HTML the transcription prompt invites.

    The references are plain text. Left in, a table's ``|-----|`` rows alone
    scored a perfect read of a French invoice at CER 0.72, so the metric would
    move with the number of dashes rather than with what was read.
    """
    text = _HTML_TAG.sub(" ", text)
    text = _RULE_ROW.sub(" ", text)
    text = _HEADING.sub("", text)
    return _EMPHASIS.sub("", text).replace("|", " ")


def levenshtein(a: str, b: str) -> int:
    """Edit distance, two rows at a time."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def cer_raw(reference: str, hypothesis: str) -> float:
    """Character error rate after ``normalise``. Can exceed 1 on runaway output."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """``cer_raw`` of the text the model read, with its markup removed."""
    return cer_raw(reference, strip_markup(hypothesis))


def field_recall(fields: Sequence[str], text: str) -> float:
    """The fraction of ``fields`` present verbatim, whitespace and case aside."""
    if not fields:
        return 1.0
    haystack = normalise(strip_markup(text))
    return sum(normalise(field) in haystack for field in fields) / len(fields)
