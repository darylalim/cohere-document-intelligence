"""Parse R7B's citation tags out of a response. Pure: no MLX, no Streamlit.

The format comes from the checkpoint's ``rag`` template::

    The total is <co>$5</co: 0:[1,2],1:[0]>.

Each group is ``{tool_call_id}:[{positions}]``. Directly injected documents are
always tool call 0, so any other id is not ours and is dropped. The positions
index the document list **as sent**; ``Turn.page_for`` turns them into pages.

The input is the response body only, never the thinking block, and it is untrusted
model output: nothing here may raise on it.
"""

from __future__ import annotations

import re

from nmv.domain import Citation

_DIRECT = "0"
_MAX_INDEX_DIGITS = 9  # int() raises on thousands of digits, and no list is that long

_TOKEN = re.compile(r"<co>|</co(?::[^<>]*)?>|<\|[A-Z_]+\|>")
_GROUP = re.compile(r"(\d+):\[([\d,\s]*)\]")
_CUT_OFF = re.compile(r"<(?:c|co|/|/c|/co|/co:[^<>]*|\|[A-Z_]*\|?)?\Z")


def _indices(closing_tag: str) -> tuple[int, ...]:
    """The tool-call-0 positions in a closing tag, in order, without repeats."""
    found = [
        int(number)
        for tool_call, positions in _GROUP.findall(closing_tag)
        if tool_call == _DIRECT
        for number in re.findall(r"\d+", positions)
        if len(number) <= _MAX_INDEX_DIGITS
    ]
    return tuple(dict.fromkeys(found))


def parse(response_text: str) -> tuple[str, list[Citation]]:
    """Return the clean text and its citations, outermost first at each offset.

    Every tag and control token leaves the text. A span whose tag never closed, or
    closed without a usable position, stays as plain text with no citation, and so
    does a closing tag with nothing open. A stream cut off inside a tag loses that
    partial tag.
    """
    text = _CUT_OFF.sub("", response_text)
    clean: list[str] = []
    length = 0
    position = 0
    open_at: list[int] = []
    spans: list[tuple[int, int, tuple[int, ...]]] = []

    for token in _TOKEN.finditer(text):
        between = text[position : token.start()]
        clean.append(between)
        length += len(between)
        position = token.end()
        tag = token.group()
        if tag == "<co>":
            open_at.append(length)
        elif tag.startswith("</co") and open_at:
            start = open_at.pop()
            indices = _indices(tag)
            if indices and length > start:
                spans.append((start, length, indices))

    clean.append(text[position:])
    result = "".join(clean)
    spans.sort(key=lambda span: (span[0], -span[1]))
    return result, [
        Citation(result[start:end], indices, start, end) for start, end, indices in spans
    ]
