"""Command R7B's prompt and output format. Pure: no MLX, no Streamlit.

Everything specific to R7B's format lives here, so a fallback reader is a second
module rather than a rewrite.

R7B writes a plan and then an answer::

    <|START_THINKING|>...<|END_THINKING|><|START_RESPONSE|>...<|END_RESPONSE|>

Those markers are not EOS tokens, so they arrive in the text stream. Only the
response is shown, and only the response may become history: the template wraps
past assistant turns in response markers itself, so storing the raw stream would
nest them. The thinking can contain ``<co>`` tags too, and they mean nothing.

The prompt opens the response itself. Left to choose, R7B sometimes calls a tool on a
follow-up question, though it has none: 9 of 105 conversation replies were an action
block and no answer. So the plan above normally never appears, and an action block is
still caught if one comes, because that JSON must not reach the page or the history.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Any, NamedTuple, cast

START_THINKING = "<|START_THINKING|>"
END_THINKING = "<|END_THINKING|>"
START_RESPONSE = "<|START_RESPONSE|>"
END_RESPONSE = "<|END_RESPONSE|>"
START_ACTION = "<|START_ACTION|>"
END_ACTION = "<|END_ACTION|>"

# The marker the prompt ends with. START_THINKING lets R7B plan before it answers;
# ``evals/run_reader.py --think`` sets it to measure that, and the app keeps the
# response opening until a measurement says otherwise.
OPENING: str = START_RESPONSE
# Tools R7B may call, in the template's function format. Empty means the ``rag``
# template, which declares none. Anything here switches to ``tool_use``, which is
# ``rag`` without the line that empties the list. Only the evals set it so far.
TOOLS: list[dict] = []


class Phase(StrEnum):
    IDLE = "idle"  # before any text, or between the plan and the answer
    THINKING = "thinking"
    RESPONSE = "response"
    DONE = "done"


_NEXT = {
    START_THINKING: Phase.THINKING,
    END_THINKING: Phase.IDLE,
    START_RESPONSE: Phase.RESPONSE,
    END_RESPONSE: Phase.DONE,
    START_ACTION: Phase.DONE,
}


class Piece(NamedTuple):
    phase: Phase
    text: str


def render(tokenizer: Any, history: list[dict], documents: list[dict]) -> str:
    """The full prompt, through the checkpoint's own ``rag`` template, up to the reply."""
    prompt = tokenizer.apply_chat_template(
        history,
        documents=documents,
        chat_template="tool_use" if TOOLS else "rag",
        tools=TOOLS or None,
        enable_citations=True,
        add_generation_prompt=True,
        tokenize=False,
    )
    if TOOLS and documents:
        prompt = _renumber_results(cast(str, prompt))
    return cast(str, prompt) + OPENING


_RESULT_BLOCK = re.compile(r"<\|START_TOOL_RESULT\|>.*?<\|END_TOOL_RESULT\|>", re.DOTALL)
_RESULT_ID = re.compile(r'("tool_call_id": ")(\d+)(")')
# The documents' own call. The preamble names the tool too, but never in this form.
_DOCUMENT_CALL = '"tool_name": "direct-injected-document", "parameters": {}}'


def _renumber_results(prompt: str) -> str:
    """Give each tool result the id of the call it answers.

    The template numbers calls after the injected documents, which are call 0, but
    numbers results by counting calls alone, so the first result says it is call 0:
    the documents. Every result block after the documents' own moves up by one. The
    preamble quotes the result marker, so blocks are counted from the documents'
    call, not from the top.
    """
    anchor = prompt.find(_DOCUMENT_CALL)
    if anchor == -1:
        return prompt
    blocks = [b for b in _RESULT_BLOCK.finditer(prompt) if b.start() > anchor]
    for block in reversed(blocks[1:]):
        fixed = _RESULT_ID.sub(lambda m: f"{m[1]}{int(m[2]) + 1}{m[3]}", block[0])
        prompt = prompt[: block.start()] + fixed + prompt[block.end() :]
    return prompt


def _held_back(buffer: str) -> int:
    """How many trailing characters could be the start of a marker."""
    for size in range(min(len(buffer), max(map(len, _NEXT)) - 1), 0, -1):
        if any(marker.startswith(buffer[-size:]) for marker in _NEXT):
            return size
    return 0


class PhaseParser:
    """Split a text stream into phases. A marker may span two chunks.

    Text outside any block counts as response, because a reply that skips the
    format is still the reply. Anything after the response is dropped.
    """

    def __init__(self) -> None:
        # A prompt that opened the plan sends no marker for it.
        self.phase = Phase.THINKING if OPENING == START_THINKING else Phase.IDLE
        self._buffer = ""

    def _piece(self, text: str) -> list[Piece]:
        if self.phase is Phase.IDLE:
            if not text.strip():
                return []
            self.phase = Phase.RESPONSE
            text = text.lstrip()
        if self.phase is Phase.DONE or not text:
            return []
        return [Piece(self.phase, text)]

    def feed(self, text: str) -> list[Piece]:
        self._buffer += text
        pieces: list[Piece] = []
        while True:
            found = [
                (at, marker)
                for marker in _NEXT
                if (at := self._buffer.find(marker)) != -1
            ]
            if not found:
                break
            at, marker = min(found)
            pieces += self._piece(self._buffer[:at])
            self.phase = _NEXT[marker]
            self._buffer = self._buffer[at + len(marker) :]
        keep = _held_back(self._buffer)
        ready = self._buffer[: len(self._buffer) - keep]
        self._buffer = self._buffer[len(self._buffer) - keep :]
        return pieces + self._piece(ready)

    def close(self) -> list[Piece]:
        """Flush what was held back as a possible marker. It was not one."""
        text, self._buffer = self._buffer, ""
        return self._piece(text)


def stream_phases(chunks: Iterable[str]) -> Iterator[Piece]:
    parser = PhaseParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.close()
