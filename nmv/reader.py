"""The document reader: Command R7B, answering questions over page transcripts.

Mirrors ``nmv/model.py``. Every mlx-lm import is function-local and runs on the
worker from ``nmv.runtime``, for the same reason as mlx-vlm's. Fitting the prompt
into the context window only needs the tokenizer, which is not MLX, so that runs
on the calling thread and leaves the worker free.

"Never send a system prompt" is a North Micro Vision rule. R7B does accept a
preamble, but none is sent: the ``rag`` template already instructs the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import streamlit as st

from nmv import runtime, trace
from nmv.domain import PageId
from nmv.model import RunStats
from nmv.r7b_format import Phase, PhaseParser, Piece, render

READER_REPO = "mlx-community/c4ai-command-r7b-12-2024-4bit"
# A commit, not a branch, for the reasons at MODEL_REVISION. This commit has no .py
# files, no `model_file` and no `auto_map`; check all three before moving it.
READER_REVISION = "2ae601b3d780410caacd2148c6c96e620144f22d"
# The config's max_position_embeddings, with no rope_scaling. A hard limit on the
# whole rendered prompt plus the answer.
READER_CONTEXT = 8192
# Thinking counts against this, so it is the reader's own and not the sidebar's.
READER_MAX_TOKENS = 1024
SAFETY_MARGIN = 64

# Fetching only these means no .py file can arrive with the weights.
_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.jinja"]


class ContextExceeded(ValueError):
    """The conversation cannot fit the reader's context window."""


@dataclass
class Reader:
    model: Any
    tokenizer: Any


@dataclass(frozen=True)
class ReaderSampling:
    """The reader's own settings. It never uses North Micro Vision's sliders."""

    temperature: float = 0.3
    top_p: float = 0.75
    repetition_penalty: float = 1.0
    max_tokens: int = READER_MAX_TOKENS


def reader_sampling(deterministic: bool) -> ReaderSampling:
    """Temperature 0 under the sidebar's toggle, so the evals and the app agree."""
    return ReaderSampling(temperature=0.0) if deterministic else ReaderSampling()


def download_reader() -> str:
    """Fetch the weights, returning their path. Runs page-side, never on the
    worker, so a long download cannot queue every tab's work behind it."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        READER_REPO, revision=READER_REVISION, allow_patterns=_ALLOW_PATTERNS
    )


def _load(local_path: str) -> Reader:
    from mlx_lm import load

    loaded = load(local_path)
    # Here and not at boot: only with both models resident does the cache matter.
    # This already runs on the worker, so it calls limit_cache directly.
    runtime.limit_cache()
    return Reader(model=loaded[0], tokenizer=loaded[1])


@st.cache_resource(show_spinner=False)
def load_reader() -> Reader:
    """Load once per server process, as ``load_studio`` does."""
    return runtime.call(_load, download_reader())


def is_reader_cached() -> bool:
    """True when weights are on disk, so the UI can warn before a 4.5 GB pull."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(READER_REPO, "config.json", revision=READER_REVISION)
    return isinstance(cached, str)


@dataclass(frozen=True)
class Fit:
    """What will actually be sent."""

    history: list[dict]
    pages: list[tuple[PageId, str]]
    dropped_pages: tuple[PageId, ...]
    dropped_turns: int


def fit_context(
    history: list[dict],
    pages: list[tuple[PageId, str]],
    render_prompt: Callable[[list[dict], list[tuple[PageId, str]]], str],
    count: Callable[[str], int],
    max_tokens: int,
    context: int = READER_CONTEXT,
) -> Fit:
    """Choose the history and pages to send, measured on the rendered prompt.

    Measured rather than estimated because the template alone costs ~1,650 tokens
    and JSON escaping inflates tables. Pages join in order until one does not fit,
    and the rest are reported as dropped, never lost silently.

    Trims the history only to make room for the first page, since a question with
    no documents has nothing to answer from. The first user turn stays because the
    template attaches the documents to it, and so does the latest exchange. The
    middle goes oldest first, in assistant-and-user pairs so roles keep alternating.
    """

    def fits(turns: list[dict], sent: list[tuple[PageId, str]]) -> bool:
        return count(render_prompt(turns, sent)) + max_tokens + SAFETY_MARGIN <= context

    kept = list(history)
    first = pages[:1]
    dropped_turns = 0
    while not fits(kept, first):
        # Five turns is the floor: the first turn, the assistant turn that keeps
        # roles alternating after it, the latest exchange, and the new question.
        if len(kept) < 7:
            raise ContextExceeded(
                "This conversation no longer fits the reader. Start a new one, or "
                "ask about fewer pages."
            )
        del kept[1:3]
        dropped_turns += 2

    sent = list(first)
    for page in pages[1:]:
        if not fits(kept, [*sent, page]):
            break
        sent.append(page)
    dropped = tuple(page_id for page_id, _ in pages[len(sent) :])
    return Fit(kept, sent, dropped, dropped_turns)


def check_fits(
    prompt_tokens: int, max_tokens: int, context: int = READER_CONTEXT
) -> None:
    """The last check before generating. Never decode past the window."""
    if prompt_tokens + max_tokens > context:
        raise ContextExceeded(
            f"The prompt needs {prompt_tokens} tokens and the answer up to "
            f"{max_tokens}, past the reader's limit of {context}."
        )


def documents_for(pages: list[tuple[PageId, str]]) -> list[dict]:
    """The template's document list. Citations index this list as sent; the title
    is only for reading, so its number must never be used to look a page up."""
    return [
        {"title": f"Page {number}", "text": text}
        for number, (_, text) in enumerate(pages, 1)
    ]


@dataclass
class Answer:
    """Filled in while ``stream_answer`` runs, like ``RunStats``."""

    sent_pages: tuple[PageId, ...] = ()
    dropped_pages: tuple[PageId, ...] = ()
    dropped_turns: int = 0
    response: str = ""
    prompt: str = ""  # as rendered, for Inspect
    raw: str = ""  # everything decoded, plan included, for Inspect
    thinking_tokens: int = 0
    phase: Phase = Phase.IDLE

    @property
    def cut_off_planning(self) -> bool:
        """True when decoding ended before any answer. Store nothing as history."""
        return not self.response.strip()


def stream_answer(
    reader: Reader,
    history: list[dict],
    pages: list[tuple[PageId, str]],
    sampling: ReaderSampling,
    stats: RunStats | None = None,
    answer: Answer | None = None,
) -> Iterator[Piece]:
    """Yield the plan and the answer as ``Piece``s.

    ``answer.sent_pages`` is what the citations index, so the page stores it on
    the ``Turn``. Without it, a dropped page shifts every later citation.
    """
    answer = answer if answer is not None else Answer()
    tokenizer = reader.tokenizer

    def render_prompt(turns: list[dict], sent: list[tuple[PageId, str]]) -> str:
        return render(tokenizer, turns, documents_for(sent))

    def count(prompt: str) -> int:
        # Special tokens included: the template emits no BOS, so stream_generate's
        # own encode adds one, and this has to count what is really decoded against.
        return len(tokenizer.encode(prompt))

    fit = fit_context(history, pages, render_prompt, count, sampling.max_tokens)
    prompt = render_prompt(fit.history, fit.pages)
    check_fits(count(prompt), sampling.max_tokens)
    answer.sent_pages = tuple(page_id for page_id, _ in fit.pages)
    answer.dropped_pages = fit.dropped_pages
    answer.dropped_turns = fit.dropped_turns
    answer.prompt = prompt
    run = trace.active()
    if run is not None:
        # Here and not after the stream, so a failed answer still records them.
        run.set(pages_used=len(fit.pages), pages_dropped=len(fit.dropped_pages))

    def produce(emit):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # A repetition penalty is a logits processor in mlx-lm; make_sampler has no
        # such argument, so never splat Sampling.as_kwargs() in here.
        penalty = sampling.repetition_penalty
        for chunk in stream_generate(
            reader.model,
            tokenizer,
            prompt,
            max_tokens=sampling.max_tokens,
            sampler=make_sampler(temp=sampling.temperature, top_p=sampling.top_p),
            logits_processors=(
                make_logits_processors(repetition_penalty=penalty)
                if penalty != 1.0
                else None
            ),
        ):
            if not emit(chunk):  # consumer walked away, so stop decoding
                return

    parser = PhaseParser()

    def absorb(pieces: list[Piece]) -> Iterator[Piece]:
        for piece in pieces:
            if piece.phase is Phase.RESPONSE:
                answer.response += piece.text
            yield piece

    for chunk in runtime.iterate(produce):
        trace.first_token()
        if stats is not None:
            stats.absorb(chunk)
        answer.raw += chunk.text
        if parser.phase is Phase.THINKING:
            answer.thinking_tokens += 1
            if run is not None:
                run.set(thinking_tokens=answer.thinking_tokens)
        yield from absorb(parser.feed(chunk.text))
        answer.phase = parser.phase
    yield from absorb(parser.close())
    answer.phase = parser.phase
