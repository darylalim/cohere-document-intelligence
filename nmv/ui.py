"""Shared chrome: the model gate and the sidebar controls both pages use."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import get_script_run_ctx

from nmv import runtime, system, trace, transcripts
from nmv.citations import parse
from nmv.errors import Failure, classify
from nmv.grounding import contains_json, is_point_answer
from nmv.imaging import MAX_PIXELS, MEGAPIXEL, PIXELS_PER_TOKEN
from nmv.model import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MODEL_REPO,
    RunStats,
    Sampling,
    Studio,
    is_cached,
    load_studio,
)
from nmv.r7b_format import Phase, Piece
from nmv.reader import READER_REPO, Reader, is_reader_cached, load_reader
from nmv.render import safe_markdown

logger = logging.getLogger(__name__)

MAX_MEGAPIXELS = MAX_PIXELS / MEGAPIXEL
MAX_PAGES = 10  # about memory and transcription time; the reader's budget is its own
READING_WIDTH = 760  # about 80 characters at the theme's 15 px base size
PAGE_IMAGE_WIDTH = 880  # above this an A4 page is taller than a 1440p screen
INSPECT_KEEP = 5  # per page: the content lives in session state, and only this much
MODEL_NAMES: dict[runtime.ModelName, str] = {
    "nmv": "North Micro Vision",
    "reader": "Command R7B",
}

# Every widget below carries `persist_state="session"`, and a bare `key` is not
# a substitute. Streamlit folds the active page's script hash into every element
# id -- "even if key_as_main_identity is specified", per
# streamlit/elements/lib/utils.py -- so `budget_mp` on Chat and `budget_mp` on
# Grounding are two different widgets. Without session scope, switching pages
# silently reset the budget to the ceiling and every sampling control to its
# default. Session scope also covers the sliders `_sampling_control` hides in
# deterministic mode, which were dropped the same way on each toggle.


@dataclass(frozen=True)
class SidebarSlots:
    """The parts of the sidebar that a finished run redraws."""

    stats: DeltaGenerator
    system: DeltaGenerator


def begin_run(kind: trace.Kind) -> trace.RunBuilder:
    """Start the run record, with the first 6 characters of the session id.

    Every job starts here, so this is also where a wait gets explained: with one
    worker, another tab's job would otherwise look like a hang.
    """
    ahead = runtime.pending()
    if ahead:
        st.toast(
            f"Waiting for another tab's job ({ahead} ahead)",
            icon=":material/hourglass_top:",
        )
    context = get_script_run_ctx()
    return trace.start(kind, context.session_id[:6] if context else trace.NO_ID)


@dataclass(frozen=True)
class Inspection:
    """What the model saw and said. Content, so it stays in session state and never
    reaches a log, the run record or the disk. A refresh clears it."""

    prompt: str
    raw: str
    record: trace.RunRecord
    notes: tuple[str, ...] = ()


def keep_inspection(scope: str, inspection: Inspection) -> None:
    kept = st.session_state.setdefault(f"inspect_{scope}", {})
    kept[inspection.record.run_id] = inspection
    while len(kept) > INSPECT_KEEP:
        kept.pop(next(iter(kept)))


def forget_inspections(scope: str) -> None:
    st.session_state.pop(f"inspect_{scope}", None)


def _record_rows(record: trace.RunRecord) -> list[dict[str, str]]:
    values = asdict(record)
    spans = values.pop("spans")
    values.update({f"{name} (s)": seconds for name, seconds in spans.items()})
    return [
        {
            "Field": name,
            "Value": f"{value:.3f}" if isinstance(value, float) else str(value),
        }
        for name, value in values.items()
    ]


def show_inspect(scope: str, run_id: str) -> None:
    """Draw a kept run, or nothing once it has aged out.

    ``st.code`` renders nothing, which is what makes it safe for hostile text: this
    is where a user checks whether a document carried an injected instruction.
    """
    inspection = st.session_state.get(f"inspect_{scope}", {}).get(run_id)
    if inspection is None:
        return
    with st.expander("Inspect"):
        for note in inspection.notes:
            st.caption(safe_markdown(note))
        st.caption("Prompt, as the chat template rendered it")
        st.code(inspection.prompt or "(empty)", language=None, wrap_lines=True)
        st.caption("Raw model output")
        st.code(inspection.raw or "(empty)", language=None, wrap_lines=True)
        st.caption("Run record")
        st.table(_record_rows(inspection.record))


def show_reply(chunks: Iterable[str]) -> str:
    """Stream model output onto the page and return the raw, unfiltered text.

    Not ``st.write_stream``: that renders raw chunks with
    ``unterminated_parsing=True``, so an injected Markdown image could load while
    it is still arriving. Everything shown here has been through ``safe_markdown``
    first. The return value has not, because the conversation history must keep
    what the model actually said; run it through ``safe_markdown`` to redraw it.
    """
    slot = st.empty()
    reply = ""
    for chunk in chunks:
        reply += chunk
        slot.markdown(safe_markdown(reply))
    return reply


def show_answer(pieces: Iterable[Piece]) -> None:
    """Stream the reader's response, without its plan and its citation tags.

    ``parse`` tolerates a tag cut off mid-stream, so no half-written ``<co>`` flashes
    past. The reader's ``Answer`` keeps the raw response for the history.
    """
    slot = st.empty()
    slot.caption("Planning the answer…")
    reply = ""
    for piece in pieces:
        if piece.phase is Phase.RESPONSE:
            reply += piece.text
            slot.markdown(safe_markdown(parse(reply)[0]))


def show_failure(
    exc: Exception,
    retry: Callable[[], None] | None = None,
    *,
    retry_key: str = "retry",
    first_report: bool = True,
    run_id: str = "",
) -> Failure:
    """The one way a page reports a load or generation failure.

    Wrap the call in ``except Exception`` and nothing broader: ``st.stop()`` and
    ``st.rerun()`` raise a ``BaseException``. ``retry`` runs when the user presses
    Retry, before the rerun, so it can put back what the run was doing. Pass
    ``first_report=False`` to redraw a failure that was already logged. ``run_id``
    goes in the details, so one search of the terminal finds the whole story.
    """
    failure = classify(exc)
    if first_report:
        # The class only: an exception's message can carry a file name. Expected
        # kinds get no traceback, and a run's own record already reports them.
        if failure.kind == "internal":
            logger.error("%s: %s", failure.kind, type(exc).__name__, exc_info=exc)
        elif not run_id:
            logger.warning("%s: %s", failure.kind, type(exc).__name__)
        if failure.kind == "out_of_memory":
            with suppress(Exception):  # the page must still report the first failure
                runtime.clear_cache()
    st.error(f"**{failure.title}.** {failure.remedy}", icon=":material/error:")
    with st.expander("Details"):
        if run_id:
            st.caption(f"run {run_id}")
        st.exception(exc)
    if retry is not None:
        st.button("Retry", icon=":material/refresh:", on_click=retry, key=retry_key)
    return failure


def explain_no_boxes(raw: str, truncated: bool, *, fixed_limit: bool = False) -> None:
    """Say why a reply drew no boxes. Their remedies point in different directions.

    Truncation comes first: a reply cut off at max tokens leaves a half-written
    array that decodes as nothing, and would otherwise be misread as prose. A point
    comes before the JSON test, because it decodes as JSON. ``fixed_limit`` is for
    callers that ignore the sidebar's max tokens, so the slider is no remedy.
    """
    if truncated:
        remedy = (
            "Try another citation."
            if fixed_limit
            else "Raise **Max new tokens** in the sidebar."
        )
        st.warning(
            f"The response stopped at the max-tokens limit, so the box list was cut "
            f"off mid-array. {remedy}",
            icon=":material/content_cut:",
        )
    elif is_point_answer(raw):
        st.info(
            "The model could not locate this precisely: it gave a single point, not "
            "a box, so nothing is drawn. Raising the image budget can help.",
            icon=":material/my_location:",
        )
    elif contains_json(raw):
        # The model answered, and the answer is "nothing here matches": a correct
        # result, not a failure, so this says nothing about sampling.
        st.info(
            "The model returned an empty result — it found nothing matching this "
            "instruction. Try different wording or another task preset, or raise "
            "the image budget if the target is small.",
            icon=":material/search_off:",
        )
    else:
        remedy = "" if fixed_limit else " Try the deterministic toggle, or ask for JSON."
        st.warning(
            f"No boxes parsed from the response. The model sometimes answers in "
            f"prose.{remedy}",
            icon=":material/format_quote:",
        )


def ensure_studio() -> Studio:
    """Load the checkpoint, warning first if it still has to be downloaded.

    Called before the page body paints, which looks like it breaks Streamlit's
    "render stable UI before slow work" guidance. It does not. That rule targets
    work that repeats on every rerun, and ``load_studio`` is ``st.cache_resource``
    -- 3.4 s once per server process, nothing after. The sidebar and page chrome
    are already up by this point, and the widgets that wait could not be used
    during the load anyway, because the session's script thread is blocked inside
    it. Reserving slots to paint them earlier would buy a control nobody can
    click, at the cost of routing both page bodies through ``slot.foo()`` writes
    instead of plain ``st.foo()``. Deliberate; please leave it.
    """
    if not is_cached():
        st.warning(
            f"`{MODEL_REPO}` is not in the Hugging Face cache yet — the first run "
            "downloads about 3.2 GB.",
            icon=":material/cloud_download:",
        )
    try:
        with st.spinner("Loading North Micro Vision (8-bit)…"):
            return load_studio()
    except Exception as exc:
        # cache_resource does not cache a function that raised, so the rerun that
        # Retry causes genuinely tries the load again.
        show_failure(exc, retry=lambda: None)
        st.stop()


def ensure_reader() -> Reader:
    """Load the reader, on the first question and never at page render.

    Raises what the load raises, so the page keeps the question it was answering.
    """
    if not is_reader_cached():
        st.warning(
            f"`{READER_REPO}` is not in the Hugging Face cache yet — the first "
            "question downloads about 4.5 GB.",
            icon=":material/cloud_download:",
        )
        show_reader_licence()
    with st.spinner("Loading Command R7B…"):
        return load_reader()


def show_reader_licence() -> None:
    st.info(
        "Command R7B is licensed CC-BY-NC-4.0 under [Cohere's C4AI licence "
        "agreement](https://cohere.com/c4ai-cc-by-nc-license) — non-commercial use "
        "only.",
        icon=":material/gavel:",
    )


def _budget_control() -> int:
    """Pixel budget per image, the single biggest lever on speed and memory."""
    st.subheader("Image budget", divider="gray")
    megapixels = st.slider(
        "Pixels per image",
        min_value=0.10,
        max_value=MAX_MEGAPIXELS,
        value=MAX_MEGAPIXELS,
        step=0.05,
        format="%.2f MP",
        key="budget_mp",
        persist_state="session",
        help=(
            "The encoder spends one token per 32×32 px block, so tokens scale "
            "with area. The ceiling is an A4 page at 200 dpi — the largest "
            "resolution this model was trained on."
        ),
    )
    budget = int(megapixels * MEGAPIXEL)
    st.caption(
        f"Up to **{budget // PIXELS_PER_TOKEN:,} tokens** per image"
        + ("  ·  native ceiling" if megapixels >= MAX_MEGAPIXELS else "")
    )
    return budget


def _sampling_control() -> Sampling:
    st.subheader("Sampling", divider="gray")
    st.caption("Applies to North Micro Vision")
    greedy = st.toggle(
        "Deterministic",
        value=False,
        key="greedy",
        persist_state="session",
        help="Temperature 0 — repeatable runs, better for grounding and OCR.",
    )
    max_tokens = st.slider(
        "Max new tokens",
        64,
        2048,
        512,
        step=64,
        key="max_tokens",
        persist_state="session",
    )

    if greedy:
        st.caption("Temperature pinned to 0; nucleus and top-k inactive.")
        temperature, top_p, top_k = 0.0, 1.0, 0
    else:
        temperature = st.slider(
            "Temperature",
            0.0,
            1.5,
            DEFAULT_TEMPERATURE,
            0.05,
            key="temp",
            persist_state="session",
        )
        top_p = st.slider(
            "Top-p",
            0.05,
            1.0,
            DEFAULT_TOP_P,
            0.05,
            key="top_p",
            persist_state="session",
        )
        top_k = st.slider(
            "Top-k",
            0,
            100,
            DEFAULT_TOP_K,
            1,
            key="top_k",
            persist_state="session",
        )

    # Rendered in both modes deliberately: a repetition penalty is orthogonal
    # to temperature, and greedy argmax decoding is exactly where repetition
    # loops bite hardest.
    penalty = st.slider(
        "Repetition penalty",
        1.0,
        1.5,
        1.0,
        0.01,
        key="rep_pen",
        persist_state="session",
        help="1.0 disables it. Cohere suggests a mild penalty for long outputs.",
    )
    return Sampling(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
    )


def _run_rows() -> list[dict[str, str | int | float | None]]:
    """The recent runs, newest first. Run records hold numbers only."""
    return [
        {
            "Kind": record.kind,
            "Queue s": round(record.queue_wait_s, 2),
            "First token s": (
                None if record.first_token_s is None else round(record.first_token_s, 2)
            ),
            "Prompt tokens": record.prompt_tokens,
            "Generated": record.generated_tokens,
            "Thinking": record.thinking_tokens,
            "Peak GB": round(record.peak_gb, 2),
        }
        for record in reversed(trace.recent())
    ]


def _render_system(slot: DeltaGenerator) -> None:
    """Fill the System panel. Nothing here waits for the worker, so a tab stays
    responsive during another tab's answer."""
    snapshot = runtime.snapshot()
    pressure = system.pressure_level()
    swap = system.swap_used_mb()
    pending = runtime.pending()
    count, size = transcripts.usage()
    models = [MODEL_NAMES[name] for name, loaded in snapshot["models"].items() if loaded]
    runs = _run_rows()

    with slot.container():
        limit = snapshot["limit_gb"]
        if limit:
            used = snapshot["active_gb"] + snapshot["cache_gb"]
            # The limit is soft, and going past it is when this bar matters most.
            # st.progress raises above 1.0, so only the bar is clamped.
            st.progress(min(used / limit, 1.0), text=f"{used:.1f} / {limit:.1f} GB")
        else:
            st.caption("Memory appears once a job has finished.")
        label = system.PRESSURE_LABELS.get(pressure or 0, "Unknown")
        st.caption(f"Memory pressure: {label}")
        st.caption(f"Swap used: {'Unknown' if swap is None else f'{swap:,.0f} MB'}")
        st.caption(f"Models loaded: {', '.join(models) or 'none'}")
        st.caption(f"Worker: {f'busy · {pending - 1} queued' if pending else 'idle'}")
        st.caption(f"Saved transcripts: {count} · {size / 1000:,.1f} KB")
        if runs:
            st.dataframe(runs, hide_index=True, alt="The last 10 runs, newest first")


def render_stats(slots: SidebarSlots, stats: RunStats | None) -> None:
    """Fill the reserved sidebar slots with timings from a run.

    The slots are claimed while the sidebar draws, before generation starts, so
    a fresh run can write into them afterwards. Without that, the panels would
    always trail one interaction behind — they would render the *previous* run's
    numbers under a "Last run" heading.

    They are ``st.empty()`` rather than ``st.container()`` because this runs
    twice per interaction: once from ``sidebar()`` carrying the previous run's
    numbers, then again from the page once generation finishes. ``st.empty()``
    replaces on write; a container appends, stacking two "Last run" panels.
    Streamlit's own guidance discourages ``st.empty()`` slots filled after slow
    work because remounting resets stateful widgets — ``st.metric`` holds no
    such state, so that hazard does not apply here. The System slot sits inside
    its expander for the same reason: the expander keeps its open state.
    """
    _render_system(slots.system)
    if stats is None:
        return
    with slots.stats.container():
        st.subheader("Last run", divider="gray")
        if stats.model:
            st.caption(stats.model)
        with st.container(horizontal=True):
            st.metric("Prefill tokens", f"{stats.prompt_tokens:,}")
            st.metric("Decode tok/s", f"{stats.generation_tps:.0f}")
        with st.container(horizontal=True):
            st.metric("Prefill tok/s", f"{stats.prompt_tps:.0f}")
            st.metric("Peak memory GB", f"{stats.peak_memory:.2f}")


def sidebar(*, reader: bool = False) -> tuple[int, Sampling, SidebarSlots]:
    """Render every shared control. ``reader`` adds Command R7B's model caption.

    Returns the image budget, the sampling config, and the slots the caller hands
    to ``render_stats`` once generation finishes.
    """
    with st.sidebar:
        st.caption("Model")
        st.code(MODEL_REPO, language=None, wrap_lines=True)
        st.caption("2.4B · 8-bit · Apache 2.0 · native resolution")
        if reader:
            st.code(READER_REPO, language=None, wrap_lines=True)
            st.caption("8B · 4-bit · CC-BY-NC-4.0 · non-commercial")
        budget = _budget_control()
        sampling = _sampling_control()
        stats_slot = st.empty()
        with st.expander("System"):
            system_slot = st.empty()

    slots = SidebarSlots(stats_slot, system_slot)
    render_stats(slots, st.session_state.get("last_stats"))
    return budget, sampling, slots
