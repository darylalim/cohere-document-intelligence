"""Ask documents: read every page, reason over the text, locate the answer.

North Micro Vision transcribes each page once and the store keeps the result.
Command R7B answers over those transcripts with citations, and a citation goes back
to North Micro Vision to be boxed on the page. Both stages ignore the sidebar's max
tokens, and Transcribe and Locate ignore its sampling too: a transcript is cached
and a box is a measurement, so neither may vary from click to click.
"""

import logging
from dataclasses import replace
from io import BytesIO

import streamlit as st
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.typing import UploadedFile

from nmv import trace, transcripts
from nmv.citations import parse
from nmv.documents import best_transcript, match_line, natural_key
from nmv.domain import Derivation, Grounding, Page, PageId, Transcript, Turn, page_id
from nmv.grounding import describe, draw_boxes, parse_boxes
from nmv.imaging import (
    IMAGE_ERRORS,
    IMAGE_REMEDY,
    IMAGE_TYPES,
    ImagePlan,
    encode,
    normalize,
    open_upload,
    plan,
)
from nmv.model import (
    MODEL_REVISION,
    Rendered,
    RunStats,
    Sampling,
    stream_reply,
    user_turn,
)
from nmv.prompts import (
    LOCATE_MAX_TOKENS,
    LOCATE_PROMPT,
    LOCATE_PROMPT_VERSION,
    TRANSCRIBE_MAX_TOKENS,
    TRANSCRIBE_PROMPT,
    TRANSCRIBE_PROMPT_VERSION,
)
from nmv.reader import (
    Answer,
    ContextExceeded,
    download_reader,
    is_reader_cached,
    reader_sampling,
    stream_answer,
)
from nmv.render import safe_markdown
from nmv.ui import (
    MAX_PAGES,
    PAGE_IMAGE_WIDTH,
    READING_WIDTH,
    Inspection,
    begin_run,
    ensure_reader,
    ensure_studio,
    explain_no_boxes,
    forget_inspections,
    keep_inspection,
    render_stats,
    show_answer,
    show_failure,
    show_inspect,
    show_reader_licence,
    sidebar,
)

logger = logging.getLogger("nmv.pages.documents")

LOCATE_CACHE_ENTRIES = 64
NMV = "North Micro Vision"
R7B = "Command R7B"

state = st.session_state


@st.cache_data(show_spinner=False, max_entries=MAX_PAGES)
def _read_upload(_upload: UploadedFile, file_id: str) -> Page:
    """The normalised page. Both the encoded copy and the overlay come from this
    one image, so EXIF rotation cannot put a box on the wrong axis."""
    data = _upload.getvalue()
    image = normalize(open_upload(BytesIO(data)))
    return Page(page_id(data), "", image.size, image)


@st.cache_data(show_spinner=False, max_entries=MAX_PAGES)
def _encoded(
    _image: Image.Image, page: PageId, budget: int
) -> tuple[Image.Image, ImagePlan]:
    """The page as the model sees it, once per (page, budget) pair, so every Locate
    on a page reuses the resize its transcription paid for.

    ``_image`` is underscore-prefixed so Streamlit skips hashing its pixels, and
    ``page`` already identifies them. An entry holds only the resized copy, because
    :func:`_read_upload` keeps the original.
    """
    return encode(_image, budget)


@st.cache_resource(show_spinner=False)
def _prune_once() -> int:
    """Once per server process, never on every rerun."""
    return transcripts.prune(
        MODEL_REVISION, TRANSCRIBE_PROMPT_VERSION, TRANSCRIBE_MAX_TOKENS
    )


def _read_pages(uploads: list[UploadedFile], budget: int) -> list[Page]:
    """Unique pages in natural order, saying what happened to every other file."""
    pages: list[Page] = []
    numbers: dict[PageId, int] = {}
    skipped: list[str] = []
    for upload in sorted(uploads, key=lambda item: natural_key(item.name)):
        try:
            page = _read_upload(upload, upload.file_id)
            plan(*page.source_size, budget)  # refuses a page too elongated to encode
        except IMAGE_ERRORS as error:
            st.error(
                f"**{upload.name}** — {error} {IMAGE_REMEDY}",
                icon=":material/broken_image:",
            )
            continue
        if page.id in numbers:
            st.caption(f"{upload.name} is a duplicate of p.{numbers[page.id]}.")
        elif len(pages) == MAX_PAGES:
            skipped.append(upload.name)
        else:
            pages.append(replace(page, display_name=upload.name))
            numbers[page.id] = len(pages)
    if skipped:
        st.warning(
            f"Only the first {MAX_PAGES} pages are used. Skipped: {', '.join(skipped)}. "
            "Remove some pages to make room.",
            icon=":material/content_cut:",
        )
    return pages


def _transcribe_derivation(page: Page, budget: int) -> Derivation:
    resolved = plan(*page.source_size, budget)
    return Derivation(
        page.id,
        resolved.target_pixels,
        MODEL_REVISION,
        TRANSCRIBE_PROMPT_VERSION,
        TRANSCRIBE_MAX_TOKENS,
    )


def _usable_transcript(page: Page, budget: int) -> Transcript | None:
    """An exact hit at this budget, or else the best one saved at another."""
    wanted = _transcribe_derivation(page, budget)
    key = wanted.key()
    exact = state.ask_held.get(key) or transcripts.get(key)
    if exact is not None:
        return exact
    return best_transcript(
        [*state.ask_held.values(), *transcripts.by_page(page.id)], wanted
    )


def _transcribe(
    page: Page, budget: int, save: bool, action: trace.RunBuilder, stats: RunStats
) -> str:
    """One page, kept the moment it finishes, so a later failure cannot lose it.

    Returns the run record's ``store`` value.
    """
    assert page.image is not None
    with trace.span("prep"):
        encoded, resolved = _encoded(page.image, page.id, budget)
    action.set(
        image_tokens=resolved.tokens,
        budget_px=budget,
        resolved_px=resolved.target_pixels,
    )
    text = "".join(
        stream_reply(
            studio,
            [user_turn(TRANSCRIBE_PROMPT, 1)],
            [encoded],
            Sampling(max_tokens=TRANSCRIBE_MAX_TOKENS, temperature=0.0),
            stats,
        )
    )
    transcript = Transcript(
        _transcribe_derivation(page, budget),
        text,
        stats.finish_reason,
        *resolved.target,
    )
    with trace.span("store"):
        saved = save and transcripts.put(transcript, page.display_name)
    if not saved:
        state.ask_held[transcript.derivation.key()] = transcript
    state.last_stats = stats
    outcome = "miss" if saved else "not-saved"
    action.set(store=outcome)
    return outcome


def _reuse(
    found: dict[PageId, Transcript | None], by_id: dict[PageId, Page], budget: int
) -> str:
    """The run record's ``store``: did every page come from this budget's transcript?"""
    for page, transcript in found.items():
        wanted = _transcribe_derivation(by_id[page], budget).key()
        if transcript is not None and transcript.derivation.key() != wanted:
            return "fallback-budget"
    return "hit"


def _panel_only() -> bool:
    """Whether this is a fragment's own rerun, not a full run passing through it.

    Such a run cannot write to the sidebar: Streamlit reserves a fragment's place in
    an outside container during a full run, and these fragments never claim one.
    """
    context = get_script_run_ctx()
    return bool(context and context.fragment_ids_this_run)


def _locate(page: Page, span: str, transcript: Transcript | None, budget: int):
    """Box the document's text for a cited span, cached and always greedy.

    A cached result is no action: this runs on every rerun while a citation is
    selected, and a record each time would push real runs out of the ring.
    """
    assert page.image is not None
    sent = match_line(span, transcript.text if transcript else "")
    resolved = plan(*page.source_size, budget)
    derivation = Derivation(
        page.id,
        resolved.target_pixels,
        MODEL_REVISION,
        LOCATE_PROMPT_VERSION,
        LOCATE_MAX_TOKENS,
    )
    key = (derivation.key(), sent)
    if key not in state.locate_cache:
        action = begin_run("locate")
        action.set(
            image_tokens=resolved.tokens,
            budget_px=budget,
            resolved_px=resolved.target_pixels,
            store="miss",
        )
        with trace.span("prep"):
            encoded, _ = _encoded(page.image, page.id, budget)
        stats = RunStats(model=f"{NMV} · locate")
        rendered = Rendered()
        raw = "".join(
            stream_reply(
                studio,
                [user_turn(LOCATE_PROMPT.format(text=sent), 1)],
                [encoded],
                Sampling(max_tokens=LOCATE_MAX_TOKENS, temperature=0.0),
                stats,
                rendered,
            )
        )
        state.last_stats = stats
        with trace.span("parse"):
            boxes = tuple(parse_boxes(raw))
        result = Grounding(derivation, sent, boxes, stats.truncated)
        while len(state.locate_cache) >= LOCATE_CACHE_ENTRIES:
            state.locate_cache.pop(next(iter(state.locate_cache)))
        record = trace.finish(action, stats=stats)
        size = f"{resolved.target[0]}×{resolved.target[1]}"
        keep_inspection(
            "ask",
            Inspection(
                rendered.prompt,
                raw,
                record,
                (f"Image: {size}, {resolved.tokens:,} tokens.",),
            ),
        )
        state.locate_cache[key] = (result, raw, record.run_id)
        # The rerun's Locate is a cache hit, and it costs nothing next to the model.
        if _panel_only():
            st.rerun()
        render_stats(stats_slot, stats)
    return state.locate_cache[key]


def _new_conversation() -> None:
    state.ask_turns = []
    state.ask_notes = {}
    state.locate_target = None
    forget_inspections("ask")


def _pick_citation(widget: str, entries: list[tuple[str, PageId]], view: str) -> None:
    """Move the page viewer to a citation, and leave only that pill selected.

    Redraws the page panel and the turn that lost its selection, and nothing else.
    The exception is a click during a transcription or an answer. It joins the full
    rerun that ends that run, and a keyed rerun here would replace it: the queue
    would stop with pages left, or a new answer would have no citations.
    """
    choice = state[widget]
    panels = ["page_panel"]
    for index in range(len(state.ask_turns)):
        other = f"ask_cite_{index}"
        if other != widget and state.get(other) is not None:
            state[other] = None
            panels.append(other)
    state.locate_target = None if choice is None else entries[choice]
    if choice is not None:
        state[view] = entries[choice][1]
    if state.transcribe_queue or state.ask_answering:
        st.rerun()
    st.rerun(panels)


def _citations(index: int, turn: Turn, numbers: dict[PageId, int], view: str) -> None:
    if not turn.citations:
        st.caption("No page citations — check the answer against the pages.")
        return
    entries: list[tuple[str, PageId]] = []
    missing: list[str] = []
    unknown = 0
    for citation in turn.citations:
        for position in citation.doc_indices:
            cited = turn.page_for(position)
            if cited is None:
                unknown += 1
            elif cited not in numbers:
                missing.append(citation.span)
            elif (citation.span, cited) not in entries:
                entries.append((citation.span, cited))

    def label(entry: int) -> str:
        span, cited = entries[entry]
        short = span if len(span) <= 40 else f"{span[:39]}…"
        # A widget label renders Markdown too, images included.
        return safe_markdown(f"p.{numbers[cited]}: “{short}”")

    if entries:
        widget = f"ask_cite_{index}"
        st.pills(
            "Citations",
            list(range(len(entries))),
            format_func=label,
            key=widget,
            on_change=_pick_citation,
            args=(widget, entries, view),
            # Each pill names its page and quote, and a screen reader still gets this.
            label_visibility="collapsed",
            wrap=True,
        )
    if unknown:
        st.pills(
            "Unknown citations",
            ["source not found"],
            key=f"ask_unknown_{index}",
            disabled=True,
            # The pill says it all, and a screen reader still gets this label.
            label_visibility="collapsed",
        )
    if missing:
        st.caption("Re-upload this page to locate it.")


@st.fragment(key="page_panel")
def _page_panel(
    by_id: dict[PageId, Page],
    numbers: dict[PageId, int],
    found: dict[PageId, Transcript | None],
    budget: int,
    busy: bool,
    view_key: str,
) -> None:
    """The viewer column, so picking a page or citation reruns only this much.

    A click here also leaves a transcription or answer in progress running, where a
    full rerun would abandon it and start the page again.
    """
    first = next(iter(by_id))
    # Through session state and not `default=`: a citation sets this key too,
    # and Streamlit warns about a widget that has both.
    state.setdefault(view_key, first)
    shown = (
        st.pills(
            "Page",
            list(by_id),
            format_func=lambda page: f"p.{numbers[page]}",
            key=view_key,
            wrap=True,
        )
        or first
    )
    page = by_id[shown]
    assert page.image is not None
    transcript = found.get(shown) or _usable_transcript(page, budget)

    image = page.image
    target = state.locate_target
    located = None
    if target is not None and target[1] == shown:
        try:
            with st.spinner("Locating…"):
                located = _locate(page, target[0], transcript, budget)
        except Exception as exc:
            action = trace.active()  # None if it failed before reaching the model
            failure = show_failure(
                exc,
                retry=lambda: None,
                retry_key="retry_locate",
                run_id=action.run_id if action else "",
            )
            if action is not None:
                trace.finish(action, failure.kind)
                if not _panel_only():
                    render_stats(stats_slot, None)
    if located is not None and located[0].boxes:
        image = draw_boxes(page.image, list(located[0].boxes))
    # st.image has no alt text, so a caption stands in. It renders Markdown and skips
    # safe_markdown, so the model's span stays out of it, in the caption below.
    boxed = located is not None and located[0].boxes
    alt = f"Page {numbers[shown]} of {len(numbers)}" + (", box drawn" if boxed else "")
    st.image(image, caption=alt, width=PAGE_IMAGE_WIDTH)
    if located is not None:
        result, raw, located_run = located
        found_at = [describe(box) for box in result.boxes]
        if len(found_at) > 1:
            found_at = [f"#{n} {place}" for n, place in enumerate(found_at, 1)]
        where = "; ".join(found_at) or "no box"
        st.caption(
            safe_markdown(f"Page {numbers[shown]} · “{result.instruction}” · {where}.")
            + " Greedy, whatever the sidebar says."
        )
        if not result.boxes:
            explain_no_boxes(raw, result.truncated, fixed_limit=True)
        show_inspect("ask", located_run)

    with st.expander("Transcript"):
        if transcript is None:
            st.caption("Not transcribed yet.")
        else:
            wanted = _transcribe_derivation(page, budget)
            if transcript.derivation.resolved_px != wanted.resolved_px:
                megapixels = transcript.derivation.resolved_px / 1_000_000
                st.caption(f"Saved at {megapixels:.1f} MP.")
            if transcript.truncated:
                st.warning(
                    f"This page stopped at {TRANSCRIBE_MAX_TOKENS:,} tokens, so "
                    "its transcript is incomplete. Crop it into two images to read all "
                    "of it.",
                    icon=":material/content_cut:",
                )
            if transcript.empty:
                st.warning(
                    "This page looks empty, so it is not sent to the reader. Check the "
                    "image, then re-transcribe it.",
                    icon=":material/hide_image:",
                )
            st.markdown(safe_markdown(transcript.text))
            if st.button(
                "Re-transcribe at the current budget",
                icon=":material/refresh:",
                disabled=busy,
            ):
                state.transcribe_queue.append(shown)
                state.transcribe_total = len(state.transcribe_queue)
                st.rerun()


for name, value in (
    ("ask_turns", []),
    ("ask_notes", {}),
    ("ask_held", {}),  # transcripts kept in session: save off, or the store failed
    ("ask_failed", []),
    ("locate_cache", {}),
    ("locate_target", None),
    ("transcribe_queue", []),
    ("transcribe_total", 0),
):
    state.setdefault(name, value)
state.ask_answering = False  # after the callbacks, which read it from the last run

st.title("Ask documents")
st.caption(
    "Transcribe your pages once, ask about them, and follow each citation back to "
    "the page. Page images only: PNG, JPEG, WebP, BMP or TIFF."
)

budget, sampling, stats_slot = sidebar(reader=True)
studio = ensure_studio()
_prune_once()

queue: list[PageId] = state.transcribe_queue
busy = bool(queue)

with st.container(horizontal=True):
    uploads = st.file_uploader(
        "Page images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="ask_uploads",
        disabled=busy,
    )
    save = st.toggle(
        "Save transcripts",
        value=True,
        key="save_transcripts",
        persist_state="session",
        bind="query-params",
        help="Off keeps this session's transcripts in memory, so nothing reaches disk.",
    )
    with st.popover(
        "Clear saved transcripts",
        icon=":material/delete:",
        disabled=busy,
        help=(
            "Unavailable while pages are being transcribed."
            if busy
            else "Delete saved transcripts from this Mac."
        ),
    ):
        st.caption(f"Saved in `{transcripts.db_path()}`.")
        if st.button("Delete every saved transcript", type="primary"):
            try:
                removed = transcripts.clear()
            except transcripts.STORE_ERRORS as error:
                st.error(
                    f"Nothing was cleared: {error}. Let any transcription in another "
                    "tab finish, then try again.",
                    icon=":material/error:",
                )
            else:
                st.toast(f"Removed {removed}. {transcripts.usage()[0]} remain.")
        others = transcripts.other_versions(
            MODEL_REVISION, TRANSCRIBE_PROMPT_VERSION, TRANSCRIBE_MAX_TOKENS
        )
        if others and st.button(f"Remove transcripts from other versions ({others})"):
            transcripts.prune(
                MODEL_REVISION,
                TRANSCRIBE_PROMPT_VERSION,
                TRANSCRIBE_MAX_TOKENS,
                older_than_days=None,
            )
            st.rerun()

if save:
    count, size = transcripts.usage()
    noun = "transcript" if count == 1 else "transcripts"
    st.caption(f"{count} {noun} saved · {size / 1000:,.1f} KB")
else:
    st.caption(
        "Not saving — this session's transcripts are lost on refresh. Already-saved "
        "ones stay until cleared."
    )
if not transcripts.available():
    st.warning(
        f"Transcripts are not being saved: `{transcripts.db_path()}` could not be "
        "opened. They last until this session ends. Check that file, or set "
        "`NMV_TRANSCRIPT_DB` to another path and restart the app.",
        icon=":material/database_off:",
    )

pages = _read_pages(list(uploads or []), budget)
by_id = {page.id: page for page in pages}
numbers = {page.id: number for number, page in enumerate(pages, 1)}
digest = "-".join(page.id[:8] for page in pages)

selected: list[PageId] = []
if pages:
    selected = st.multiselect(
        "Pages to ask about",
        list(by_id),
        default=list(by_id),
        format_func=lambda page: f"p.{numbers[page]} · {by_id[page].display_name}",
        key=f"ask_selected_{digest}",
        disabled=busy,
    )

# Earlier turns' citations index the old document list. An empty selection is left
# out, because the uploader forgets its files when the user visits another page.
if selected and state.get("ask_signature") != selected:
    if state.ask_turns and "ask_signature" in state:
        _new_conversation()
        st.info("The pages changed, so this is a new conversation.")
    state.ask_signature = list(selected)

found = {page: _usable_transcript(by_id[page], budget) for page in selected}
needed = [page for page, transcript in found.items() if transcript is None]
sendable = [
    (page, transcript.text)
    for page, transcript in found.items()
    if transcript is not None and not transcript.empty
]

with st.container(horizontal=True):
    if st.button(
        f"Transcribe · {'saving to disk' if save else 'not saving'}",
        icon=":material/document_scanner:",
        type="primary",
        disabled=busy or not needed,
    ):
        queue.extend(needed)
        state.transcribe_total = len(queue)
        state.ask_failed = []
        st.rerun()
    if busy:
        st.button("Stop", icon=":material/stop:", on_click=queue.clear)
    if not is_reader_cached() and st.button(
        "Download reader (4.5 GB)", icon=":material/download:", disabled=busy
    ):
        with st.spinner("Downloading Command R7B…"):
            try:
                download_reader()
            except Exception as exc:
                show_failure(exc, retry_key="retry_download")
if not is_reader_cached():
    show_reader_licence()
st.caption(
    f"Transcription is greedy with a fixed limit of {TRANSCRIBE_MAX_TOKENS:,} tokens, "
    "whatever the sidebar's sampling and max tokens say."
)
progress = st.empty()
for failed_name, failure, failed_run in state.ask_failed:
    st.caption(f"Page **{failed_name}** was not transcribed:")
    show_failure(failure, first_report=False, run_id=failed_run)
for page in needed:
    if page not in queue:
        st.caption(f"p.{numbers[page]} has no transcript yet, so it is not sent.")

conversation, viewer = st.columns(2, gap="medium")
view_key = f"ask_view_{digest}"

with conversation:
    messages = st.container(width=READING_WIDTH)
    with messages:
        if state.ask_turns:
            st.button(
                "New conversation",
                icon=":material/delete_sweep:",
                on_click=_new_conversation,
            )
        for index, turn in enumerate(state.ask_turns):
            with st.chat_message(turn.role):
                st.markdown(turn.display_text)
                if index in state.ask_notes:
                    st.caption(state.ask_notes[index])
                if turn.role == "assistant":
                    # Keyed by turn, so a callback can redraw just the one it cleared.
                    st.fragment(_citations, key=f"ask_cite_{index}")(
                        index, turn, numbers, view_key
                    )
                    show_inspect("ask", turn.run_id)

with viewer:
    if pages:
        _page_panel(by_id, numbers, found, budget, busy, view_key)
    else:
        st.info("Upload page images to begin.", icon=":material/description:")

question = st.chat_input(
    "Ask about the selected pages",
    disabled=not sendable,
    submit_mode="disable",
)
question = question or state.pop("ask_retry", None)

if question:
    if busy:
        progress.caption("Transcription is paused until this answer finishes.")
    with messages:
        with st.chat_message("user"):
            st.markdown(safe_markdown(question))
        with st.chat_message("assistant"):
            history = [turn.message() for turn in state.ask_turns]
            history.append({"role": "user", "content": question})
            answer = Answer()
            stats = RunStats(model=R7B)
            deterministic = sampling.temperature == 0.0
            action = begin_run("ask")
            action.set(store=_reuse(found, by_id, budget))
            state.ask_answering = True
            try:
                reader = ensure_reader()
                show_answer(
                    stream_answer(
                        reader,
                        history,
                        sendable,
                        reader_sampling(deterministic),
                        stats,
                        answer,
                    ),
                    st.session_state.stream_responses,
                )
            except ContextExceeded as error:
                trace.finish(action, "context_exceeded", stats)
                st.warning(str(error), icon=":material/content_cut:")
                st.stop()
            except Exception as exc:
                # Nothing joins the history unless the reply arrived: an orphaned
                # user turn would corrupt every later one.
                failure = show_failure(
                    exc,
                    retry=lambda: state.update(ask_retry=question),
                    retry_key="retry_question",
                    run_id=action.run_id,
                )
                trace.finish(action, failure.kind, stats)
                render_stats(stats_slot, None)
                st.stop()

    state.last_stats = stats
    render_stats(stats_slot, stats)
    if answer.cut_off_planning:
        trace.finish(action, stats=stats)
        with messages:
            st.warning(
                "The answer was cut off while planning. Ask about fewer pages, or ask "
                "something narrower.",
                icon=":material/content_cut:",
            )
        st.stop()

    with trace.span("parse"):
        clean, citations = parse(answer.response)
    notes = []
    if answer.dropped_pages:
        names = ", ".join(f"p.{numbers[page]}" for page in answer.dropped_pages)
        notes.append(f"{names} did not fit the reader, so they were left out.")
    if answer.dropped_turns:
        notes.append(f"The {answer.dropped_turns} oldest turns were left out.")
    if stats.truncated:
        notes.append("The answer stopped at the reader's token limit.")
    record = trace.finish(action, stats=stats)
    sent = ", ".join(f"p.{numbers[page]}" for page in answer.sent_pages)
    left_out = ", ".join(f"p.{numbers[page]}" for page in answer.dropped_pages)
    keep_inspection(
        "ask",
        Inspection(
            answer.prompt,
            answer.raw,
            record,
            (
                f"Sent to the reader: {sent or 'none'}.",
                f"Left out by the context budget: {left_out or 'none'}.",
            ),
        ),
    )
    state.ask_turns.append(Turn("user", safe_markdown(question), question))
    state.ask_turns.append(
        Turn(
            "assistant",
            safe_markdown(clean),
            answer.response,
            tuple(citations),
            answer.sent_pages,
            record.run_id,
        )
    )
    if notes:
        state.ask_notes[len(state.ask_turns) - 1] = " ".join(notes)
    st.rerun()

# Last in the script, so a question asked mid-queue is answered first. One page per
# run: a rerun resumes from the queue, and each finished page appears at once.
while queue and queue[0] not in by_id:
    queue.pop(0)  # its upload was removed
if queue:
    done = state.transcribe_total - len(queue) + 1
    page = by_id[queue[0]]
    progress.caption(
        f"Page {done} of {state.transcribe_total} · asking a question pauses this."
    )
    action = begin_run("transcribe")
    stats = RunStats(model=f"{NMV} · transcribe")
    position = f"{done}/{state.transcribe_total}"
    try:
        stored = _transcribe(page, budget, save, action, stats)
        logger.info(
            "page %s %s %s %.1f s", position, page.id[:8], stored, action.elapsed()
        )
        trace.finish(action, stats=stats)
    except Exception as exc:
        failure = show_failure(exc, run_id=action.run_id)
        logger.warning("page %s %s %s", position, page.id[:8], failure.kind)
        trace.finish(action, failure.kind, stats)
        state.ask_failed.append((page.display_name, exc, action.run_id))
    queue.pop(0)
    st.rerun()
