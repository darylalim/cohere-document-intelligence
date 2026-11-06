"""Multi-turn visual chat.

Images are attached to the turn they arrived on. mlx-vlm reads the per-message
image markers we emit and hands each picture to the right turn, so a follow-up
question can refer back to a page uploaded three turns ago.
"""

from typing import Any

import streamlit as st

from nmv import trace
from nmv.imaging import (
    IMAGE_ERRORS,
    IMAGE_REMEDY,
    IMAGE_TYPES,
    VALIDATED_CONTEXT_TOKENS,
    open_upload,
    prepare,
)
from nmv.model import Rendered, RunStats, assistant_turn, stream_reply, user_turn
from nmv.render import safe_markdown
from nmv.ui import (
    READING_WIDTH,
    Inspection,
    begin_run,
    ensure_studio,
    forget_inspections,
    keep_inspection,
    render_stats,
    show_failure,
    show_inspect,
    show_reply,
    sidebar,
)

SUGGESTIONS = {
    ":blue[:material/description:] Read a document": (
        "Transcribe all text in this image, preserving the layout."
    ),
    ":green[:material/bar_chart:] Explain a chart": (
        "Describe this chart and state the trend it shows."
    ),
    ":orange[:material/table:] Extract a table": (
        "Extract every table in this image as markdown."
    ),
}


def _clear_conversation() -> None:
    st.session_state.chat.clear()
    forget_inspections("chat")


def _use_suggestion() -> None:
    """Fill the input and leave the send to the user, who still has to attach an image.

    A callback, because the input is already drawn by the time the pills are.
    """
    picked = st.session_state.chat_suggestion
    if picked:
        st.session_state.chat_box = SUGGESTIONS[picked]
    st.session_state.chat_suggestion = None


st.title("Chat")
st.caption(
    "Document Q&A, OCR, charts and captioning. Attach images with the **+** button; "
    "they stay attached to that turn for the rest of the conversation."
)

budget, sampling, stats_slot = sidebar()

if "chat" not in st.session_state:
    st.session_state.chat = []

studio = ensure_studio()

if st.session_state.chat:
    st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        on_click=_clear_conversation,
    )

# Replayed and new turns share it, so a long line reads the same on any display.
messages_area = st.container(width=READING_WIDTH)
with messages_area:
    for record in st.session_state.chat:
        with st.chat_message(record["role"]):
            if record["images"]:
                st.image(record["images"], width=180)
            if record.get("note"):
                st.caption(record["note"])
            # User turns too: text pasted from a document can carry an image just as
            # a reply can. The record keeps the raw text for the conversation history.
            st.markdown(safe_markdown(record["text"]))
            if record.get("stats_line"):
                st.caption(record["stats_line"])
            if record.get("run_id"):
                show_inspect("chat", record["run_id"])

queued = st.session_state.pop("queued_prompt", None)

submission = st.chat_input(
    "Ask about a page, chart, or photo",
    key="chat_box",
    accept_file="multiple",
    file_type=IMAGE_TYPES,
    submit_mode="disable",
)

if submission is not None:
    text = (submission.text or "").strip()
    uploads = list(submission.files or [])
elif queued:
    text, uploads = queued, []
else:
    text, uploads = "", []

# Every starter prompt names an image. Sent without one, the model describes a chart
# that does not exist, with invented axes and values.
has_image = uploads or any(record["images"] for record in st.session_state.chat)
if text in SUGGESTIONS.values() and not has_image:
    st.info(
        "Attach an image with the **+** button, then send.",
        icon=":material/add_photo_alternate:",
    )
    text, submission = "", None

# After the input, so a first message removes them: the conversation container sits
# above this point, and they would otherwise linger under the reply.
if not st.session_state.chat and not queued and submission is None:
    st.pills(
        "Try asking",
        list(SUGGESTIONS),
        # Each pill names its task, and a screen reader still gets this label.
        label_visibility="collapsed",
        key="chat_suggestion",
        on_change=_use_suggestion,
    )

if text or uploads:
    action = begin_run("chat")
    images, notes, failures = [], [], []
    turn_tokens = 0
    largest = 0
    with trace.span("prep"):
        for upload in uploads:
            try:
                image, resolved = prepare(open_upload(upload), budget)
            except IMAGE_ERRORS as error:
                failures.append(f"**{upload.name}** — {error} {IMAGE_REMEDY}")
                continue
            images.append(image)
            turn_tokens += resolved.tokens
            largest = max(largest, resolved.target_pixels)
            note = (
                f"{upload.name} · {resolved.target[0]}×{resolved.target[1]} · "
                f"{resolved.tokens:,} tokens"
            )
            if resolved.resized:
                note += f" (from {resolved.source[0]}×{resolved.source[1]})"
            notes.append(note)

    for failure in failures:
        st.error(failure, icon=":material/broken_image:")

    if not text:
        text = "Describe this image." if images else ""

    if not text:
        trace.finish(action, "bad_image")  # every upload failed, and nothing was asked

    if text:
        # Held out of session_state until the reply lands. Committing the user
        # turn first would leave an orphan behind any failure or interruption,
        # and the rebuild below would then emit two consecutive user messages
        # whose image markers no longer match `ordered_images` — corrupting
        # every later turn with no way back but "Clear conversation".
        pending: dict[str, Any] = {
            "role": "user",
            "text": text,
            "images": images,
            "note": " · ".join(f"`{n}`" for n in notes) if notes else "",
            "tokens": turn_tokens,
        }
        with messages_area.chat_message("user"):
            if images:
                st.image(images, width=180)
            if notes:
                st.caption(" · ".join(f"`{n}`" for n in notes))
            st.markdown(safe_markdown(text))

        # Rebuild the whole conversation each turn. Images are collected in the
        # same order the markers appear, which is how they get matched up.
        messages, ordered_images = [], []
        for record in (*st.session_state.chat, pending):
            if record["role"] == "user":
                messages.append(user_turn(record["text"], len(record["images"])))
                ordered_images.extend(record["images"])
            else:
                messages.append(assistant_turn(record["text"]))

        # Every image from every prior turn is re-sent and re-encoded each
        # time, so image cost accumulates across the conversation. Warn once it
        # passes the multimodal window Cohere actually validated — the sidebar
        # figure is per-image and cannot show this.
        carried = sum(r.get("tokens", 0) for r in st.session_state.chat) + turn_tokens
        if carried > VALIDATED_CONTEXT_TOKENS:
            messages_area.warning(
                f"Images in this conversation now total ~{carried:,} tokens, past the "
                f"{VALIDATED_CONTEXT_TOKENS:,}-token multimodal context this model was "
                "validated at. Start a new conversation for reliable answers.",
                icon=":material/warning:",
            )

        action.set(image_tokens=carried, budget_px=budget, resolved_px=largest)
        stats = RunStats()
        rendered = Rendered()
        with messages_area.chat_message("assistant"):
            try:
                reply = show_reply(
                    stream_reply(
                        studio, messages, ordered_images, sampling, stats, rendered
                    ),
                    st.session_state.stream_responses,
                )
            except Exception as exc:
                # The pending turn is simply never committed. Only a text turn can
                # be put back: the queue holds no images.
                failure = show_failure(
                    exc,
                    retry=None
                    if images
                    else lambda: st.session_state.update(queued_prompt=text),
                    run_id=action.run_id,
                )
                trace.finish(action, failure.kind, stats)
                render_stats(stats_slot, None)
                st.stop()
            stats_line = (
                f"{stats.prompt_tokens:,} prompt tokens · "
                f"{stats.generation_tokens:,} generated at "
                f"{stats.generation_tps:.0f} tok/s · peak {stats.peak_memory:.2f} GB"
            )
            if stats.truncated:
                stats_line += " · **stopped at max tokens**"
            st.caption(stats_line)
            run_record = trace.finish(action, stats=stats)
            keep_inspection(
                "chat", Inspection(rendered.prompt, reply, run_record, tuple(notes))
            )
            show_inspect("chat", run_record.run_id)

        st.session_state.last_stats = stats
        render_stats(stats_slot, stats)
        # Both turns commit together, so history is never left half-written.
        st.session_state.chat.append(pending)
        st.session_state.chat.append(
            {
                "role": "assistant",
                "text": reply,
                "images": [],
                "stats_line": stats_line,
                "run_id": run_record.run_id,
            }
        )
