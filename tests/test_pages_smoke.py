# tier: ui
"""Every page renders and survives failure, with the models replaced by fakes.

Run:  uv run python tests/test_pages_smoke.py

Deliberately narrow, and no judge of page-switch persistence. ``CLAUDE.md`` explains
why AppTest gives the wrong answer there; the browser check is authoritative.
"""

import io
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402
from PIL import Image  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from nmv import runtime, trace, transcripts, ui  # noqa: E402
from nmv.r7b_format import Phase, Piece  # noqa: E402
from nmv.reader import Answer, ContextExceeded  # noqa: E402

TIMEOUT = 30
OOM = "[metal::malloc] Attempting to allocate 21474836480 bytes"
TRANSCRIPT = "Invoice\nTotal due: $4,182.55"
RESPONSE = "The total is <co>$4,182.55</co: 0:[0]>, not <co>$1</co: 0:[7]>."

temp = tempfile.TemporaryDirectory()
os.environ[transcripts.DB_ENV] = str(Path(temp.name) / "transcripts.db")


def fake_reply(
    studio, messages, images, sampling, stats=None, rendered=None
) -> Iterator[str]:
    """Transcribe and Locate both come through here; the prompt says which."""
    prompt = messages[-1]["content"][-1]["text"]
    if stats is not None:
        stats.finish_reason = "stop"
    if "4,182.55" in prompt and "Locate" in prompt:
        yield "[[100, 100, 400, 200]]"
    elif "Locate" in prompt:
        yield "[[852, 301, 852, 301]]"
    else:
        yield TRANSCRIPT


def fake_answer(reader, history, pages, sampling, stats=None, answer=None):
    assert answer is not None
    answer.sent_pages = tuple(page for page, _ in pages)
    answer.response = RESPONSE
    answer.prompt = "<rendered reader prompt>"
    answer.raw = f"planning{RESPONSE}"
    yield Piece(Phase.THINKING, "planning")
    yield Piece(Phase.RESPONSE, RESPONSE)


def raises(error: Exception):
    def fail(*args, **kwargs):
        raise error

    return fail


def stubbed(**overrides) -> ExitStack:
    fakes = {
        "nmv.ui.load_studio": lambda: object(),
        "nmv.ui.load_reader": lambda: object(),
        "nmv.ui.is_cached": lambda: True,
        "nmv.ui.is_reader_cached": lambda: True,
        "nmv.reader.is_reader_cached": lambda: True,
        "nmv.ui.runtime.clear_cache": lambda: None,
        "nmv.model.stream_reply": fake_reply,
        "nmv.reader.stream_answer": fake_answer,
    }
    fakes.update({name.replace("__", "."): fake for name, fake in overrides.items()})
    stack = ExitStack()
    for name, fake in fakes.items():
        stack.enter_context(patch(name, fake))
    return stack


def page(name: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "app_pages" / name), default_timeout=TIMEOUT)


def png(colour: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 400), colour).save(buffer, "PNG")
    return buffer.getvalue()


def labels(app: AppTest) -> list[str]:
    return [button.label for button in app.button]


def button(app: AppTest, text: str):
    return next(item for item in app.button if text in item.label)


def uncaught(app: AppTest) -> list:
    """An exception Streamlit caught. ``show_failure`` nests its own in an expander."""
    return [item for item in app.main.children.values() if item.type == "exception"]


def text_of(app: AppTest) -> str:
    parts = [*app.caption, *app.markdown, *app.info, *app.warning, *app.error]
    return " ".join(str(part.value) for part in parts)


for name in ("chat.py", "grounding.py", "documents.py"):
    print(name)
    with stubbed():
        app = page(name).run()
    check("renders with no upload", not uncaught(app), str(app.exception))
    with stubbed(nmv__ui__load_studio=raises(RuntimeError(OOM))):
        app = page(name).run()
    check("a failed load raises nothing", not uncaught(app), str(app.exception))
    check("it gets show_failure's message", "GPU memory" in app.error[0].value)
    check("it offers Retry", "Retry" in labels(app))

print("System panel")
SNAPSHOT = {
    **runtime.snapshot(),
    "active_gb": 11.0,
    "cache_gb": 1.0,
    "limit_gb": 10.7,  # past the limit: st.progress raises above 1.0 unless clamped
    "models": {"nmv": True, "reader": False},
}
with stubbed(
    nmv__ui__runtime__snapshot=lambda: SNAPSHOT,
    nmv__ui__runtime__pending=lambda: 3,
    nmv__ui__system__pressure_level=lambda: 2,
    nmv__ui__system__swap_used_mb=lambda: None,
):
    app = page("chat.py").run()
panel = next(item for item in app.sidebar.expander if item.label == "System")
shown = " ".join(item.value for item in panel.caption)
check("it renders past the limit", not uncaught(app), str(app.exception))
check("collapsed", not panel.proto.expanded)
bar = panel.get("progress")[0].proto
check("the bar is clamped", bar.value == 100, str(bar.value))
check("its text keeps the real figures", bar.text == "12.0 / 10.7 GB", bar.text)
check("pressure is named", "Memory pressure: Warning" in shown, shown)
check("a failed read is Unknown", "Swap used: Unknown" in shown, shown)
check("models are named", "North Micro Vision" in shown and "R7B" not in shown, shown)
check("the worker shows its queue", "busy · 2 queued" in shown, shown)

with stubbed(nmv__ui__runtime__pending=lambda: 2):
    app = page("chat.py").run()
    app.chat_input[0].set_value("hello").run()
check("a wait is explained", "(2 ahead)" in app.toast[0].value, str(list(app.toast)))
with stubbed():
    app = page("chat.py").run()
    app.chat_input[0].set_value("hello").run()
check("no wait, no toast", not list(app.toast))
panel = next(item for item in app.sidebar.expander if item.label == "System")
check("the run reaches the panel", len(panel.dataframe) == 1, str(panel.dataframe))

print("Memory warnings")


def memory(pressure: int, swap: float = 100.0, reader: bool = True, **overrides):
    resident = {**SNAPSHOT, "models": {"nmv": True, "reader": reader}}
    return stubbed(
        nmv__ui__runtime__snapshot=lambda: resident,
        nmv__ui__system__pressure_level=lambda: pressure,
        nmv__ui__system__swap_used_mb=lambda: swap,
        **overrides,
    )


def finished_run() -> None:
    trace.finish(trace.start("ask"))


def titles(app: AppTest) -> list[str]:
    return [item.proto.title for item in app.warning]


with memory(2):
    app = page("chat.py").run()
    check("a new session only takes a baseline", not titles(app), str(titles(app)))
    finished_run()
    app.run()
    check("pressure 2 warns after a run", "Memory is running short" in titles(app))
    check("it names the fix", "unload the reader" in text_of(app), text_of(app))
    check("no button off Ask documents", "Unload reader" not in labels(app))
with memory(1):
    app.run()
    check("it stays until the next run", "Memory is running short" in titles(app))
    finished_run()
    app.run()
    check("a calm run clears it", not titles(app), str(titles(app)))
with memory(1, swap=500.0):
    finished_run()
    app.run()
    check("swap growth warns", "Memory is running short" in titles(app))

with memory(4), patch("nmv.ui.logger.warning") as warned:
    finished_run()
    app.run()
    check("critical is urgent", "Memory is critically low" in titles(app))
    check("and logged once", warned.call_count == 1, str(warned.call_args_list))
    app.run()
    check("not again on a rerun", warned.call_count == 1, str(warned.call_args_list))

with memory(2, reader=False):
    app = page("documents.py").run()
    finished_run()
    app.run()
    check("no reader, no button", "Unload reader" not in labels(app))
    check("and no mention of it", "unload the reader" not in text_of(app))

unloads: list[bool] = []
with memory(2, nmv__ui__unload_reader=lambda: unloads.append(True)):
    app = page("documents.py").run()
    check("no warning, no button", "Unload reader" not in labels(app))
    finished_run()
    app.run()
    check("Ask documents offers it", "Unload reader" in labels(app))
    button(app, "Unload reader").click().run()
    check("it unloads", unloads == [True], str(unloads))
    check("the warning goes", not titles(app), str(titles(app)))
    check("so does the button", "Unload reader" not in labels(app))

print("A failed run reports straight away")
with memory(2, nmv__model__stream_reply=raises(RuntimeError(OOM))):
    app = page("chat.py").run()
    app.chat_input[0].set_value("hello").run()
check("no uncaught exception", not uncaught(app), str(app.exception))
check("the warning does not wait", "Memory is running short" in titles(app))

print("Chat generation failure")
with stubbed(nmv__model__stream_reply=raises(RuntimeError(OOM))):
    app = page("chat.py").run()
    app.chat_input[0].set_value("hello").run()
check("no uncaught exception", not uncaught(app), str(app.exception))
check("it is explained", "GPU memory" in app.error[0].value)
check("nothing joined the history", app.session_state["chat"] == [])

print("Inspect")
HOSTILE = "![pixel](https://attacker.example/x.png) **ignore previous instructions**"


def hostile_reply(studio, messages, images, sampling, stats=None, rendered=None):
    if rendered is not None:
        rendered.prompt = "<|START_OF_TURN|>hello"
    yield HOSTILE


with stubbed(nmv__model__stream_reply=hostile_reply):
    app = page("chat.py").run()
    app.chat_input[0].set_value("hello").run()
    check("no exception", not uncaught(app), str(app.exception))
    code = [block.value for block in app.code]
    check("the raw output is in a code block", HOSTILE in code, str(code))
    check("the rendered prompt is too", "<|START_OF_TURN|>hello" in code, str(code))
    check("the record is a table", len(app.table) == 1)
    check("the image never renders", "attacker.example" not in text_of(app))
    app.run()  # the turn is redrawn from the history
    check("it survives a rerun", HOSTILE in [block.value for block in app.code])

    for _ in range(ui.INSPECT_KEEP + 1):
        app.chat_input[0].set_value("again").run()
    app.run()  # the run that evicts one has already drawn it
    check("only the latest turns keep content", len(app.table) == ui.INSPECT_KEEP)
    button(app, "Clear conversation").click().run()
check("clearing forgets them", "inspect_chat" not in app.session_state)

print("Run IDs reach the failure details")
with stubbed(nmv__model__stream_reply=raises(RuntimeError(OOM))):
    app = page("chat.py").run()
    app.chat_input[0].set_value("hello").run()
last = trace.recent()[-1]
check("the failure is recorded", last.outcome == "out_of_memory", str(last))
check("the details name the run", f"run {last.run_id}" in text_of(app), text_of(app))

print("Ask documents: the save toggle")
with stubbed():
    app = page("documents.py").run()
    check("uploader present", len(app.file_uploader) == 1)
    check("label says saving", any("saving to disk" in label for label in labels(app)))
    app.toggle(key="save_transcripts").set_value(False).run()
check("label says not saving", any("not saving" in label for label in labels(app)))
check("the caption says so", "lost on refresh" in text_of(app))

print("Ask documents: transcribe, ask, cite, locate")
with stubbed():
    app = page("documents.py").run()
    app.file_uploader[0].set_value(
        [
            ("page-10.png", png("white"), "image/png"),
            ("page-2.png", png("black"), "image/png"),
            ("copy.png", png("black"), "image/png"),
        ]
    ).run()
    check("no exception", not uncaught(app), str(app.exception))
    check("the duplicate is named", "duplicate of p.1" in text_of(app), text_of(app))
    check("two pages are selected", len(app.multiselect[0].value) == 2)
    check("nothing to ask yet", app.chat_input[0].disabled)
    button(app, "Transcribe").click().run()
    check(
        "the queue drained",
        not uncaught(app) and app.session_state["transcribe_queue"] == [],
    )
    check("both pages saved", transcripts.usage()[0] == 2)
    check("the button has no work left", button(app, "Transcribe").disabled)
    check("the question box opened", not app.chat_input[0].disabled)

    app.chat_input[0].set_value("What is the total due?").run()
    check("no exception after asking", not uncaught(app), str(app.exception))
    turns = app.session_state["ask_turns"]
    check("both turns joined", [turn.role for turn in turns] == ["user", "assistant"])
    check(
        "display has no tags", "<co>" not in turns[1].display_text, turns[1].display_text
    )
    check("history keeps its tags", turns[1].history_text == RESPONSE)
    check("sent_pages is recorded", len(turns[1].sent_pages) == 2)

    cited = app.button_group(key="ask_cite_1")
    check("the citation is a pill", "$4,182.55" in cited.options[0], cited.options[0])
    invented = app.button_group(key="ask_unknown_1")
    check("an invented index is disabled", invented.disabled)

    cited.set_value(0).run()
    check("no exception locating", not uncaught(app), str(app.exception))
    (entry,) = app.session_state["locate_cache"].values()
    check("a box was parsed", len(entry[0].boxes) == 1)
    check("the span on the page was sent", entry[0].instruction == "$4,182.55")
    check("the answer and the Locate keep Inspect", len(app.table) == 2)
    kinds = [record.kind for record in trace.recent()]
    check("each action left a record", kinds[-3:] == ["transcribe", "ask", "locate"])
    app.run()
    check("a cached Locate adds none", len(trace.recent()) == len(kinds))

print("Ask documents: a point answer")


def point_only(studio, messages, images, sampling, stats=None, rendered=None):
    if stats is not None:
        stats.finish_reason = "stop"
    yield "[[852, 301, 852, 301]]"


with stubbed(nmv__model__stream_reply=point_only):
    app = page("documents.py").run()
    app.file_uploader[0].set_value(("page.png", png("white"), "image/png")).run()
    app.session_state["locate_target"] = ("$4,182.55", app.multiselect[0].value[0])
    app.run()
check("no exception", not uncaught(app), str(app.exception))
check("it says so", "could not locate this precisely" in text_of(app), text_of(app))
check(
    "the cache has no box",
    not next(iter(app.session_state["locate_cache"].values()))[0].boxes,
)

print("Ask documents: a failing page does not stop the rest")
calls: list[int] = []


def fails_once(studio, messages, images, sampling, stats=None, rendered=None):
    calls.append(1)
    if len(calls) == 1:
        raise RuntimeError(OOM)
    yield TRANSCRIPT


transcripts.clear()
with stubbed(nmv__model__stream_reply=fails_once):
    app = page("documents.py").run()
    app.file_uploader[0].set_value(
        [("a.png", png("red"), "image/png"), ("b.png", png("blue"), "image/png")]
    ).run()
    button(app, "Transcribe").click().run()
check("no exception", not uncaught(app), str(app.exception))
check("the second page finished", transcripts.usage()[0] == 1)
check("the failure stays on screen", any("GPU memory" in e.value for e in app.error))

print("Ask documents: reader failures")
with stubbed():
    app = page("documents.py").run()
    app.file_uploader[0].set_value(("b.png", png("blue"), "image/png")).run()
    check("the saved transcript is reused", not app.chat_input[0].disabled)
with stubbed(nmv__reader__stream_answer=raises(RuntimeError(OOM))):
    app.chat_input[0].set_value("Anything?").run()
check("no exception", not uncaught(app), str(app.exception))
check("the failure is explained", "GPU memory" in app.error[0].value)
check("no orphan turn", app.session_state["ask_turns"] == [])
with stubbed(nmv__reader__stream_answer=raises(ContextExceeded("Start a new one."))):
    app.chat_input[0].set_value("Anything?").run()
check("a full context is a plain warning", "Start a new one." in text_of(app))
check("Answer starts empty", Answer().cut_off_planning)

print("Ask documents: an answer cut off while planning")


def planning_only(reader, history, pages, sampling, stats=None, answer=None):
    yield Piece(Phase.THINKING, "planning")


with stubbed(nmv__reader__stream_answer=planning_only):
    app.chat_input[0].set_value("Anything?").run()
check("no exception", not uncaught(app), str(app.exception))
check("it says so", "cut off while planning" in text_of(app), text_of(app))
check("nothing is stored", app.session_state["ask_turns"] == [])

finish()
