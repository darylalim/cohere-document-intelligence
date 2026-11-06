"""The run record: spans, the ring, the trace file, and that it can hold no content.

Run:  uv run python tests/test_trace.py
"""

import json
import logging
import os
import re
import sys
import tempfile
import typing
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from nmv import errors, trace  # noqa: E402


class FakeClock:
    now = 100.0

    def __call__(self) -> float:
        return self.now


class Stats:
    prompt_tokens = 1000
    generation_tokens = 100
    prompt_tps = 500.0
    generation_tps = 50.0
    peak_memory = 4.5
    finish_reason = "stop"


fake = FakeClock()
trace.clock = fake

print("spans")
run = trace.start("transcribe", "abc123")
with trace.span("store"):
    fake.now += 1.0
    with trace.span("parse"):
        fake.now += 2.0
with trace.span("parse"):
    fake.now += 0.5
trace.first_token()
fake.now += 1.0
trace.first_token()
record = trace.finish(run)
check("the outer span holds the inner", record.spans["store"] == 3.0, str(record.spans))
check("repeats add up", record.spans["parse"] == 2.5, str(record.spans))
check("only the first token counts", record.first_token_s == 3.5)
check("duration runs to finish", record.duration == 4.5)
check("finish closes the run", trace.active() is None)
with trace.span("parse"):
    pass
check("a span with no run does nothing", trace.active() is None)

print("statistics")
run = trace.start("ask")
run.set(pages_used=2, pages_dropped=1, store="hit")
record = trace.finish(run, "ok", Stats())
check("prefill comes from MLX's figures", record.spans["prefill"] == 2.0)
check("so does decode", record.spans["decode"] == 2.0)
check("tokens carry over", (record.prompt_tokens, record.generated_tokens) == (1000, 100))
check("pages carry over", (record.pages_used, record.pages_dropped) == (2, 1))
check("the prefix is not built yet", record.prefix == "n/a")


class Stalled(Stats):
    prompt_tps = 0.0


record = trace.finish(trace.start("chat"), "internal", Stalled())
check("no speed means no prefill span", "prefill" not in record.spans)
run = trace.start("chat")
try:
    run.set(prompt="what the document said")
except TypeError:
    check("a page cannot set a field that does not exist", True)
else:
    check("a page cannot set a field that does not exist", False)
trace.finish(run)

print("the logging filter")
log = logging.makeLogRecord({})
trace.RunFilter().filter(log)
check("unset is a dash", (log.__dict__["run"], log.__dict__["session"]) == ("-", "-"))
run = trace.start("chat", "abc123")
log = logging.makeLogRecord({})
trace.RunFilter().filter(log)
stamped = (log.__dict__["run"], log.__dict__["session"])
check("it stamps the run", stamped == (run.run_id, "abc123"))
trace.finish(run)

print("the log line")
lines: list[logging.LogRecord] = []


class Collect(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        lines.append(record)


trace.logger.addHandler(Collect())
trace.logger.setLevel(logging.INFO)
trace.finish(trace.start("chat"))
trace.finish(trace.start("chat"), "out_of_memory")
check("one line per record", len(lines) == 2)
check("success is INFO", lines[0].levelno == logging.INFO)
check("a failure is WARNING", lines[1].levelno == logging.WARNING)
check("the line says how it ended", "outcome=out_of_memory" in lines[1].getMessage())

print("the ring")
for _ in range(trace.RING_SIZE + 5):
    newest = trace.finish(trace.start("chat"))
check("it keeps the last ten", len(trace.recent()) == trace.RING_SIZE)
check("newest last", trace.recent()[-1] == newest)

print("the trace file")
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "trace.jsonl"
    os.environ[trace.TRACE_FILE_ENV] = str(path)
    run = trace.start("locate")
    run.set(store="miss")
    first = trace.finish(run, stats=Stats())
    second = trace.finish(trace.start("ground"))
    written = path.read_text(encoding="utf-8").splitlines()
    check("one line per record", len(written) == 2)
    check("each is valid JSON", json.loads(written[0]) == asdict(first))
    check("in order", json.loads(written[1])["run_id"] == second.run_id)
    os.environ[trace.TRACE_FILE_ENV] = str(Path(folder) / "missing" / "trace.jsonl")
    trace.finish(trace.start("chat"))
    check("a bad path does not raise", True)
    del os.environ[trace.TRACE_FILE_ENV]

print("the record cannot hold content")
ALLOWED_TEXT = {
    "run_id",
    "session",
    "kind",
    "outcome",
    "store",
    "prefix",
    "finish_reason",
}
hints = typing.get_type_hints(trace.RunRecord)


def holds_text(hint: object) -> bool:
    return hint is str or str in typing.get_args(hint) and hint != SPANS


SPANS = dict[str, float]
text_fields = {name for name, hint in hints.items() if holds_text(hint)}
check("every string is an ID or an enum", text_fields == ALLOWED_TEXT, str(text_fields))
check("spans map a fixed name to seconds", hints["spans"] == SPANS)
check(
    "the span names are the plan's five",
    set(typing.get_args(trace.SpanName))
    == {"prep", "prefill", "decode", "parse", "store"},
)
check("the run ID is 8 hex", re.fullmatch(r"[0-9a-f]{8}", newest.run_id) is not None)
kinds = {
    value.kind for value in vars(errors).values() if isinstance(value, errors.Failure)
}
check("every failure kind is an outcome", kinds <= set(trace.OUTCOMES), str(kinds))
default_store = trace.RunRecord("a", "b", "c").store
check("the default store state is listed", default_store in trace.STORE_STATES)

finish()
