"""Fitting pages and history into the reader's context, with a stub renderer.

Run:  uv run python tests/test_reader_budget.py

One token per character keeps the arithmetic readable. The real template and
tokenizer are checked in the weights tier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv.domain import PageId  # noqa: E402
from nmv.reader import (  # noqa: E402
    SAFETY_MARGIN,
    ContextExceeded,
    check_fits,
    documents_for,
    fit_context,
    reader_sampling,
)

MAX = 100


def turn(role: str, size: int) -> dict:
    return {"role": role, "content": role[0] * size}


def render(history: list[dict], pages: list) -> str:
    return "".join(t["content"] for t in history) + "".join(text for _, text in pages)


def fit(history: list[dict], pages: list, context: int):
    return fit_context(history, pages, render, len, MAX, context)


def page(name: str, size: int) -> tuple[PageId, str]:
    return PageId(name), "p" * size


question = [turn("user", 10)]
overhead = MAX + SAFETY_MARGIN

print("pages")
three = [page("a", 50), page("b", 50), page("c", 50)]
everything = fit(question, three, 160 + overhead)
check("all fit", [p for p, _ in everything.pages] == ["a", "b", "c"])
check("none dropped", everything.dropped_pages == ())
exact = fit(question, three, 110 + overhead)
check("the boundary is inclusive", len(exact.pages) == 2, str(len(exact.pages)))
short = fit(question, three, 109 + overhead)
check("one token over drops a page", [p for p, _ in short.pages] == ["a"])
check("dropped pages are reported in order", short.dropped_pages == ("b", "c"))
gap = fit(question, [page("a", 50), page("b", 500), page("c", 1)], 70 + overhead)
check("pages join in order and stop", gap.dropped_pages == ("b", "c"))

print("history")
long_chat = [
    turn("user", 10),
    turn("assistant", 100),
    turn("user", 11),
    turn("assistant", 101),
    turn("user", 12),
    turn("assistant", 13),
    turn("user", 14),
]
total = sum(len(t["content"]) for t in long_chat)
roomy = fit(long_chat, [page("a", 50)], total + 50 + overhead)
check("history is left alone when it fits", roomy.dropped_turns == 0)
one = fit(long_chat, [page("a", 50)], total + 49 + overhead)
check("the oldest middle pair goes first", one.dropped_turns == 2)
check(
    "the first turn and the latest exchange stay",
    [len(t["content"]) for t in one.history] == [10, 101, 12, 13, 14],
)
check(
    "roles still alternate after the first turn's pair",
    [t["role"][0] for t in one.history] == list("uauau"),
)
longer = [*long_chat[:1], turn("assistant", 90), turn("user", 91), *long_chat[1:]]
two = fit(longer, [page("a", 50)], 10 + 101 + 12 + 13 + 14 + 50 + overhead)
check("two pairs can go", two.dropped_turns == 4, str(two.dropped_turns))
check(
    "five turns are left",
    [len(t["content"]) for t in two.history] == [10, 101, 12, 13, 14],
)
check("the first page survives trimming", [p for p, _ in two.pages] == ["a"])
try:
    fit(longer, [page("a", 50)], 10 + 101 + 12 + 13 + 14 + 49 + overhead)
    check("five turns is the floor", False)
except ContextExceeded:
    check("five turns is the floor", True)
kept = fit(long_chat, [page("a", 50), page("b", 500)], total + 60 + overhead)
check("a later page never costs history", kept.dropped_turns == 0)
check("that page is dropped instead", kept.dropped_pages == ("b",))

print("refusal")
try:
    fit(question, [page("a", 500)], 100 + overhead)
    check("refuses when nothing fits", False)
except ContextExceeded:
    check("refuses when nothing fits", True)
try:
    fit(long_chat, [page("a", 500)], 100 + overhead)
    check("refuses even after trimming", False)
except ContextExceeded:
    check("refuses even after trimming", True)

print("check_fits")
check_fits(8000, 192)
check("the limit itself is allowed", True)
try:
    check_fits(8001, 192)
    check("past the limit raises", False)
except ContextExceeded:
    check("past the limit raises", True)

print("documents_for and sampling")
docs = documents_for([page("a", 1), page("b", 2)])
check("titles number from one", [d["title"] for d in docs] == ["Page 1", "Page 2"])
check("text is passed through", docs[1]["text"] == "pp")
check("deterministic pins temperature", reader_sampling(True).temperature == 0.0)
check("otherwise 0.3", reader_sampling(False).temperature == 0.3)

finish()
