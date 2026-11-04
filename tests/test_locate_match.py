"""The pure decisions behind Ask documents: page order, Locate's text, fallbacks.

Run:  uv run python tests/test_locate_match.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv.documents import best_transcript, match_line, natural_key  # noqa: E402
from nmv.domain import Derivation, Transcript, page_id  # noqa: E402

print("match_line")
PAGE = "# Invoice\n\nBill to: Acme\nTotal due: $4,182.55\n\n| Item | Cost |\n"
check(
    "a paraphrase finds the verbatim line",
    match_line("The total amount due is $4,182.55", PAGE) == "Total due: $4,182.55",
    match_line("The total amount due is $4,182.55", PAGE),
)
check(
    "an unrelated span falls back", match_line("zebra crossing", PAGE) == "zebra crossing"
)
check("no transcript falls back", match_line("anything", "") == "anything")
check(
    "Markdown decoration is not sent",
    match_line("The total cost", "| Total cost |\n") == "Total cost",
)
check("a heading loses its marks", match_line("The Invoice", PAGE) == "Invoice")

print("match_line: a span already on the page")
check("it is sent as it stands", match_line("$4,182.55", PAGE) == "$4,182.55")
check("R7B's trailing full stop goes", match_line("$4,182.55.", PAGE) == "$4,182.55")
check("the page's casing wins", match_line("total DUE", PAGE) == "Total due")
check("a span of punctuation is not on the page", match_line("...", PAGE) == "...")

print("natural_key")
names = ["page-10.png", "page-2.png", "Page-1.png", "cover.png"]
check(
    "numbers sort as numbers",
    sorted(names, key=natural_key)
    == ["cover.png", "Page-1.png", "page-2.png", "page-10.png"],
    str(sorted(names, key=natural_key)),
)
check(
    "a leading number is comparable",
    sorted(["2.png", "a.png"], key=natural_key)[0] == "2.png",
)

print("best_transcript")
wanted = Derivation(page_id(b"p"), 1_000_000, "a" * 40, 1, 2048)


def made(**changes) -> Transcript:
    return Transcript(replace(wanted, **changes), "text", "stop", 1, 1)


low, high = made(resolved_px=500_000), made(resolved_px=2_000_000)
check("nothing saved", best_transcript([], wanted) is None)
check("the highest budget wins", best_transcript([low, high], wanted) == high)
for name, change in {
    "another model": {"model_revision": "b" * 40},
    "another prompt": {"prompt_version": 2},
    "another limit": {"max_tokens": 1},
    "another page": {"page": page_id(b"q")},
}.items():
    check(f"{name} is never served", best_transcript([made(**change)], wanted) is None)

finish()
