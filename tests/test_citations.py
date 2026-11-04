"""``citations.parse`` against the template's format and what R7B really emits.

Run:  uv run python tests/test_citations.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv.citations import parse  # noqa: E402
from nmv.domain import Citation  # noqa: E402
from nmv.r7b_format import Phase, stream_phases  # noqa: E402


def indices(text: str) -> list[tuple[int, ...]]:
    return [citation.doc_indices for citation in parse(text)[1]]


print("one span")
clean, found = parse("The total is <co>$5</co: 0:[1]>.")
check("the tags leave the text", clean == "The total is $5.", clean)
check("the citation is exact", found == [Citation("$5", (1,), 13, 15)], str(found))
check("the offsets index the clean text", clean[13:15] == "$5")

print("several spans and groups")
clean, found = parse("<co>a</co: 0:[0]> and <co>b</co: 0:[2]>")
check("offsets account for earlier tags", clean[found[1].start : found[1].end] == "b")
check("several positions", indices("<co>a</co: 0:[1,2]>") == [(1, 2)])
check("spaces in the list", indices("<co>a</co: 0:[1, 2]>") == [(1, 2)])
check("several groups", indices("<co>a</co: 0:[1],0:[3]>") == [(1, 3)])
check("repeats collapse", indices("<co>a</co: 0:[2,2],0:[2]>") == [(2,)])

print("nested")
clean, found = parse("<co>a <co>b</co: 0:[1]> c</co: 0:[2]>")
check("nested tags leave the text", clean == "a b c", clean)
check("the outer span comes first", [c.span for c in found] == ["a b c", "b"])
check("each keeps its own positions", [c.doc_indices for c in found] == [(2,), (1,)])

print("tool call ids")
check("id 1 is dropped", indices("<co>a</co: 1:[0]>") == [])
check("id 1 leaves the span", parse("<co>a</co: 1:[0]>")[0] == "a")
check("id 0 survives beside id 1", indices("<co>a</co: 0:[1,2],1:[0]>") == [(1, 2)])
check("id 10 is not id 0", indices("<co>a</co: 10:[0]>") == [])

print("cut off")
for text in (
    "Total <co>$5",
    "Total <co>$5<",
    "Total <co>$5</",
    "Total <co>$5</co",
    "Total <co>$5</co:",
    "Total <co>$5</co: 0:[",
    "Total <co>$5</co: 0:[1",
    "Total <co>$5</co: 0:[1]",
    "Total <co>$5<|END_RESP",
):
    check(f"{text!r}", parse(text) == ("Total $5", []), repr(parse(text)))
for text in ("Total $5<c", "Total $5<co"):
    check(f"{text!r}", parse(text) == ("Total $5", []), repr(parse(text)))
clean, found = parse("<co>a</co: 0:[0]> then <co>b")
check("spans before the cut survive", clean == "a then b" and len(found) == 1, clean)

print("no citations")
check("plain text is untouched", parse("Nothing cited.") == ("Nothing cited.", []))
check("empty text", parse("") == ("", []))
check("[1] in prose is not a citation", parse("See [1] and 0:[1].")[1] == [])
check("[1] in prose stays", parse("See [1] and 0:[1].")[0] == "See [1] and 0:[1].")
check("a comparison stays", parse("a < b > c")[0] == "a < b > c")
check("other tags stay", parse("<b>x</b>")[0] == "<b>x</b>")

print("malformed")
check("a stray closing tag leaves", parse("a</co: 0:[1]>b") == ("ab", []))
check("a bare closing tag cites nothing", parse("<co>a</co>") == ("a", []))
check("no group cites nothing", parse("<co>a</co: 0>") == ("a", []))
check("an empty list cites nothing", parse("<co>a</co: 0:[]>") == ("a", []))
check("an empty span cites nothing", parse("<co></co: 0:[1]>") == ("", []))
check("a negative position is dropped", indices("<co>a</co: 0:[-1]>") == [])
check("a huge position is dropped", indices(f"<co>a</co: 0:[{'9' * 5000}]>") == [])

print("stray control tokens")
clean, found = parse("<|START_RESPONSE|>a <co>b</co: 0:[0]><|END_RESPONSE|>")
check("control tokens leave the text", clean == "a b", clean)
check("offsets ignore them", found == [Citation("b", (0,), 2, 3)], str(found))
check("inside a span", parse("<co>a<|END_OF_TURN_TOKEN|>b</co: 0:[0]>")[0] == "ab")

print("captured stream")
fixture = Path(__file__).parent / "fixtures" / "r7b" / "response-only.json"
pieces = stream_phases(json.loads(fixture.read_text(encoding="utf-8")))
reply = "".join(piece.text for piece in pieces if piece.phase is Phase.RESPONSE)
clean, found = parse(reply)
check(
    "the clean text is exact",
    clean
    == "The total due is $4,182.55 and Brackenfold Freight Ltd carried the parcels.",
    clean,
)
check("both spans", [c.span for c in found] == ["$4,182.55", "Brackenfold Freight Ltd"])
check("both positions", [c.doc_indices for c in found] == [(0,), (1,)])

finish()
