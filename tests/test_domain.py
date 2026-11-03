"""The document pipeline's data model: identity, derivation keys, citation mapping.

Run:  uv run python tests/test_domain.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402
from PIL import Image  # noqa: E402

from nmv.domain import Derivation, Page, PageId, Turn, page_id  # noqa: E402

print("page_id")
first = page_id(b"page bytes")
check("is 64 hex characters", len(first) == 64 and int(first, 16) >= 0, first)
check("is stable for the same bytes", first == page_id(b"page bytes"))
check("changes with one byte", first != page_id(b"page byteS"))

print("Derivation.key")
base = Derivation(first, 1_000_000, "a" * 40, 1, 2048)
check("is stable", base.key() == replace(base).key())
changes = {
    "page": page_id(b"other"),
    "resolved_px": 999_999,
    "model_revision": "b" * 40,
    "prompt_version": 2,
    "max_tokens": 4096,
}
for name, value in changes.items():
    changed = replace(base, **{name: value})
    check(f"changes with {name}", changed.key() != base.key())
check(
    "fields cannot run together",
    Derivation(PageId("p1"), 1, "r", 1, 1).key()
    != Derivation(PageId("p"), 11, "r", 1, 1).key(),
)

print("Turn.page_for")
p0, p1, p2 = (page_id(bytes([n])) for n in range(3))
dropped = Turn("assistant", "", "", sent_pages=(p0, p2))
check("a dropped page shifts the mapping", dropped.page_for(1) == p2)
hit = page_id(b"search hit")
with_hit = Turn("assistant", "", "", sent_pages=(p0, hit, p1))
check("a search hit in the sent list", with_hit.page_for(1) == hit)
check("an invented index is None", dropped.page_for(7) is None)
check("an index one past the end is None", dropped.page_for(2) is None)
check("a negative index is None", dropped.page_for(-1) is None)
check("nothing sent is None", Turn("user", "", "").page_for(0) is None)

print("Turn.message")
turn = Turn("assistant", "shown", "<co>kept</co: 0>")
check(
    "uses history_text, never display_text",
    turn.message() == {"role": "assistant", "content": "<co>kept</co: 0>"},
)

print("Page")
image = Image.new("RGB", (1, 1))
check(
    "equality ignores the image",
    Page(p0, "a.png", (1, 1), image) == Page(p0, "a.png", (1, 1)),
)

finish()
