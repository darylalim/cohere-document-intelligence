"""``parse_boxes`` against what the model really emits.

Files in ``tests/fixtures/grounding/`` are its output at temperature 0 on
``assets/sample-invoice.png``. ``*.handmade.txt`` marks the cases it will not
produce on demand. Fixtures come only from ``assets/``: committed test data is
published.

Run:  uv run python tests/test_grounding_parse.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from nmv.grounding import Box, is_point_answer, parse_boxes  # noqa: E402

FIXTURES = ROOT / "tests/fixtures/grounding"


def boxes_in(name: str) -> list[Box]:
    return parse_boxes((FIXTURES / name).read_text(encoding="utf-8"))


stamp = Box(660, 672, 920, 798)
labelled = Box(659, 671, 921, 798, "PAID")

check("a bare array has no label", boxes_in("bare-array.txt") == [stamp])
check("labelled JSON keeps its label", boxes_in("labelled-json.txt") == [labelled])
check("a prose answer has no boxes", boxes_in("prose.txt") == [])
check("an empty list has no boxes", boxes_in("empty.handmade.txt") == [])
check(
    "a reply cut off mid-array keeps the boxes that finished",
    boxes_in("truncated.handmade.txt") == [labelled],
    str(boxes_in("truncated.handmade.txt")),
)
check("swapped corners are put in order", boxes_in("swapped.handmade.txt") == [stamp])

check("a point is not a box", parse_boxes("[[852, 301, 852, 301]]") == [])
check("a point is recognised", is_point_answer("[[852, 301, 852, 301]]"))
check("a real box is not a point", not is_point_answer("[[660, 672, 920, 798]]"))
check("a zero-width line is not a point", not is_point_answer("[[10, 10, 10, 900]]"))
check("prose is not a point", not is_point_answer("I see [nothing] here"))
check("no reply is not a point", not is_point_answer(""))

pixels = stamp.to_pixels(1240, 1754)
check(
    "the grid maps onto the PAID stamp within a few pixels",
    all(abs(a - b) <= 3 for a, b in zip(pixels, (820, 1180, 1140, 1400), strict=True)),
    str(pixels),
)

finish()
