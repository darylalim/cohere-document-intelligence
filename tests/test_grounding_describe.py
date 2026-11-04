"""A box in words, for anyone who cannot see the overlay.

Run:  uv run python tests/test_grounding_describe.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nmv.grounding import Box, describe, draw_boxes, parse_boxes  # noqa: E402


def place(x: float, y: float) -> str:
    """The position clause of a 100-unit box centred on (x, y)."""
    return describe(Box(x - 50, y - 50, x + 50, y + 50)).split(" · ")[0]


print("corners and centre")
check("upper left", place(100, 100) == "upper left")
check("upper right", place(900, 100) == "upper right")
check("lower left", place(100, 900) == "lower left")
check("lower right", place(900, 900) == "lower right")
check("the exact centre", place(500, 500) == "centre")
check("upper centre", place(500, 100) == "upper centre")
check("middle left", place(100, 500) == "middle left")

print("threshold edges")
check("332 is left", place(332, 500) == "middle left")
check("333 is centre", place(333, 500) == "centre")
check("667 is centre", place(667, 500) == "centre")
check("668 is right", place(668, 500) == "middle right")
check("332 is upper", place(500, 332) == "upper centre")
check("668 is lower", place(500, 668) == "lower centre")

print("sizes")
check("a full page", describe(Box(0, 0, 1000, 1000)) == "most of the page")
check("exactly 90% both ways", describe(Box(50, 50, 950, 950)) == "most of the page")
wide = describe(Box(0, 400, 1000, 600))
check(
    "full width alone keeps its size",
    wide == "centre · about 100% of the page wide, 20% tall",
    wide,
)
tiny = describe(Box(100, 100, 120, 110))
check("a tiny box", tiny == "upper left · under 5% of the page wide, under 5% tall", tiny)
narrow = describe(Box(100, 100, 120, 400))
check(
    "one small side", narrow == "upper left · under 5% of the page wide, 30% tall", narrow
)

print("swapped coordinates")
check(
    "corners in the wrong order read the same",
    describe(Box(919, 798, 661, 673)) == describe(Box(661, 673, 919, 798)),
)
parsed = parse_boxes("[919, 798, 661, 673]")[0]
check(
    "and match what parse_boxes keeps",
    describe(parsed) == describe(Box(919, 798, 661, 673)),
)

print("labels")
check(
    "a label changes nothing",
    describe(Box(0, 0, 100, 100, "stamp")) == describe(Box(0, 0, 100, 100)),
)

print("every box keeps its number")


def captions(boxes: list[Box], show_labels: bool) -> list[str]:
    drawn: list[str] = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return real(self, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", spy):
        draw_boxes(Image.new("RGB", (400, 400), "white"), boxes, show_labels=show_labels)
    return drawn


pair = [Box(100, 100, 300, 300, "stamp"), Box(500, 500, 700, 700)]
check("text on", captions(pair, True) == ["#1 stamp", "#2"], str(captions(pair, True)))
check("text off", captions(pair, False) == ["#1", "#2"], str(captions(pair, False)))

print("the sample invoice")
stamp = describe(Box(661, 673, 919, 798))
expected = "lower right · about 25% of the page wide, 15% tall"
check("the PAID stamp, which needs halves rounding up", stamp == expected, stamp)

finish()
