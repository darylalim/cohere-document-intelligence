"""The calculator R7B may call, on sums it got wrong and on hostile input.

Run:  uv run python tests/test_calculator.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from evals.calculator import (  # noqa: E402
    DIVIDE_BY_ZERO,
    REFUSED,
    TOO_BIG,
    TOO_LONG,
    calculate,
)

print("the sums R7B got wrong on set-v3")
check("a monthly fee over a term", calculate("2563.72 * 36") == "92293.92")
check("written as money", calculate("$2,563.72 x 36") == "92293.92")
check("an 18-month term", calculate("4499.75 × 18") == "80995.5")
check("three payments", calculate("828.97 + 18.32 + 771.16") == "1618.45")
check("a difference", calculate("(2319.40 - 394.68)") == "1924.72")
check("exact, not binary", calculate("0.1 + 0.2") == "0.3")
check("a trailing equals sign", calculate("9.60 + 643.26 =") == "652.86")

print("ordinary arithmetic")
check("precedence", calculate("2 + 3 * 4") == "14")
check("brackets", calculate("(2 + 3) * 4") == "20")
check("unary minus", calculate("-5 + 2") == "-3")
check("division", calculate("10 / 4") == "2.5")
check("repeating decimals stop at six places", calculate("1 / 3") == "0.333333")
check("no negative zero", calculate("-0") == "0")
check("division by zero", calculate("1 / 0") == DIVIDE_BY_ZERO)

print("refused, never run")
for label, text in (
    ("a name", "x"),
    ("a call", "__import__('os').system('true')"),
    ("an attribute", "(1).real"),
    ("a power", "9 ** 9 ** 9"),
    ("a string", "'a' * 3"),
    ("a comparison", "1 < 2"),
    ("a list", "[1, 2]"),
    ("a lambda", "(lambda: 1)()"),
    ("a date", "30 days from 8/03/2026"),
    ("empty", ""),
    ("a boolean", "True + 1"),
    ("a complex number", "1j * 2"),
    ("floor division", "7 // 2"),
):
    check(f"{label} is refused", calculate(text) == REFUSED)
check("not a string", calculate({"expression": "1"}) == REFUSED)
check("too long", calculate("1+" * 150 + "1") == TOO_LONG)
check("deep brackets", calculate("(" * 99 + "1" + ")" * 99) in ("1", REFUSED))
started = time.monotonic()
huge = calculate("9" * 90 + " * " + "9" * 90)
check("huge numbers end quickly", time.monotonic() - started < 1)
check("and say they are too large", huge == TOO_BIG)
check(
    "a large but writable result",
    calculate("123456789 * 987654321") == "121932631112635269",
)

print("output is safe to paste into the template")
for text in ("1 / 7", "-2.5 * 4", "123456789 * 987654321", "1 / 0", "x"):
    result = calculate(text)
    check(
        f"{text!r} has no quote or brace",
        not any(c in result for c in '"\\{}<>'),
    )

finish()
