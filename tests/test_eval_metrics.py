"""The eval metrics, on cases whose answers are known by hand.

Run:  uv run python tests/test_eval_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from evals.metrics import (  # noqa: E402
    best_iou,
    cer,
    cer_raw,
    field_recall,
    iou,
    levenshtein,
    normalise,
    strip_markup,
)
from evals.report import compare  # noqa: E402

print("iou")
check("identical is 1", iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0)
check("disjoint is 0", iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0)
check("touching edges is 0", iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0)
check("nested is the area ratio", iou((0, 0, 10, 10), (0, 0, 5, 10)) == 0.5)
check("half overlap is a third", abs(iou((0, 0, 2, 1), (1, 0, 3, 1)) - 1 / 3) < 1e-9)

print("best_iou")
check("nothing parsed is None, not 0", best_iou((0, 0, 1, 1), []) is None)
check("a parsed miss is 0", best_iou((0, 0, 1, 1), [(5, 5, 6, 6)]) == 0.0)
check("takes the best", best_iou((0, 0, 10, 10), [(5, 5, 6, 6), (0, 0, 10, 10)]) == 1.0)

print("levenshtein and cer")
check("kitten/sitting is 3", levenshtein("kitten", "sitting") == 3)
check("empty against text", levenshtein("", "abc") == 3)
check("exact is 0", cer("Total due", "Total due") == 0.0)
check("one substitution in four", cer("abcd", "abxd") == 0.25)
check("case and whitespace do not count", cer("Total  Due\n", "total due") == 0.0)
check("whitespace collapses", normalise(" a\t b\n\nc ") == "a b c")
check("empty reference, empty output", cer("", " ") == 0.0)
check("empty reference, some output", cer("", "x") == 1.0)
check("runaway output exceeds 1", cer("ab", "abcdefgh") == 3.0)

print("strip_markup")
table = "# Bill\n\n| Item | Qty |\n|------|:---:|\n| **Pens** | 2 |\n"
check(
    "a table reads as its cells", normalise(strip_markup(table)) == "bill item qty pens 2"
)
check("html tags go, their text stays", normalise(strip_markup("<p>PAID</p>")) == "paid")
check("a hyphen in text survives", strip_markup("co-operation") == "co-operation")
check("a negative number survives", normalise(strip_markup("-4.50")) == "-4.50")
check("a line of only dashes goes", normalise(strip_markup("a\n---\nb")) == "a b")
check(
    "markup does not count in cer",
    cer("Item Qty Pens 2", "| Item | Qty |\n|--|--|\n| Pens | 2 |") == 0.0,
)
check(
    "but it does in cer_raw",
    cer_raw("Item Qty Pens 2", "| Item | Qty |\n|--|--|\n| Pens | 2 |") > 0.3,
)
check("a real misread still counts", cer("Pens 2", "| Pans | 2 |") > 0)

print("field_recall")
check("all present", field_recall(["ACME", "4,182.55"], "acme due 4,182.55") == 1.0)
check("half present", field_recall(["ACME", "9.99"], "ACME") == 0.5)
check("whitespace inside a field", field_recall(["Total due"], "total\n due") == 1.0)
check("no fields is vacuous", field_recall([], "x") == 1.0)
check(
    "bold does not hide a field", field_recall(["Total due"], "**Total** **due**") == 1.0
)

print("compare")


def run(hits: list[bool], page_cer: float) -> dict:
    return {
        "grounding": [
            {"doc": "d", "page": 0, "target": str(i), "hit": hit}
            for i, hit in enumerate(hits)
        ],
        "transcription": [{"doc": "d", "page": 0, "cer": page_cer}],
        "metrics": {"cer_mean": page_cer},
    }


same = compare(run([True] * 4, 0.1), run([True] * 4, 0.1))
check("identical runs raise no flag", same["flags"] == [] and same["lost"] == [])
two = compare(run([True] * 4, 0.1), run([True, True, False, False], 0.1))
check("losing exactly 2 targets is within the limit", two["flags"] == [])
three = compare(run([True] * 4, 0.1), run([True, False, False, False], 0.1))
check("losing 3 targets flags", len(three["flags"]) == 1 and len(three["lost"]) == 3)
swap = compare(run([True, False], 0.1), run([False, True], 0.1))
check("a gain is not netted against a loss", swap["lost"] and swap["gained"])
check(
    "a 1 point rise is within the limit",
    not compare(run([], 0.1), run([], 0.11))["flags"],
)
check("a 2 point rise flags", len(compare(run([], 0.1), run([], 0.12))["flags"]) == 1)
check("a fall never flags", compare(run([], 0.5), run([], 0.1))["flags"] == [])
try:
    compare(run([True], 0.1), run([True, True], 0.1))
    check("different targets refuse to compare", False)
except ValueError:
    check("different targets refuse to compare", True)

finish()
