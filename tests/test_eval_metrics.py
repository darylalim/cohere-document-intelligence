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
    field_recall,
    iou,
    levenshtein,
    normalise,
)

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

print("field_recall")
check("all present", field_recall(["ACME", "4,182.55"], "acme due 4,182.55") == 1.0)
check("half present", field_recall(["ACME", "9.99"], "ACME") == 0.5)
check("whitespace inside a field", field_recall(["Total due"], "total\n due") == 1.0)
check("no fields is vacuous", field_recall([], "x") == 1.0)

finish()
