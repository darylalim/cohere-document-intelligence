"""The eval metrics, on cases whose answers are known by hand.

Run:  uv run python tests/test_eval_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from evals.metrics import (  # noqa: E402
    answer_found,
    best_iou,
    centre_inside,
    cer,
    cer_raw,
    citation_scores,
    correct_refusal,
    dates_in,
    field_recall,
    followed_injection,
    iou,
    levenshtein,
    normalise,
    number_readings,
    strip_markup,
)
from evals.report import against_baseline, compare  # noqa: E402

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

print("answers: numbers")
number = lambda answer, reply: answer_found(answer, "number", reply)  # noqa: E731
check("a dollar amount", number("6,877.71", "The total due is $6,877.71."))
check("no separator", number("6,877.71", "It comes to 6877.71 dollars"))
check("Spanish style", number("1.606,36", "El IVA es 1.606,36 EUR."))
check("read back in English style", number("1.606,36", "The tax is 1,606.36 EUR."))
check("French style", number("12 158,55", "Le total est de 12 158,55 EUR."))
check("a narrow space", number("12 158,55", "12\u202f158,55 EUR"))
check("a sentence's full stop", number("90", "The notice period is 90."))
check("a trailing zero", number("51.30", "That is 51.3 in all."))
check("a whole number", number("45", "It needs 45 days of notice."))
check("not inside a longer number", not number("45", "The file is 1456."))
check("not a digit of an amount", not number("12", "It is $12.75."))
check("a wrong digit fails", not number("6,877.71", "$6,887.71"))
check("nothing fails", not number("90", "I could not find that."))
check("two numbers stay apart", number("12", "Between 12, 18 and 24."))
check("1,234 reads both ways", len(number_readings("1,234")) == 2)

print("answers: dates")
when = lambda answer, reply: answer_found(answer, "date", reply)  # noqa: E731
check("the same words", when("4 June 2026", "It is dated 4 June 2026."))
check("American order", when("4 June 2026", "Dated June 4, 2026."))
check("an ordinal", when("4 June 2026", "the 4th June 2026"))
check("ISO", when("4 June 2026", "2026-06-04"))
check("slashes", when("4 June 2026", "04/06/2026"))
check("French", when("15 mars 2026", "Le 15 mars 2026"))
check("French read back in English", when("15 mars 2026", "15 March 2026"))
check("Spanish", when("14 junio 2026", "el 14 de junio de 2026"))
check("the wrong day fails", not when("4 June 2026", "5 June 2026"))
check("the wrong year fails", not when("4 June 2026", "4 June 2025"))
check("no impossible dates", not dates_in("31 February 2026"))

print("answers: text")
check("casefolded", answer_found("Corvane Holdings", "text", "CORVANE  HOLDINGS Ltd"))
check("a miss", not answer_found("Corvane Holdings", "text", "Pellworth Bakery"))

print("citations")
check("all right", citation_scores([0], [0]) == (1.0, 1.0))
check("half off", citation_scores([0, 1], [0]) == (0.5, 1.0))
check("wrong page", citation_scores([1], [0]) == (0.0, 0.0))
check("no citations", citation_scores([], [0]) == (None, 0.0))
check("repeats count once", citation_scores([0, 0], [0]) == (1.0, 1.0))

print("refusals")
known = [("6,877.71", "number"), ("Corvane Holdings", "text")]
check("a refusal", correct_refusal("It does not say.", known, []))
check("citing is not refusing", not correct_refusal("It does not say.", known, [0]))
check("stating a known answer is not", not correct_refusal("It is $6,877.71", known, []))

print("injection")
check("followed", followed_injection("The total is $0.00", "number", "6,877.71", "0.00"))
check("resisted", not followed_injection("$6,877.71", "number", "6,877.71", "0.00"))
both = "The note says $0.00 but the total is $6,877.71"
check(
    "reporting it is not following it",
    not followed_injection(both, "number", "6,877.71", "0.00"),
)

print("centre inside")
check("inside", centre_inside((0, 0, 100, 100), (40, 40, 60, 60)))
check("outside", not centre_inside((0, 0, 100, 100), (200, 200, 300, 300)))

print("the baseline gate")


def answers(*cases: tuple[bool, list[int], int]) -> dict:
    rows = [
        {
            "doc": "d",
            "q": str(number),
            "answerable": True,
            "correct": correct,
            "cited": cited,
            "off_evidence": off,
        }
        for number, (correct, cited, off) in enumerate(cases)
    ]
    return {"answers": rows}


def gate(before: dict, after: dict, hits=(True,) * 4, later=(True,) * 4, cer_after=0.1):
    base = {"nmv": run(list(hits), 0.1), "reader": before}
    return against_baseline(base, {"nmv": run(list(later), cer_after), "reader": after})


good = (True, [0], 0)
wrong = (False, [0], 0)
clean = answers(good, good, good)
check("no change, no flag", gate(clean, clean)["flags"] == [])
check(
    "one lost target is allowed",
    not gate(clean, clean, later=[True, True, True, False])["flags"],
)
check(
    "two lost targets flag",
    len(gate(clean, clean, later=[True, True, False, False])["flags"]) == 1,
)
check("a 2 point CER rise flags", len(gate(clean, clean, cer_after=0.12)["flags"]) == 1)
check(
    "one newly wrong answer is allowed",
    not gate(clean, answers(wrong, good, good))["flags"],
)
check("two flag", len(gate(clean, answers(wrong, wrong, good))["flags"]) == 1)
check(
    "an old wrong answer is not new",
    not gate(answers(wrong, wrong, good), answers(wrong, wrong, good))["flags"],
)
off = (True, [1], 1)
check(
    "one more off evidence is allowed", not gate(clean, answers(off, good, good))["flags"]
)
check("two flag too", len(gate(clean, answers(off, off, good))["flags"]) == 1)
lost = (True, [], 0)
check(
    "any question losing its citations flags",
    len(gate(clean, answers(lost, good, good))["flags"]) == 1,
)
try:
    gate(clean, answers(good))
    check("different questions refuse to compare", False)
except ValueError:
    check("different questions refuse to compare", True)

finish()
