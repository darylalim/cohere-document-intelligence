"""The eval sets on disk are what the generator draws, and set-v3's summary adds up.

Run:  uv run python tests/test_eval_sets.py

A set is never edited in place, so a change to ``evals/make_set.py`` that moves a
committed set's truth is a bug: results would stop being comparable without anything
saying so. Only the truth is compared, because writing PNGs is slow and the truth
records every string drawn and where.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from evals import make_set  # noqa: E402
from evals.run_reader import summarise  # noqa: E402

print("committed sets match the generator")
for name, build in make_set.BUILDERS.items():
    folder = ROOT / "evals" / name
    docs = build()
    on_disk = sorted(path.parent.name for path in folder.glob("*/truth.json"))
    check(f"{name} has the same documents", on_disk == sorted(d.id for d in docs))
    for doc in docs:
        saved = json.loads((folder / doc.id / "truth.json").read_text("utf-8"))
        # The round trip turns tuples into lists, as saving does.
        drawn = json.loads(json.dumps(doc.truth(), ensure_ascii=False))
        check(f"{name}/{doc.id} truth unchanged", saved == drawn)

print("set-v3")
steps = make_set.build_steps()
questions = [q for doc in steps for q in doc.qa if q["answer"] is not None]
skills = {q["skill"] for q in questions}
check("every skill is asked", skills == set(make_set.SKILLS))
check("some answers are computed", any(q["computed"] for q in questions))
check("some questions need two pages", any(len(q["pages"]) > 1 for q in questions))
check(
    "no page past three, so every page fits the reader",
    all(len(doc.sheets) <= 3 for doc in steps),
)
check(
    "the seeds are the first that pass", make_set.find_step_seeds() == make_set.STEP_SEEDS
)


def case(skill=None, correct=True, distractor=False, dropped=0, answerable=True):
    row = {
        "answerable": answerable,
        "correct": correct,
        "cited": [0],
        "precision": 1.0,
        "recall": 1.0,
        "off_evidence": 0,
        "refused": True,
        "action_channel": False,
        "thinking_tokens": 0,
        "truncated": False,
        "pages_dropped": dropped,
    }
    if skill:
        row |= {"skill": skill, "gave_distractor": distractor}
    return row


print("summarise")
plain = summarise([case(), case(correct=False)])
check("a set without skills has no skill table", "by_skill" not in plain)
check("nor a distractor count", "gave_distractor" not in plain)
mixed = summarise(
    [
        case("compare"),
        case("compare", correct=False, distractor=True),
        case("arithmetic", distractor=True),
        case("chain", dropped=1),
        case(answerable=False),
    ]
)
table = mixed["by_skill"]
check("skills are counted apart", table["compare"]["questions"] == 2)
check("correct per skill", table["compare"]["correct"] == 1)
check("distractor per skill", table["arithmetic"]["gave_distractor"] == 1)
check("distractor overall", mixed["gave_distractor"] == 2)
check("a right answer may still name the distractor", table["arithmetic"]["correct"] == 1)
check("a dropped page is counted per question", mixed["pages_dropped"] == 1)

finish()
