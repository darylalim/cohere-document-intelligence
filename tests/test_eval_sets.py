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

from evals import make_set, run_reader  # noqa: E402
from evals.calculator import TOOL  # noqa: E402
from evals.run_reader import summarise  # noqa: E402
from nmv import r7b_format  # noqa: E402
from nmv.reader import Answer  # noqa: E402

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


print("the calculator loop, on scripted replies")
CALL = (
    "Plan {n}.<|END_THINKING|><|START_ACTION|>["
    '{{"tool_call_id": "1", "tool_name": "{tool}", "parameters": {params}}}'
    "]<|END_ACTION|>"
)
ANSWER = "Done.<|END_THINKING|><|START_RESPONSE|>It is 5.<|END_RESPONSE|>"


def scripted(*raws: str):
    """Stand in for R7B: each call returns the next reply and records its input."""
    seen: list[tuple[list[dict], str]] = []
    queue = list(raws)

    def ask_once(loaded, history: list[dict], pages: list, deterministic: bool):
        seen.append((list(history), r7b_format.OPENING))
        raw = queue.pop(0)
        response = raw.split("<|START_RESPONSE|>")[-1] if "RESPONSE" in raw else ""
        answer = Answer(raw=raw, response=response.split("<|END")[0], thinking_tokens=3)
        return answer, run_reader.model.RunStats(), 1.0

    return seen, ask_once


def calc(n: int, expression: str = "2+3") -> str:
    params = f'{{"expression": "{expression}"}}'
    return CALL.format(n=n, tool="calculate", params=params)


r7b_format.TOOLS = [TOOL]
r7b_format.OPENING = r7b_format.START_THINKING
question = [{"role": "user", "content": "Q?"}]

seen, once = scripted(calc(1), ANSWER)
answer, _, seconds, used = run_reader.ask(None, question, [], True, once)
check("a calculation then an answer", answer.response == "It is 5.")
check("one round", used.rounds == 1 and not used.forced)
check("the result is recorded", used.calculations == [["2+3", "5"]])
turns = seen[1][0]
check("the plan goes back as the call's plan", turns[1]["tool_plan"] == "Plan 1.")
check(
    "the result goes back as a tool turn",
    turns[2] == {"role": "tool", "tool_call_id": "calc-1-0", "content": "5"},
)
check("time adds up", seconds == 2.0)
check("thinking adds up", answer.thinking_tokens == 6)
check(
    "the history passed in is not changed",
    question == [{"role": "user", "content": "Q?"}],
)

seen, once = scripted(
    CALL.format(n=1, tool="direct-injected-document", params="{}"), "It is 5."
)
answer, _, _, used = run_reader.ask(None, question, [], True, once)
check("another tool makes it answer", used.forced and used.rounds == 0)
forced_opening = seen[1][1]
check(
    "from its plan, at the response",
    forced_opening.endswith("Plan 1.<|END_THINKING|><|START_RESPONSE|>"),
)
check("the opening is put back", r7b_format.OPENING == r7b_format.START_THINKING)
check("the forced reply is the answer", answer.raw == "It is 5.")

seen, once = scripted(calc(1), calc(2), calc(3), "It is 5.")
answer, _, _, used = run_reader.ask(None, question, [], True, once)
check("two rounds at most", used.rounds == 2 and used.forced)
check("four calls to R7B", len(seen) == 4)
check("every round is logged", len(used.log) == 4)
check(
    "every call is timed, by step",
    [t["step"] for t in used.timings] == ["first", "tool", "tool", "forced"],
)
check("a zero rate times nothing", used.timings[0]["prefill_seconds"] == 0.0)

seen, once = scripted("Plan.<|END_THINKING|><|START_ACTION|>[not json", "It is 5.")
answer, _, _, used = run_reader.ask(None, question, [], True, once)
check("an unreadable call makes it answer", used.forced and used.calculations == [])

seen, once = scripted(calc(1, "__import__('os')"), ANSWER)
answer, _, _, used = run_reader.ask(None, question, [], True, once)
check(
    "a hostile expression is refused, not run",
    used.calculations[0][1].startswith("error"),
)

print("calculator citations, mapped to the plans' pages")
cited = "Plan <co>4</co: 0:[1]> and <co>1</co: 0:[0,1]>.<|END_THINKING|>"
seen, once = scripted(
    cited + calc(1)[len("Plan 1.<|END_THINKING|>") :],
    "Nothing new.<|END_THINKING|>" + calc(2)[len("Plan 2.<|END_THINKING|>") :],
    ANSWER,
)
answer, _, _, used = run_reader.ask(None, question, [], True, once)
check("each call keeps what the plans cited so far", used.sources == [(1, 0), (1, 0)])

text, mapped, unmapped = run_reader.resolve(
    "It is <co>5</co: 1:[0]>, <co>6</co: 0:[2],2:[0]> and <co>7</co: 3:[0]>.",
    [(1, 0), ()],
)
check(
    "a result citation becomes the plan's pages",
    text == "It is <co>5</co: 0:[1,0]>, <co>6</co: 0:[2],2:[0]> and <co>7</co: 3:[0]>.",
)
check("counts mapped and left", (mapped, unmapped) == (1, 2))
check(
    "document citations are untouched",
    run_reader.resolve("<co>a</co: 0:[3]>", [])[1:] == (0, 0),
)
check(
    "a huge id is not an int() bomb",
    run_reader.resolve(f"<co>a</co: {'9' * 5000}:[0]>", [(1,)])[1] == 0,
)

r7b_format.TOOLS = []
r7b_format.OPENING = r7b_format.START_RESPONSE
seen, once = scripted(calc(1))
answer, _, _, used = run_reader.ask(None, question, [], True, once)
check("without tools a call is left alone", used.rounds == 0 and len(seen) == 1)

finish()
