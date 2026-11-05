"""Compare two runs: a per-metric table, the changed cases, and any flags.

    uv run python evals/report.py <baseline.json> <candidate.json>
    uv run python evals/report.py --against baseline <nmv.json> <reader.json>
    uv run python evals/report.py --write-baseline <nmv.json> <reader.json>

``--against baseline`` holds a full run to ``evals/baseline.json`` and exits 1 on a
flag. Refreshing the baseline is a deliberate commit with the reason in its message:
never do it to make a red report pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Counts, not percentages: on a small set one case is several points.
MAX_LOST_HITS = 2
MAX_CER_RISE = 0.01  # one point, absolute
# Below this a transcription change is decoding noise, not worth listing.
CER_NOTICE = 0.005
BASELINE = Path(__file__).resolve().parent / "baseline.json"
# The baseline gate is tighter than an A/B of two checkpoints: decoding is greedy, so
# the same code and weights give the same numbers.
MAX_BASELINE_LOST_HITS = 1
MAX_NEW_WRONG_ANSWERS = 1
MAX_NEW_OFF_EVIDENCE = 1


def _target_key(case: dict) -> tuple:
    return case["doc"], case["page"], case["target"]


def _page_key(case: dict) -> tuple:
    return case["doc"], case["page"]


def compare(base: dict, new: dict) -> dict:
    """The differences that bear on the decision, as plain data."""
    before = {_target_key(c): c for c in base["grounding"]}
    after = {_target_key(c): c for c in new["grounding"]}
    if before.keys() != after.keys():
        raise ValueError("the runs covered different grounding targets")
    lost = [k for k in before if before[k]["hit"] and not after[k]["hit"]]
    gained = [k for k in before if after[k]["hit"] and not before[k]["hit"]]

    pages_before = {_page_key(c): c for c in base["transcription"]}
    pages_after = {_page_key(c): c for c in new["transcription"]}
    if pages_before.keys() != pages_after.keys():
        raise ValueError("the runs covered different pages")
    pages = [
        (key, pages_before[key]["cer"], pages_after[key]["cer"])
        for key in pages_before
        if abs(pages_before[key]["cer"] - pages_after[key]["cer"]) >= CER_NOTICE
    ]

    cer_rise = new["metrics"]["cer_mean"] - base["metrics"]["cer_mean"]
    flags = []
    if len(lost) > MAX_LOST_HITS:
        flags.append(f"lost {len(lost)} grounding targets (limit {MAX_LOST_HITS})")
    if cer_rise > MAX_CER_RISE:
        flags.append(f"CER rose {cer_rise * 100:.2f} points (limit 1)")
    return {
        "lost": lost,
        "gained": gained,
        "pages": pages,
        "cer_rise": cer_rise,
        "flags": flags,
    }


def _question_key(case: dict) -> tuple:
    return case["doc"], case["q"]


def against_baseline(base: dict, new: dict) -> dict:
    """Flags for ``new`` against the baseline, each a ``{"nmv", "reader"}`` pair."""
    vision = compare(base["nmv"], new["nmv"])
    before = {_question_key(c): c for c in base["reader"]["answers"]}
    after = {_question_key(c): c for c in new["reader"]["answers"]}
    if before.keys() != after.keys():
        raise ValueError("the runs covered different questions")
    answerable = [k for k in before if before[k]["answerable"]]
    wrong = [k for k in answerable if before[k]["correct"] and not after[k]["correct"]]
    off = sum(after[k]["off_evidence"] - before[k]["off_evidence"] for k in answerable)
    uncited = [k for k in answerable if before[k]["cited"] and not after[k]["cited"]]

    flags = []
    if len(vision["lost"]) > MAX_BASELINE_LOST_HITS:
        flags.append(f"lost {len(vision['lost'])} grounding targets (limit 1)")
    if vision["cer_rise"] > MAX_CER_RISE:
        flags.append(f"CER rose {vision['cer_rise'] * 100:.2f} points (limit 1)")
    if len(wrong) > MAX_NEW_WRONG_ANSWERS:
        flags.append(f"{len(wrong)} more answers wrong (limit 1)")
    if off > MAX_NEW_OFF_EVIDENCE:
        flags.append(f"{off} more citations off their evidence page (limit 1)")
    if uncited:
        flags.append(f"{len(uncited)} answerable questions lost all their citations")
    return {"wrong": wrong, "uncited": uncited, "flags": flags, "vision": vision}


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text("utf-8"))


def _gate(nmv: str, reader: str) -> int:
    result = against_baseline(
        _read(str(BASELINE)), {"nmv": _read(nmv), "reader": _read(reader)}
    )
    for doc, question in result["wrong"]:
        print(f"newly wrong: {doc}: {question}")
    for doc, question in result["uncited"]:
        print(f"lost its citations: {doc}: {question}")
    print("FLAGS: " + ("; ".join(result["flags"]) or "none"))
    return 1 if result["flags"] else 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 4 and args[:2] == ["--against", "baseline"]:
        return _gate(args[2], args[3])
    if len(args) == 3 and args[0] == "--write-baseline":
        pair = {"nmv": _read(args[1]), "reader": _read(args[2])}
        BASELINE.write_text(json.dumps(pair, indent=2) + "\n", encoding="utf-8")
        print(BASELINE.name)
        return 0
    if len(args) != 2:
        print(__doc__)
        return 2
    base, new = _read(args[0]), _read(args[1])
    result = compare(base, new)

    print(f"{'metric':<28}{base['config']:>12}{new['config']:>12}")
    for metric, value in base["metrics"].items():
        other = new["metrics"][metric]
        if isinstance(value, float):
            print(f"{metric:<28}{value:>12.4f}{other:>12.4f}")
        else:
            print(f"{metric:<28}{value:>12}{other:>12}")

    print(f"\nlost targets ({len(result['lost'])})")
    for doc, page, target in result["lost"]:
        print(f"  {doc} p{page + 1}: {target}")
    print(f"gained targets ({len(result['gained'])})")
    for doc, page, target in result["gained"]:
        print(f"  {doc} p{page + 1}: {target}")
    print(f"pages whose CER moved ({len(result['pages'])})")
    for (doc, page), before, after in result["pages"]:
        print(f"  {doc} p{page + 1}: {before:.3f} -> {after:.3f}")

    print("\nFLAGS: " + ("; ".join(result["flags"]) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
