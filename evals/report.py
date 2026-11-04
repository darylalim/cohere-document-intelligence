"""Compare two runs: a per-metric table, the changed cases, and any flags.

uv run python evals/report.py <baseline.json> <candidate.json>
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


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base, new = (json.loads(Path(p).read_text("utf-8")) for p in sys.argv[1:3])
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
