"""Run Command R7B over an eval set's questions and write its scores.

    uv run python evals/run_reader.py --source gold
    uv run python evals/run_reader.py --source nmv --transcripts <folder>
    uv run python evals/run_reader.py --source nmv --transcripts <folder> --conversation

``gold`` reads the truth's reference text, which measures reasoning and citing alone.
``nmv`` reads a saved ``evals/run.py`` transcript, which measures the real pipeline, and
adds Locate on the first cited span. The gap between the two is what OCR costs.

Every question is asked alone at temperature 0, on all of its document's pages.
``--conversation`` instead asks each document's questions in a row with sampling on,
and counts only the replies that use R7B's action channel: the raw tool-call answer
has been seen on follow-ups, never on a first question. Sampled runs do not repeat, so
it never feeds the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.metrics import (  # noqa: E402
    IOU_HIT,
    answer_found,
    best_iou,
    centre_inside,
    citation_scores,
    correct_refusal,
    followed_injection,
)
from evals.run import PACKAGES, generate, git, package_version  # noqa: E402
from nmv import model, prompts, r7b_format, reader  # noqa: E402
from nmv.citations import parse  # noqa: E402
from nmv.documents import match_line  # noqa: E402
from nmv.domain import PageId, page_id  # noqa: E402
from nmv.grounding import parse_boxes  # noqa: E402
from nmv.imaging import MAX_PIXELS, encode, normalize  # noqa: E402
from nmv.r7b_format import START_ACTION  # noqa: E402

INVENTED = -1  # a citation to a position that was never sent


def sent_pages(truth: dict, source: str, transcripts: Path | None) -> list:
    """``(id, text)`` per page. The id takes the position, as two pages can match."""
    pages: list[tuple[PageId, str]] = []
    for index, entry in enumerate(truth["text"]):
        if source == "gold":
            text = entry["reference"]
        else:
            assert transcripts is not None
            path = transcripts / truth["id"] / f"page-{index + 1}.md"
            text = path.read_text(encoding="utf-8")
        pages.append((page_id(f"{truth['id']}/{index}\n{text}".encode()), text))
    return pages


def ask(loaded, history: list[dict], pages: list, deterministic: bool):
    stats = model.RunStats()
    answer = reader.Answer()
    sampling = reader.reader_sampling(deterministic)
    started = time.monotonic()
    list(reader.stream_answer(loaded, history, pages, sampling, stats, answer))
    return answer, stats, time.monotonic() - started


def cited_pages(citations: list, answer: reader.Answer, pages: list) -> list[int]:
    """Citations as page positions in the document, in reply order."""
    position = {page: index for index, (page, _) in enumerate(pages)}
    found: list[int] = []
    for citation in citations:
        for index in citation.doc_indices:
            sent = answer.sent_pages
            found.append(position[sent[index]] if index < len(sent) else INVENTED)
    return found


def locate(
    studio, folder: Path, truth: dict, case: dict, citation, pages: list, saved: Path
) -> dict:
    """Box the first cited span as the app does, on the page it cites."""
    wanted = truth_box(truth, case["q"])
    if wanted is None:  # a computed answer, or one the generator gave no box
        return {}
    page = case["cited"][0]
    if page == INVENTED or page != wanted["page"]:
        return {"located": False, "locate_hit": False, "locate_centre": False}
    original = normalize(Image.open(folder / truth["pages"][page]))
    image, resolved = encode(original, MAX_PIXELS)
    sent = match_line(citation.span, pages[page][1])
    prompt = prompts.LOCATE_PROMPT.format(text=sent)
    limit = prompts.LOCATE_MAX_TOKENS
    text, _, _ = generate(studio, image, resolved, prompt, limit, "locate")
    saved.write_text(text, encoding="utf-8")
    boxes = [(b.x1, b.y1, b.x2, b.y2) for b in parse_boxes(text)]
    score = best_iou(wanted["bbox"], boxes)
    return {
        "located": bool(boxes),
        "locate_iou": score,
        "locate_hit": score is not None and score >= IOU_HIT,
        "locate_centre": any(centre_inside(wanted["bbox"], box) for box in boxes),
    }


def truth_box(truth: dict, question: str) -> dict | None:
    return next(q["answer_bbox"] for q in truth["qa"] if q["q"] == question)


def score(truth: dict, question: dict, answer, stats, seconds, pages) -> tuple:
    clean, citations = parse(answer.response)
    cited = cited_pages(citations, answer, pages)
    answerable = question["answer"] is not None
    case = {
        "doc": truth["id"],
        "q": question["q"],
        "kind": question.get("kind"),
        "answerable": answerable,
        "cited": cited,
        "thinking_tokens": answer.thinking_tokens,
        "action_channel": START_ACTION in answer.raw,
        "truncated": stats.truncated,
        "pages_dropped": len(pages) - len(answer.sent_pages),
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generation_tokens,
        "seconds": round(seconds, 2),
    }
    if answerable:
        precision, recall = citation_scores(cited, question["pages"])
        case |= {
            "correct": answer_found(question["answer"], question["kind"], clean),
            "precision": precision,
            "recall": recall,
            "off_evidence": sum(page not in question["pages"] for page in cited),
        }
        if "skill" in question:  # set-v3: what the question takes, and the wrong turn
            case |= {
                "skill": question["skill"],
                "gave_distractor": answer_found(
                    question["distractor"], question["kind"], clean
                ),
            }
    else:
        known = [(q["answer"], q["kind"]) for q in truth["qa"] if q["answer"] is not None]
        case |= {"refused": correct_refusal(clean, known, cited), "off_evidence": 0}
    injection = truth.get("injection")
    if injection and injection["q"] == question["q"]:
        case["followed_injection"] = followed_injection(
            clean, question["kind"], injection["answer"], injection["planted"]
        )
    return case, citations


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(cases: list[dict]) -> dict:
    answerable = [c for c in cases if c["answerable"]]
    refusals = [c for c in cases if not c["answerable"]]
    located = [c for c in answerable if "locate_hit" in c]
    injected = [c for c in cases if "followed_injection" in c]
    precision = [c["precision"] for c in answerable if c["precision"] is not None]
    return {
        "answerable": len(answerable),
        "correct": sum(c["correct"] for c in answerable),
        "uncited": sum(not c["cited"] for c in answerable),
        "citation_precision_mean": mean(precision),
        "citation_recall_mean": mean([c["recall"] for c in answerable]),
        "off_evidence": sum(c["off_evidence"] for c in answerable),
        "unanswerable": len(refusals),
        "refused": sum(c["refused"] for c in refusals),
        "located_cases": len(located),
        "locate_no_box": sum(not c["located"] for c in located),
        "locate_hits": sum(c["locate_hit"] for c in located),
        "locate_centre_hits": sum(c["locate_centre"] for c in located),
        "injection_cases": len(injected),
        "injection_followed": sum(c["followed_injection"] for c in injected),
        "action_channel": sum(c["action_channel"] for c in cases),
        "with_thinking": sum(c["thinking_tokens"] > 0 for c in cases),
        "truncated": sum(c["truncated"] for c in cases),
        "pages_dropped": sum(c["pages_dropped"] > 0 for c in cases),
        **by_skill(answerable),
    }


def by_skill(answerable: list[dict]) -> dict:
    """Per skill: answered, correct, and how often the reply gave the distractor, the
    answer one step short. Empty for sets without skills, so their summary is as
    before."""
    skills = sorted({c["skill"] for c in answerable if "skill" in c})
    if not skills:
        return {}
    table = {}
    for skill in skills:
        mine = [c for c in answerable if c.get("skill") == skill]
        table[skill] = {
            "questions": len(mine),
            "correct": sum(c["correct"] for c in mine),
            "gave_distractor": sum(c["gave_distractor"] for c in mine),
        }
    return {
        "gave_distractor": sum(c.get("gave_distractor", False) for c in answerable),
        "by_skill": table,
    }


def conversation(loaded, truth: dict, pages: list, out: Path) -> list[dict]:
    """Every question in a row, as a follow-up to the last."""
    history: list[dict] = []
    rows = []
    for turn, question in enumerate(truth["qa"], 1):
        history.append({"role": "user", "content": question["q"]})
        answer, _, _ = ask(loaded, history, pages, False)
        (out / f"turn-{turn}.txt").write_text(answer.raw, encoding="utf-8")
        history.append({"role": "assistant", "content": answer.response})
        rows.append({"turn": turn, "action": START_ACTION in answer.raw})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("gold", "nmv"), required=True)
    parser.add_argument("--set", default="set-v1", dest="eval_set")
    parser.add_argument("--transcripts", type=Path, help="an evals/runs/ folder")
    parser.add_argument("--only", default="", help="only document ids with this prefix")
    parser.add_argument("--conversation", action="store_true")
    parser.add_argument(
        "--think", action="store_true", help="open the plan, not the answer"
    )
    args = parser.parse_args()
    if args.think:
        r7b_format.OPENING = r7b_format.START_THINKING
    if args.source == "nmv" and args.transcripts is None:
        parser.error("--source nmv needs --transcripts")

    folder = ROOT / "evals" / args.eval_set
    sha = git("rev-parse", "--short", "HEAD")
    name = f"{date.today().isoformat()}-{sha}-reader-r7b-{args.source}"
    if args.eval_set != "set-v1":
        name += f"-{args.eval_set}"
    if args.think:
        name += "-think"
    if args.conversation:
        name += "-conversation"
    raw = ROOT / "evals" / "runs" / name
    started = time.monotonic()
    loaded = reader.load_reader()
    locating = args.source == "nmv" and not args.conversation
    studio = model.load_studio() if locating else None

    cases: list[dict] = []
    turns: list[dict] = []
    for path in sorted(folder.glob("*/truth.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        doc = truth["id"]
        if not doc.startswith(args.only):
            continue
        out = raw / doc
        out.mkdir(parents=True, exist_ok=True)
        pages = sent_pages(truth, args.source, args.transcripts)
        if args.conversation:
            rows = conversation(loaded, truth, pages, out)
            turns += [{"doc": doc, **row} for row in rows]
            print(doc, [row["turn"] for row in rows if row["action"]], flush=True)
            continue
        for number, question in enumerate(truth["qa"]):
            history = [{"role": "user", "content": question["q"]}]
            answer, stats, seconds = ask(loaded, history, pages, True)
            (out / f"q{number}.txt").write_text(answer.raw, encoding="utf-8")
            case, citations = score(truth, question, answer, stats, seconds, pages)
            if studio is not None and case["answerable"] and citations:
                saved = out / f"locate-q{number}.txt"
                case |= locate(
                    studio, path.parent, truth, case, citations[0], pages, saved
                )
            cases.append(case)
            print(
                json.dumps(
                    {
                        k: case.get(k)
                        for k in ("doc", "q", "correct", "cited", "refused", "locate_hit")
                    }
                ),
                flush=True,
            )

    summary = {
        "config": "reader-r7b",
        "source": args.source,
        "set": args.eval_set,
        "only": args.only,
        "repo": reader.READER_REPO,
        "revision": reader.READER_REVISION,
        "code": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", ".", ":!evals/results")),
        "packages": {package: package_version(package) for package in PACKAGES},
        "settings": asdict(reader.reader_sampling(not args.conversation)),
        "opening": r7b_format.OPENING,
        "seconds": round(time.monotonic() - started, 1),
    }
    if args.conversation:
        summary["turns"] = turns
        summary["metrics"] = {
            "replies": len(turns),
            "action_channel": sum(row["action"] for row in turns),
            "first_turn_action": sum(r["action"] for r in turns if r["turn"] == 1),
        }
    else:
        summary["metrics"] = summarise(cases)
        summary["answers"] = cases
    target = ROOT / "evals" / "results" / f"{name}.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
