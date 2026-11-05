"""Run one model config over an eval set and write its scores.

    uv run python evals/run.py --config nmv-8bit --set set-v1

One config per process, so two models are never loaded at once. All MLX work
goes through ``nmv.runtime``, like the app. Raw output lands in the git-ignored
``evals/runs/``; a numbers-only summary lands in ``evals/results/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.metrics import IOU_HIT, best_iou, cer, cer_raw, field_recall  # noqa: E402
from nmv import model, prompts, trace  # noqa: E402
from nmv.grounding import parse_boxes  # noqa: E402
from nmv.imaging import MAX_PIXELS, ImagePlan, encode, normalize  # noqa: E402


@dataclass(frozen=True)
class Config:
    repo: str
    revision: str


CONFIGS = {
    # The checkpoint the app used before the 8-bit switch, kept as the yardstick.
    "nmv-bf16": Config(
        "mlx-community/North-Micro-Vision-Instruct-bf16",
        "614b36574d6ecf1c2a79ff3ea28aa89bbc7fcd13",
    ),
    "nmv-8bit": Config(model.MODEL_REPO, model.MODEL_REVISION),
}
PACKAGES = ("mlx", "mlx-vlm", "mlx-lm", "pillow")


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def generate(
    studio,
    image: Image.Image,
    resolved: ImagePlan,
    prompt: str,
    limit: int,
    kind: trace.Kind,
):
    """One greedy reply and its run record. ``Sampling`` has no seed, so temperature 0
    is what makes a run repeatable."""
    run = trace.start(kind)
    run.set(
        image_tokens=resolved.tokens,
        budget_px=MAX_PIXELS,
        resolved_px=resolved.target_pixels,
    )
    stats = model.RunStats()
    sampling = model.Sampling(max_tokens=limit, temperature=0.0)
    turn = [model.user_turn(prompt, 1)]
    text = "".join(model.stream_reply(studio, turn, [image], sampling, stats))
    return text, stats, asdict(trace.finish(run, stats=stats))


def ground(studio, image, resolved, target: dict, doc: str) -> tuple[dict, str]:
    prompt = prompts.LOCATE_PROMPT.format(text=target["target"])
    text, stats, record = generate(
        studio, image, resolved, prompt, prompts.LOCATE_MAX_TOKENS, "locate"
    )
    boxes = [(b.x1, b.y1, b.x2, b.y2) for b in parse_boxes(text)]
    score = best_iou(target["bbox"], boxes)
    case = {
        "doc": doc,
        "page": target["page"],
        "target": target["target"],
        "iou": score,
        "hit": score is not None and score >= IOU_HIT,
        "parse_failure": score is None,
        "truncated": stats.truncated,
        "peak_gb": stats.peak_memory,
        "run": record,
    }
    return case, text


def transcribe(studio, image, resolved, truth: dict, doc: str) -> tuple[dict, str]:
    text, stats, record = generate(
        studio,
        image,
        resolved,
        prompts.TRANSCRIBE_PROMPT,
        prompts.TRANSCRIBE_MAX_TOKENS,
        "transcribe",
    )
    case = {
        "doc": doc,
        "page": truth["page"],
        "cer": cer(truth["reference"], text),
        "cer_raw": cer_raw(truth["reference"], text),
        "field_recall": field_recall(truth["fields"], text),
        "truncated": stats.truncated,
        "generated_tokens": stats.generation_tokens,
        "prefill_tps": stats.prompt_tps,
        "decode_tps": stats.generation_tps,
        "peak_gb": stats.peak_memory,
        "run": record,
    }
    return case, text


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(grounding: list[dict], transcription: list[dict]) -> dict:
    scored = [c["iou"] for c in grounding if c["iou"] is not None]
    every = grounding + transcription
    return {
        "grounding_targets": len(grounding),
        "grounding_hits": sum(c["hit"] for c in grounding),
        "grounding_parse_failures": sum(c["parse_failure"] for c in grounding),
        "grounding_mean_iou": mean(scored),
        "pages": len(transcription),
        "cer_mean": mean([c["cer"] for c in transcription]),
        "cer_raw_mean": mean([c["cer_raw"] for c in transcription]),
        "field_recall_mean": mean([c["field_recall"] for c in transcription]),
        "truncated": sum(c["truncated"] for c in every),
        "peak_gb": max((c["peak_gb"] for c in every), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--set", default="set-v1", dest="eval_set")
    parser.add_argument("--only", default="", help="only document ids with this prefix")
    args = parser.parse_args()

    config = CONFIGS[args.config]
    folder = ROOT / "evals" / args.eval_set
    name = (
        f"{date.today().isoformat()}-{git('rev-parse', '--short', 'HEAD')}-{args.config}"
    )
    if args.eval_set != "set-v1":  # a smaller set must not overwrite the main summary
        name += f"-{args.eval_set}"
    raw = ROOT / "evals" / "runs" / name
    started = time.monotonic()
    studio = model.load_studio(config.repo, config.revision)

    grounding: list[dict] = []
    transcription: list[dict] = []
    for path in sorted(folder.glob("*/truth.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        doc = truth["id"]
        if not doc.startswith(args.only):
            continue
        out = raw / doc
        out.mkdir(parents=True, exist_ok=True)
        for index, page in enumerate(truth["pages"]):
            original = normalize(Image.open(path.parent / page))
            image, resolved = encode(original, MAX_PIXELS)

            for number, target in enumerate(truth["boxes"]):
                if target["page"] != index:
                    continue
                case, text = ground(studio, image, resolved, target, doc)
                grounding.append(case)
                (out / f"box-{number}.txt").write_text(text, encoding="utf-8")
                print(f"{doc} {target['target'][:40]!r} iou={case['iou']}", flush=True)

            case, text = transcribe(studio, image, resolved, truth["text"][index], doc)
            transcription.append(case)
            (out / f"page-{index + 1}.md").write_text(text, encoding="utf-8")
            print(
                f"{doc} p{index + 1} cer={case['cer']:.3f} "
                f"fields={case['field_recall']:.2f} truncated={case['truncated']}",
                flush=True,
            )

    summary = {
        "config": args.config,
        "set": args.eval_set,
        "only": args.only,
        "repo": config.repo,
        "revision": config.revision,
        "code": git("rev-parse", "HEAD"),
        # Results are excluded: an earlier run's file is not a change to the code.
        "dirty": bool(git("status", "--porcelain", "--", ".", ":!evals/results")),
        "packages": {package: package_version(package) for package in PACKAGES},
        "settings": {
            "temperature": 0.0,
            "budget_pixels": MAX_PIXELS,
            "transcribe_prompt_version": prompts.TRANSCRIBE_PROMPT_VERSION,
            "transcribe_max_tokens": prompts.TRANSCRIBE_MAX_TOKENS,
            "locate_prompt_version": prompts.LOCATE_PROMPT_VERSION,
            "locate_max_tokens": prompts.LOCATE_MAX_TOKENS,
        },
        "seconds": round(time.monotonic() - started, 1),
        "metrics": summarise(grounding, transcription),
        "grounding": grounding,
        "transcription": transcription,
    }
    results = ROOT / "evals" / "results"
    results.mkdir(exist_ok=True)
    target = results / f"{name}.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
