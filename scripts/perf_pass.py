"""Step 8's performance pass, items 6-9, on a generated dense document.

    uv run python scripts/perf_pass.py

Both models stay loaded throughout, as in the app, and all MLX work goes through
``nmv.runtime``. Run it with the server stopped and nothing else heavy open.

The set-v1 pages are 1240x1754, about 2.17 MP, and ``plan`` never scales up, so a
sweep over them would compare almost nothing above 2 MP. This draws its own pages at
A4 200 dpi, which is exactly the ceiling, with ~10 pt prose and ~8.5 pt tables. Every
second page goes through ``photocopy``, because clean rendered text flatters a low
budget. Everything in it is invented.

Pages and transcripts land in the git-ignored ``local/``, with a throwaway store, so
this never touches the app's saved transcripts. A numbers-only summary lands in
``evals/results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from itertools import cycle
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.make_set import CLAUSES, Sheet, particulars, photocopy  # noqa: E402
from evals.metrics import cer, field_recall, strip_markup  # noqa: E402
from nmv import model, prompts, reader, runtime, transcripts  # noqa: E402
from nmv.domain import Derivation, PageId, Transcript, page_id  # noqa: E402
from nmv.imaging import MAX_PIXELS, encode, normalize, plan  # noqa: E402
from nmv.r7b_format import Phase  # noqa: E402

GB = 1e9
A4_200_DPI = (1654, 2339)
MARGIN = 130
SEED = 8
BUDGETS = (MAX_PIXELS, 2_000_000, 1_000_000)  # the ceiling first: it is the yardstick
QUESTIONS = (
    "What is the file reference for clause 3?",
    "Which office is responsible for that clause?",
)
ACTION_MARKER = "<|START_ACTION|>"
# The strings a misread digit changes: file references, and table codes and amounts.
FIELDS = re.compile(r"[A-H]\d{3}/\d{2}|[A-Z]{2}-\d{4}|\d{1,3},\d{3}\.\d{2}")
OUT = ROOT / "local" / "dense-v1"


@dataclass(frozen=True)
class DensePage:
    path: Path
    id: PageId
    image: Image.Image
    reference: str

    @property
    def fields(self) -> list[str]:
        return FIELDS.findall(self.reference)


def _prose(sheet: Sheet, numbered, rng: random.Random) -> None:
    width = A4_200_DPI[0] - 2 * MARGIN
    y = float(MARGIN)
    while y < A4_200_DPI[1] - MARGIN - 200:
        number, (heading, clause) = next(numbered)
        sheet.text(MARGIN, y, f"{number}. {heading}", 28)
        sentence = particulars(rng, number, heading, 0).sentence()
        y = sheet.wrap(MARGIN, y + 38, width, f"{clause} {sentence}", 26)
        y += 20


def _table(sheet: Sheet, page: int, rng: random.Random) -> None:
    sheet.text(MARGIN, MARGIN, f"Schedule of charges, part {page}", 32)
    columns = (MARGIN, MARGIN + 210, MARGIN + 800, -(A4_200_DPI[0] - MARGIN))
    y = MARGIN + 70
    sheet.row(
        y, list(zip(columns, ("Code", "Service", "Hours", "Amount"), strict=True)), 24
    )
    sheet.rule(MARGIN, y + 36, A4_200_DPI[0] - MARGIN)
    y += 50
    services = [heading for heading, _ in CLAUSES]
    while y < A4_200_DPI[1] - MARGIN - 40:
        code = f"{rng.choice('KLMNPR')}{rng.choice('STUVWX')}-{rng.randint(1000, 9999)}"
        hours = f"{rng.randint(1, 240)}.{rng.choice((0, 25, 5, 75))}"
        amount = f"{rng.randint(1, 98)},{rng.randint(100, 999)}.{rng.randint(10, 99)}"
        cells = (code, rng.choice(services), hours, amount)
        sheet.row(y, list(zip(columns, cells, strict=True)), 24)
        y += 38


def make_pages(count: int) -> list[DensePage]:
    """Every third page a table, the rest prose. Seeded, so reruns match."""
    rng = random.Random(SEED)
    clauses = rng.sample(CLAUSES, k=len(CLAUSES))
    numbered = ((n, c) for n, c in enumerate(cycle(clauses), 1))
    OUT.mkdir(parents=True, exist_ok=True)
    pages = []
    for number in range(1, count + 1):
        sheet = Sheet(A4_200_DPI)
        if number % 3 == 0:
            _table(sheet, number // 3, rng)
        else:
            _prose(sheet, numbered, rng)
        image = photocopy(sheet.image) if number % 2 == 0 else sheet.image
        path = OUT / f"page-{number:02}.png"
        image.save(path)
        pages.append(
            DensePage(
                path,
                page_id(path.read_bytes()),
                normalize(Image.open(path)),
                "\n".join(sheet.lines),
            )
        )
    return pages


def _memory() -> dict:
    import mlx.core as mx

    return {
        "active_gb": round(mx.get_active_memory() / GB, 3),
        "peak_gb": round(mx.get_peak_memory() / GB, 3),
        "cache_gb": round(mx.get_cache_memory() / GB, 3),
    }


def _reset_peak() -> None:
    import mlx.core as mx

    mx.reset_peak_memory()


def _limit() -> float:
    import mlx.core as mx

    return int(mx.device_info()["max_recommended_working_set_size"]) / GB


def swap_used_mb() -> float:
    line = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    return float(line.split("used = ")[1].split("M")[0])


def snapshot() -> dict:
    """Memory since the last snapshot, then a fresh peak for the next one."""
    row = {**runtime.call(_memory), "swap_mb": swap_used_mb()}
    runtime.call(_reset_peak)
    return row


def derivation(page: DensePage, budget: int) -> Derivation:
    resolved = plan(*page.image.size, budget)
    return Derivation(
        page.id,
        resolved.target_pixels,
        model.MODEL_REVISION,
        prompts.TRANSCRIBE_PROMPT_VERSION,
        prompts.TRANSCRIBE_MAX_TOKENS,
    )


def transcribe(studio, page: DensePage, budget: int) -> tuple[dict, str]:
    """One miss, as the app's ``_transcribe`` runs it."""
    started = time.monotonic()
    encoded, resolved = encode(page.image, budget)
    prep = time.monotonic() - started
    stats = model.RunStats()
    sampling = model.Sampling(max_tokens=prompts.TRANSCRIBE_MAX_TOKENS, temperature=0.0)
    turn = [model.user_turn(prompts.TRANSCRIBE_PROMPT, 1)]
    text = "".join(model.stream_reply(studio, turn, [encoded], sampling, stats))
    generated = time.monotonic() - started
    transcript = Transcript(
        derivation(page, budget), text, stats.finish_reason, *resolved.target
    )
    assert transcripts.put(transcript, page.path.name)
    row = {
        "page": page.path.name,
        "budget": budget,
        "resolved_px": resolved.target_pixels,
        "image_tokens": resolved.tokens,
        "seconds": round(time.monotonic() - started, 2),
        "prep_s": round(prep, 3),
        "generate_s": round(generated - prep, 2),
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generation_tokens,
        "prefill_tps": round(stats.prompt_tps, 1),
        "decode_tps": round(stats.generation_tps, 1),
        "truncated": stats.truncated,
        **snapshot(),
    }
    return row, text


def hit_seconds(page: DensePage, budget: int) -> float:
    """The app's exact-hit path: resolve the size, then read the store."""
    started = time.monotonic()
    found = transcripts.get(derivation(page, budget).key())
    elapsed = time.monotonic() - started
    assert found is not None
    return elapsed


def ask(
    loaded, history: list[dict], sent: list[tuple[PageId, str]], deterministic: bool
) -> tuple[dict, str]:
    stats = model.RunStats()
    answer = reader.Answer()
    sampling = reader.reader_sampling(deterministic)
    started = time.monotonic()
    first_piece = first_word = None
    for piece in reader.stream_answer(loaded, history, sent, sampling, stats, answer):
        now = time.monotonic() - started
        first_piece = now if first_piece is None else first_piece
        if first_word is None and piece.phase is Phase.RESPONSE and piece.text.strip():
            first_word = now
    row = {
        "first_output_s": first_piece and round(first_piece, 2),
        "first_word_s": first_word and round(first_word, 2),
        "total_s": round(time.monotonic() - started, 2),
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generation_tokens,
        "thinking_tokens": answer.thinking_tokens,
        "prefill_tps": round(stats.prompt_tps, 1),
        "decode_tps": round(stats.generation_tps, 1),
        "pages_sent": len(answer.sent_pages),
        "pages_dropped": len(answer.dropped_pages),
        "action_channel": ACTION_MARKER in answer.raw,
        **snapshot(),
    }
    return row, answer.response


def conversation(loaded, sent: list[tuple[PageId, str]], deterministic: bool) -> list:
    """Item 7: question 1, then a follow-up on the same pages."""
    history: list[dict] = []
    rows = []
    for number, question in enumerate(QUESTIONS, 1):
        history.append({"role": "user", "content": question})
        row, response = ask(loaded, history, sent, deterministic)
        history.append({"role": "assistant", "content": response})
        rows.append({"question": number, "deterministic": deterministic, **row})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def score(pages: list[DensePage], texts: dict[int, list[str]]) -> dict:
    """Item 8: each budget against the ceiling's transcript and against the truth."""
    rows = {}
    for budget, transcribed in texts.items():
        # ``cer`` strips the hypothesis only, and this reference is a transcript too.
        versus_ceiling = [
            cer(strip_markup(ceiling), text)
            for ceiling, text in zip(texts[MAX_PIXELS], transcribed, strict=True)
        ]
        versus_truth = [
            cer(page.reference, text)
            for page, text in zip(pages, transcribed, strict=True)
        ]
        recall = [
            field_recall(page.fields, text)
            for page, text in zip(pages, transcribed, strict=True)
        ]
        rows[str(budget)] = {
            "cer_vs_ceiling": [round(value, 4) for value in versus_ceiling],
            "cer_vs_truth": [round(value, 4) for value in versus_truth],
            "field_recall": [round(value, 4) for value in recall],
            "cer_vs_ceiling_mean": round(sum(versus_ceiling) / len(pages), 4),
            "cer_vs_truth_mean": round(sum(versus_truth) / len(pages), 4),
            "field_recall_mean": round(sum(recall) / len(pages), 4),
        }
        print(budget, json.dumps(rows[str(budget)]), flush=True)
    return rows


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=10)
    args = parser.parse_args()

    pages = make_pages(args.pages)
    store = OUT / "store.db"
    for leftover in OUT.glob("store.db*"):
        leftover.unlink()
    os.environ[transcripts.DB_ENV] = str(store)

    limit = runtime.call(_limit)
    swap_before = swap_used_mb()
    print(f"limit {limit:.2f} GB, swap {swap_before:.0f} MB", flush=True)

    studio = model.load_studio()
    loaded = reader.load_reader()
    resident = snapshot()
    print("resident", json.dumps(resident), flush=True)

    misses: list[dict] = []
    texts: dict[int, list[str]] = {}
    after_pass: dict[str, dict] = {}
    for budget in BUDGETS:
        texts[budget] = []
        for page in pages:
            row, text = transcribe(studio, page, budget)
            misses.append(row)
            texts[budget].append(text)
            name = f"{page.path.stem}-{budget}.md"
            (OUT / name).write_text(text, encoding="utf-8")
            print(json.dumps(row), flush=True)
        after_pass[str(budget)] = {**runtime.call(_memory), "swap_mb": swap_used_mb()}

    hits = [round(hit_seconds(page, MAX_PIXELS), 5) for page in pages]
    print("hits", hits, flush=True)

    sent = [(page.id, text) for page, text in zip(pages, texts[MAX_PIXELS], strict=True)]
    questions = {}
    for label, chosen in (("3 pages", sent[:3]), (f"{len(sent)} pages", sent)):
        for deterministic in (True, False):
            mode = "deterministic" if deterministic else "sampling"
            print(label, mode, flush=True)
            rows = conversation(loaded, chosen, deterministic)
            questions[f"{label}, {mode}"] = rows

    swap_after = swap_used_mb()
    summary = {
        "machine": "M2 Pro 16 GB",
        "code": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", ".", ":!evals/results")),
        "pages": args.pages,
        "limit_gb": round(limit, 2),
        "cache_limit_bytes": runtime.MLX_CACHE_LIMIT_BYTES,
        "resident": resident,
        "after_pass": after_pass,
        "swap_before_mb": swap_before,
        "swap_after_mb": swap_after,
        "misses": misses,
        "hit_seconds": hits,
        "questions": questions,
    }
    name = f"{date.today().isoformat()}-{git('rev-parse', '--short', 'HEAD')}-perf-pass"
    target = ROOT / "evals" / "results" / f"{name}.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary["sweep"] = score(pages, texts)  # slow, and needs no model: keep it last
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
