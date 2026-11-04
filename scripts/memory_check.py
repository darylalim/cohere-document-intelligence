"""Measure whether both models fit on this Mac: the memory half of decision 2.

    uv run python scripts/memory_check.py [--cache-limit-gb 1]

Runs through ``nmv.runtime`` like everything else, so every MLX call is on the
worker. Pass means peak plus cache at or below 90% of the working-set limit, with
no swap growth. Step 8 repeats this with the full feature set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nmv import model, prompts, reader, runtime  # noqa: E402
from nmv.domain import page_id  # noqa: E402
from nmv.grounding import parse_boxes  # noqa: E402
from nmv.imaging import MAX_PIXELS, encode, normalize  # noqa: E402

GB = 1e9
PASS_FRACTION = 0.9
SAMPLE = ROOT / "assets" / "sample-invoice.png"
CHAT_TURNS = 3
CHAT_MAX_TOKENS = 512


def _memory() -> dict:
    import mlx.core as mx

    return {
        "active": mx.get_active_memory() / GB,
        "peak": mx.get_peak_memory() / GB,
        "cache": mx.get_cache_memory() / GB,
    }


def _reset_peak() -> None:
    import mlx.core as mx

    mx.reset_peak_memory()


def _limit() -> float:
    import mlx.core as mx

    return int(mx.device_info()["max_recommended_working_set_size"]) / GB


def _set_cache_limit(gb: float) -> None:
    import mlx.core as mx

    mx.set_cache_limit(int(gb * GB))


def swap_used_mb() -> float:
    line = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    return float(line.split("used = ")[1].split("M")[0])


class Log:
    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.stages: list[dict] = []

    def stage(self, name: str, started: float, detail: str = "") -> None:
        memory = runtime.call(_memory)
        total = memory["peak"] + memory["cache"]
        row = {
            "stage": name,
            **{key: round(value, 2) for key, value in memory.items()},
            "peak_plus_cache": round(total, 2),
            "percent_of_limit": round(total / self.limit * 100, 1),
            "swap_mb": swap_used_mb(),
            "seconds": round(time.monotonic() - started, 1),
            "detail": detail,
        }
        self.stages.append(row)
        print(json.dumps(row), flush=True)
        runtime.call(_reset_peak)


def greedy(max_tokens: int) -> model.Sampling:
    return model.Sampling(max_tokens=max_tokens, temperature=0.0)


def ceiling_page() -> Image.Image:
    """The sample invoice at the ceiling. It is smaller than the budget and
    ``plan`` never scales up, so a ceiling-budget page has to be made."""
    page = normalize(Image.open(SAMPLE))
    scale = (MAX_PIXELS / (page.width * page.height)) ** 0.5
    size = (int(page.width * scale), int(page.height * scale))
    return page.resize(size, Image.Resampling.LANCZOS)


def long_documents() -> list[tuple]:
    """Eval-set transcripts, enough to run the reader out of context."""
    pages = []
    for path in sorted((ROOT / "evals" / "set-v1").glob("*/truth.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        for entry in truth["text"]:
            text = entry["reference"]
            pages.append((page_id(text.encode()), text))
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-limit-gb", type=float)
    args = parser.parse_args()

    limit = runtime.call(_limit)
    swap_before = swap_used_mb()
    print(f"working-set limit {limit:.2f} GB, swap before {swap_before:.0f} MB")
    if args.cache_limit_gb is not None:
        runtime.call(_set_cache_limit, args.cache_limit_gb)
        print(f"MLX cache limited to {args.cache_limit_gb} GB")
    log = Log(limit)

    started = time.monotonic()
    studio = model.load_studio()
    log.stage("load NMV 8-bit", started)

    started = time.monotonic()
    loaded = reader.load_reader()
    log.stage("load reader", started)

    page, _ = encode(ceiling_page(), MAX_PIXELS)
    tokens = page.width * page.height // 1024

    started = time.monotonic()
    stats = model.RunStats()
    prompt = prompts.LOCATE_PROMPT.format(text="the PAID stamp")
    turn = [model.user_turn(prompt, 1)]
    text = "".join(model.stream_reply(studio, turn, [page], greedy(512), stats))
    log.stage(
        "grounding at the ceiling",
        started,
        f"{stats.prompt_tokens} prompt tokens, {len(parse_boxes(text))} box",
    )

    started = time.monotonic()
    stats = model.RunStats()
    turn = [model.user_turn(prompts.TRANSCRIBE_PROMPT, 1)]
    limit_tokens = prompts.TRANSCRIBE_MAX_TOKENS
    transcript = "".join(
        model.stream_reply(studio, turn, [page], greedy(limit_tokens), stats)
    )
    log.stage(
        "transcribe at the ceiling",
        started,
        f"{stats.prompt_tokens} prompt, {stats.generation_tokens} generated",
    )

    started = time.monotonic()
    stats = model.RunStats()
    answer = reader.Answer()
    history = [{"role": "user", "content": "What is the total due?"}]
    pages = [(page_id(transcript.encode()), transcript)]
    sampling = reader.reader_sampling(True)
    list(reader.stream_answer(loaded, history, pages, sampling, stats, answer))
    log.stage(
        "question over the transcript",
        started,
        f"{stats.prompt_tokens} prompt tokens: {answer.response[:70]!r}",
    )

    started = time.monotonic()
    stats = model.RunStats()
    answer = reader.Answer()
    history = [{"role": "user", "content": "Which documents mention a total?"}]
    many = long_documents()
    list(reader.stream_answer(loaded, history, many, sampling, stats, answer))
    log.stage(
        "question at the reader's context limit",
        started,
        f"{stats.prompt_tokens} prompt tokens, {len(answer.sent_pages)} pages sent, "
        f"{len(answer.dropped_pages)} dropped",
    )

    # The worst case: Chat re-sends every earlier image, with the reader resident.
    messages: list[dict] = []
    images: list[Image.Image] = []
    for number in range(1, CHAT_TURNS + 1):
        started = time.monotonic()
        stats = model.RunStats()
        images.append(page)
        messages.append(model.user_turn(f"Summarise image {number} briefly.", 1))
        reply = "".join(
            model.stream_reply(studio, messages, images, greedy(CHAT_MAX_TOKENS), stats)
        )
        messages.append(model.assistant_turn(reply))
        log.stage(
            f"chat turn {number}, {number} ceiling images",
            started,
            f"{stats.prompt_tokens} prompt tokens (~{tokens} per image)",
        )

    worst = max(row["peak_plus_cache"] for row in log.stages)
    swap_after = swap_used_mb()
    passed = worst <= limit * PASS_FRACTION and swap_after <= swap_before
    print(
        json.dumps(
            {
                "limit_gb": round(limit, 2),
                "bar_gb": round(limit * PASS_FRACTION, 2),
                "worst_peak_plus_cache_gb": worst,
                "swap_before_mb": swap_before,
                "swap_after_mb": swap_after,
                "pass": passed,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
