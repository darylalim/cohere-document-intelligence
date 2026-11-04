# tier: weights
"""Both pinned models on the sample invoice, through the real worker.

Run:  uv run python tests/test_models_e2e.py

Never in CI: GitHub's 7 GB macOS runner cannot hold both models. Run it before
merging a milestone, and after any change to a model pin, mlx, mlx-vlm or mlx-lm.
Everything runs at temperature 0, because ``Sampling`` has no seed.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402
from PIL import Image  # noqa: E402

from nmv import model, reader, runtime  # noqa: E402
from nmv.citations import parse  # noqa: E402
from nmv.documents import match_line  # noqa: E402
from nmv.domain import page_id  # noqa: E402
from nmv.grounding import parse_boxes  # noqa: E402
from nmv.imaging import MAX_PIXELS, encode, normalize  # noqa: E402
from nmv.prompts import (  # noqa: E402
    LOCATE_MAX_TOKENS,
    LOCATE_PROMPT,
    TRANSCRIBE_MAX_TOKENS,
    TRANSCRIBE_PROMPT,
)
from nmv.r7b_format import Phase, render  # noqa: E402

SAMPLE = ROOT / "assets/sample-invoice.png"
# The PAID stamp on the 0-1000 grid: its pixels, (820, 1180, 1140, 1400), over the
# invoice's 1240 x 1754 (step 5d's grid maths).
PAID_STAMP = (661, 673, 919, 798)
STAMP_TOLERANCE = 20
# Printed on the invoice.
INVOICE_TEXT = ("acme logistics", "nmv-2026-0847", "4,182.55")
TOTAL = "4,182.55"
# The real rag template around a one-character document, measured in step 2.
TEMPLATE_TOKENS = 1658
GB = 1e9

if not (model.is_cached() and reader.is_reader_cached()):
    print("SKIP  the pinned weights are not in the Hugging Face cache")
    sys.exit(0)


def nmv(image: Image.Image, prompt: str, limit: int) -> str:
    sampling = model.Sampling(max_tokens=limit, temperature=0.0)
    turn = [model.user_turn(prompt, 1)]
    return "".join(model.stream_reply(studio, turn, [image], sampling))


def memory() -> tuple[float, float, float]:
    import mlx.core as mx

    limit = int(mx.device_info()["max_recommended_working_set_size"])
    return mx.get_peak_memory() / GB, mx.get_cache_memory() / GB, limit / GB


data = SAMPLE.read_bytes()
page = page_id(data)
original = normalize(Image.open(SAMPLE))
encoded, _ = encode(original, MAX_PIXELS)
studio = model.load_studio()

print("grounding")
raw = nmv(encoded, LOCATE_PROMPT.format(text="the PAID stamp"), LOCATE_MAX_TOKENS)
boxes = parse_boxes(raw)
check("one box parsed", len(boxes) >= 1, raw)
if boxes:
    edges = (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)
    check(
        f"within {STAMP_TOLERANCE} grid units of the stamp",
        all(
            abs(a - b) <= STAMP_TOLERANCE for a, b in zip(edges, PAID_STAMP, strict=True)
        ),
        str(edges),
    )

print("transcription")
transcript = nmv(encoded, TRANSCRIBE_PROMPT, TRANSCRIBE_MAX_TOKENS)
flat = re.sub(r"\s+", " ", transcript).lower()
for text in INVOICE_TEXT:
    check(f"contains {text!r}", text in flat)

print("reader")
loaded = reader.load_reader()
question = [{"role": "user", "content": "What is the total due?"}]
answer = reader.Answer()
pieces = list(
    reader.stream_answer(
        loaded,
        question,
        [(page, transcript)],
        reader.reader_sampling(True),
        answer=answer,
    )
)
clean, citations = parse(answer.response)
check("the answer has the total", TOTAL in clean, clean)
check(
    "it cites page index 0",
    any(0 in citation.doc_indices for citation in citations),
    answer.response,
)
check("sent_pages is the page", answer.sent_pages == (page,))
shown = "".join(piece.text for piece in pieces if piece.phase is Phase.RESPONSE)
check("no control tokens in the response", "<|" not in shown, shown)
check("no control tokens in the clean text", "<|" not in clean and "<co" not in clean)

print("locate")
if citations:
    sent = match_line(citations[0].span, transcript)
    raw = nmv(encoded, LOCATE_PROMPT.format(text=sent), LOCATE_MAX_TOKENS)
    located = parse_boxes(raw)
    check("a box parsed", len(located) >= 1, f"{sent!r}: {raw}")
    if located:
        centre = (located[0].x1 + located[0].x2) / 2, (located[0].y1 + located[0].y2) / 2
        check(
            "it is where the total is", centre[0] < 500 and centre[1] < 333, str(centre)
        )

print("template overhead")
tiny = reader.documents_for([(page, "x")])
tokens = len(loaded.tokenizer.encode(render(loaded.tokenizer, question, tiny)))
check(
    f"within 10% of {TEMPLATE_TOKENS}",
    abs(tokens - TEMPLATE_TOKENS) <= TEMPLATE_TOKENS * 0.1,
    str(tokens),
)

print("memory")
peak, cache, limit = runtime.call(memory)
print(f"  peak {peak:.2f} GB, cache {cache:.2f} GB, limit {limit:.2f} GB")
check("peak stays under the limit", peak <= limit)

finish()
