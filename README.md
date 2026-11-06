# Cohere Document Intelligence

[![CI](https://github.com/darylalim/cohere-document-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/cohere-document-intelligence/actions/workflows/ci.yml)

Streamlit app for running
[Cohere North Micro Vision](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
locally on Apple Silicon with MLX.

North Micro Vision is a 2.4B-parameter, Apache-2.0 vision-language model that reads
images at their **native resolution** rather than squashing them to a fixed square. Its
strengths are documents, charts, OCR and visual grounding — exactly the workload where
running locally beats shipping scans to an API.

This app runs the **8-bit** conversion only
(`mlx-community/North-Micro-Vision-Instruct-8bit`, ~3.2 GB), which leaves room for a second
model on a 16 GB Mac.

## Pages

### Chat

Multi-turn document Q&A, OCR, charts and captioning. Attach images with the **+**
button; each one stays bound to the turn it arrived on, so a later question can refer
back to a page uploaded several turns earlier. Responses stream token by token.

![The Chat page: an invoice attached to the first turn and answered, then a follow-up answered from the same image without re-attaching it](assets/screenshot-chat.png)

*The follow-up carries no image of its own. Prompt tokens go 2,166 → 2,213 across the two
turns, so the page attached on turn one is still bound to that turn.*

### Grounding

Single-shot detection with a bounding-box overlay, a table of parsed coordinates, and a
PNG download. Turn on **Deterministic** for steadier boxes.

![The Grounding page: the invoice's text block boxed, with both detections listed beside the overlay in 0-1000 grid and pixel coordinates](assets/screenshot-grounding.png)

*Grounding on `assets/sample-invoice.png` — 2,145 visual tokens at 1248×1760. Row 2's
pixel span `817, 1177, 1142, 1403` is the PAID stamp, further down the page than the crop
shows.*

`assets/sample-invoice.png` is a synthetic invoice for exercising both pages.

## Requirements

- Apple Silicon Mac. 16 GB unified memory is enough; peak usage is ~4.5 GB.
- Python 3.11–3.13 (the pinned venv uses 3.12).
- [uv](https://docs.astral.sh/uv/).

Measured quality and memory are in [`docs/performance.md`](docs/performance.md).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The first launch pulls ~3.2 GB of weights into the Hugging Face cache. The app loads a
fixed commit of the checkpoint rather than whatever `main` points at, so to fetch them
ahead of time, ask for that same commit:

```bash
uv run hf download mlx-community/North-Micro-Vision-Instruct-8bit \
  --revision 26e6b3aef507899fbf86057593d99fc12418e26f
```

Ask documents also needs Command R7B (4.5 GB), which it offers to download on the page.
To fetch that ahead of time too:

```bash
uv run hf download mlx-community/c4ai-command-r7b-12-2024-4bit \
  --revision 2ae601b3d780410caacd2148c6c96e620144f22d
```

## Accessibility

Every box is described in words and numbered to match its table row, every image has a
caption, and streaming can be turned off in the sidebar. Contrast floors are tested on both
themes. Known gaps, and what is not yet checked, are in
[`docs/architecture.md`](docs/architecture.md#accessibility).

## Documentation

- [Privacy](docs/privacy.md): what is stored, logged and sent
- [Troubleshooting](docs/troubleshooting.md): what each on-screen error means
- [Architecture](docs/architecture.md): how the app works, and the model's limits
- [Performance](docs/performance.md): measured quality, memory and speed
- [Evaluation](evals/README.md): how those measurements are made
- [Contributing](CONTRIBUTING.md): development, tests and CI
- [Releasing](RELEASING.md): the steps before a version bump

## License

The app's own code is Apache-2.0 — see [`LICENSE`](LICENSE). Apache-2.0 rather than MIT
for the patent grant (§3) and, more to the point here, for the explicit statement that no
trademark rights travel with it (§6). *Cohere* and *North Micro Vision* are Cohere's
marks; this is an unaffiliated project that runs their published checkpoint.

No weights are vendored here: `uv sync` installs no checkpoint, and the ~3.2 GB arrives
from the Hugging Face cache on first run. Both
[CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
and the [mlx-community 8-bit conversion](https://huggingface.co/mlx-community/North-Micro-Vision-Instruct-8bit)
this app loads are Apache-2.0 as well, so the two halves line up — but they are
governed by their own model cards, not by this repository's `LICENSE`.
