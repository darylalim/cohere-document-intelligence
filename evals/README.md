# Evals

These measure quality; the tests only check that the pipeline works. They need
the model weights, so they run locally and never in CI.

## The set

`set-v1/` is 21 generated documents (24 pages): invoices, bank statements,
contracts, receipts, forms, two-column articles, grey photocopies, sideways phone
photos saved with EXIF orientation 6, and French and Spanish invoices. Each folder
holds its pages and a `truth.json` with grounding targets, reference text and
questions. One question in five has no answer on the page.

`make_set.py` draws every document, so it writes the truth from what it drew, with
no hand labelling.

**The set is synthetic only.** Everything in it is invented. Never add real
documents, or anything from `local/`.

## Running it

```bash
uv run python evals/run.py --config nmv-bf16 --set set-v1
uv run python evals/run.py --config nmv-8bit --set set-v1
uv run python evals/report.py evals/results/<bf16>.json evals/results/<8bit>.json
```

Run configs one after another, never two at once. Every run uses temperature 0, the
ceiling image budget, and the prompts and limits in `nmv/prompts.py`, so it measures
what the app does.

Raw output goes to `evals/runs/`, which is git-ignored. A numbers-only summary goes to
`evals/results/`, which is committed.

## Metrics

- **Grounding:** the IoU of the best-matching parsed box, with a hit at 0.5 or above.
  Parse failures are counted apart, so a format problem is not read as a miss.
- **Transcription:** CER after normalising whitespace and case, plus field recall, the
  fraction of key strings present. Markdown syntax counts against CER and reading order
  in tables moves it too, so read field recall alongside it, and compare CER between
  runs rather than as an absolute.

## Adding a set

Never edit `set-v1`, because results would stop being comparable. `make_set.py` refuses
to write over an existing set: change the generator and pass `--out evals/set-v2`.

The bundled font is ASCII-only, so `Sheet.text` adds accents as drawn marks and amounts
use `EUR`. Text it cannot draw raises an error rather than rendering as boxes.
