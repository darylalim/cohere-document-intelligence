# Releasing

Changing `version` on `main` publishes a release once CI passes. There is no draft to
approve, so work through this list first. The version bump is always done by a person.

1. **Tests.** On an Apple Silicon Mac with both models cached:

   ```bash
   uv run python tests/run.py all --with-weights
   ```

2. **Browser checks.** Work through the manual items in the plan's
   [verification checklist](docs/plans/nmv-8bit-reader-pipeline.md#verification-checklist-run-before-calling-it-done):
   both themes, five widths, VoiceOver, axe, offline and out of memory.

3. **Evaluation.** Do the full run in [`evals/README.md`](evals/README.md#running-it), then:

   ```bash
   uv run python evals/report.py --against baseline <nmv.json> <reader.json>
   ```

   It must pass, or the baseline is refreshed on purpose, in its own commit with the reason
   in the message. Commit the run's summaries from `evals/results/`.

4. **Release notes.** If a model pin moved, draft a line saying so, with the download size,
   because the update pulls new weights.

5. **Bump.** Change `version` in `pyproject.toml` by hand, then commit both files:

   ```bash
   uv lock && git add pyproject.toml uv.lock
   ```

   A version that is not valid PEP 440, or not ahead of the highest tag, is refused. So is a
   bump without `uv lock`, which fails CI and publishes nothing.
