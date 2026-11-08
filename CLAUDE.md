# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app for reading documents locally on Apple Silicon via MLX. Cohere's North
Micro Vision (2.4B native-resolution vision-language model) reads the pages, and Command
R7B answers questions over them. Three pages: multi-turn visual chat, grounding with a
bounding-box overlay, and Ask documents, which transcribes pages, answers with citations
and boxes the cited text on the page.

**8-bit NMV and 4-bit R7B, chosen for 16 GB.**
`mlx-community/North-Micro-Vision-Instruct-8bit` (~3.2 GB) and
`mlx-community/c4ai-command-r7b-12-2024-4bit` (~4.5 GB) stay resident together and peak at
about 11 GB. Do not add other variants without being asked.

**Both checkpoints are pinned to Hub commits** — `MODEL_REVISION` in `nmv/model.py` and
`READER_REVISION` in `nmv/reader.py`, which every load, config read and cache check takes.
A checkpoint can run code here two ways: mlx-vlm executes the file `config.json` names
under `model_file`, and it loads the cohere_compass tokenizer with
`trust_remote_code=True`, so transformers runs any module an `auto_map` entry names — from
this repo, or from another repo's `main`. `load()` also downloads `*.py` with the weights,
so following `main` would run whatever that repo publishes next. Moving a pin is a code
change: confirm the new commit has no `.py` files and no `model_file` or `auto_map` key in
any `.json`, then update its `--revision` in the prefetch commands here and in the README.

## Commands

```bash
uv sync                                        # create venv, install pinned deps
uv run streamlit run streamlit_app.py          # run the app (localhost:8501)

uv run python tests/run.py all                 # every tier except weights
uv run ruff check .                            # lint  (E, F, I, UP, B, SIM, C4, S608)
uv run ruff format .                           # format (line-length 90)
uv run ty check                                # type check (targets 3.11)

# prefetch the pinned weights (MODEL_REVISION in nmv/model.py, READER_REVISION in nmv/reader.py)
uv run hf download mlx-community/North-Micro-Vision-Instruct-8bit \
  --revision 26e6b3aef507899fbf86057593d99fc12418e26f
uv run hf download mlx-community/c4ai-command-r7b-12-2024-4bit \
  --revision 2ae601b3d780410caacd2148c6c96e620144f22d
```

`tests/run.py` takes a tier: `fast` (pure Python), `mlx` (Apple Silicon, no weights), `ui`
(AppTest, models stubbed) or `weights` (real models, local only). `all` runs the first
three, and `--with-weights` adds the last. A test declares its tier on its first line as
`# tier: <name>`, and a file with no header is `fast`, so adding one needs no list edited.
Tests are plain scripts with no pytest, so `uv run python tests/<file>.py` runs just one.

ruff and ty are locked dev dependencies, so use `uv run ruff` / `uv run ty`, not the
`uvx` forms — those resolve independently of `uv.lock` and will drift the day a new
release ships. Reach for `uvx` only for tools the project does not declare.

ty runs clean. `nmv/model.py` and `nmv/r7b_format.py` use `cast` where the libraries'
annotations are narrower than their runtime behaviour (`apply_chat_template` returns a
`str` under the default `return_messages=False`; `image=` accepts PIL objects because
`prepare_inputs` routes them through `process_image`). Keep the casts and their comments
rather than replacing them with blanket ignores.

There are no `ty: ignore` suppressions. ty reports an unused one as a warning and exits
1 on warnings, so if a false positive ever needs one, keep it targeted, explain it in a
comment, and delete both the day ty stops producing the error rather than silencing the
warning.

**Restart the server after editing anything under `nmv/`.** Streamlit hot-reloads
`streamlit_app.py` and `app_pages/*`, but `nmv/*` are ordinary imported modules cached
in `sys.modules`; edits there surface as a stale `ImportError` until you restart.

When working with Python, invoke the relevant `/astral:<skill>` for `uv`, `ty`, and
`ruff` to ensure best practices are followed.

## CI and releases

`.github/workflows/ci.yml` runs every command above on an Apple Silicon runner, tier by
tier and without `weights`, plus the static greps. The tests have no Linux job because
they cannot run there. The lock does install on Linux — `mlx` publishes manylinux wheels —
but its extension links `libmlx.so`, which the wheel does not ship. `mlx-metal` supplies
the library on macOS (as `libmlx.dylib`); on Linux only the `mlx[cpu]` / `mlx[cuda]` extras
do, and they are not locked. So `import mlx.core` fails there.

`audit` is the one check on Linux, allowed because it installs nothing project-specific: it
exports the lock and runs a pinned `pip-audit` over it. (`release` runs on Linux too, but it
only publishes.)

**Editing `version` in `pyproject.toml` is a publish action.** On `main`, once CI passes,
the workflow tags that commit and publishes a GitHub release with notes built from commit
subjects — there is no draft to approve and no undo that unpublishes a tag others may
have fetched. Do not bump the version to test something or as an incidental tidy-up; bump
it when you mean to ship. The workflow refuses a version that is not valid PEP 440 or not
ahead of the highest existing tag, and marks pre-releases so they do not take "Latest",
but nothing checks whether you *meant* it.

**A bump is two files.** `uv.lock` records the project's own version alongside its
dependencies, so run `uv lock` and commit the result with the `pyproject.toml` change:

```bash
uv lock && git add pyproject.toml uv.lock
```

Skipping it fails CI before anything publishes: `uv sync --locked` reports drift
(`The lockfile at uv.lock needs to be updated`), that fails `check`, and `release`
declares `needs: check`. So the outcome of a half-finished bump is a red run and *no
release at all* rather than a broken one — safe, but worth recognising, since uv's hint
names `uv lock` without connecting it to the version you just changed.

## Claude Code hooks

`.claude/settings.json` enforces the hardest rules, so a refusal is the rule speaking. If a
guard is wrong, fix its script and re-test it with a piped payload; never delete it.
[`CONTRIBUTING.md`](CONTRIBUTING.md#claude-code-hooks) says what each refusal means.

- `version-guard.sh`: denies any change to `version`.
- MLX import guard: flags a module-scope MLX import after an edit.
- `pin-guard.sh`: flags a `*_REVISION` that is not a 40-hex commit, or a download of `*.py`.
- `stop-checks.sh`: runs ty, the fast tier and the static greps at the end of a turn.
- ruff fix and format, and `restart-reminder.sh`: never refuse.
- Permission rules: `hf cache rm` / `delete` and `rm *transcripts.db*` always ask.

## The constraint that shapes the architecture

MLX binds each thread's default stream, like any made with `mx.new_stream`, to that
thread, and an array that has not been evaluated yet is work queued on one of those
streams. Only that thread can evaluate it; anywhere else raises
`RuntimeError: There is no Stream(gpu, N) in current thread.` (N is 1 in this app).
Arrays that have been evaluated are plain data and cross threads freely.

mlx-vlm's `load()` leaves some of the model unevaluated. It calls
`mx.eval(model.parameters())`, but `parameters()` skips underscore-named attributes, so
30 rotary `_inv_freq` tables stay pending: a copy in each of the 28 language-model
layers, which the forward pass never reads, and one per attention type in
`language_model.model.rotary_embeddings`. The forward pass reads
`rotary_embeddings["sliding_attention"]`, so the first one has to run on the thread that
loaded the model; once it has, any thread can generate. Streamlit starts a fresh
ScriptRunner thread for each rerun the browser requests (`st.rerun()` and
`st.switch_page()` stay on the current one), and `st.cache_resource` loads during
whichever rerun asks first, normally the page's initial render. So a naive app fails on
its very first reply, and on every reply after it.

The suspect the name invites, `generation_stream` in `mlx_vlm/generate/common.py`, is
innocent: it is a `ThreadLocalStream`, which hands each thread a stream of its own.

`nmv/runtime.py` runs the import, the load and every later MLX call on one long-lived
worker thread, streaming tokens back over a bounded queue, so nothing pending ever
changes threads. It also serialises GPU work across browser tabs, which matters because
MLX generation is not reentrant.

**Invariant: mlx-vlm and mlx-lm are imported only inside worker functions, never at
module scope.** Importing on the worker keeps anything they queue at import time on the thread that
will evaluate it, and a module-scope import in page code is where direct, off-worker
calls begin. Check it before committing — the same pattern CI and the edit hook use:

```bash
grep -rnE --include='*.py' '^(from|import) mlx(_vlm|_lm)?\b' nmv/ app_pages/ streamlit_app.py
```

That must return nothing.

If you ever broaden the ruff `select` set, note that `PLC0415`
(`import-outside-top-level`) fires on 29 function-local imports (`uv run ruff check
--select PLC0415 --no-fix .`). Ignore the rule, never hoist them to satisfy it:

- 24 are MLX and deliberate: 5 mlx-vlm, 4 mlx-lm and 15 `mlx.core`, in `nmv/`, `scripts/`
  and `tests/`.
- 5 are `huggingface_hub`, which the invariant does not cover: `is_cached()`,
  `is_reader_cached()`, `download_reader()`, `_is_download_error()` and the doctor's
  `missing_weights()`, which must run before `uv sync`.

Consequences:

- `nmv/model.py` keeps every `from mlx_vlm import ...` inside a function body.
- `nmv/imaging.py` **mirrors** mlx-vlm's `smart_resize` as plain arithmetic rather than
  importing it. `tests/test_resize_parity.py` holds the copy to upstream across 80
  size/budget combinations — run it after touching either.
- Anything new that calls MLX goes through `runtime.call()` (blocking) or
  `runtime.iterate()` (streaming). Never call those *from* the worker: single worker,
  instant deadlock.

**The same shape applies to SQLite: never share a `sqlite3` connection across threads.**
Streamlit reruns on fresh threads and `sqlite3` refuses a connection from another one, so
`nmv/transcripts.py` opens one per call through `_connect()`.

Transcripts are the contents of the user's documents. They live in
`~/.cache/cohere-document-intelligence/transcripts.db` (`NMV_TRANSCRIPT_DB` overrides it,
so point that at a temp file when testing). Clear them with **Clear saved transcripts** on
Ask documents, which calls `transcripts.clear()`.

## Module roles

| File | Responsibility |
|---|---|
| `nmv/runtime.py` | MLX worker thread. Read first. |
| `nmv/trace.py` | One `RunRecord` per action: numbers and IDs only. **No MLX, no Streamlit.** |
| `nmv/log.py` | Terminal logging, level from `NMV_LOG_LEVEL`. Metadata only, never content. |
| `nmv/model.py` | Loading (`st.cache_resource`), `Sampling`, `RunStats`, streaming |
| `nmv/reader.py` | Command R7B: loading, context budget, streaming answers |
| `nmv/r7b_format.py` | R7B's template and thinking/response phases. **No MLX import.** |
| `nmv/prompts.py` | Every fixed prompt and its version. **No MLX, no Streamlit.** |
| `nmv/domain.py` | The document pipeline's data model. **No MLX, no Streamlit.** |
| `nmv/documents.py` | Ask documents' decisions, testable. **No MLX, no Streamlit.** |
| `nmv/citations.py` | Parse R7B's citation tags. **No MLX, no Streamlit.** |
| `nmv/transcripts.py` | The SQLite transcript store. **No MLX, no Streamlit.** |
| `nmv/errors.py` | Sort exceptions into what the user can do. **No MLX, no Streamlit.** |
| `nmv/render.py` | Make untrusted model output safe to render as Markdown |
| `nmv/system.py` | What macOS says about memory. **No MLX, no Streamlit.** Never raises. |
| `nmv/imaging.py` | Token budgeting, `encode`/`prepare`, EXIF normalisation. **No MLX import.** |
| `nmv/pdf.py` | Render PDF pages for Ask documents, under one lock. **No MLX, no Streamlit.** |
| `nmv/grounding.py` | 0–1000 box parsing and PIL overlay rendering |
| `nmv/ui.py` | Shared sidebar; returns `(budget, sampling, SidebarSlots)` |
| `app_pages/*.py` | Page scripts — UI only, direct scripts with no `main()` |
| `scripts/doctor.py` | Preflight check. **Standard library only at module level.** |
| `scripts/memory_check.py` | Whether both models fit on this Mac. Needs the weights. |
| `scripts/perf_pass.py` | The performance pass on a generated dense document |
| `evals/` | Eval sets, runners, metrics and the baseline. See `evals/README.md`. |

## Domain rules that are easy to get wrong

**Token count scales with pixel area.** One token per 32×32 px block, so
`tokens = pixels / 1024`. The image budget slider is the only lever on prefill latency
and KV-cache memory. Ceiling is 3,868,706 px (A4 @ 200 dpi). The checkpoint's
`preprocessor_config.json` advertises a far larger `size.longest_edge`, but mlx-vlm
only honours keys literally named `min_pixels`/`max_pixels`, which that file lacks — so
that larger number never applies and should not be reintroduced as a limit.

**Pre-resizing before handing images to mlx-vlm is safe only because `plan()` models
mlx-vlm's own second pass.** mlx-vlm re-runs `smart_resize` on whatever it receives
using the *checkpoint's* bounds, not our budget — so a tight budget that lands below
`MIN_PIXELS` gets scaled back up, and the size and token count shown to the user would
be wrong. `plan()` therefore applies our budget, then mlx-vlm's bounds, and reports that
fixed point. `tests/test_resize_parity.py` asserts every target survives mlx-vlm's pass
untouched and that the token count matches its own patch arithmetic.

**Grounding boxes are on a 0–1000 grid**, not in pixels, so they apply to any rendering
of the image. Overlays are therefore drawn on the full-resolution upload. Both the
encoded copy and the overlay copy must come from `imaging.normalize()` — skipping EXIF
transposition puts every box on the wrong axis for rotated phone photos.

**The model emits bare coordinate arrays** like `[659, 671, 921, 799]` even when asked
for labelled JSON. Labels are optional throughout `nmv/grounding.py`; don't add code
that assumes they exist.

**Never send a system prompt.** Cohere states the model was not trained to follow them.
Task instructions belong in the user turn.

**Per-turn image markers matter.** `model.user_turn(text, image_count)` emits
`{"type": "image"}` markers so mlx-vlm binds each image to the turn it arrived on.
Without them every image collapses onto the most recent user message and multi-turn
conversations with images break.

## Streamlit conventions in use

Targeting Streamlit 1.66 (floor 1.62, the first release with `streamlit.typing`).
`use_container_width` is deprecated — use `width="stretch"` or `width="content"`.
Sidebar stats use a reserved `st.empty()` slot filled *after*
generation, otherwise the panel trails one interaction behind. Prefer native elements
over custom HTML/CSS. `st.chat_input(accept_file=...)` renders its attach control as a
`+`, not a paperclip — check `assets/screenshot-*.png` before describing any UI
affordance in prose or captions.

**A `key` does not survive a page switch — every sidebar widget needs
`persist_state="session"`.** Streamlit folds the active page's script hash into every
element id, "even if key_as_main_identity is specified"
(`streamlit/elements/lib/utils.py`), so `budget_mp` on Chat and `budget_mp` on Grounding
are two different widgets. `nmv/ui.py` is rendered from both pages, so without session
scope a page switch silently reset the image budget to the ceiling — a ~7x change in
prefill cost with nothing on screen to say so. The same rule covers the sliders
`_sampling_control` hides in deterministic mode, which were dropped on every toggle.
Anything new in that sidebar carries `persist_state="session"` too.

Check page-switch persistence in a real browser, not with `streamlit.testing.v1.AppTest`.
On a page change the browser sends no widget states for the old page, and that is what
lets `ScriptRunner`'s pre-run cleanup ("Clear widget state on page change") hand each
`persist_state="session"` value to the new page's widget. `AppTest.switch_page().run()`
resends the old page's states instead, so the first run after *any* switch uses the
defaults. A run that finishes copies the value across afterwards, so `.value` looks right
although the run never used it; Grounding's runs end in `st.stop()` until there is a
result, which skips that copy, so there the defaults stick. A browser keeps the budget,
max tokens and temperature in both directions, and the "Up to N tokens" caption shows the
server has them. Removing Grounding's `st.stop()` calls only makes AppTest look fixed.

**The app has a dark and a light theme, built as one system (Polar Indigo).** Every number
and trade is in the comments of `.streamlit/config.toml`; read them before changing a
colour. `tests/test_theme_contrast.py` holds the contrast floors. The rules:

- Keep both `[theme.dark]` and `[theme.light]`, because that pair is what offers System,
  Light and Dark. Shared keys go in `[theme]` and `[theme.sidebar]`.
- Leave `base` unset. Streamlit forces it per mode.
- `baseFontSize`, `baseFontWeight`, `fontFaces`, `metricValueFontSize`/`Weight` and
  `showSidebarBorder` are only accepted in `[theme]`. Streamlit logs and drops them
  elsewhere.
- Both halves keep the neutral hue (~266°), the accent hue (273.5°, the widest gap in
  `PALETTE` in `nmv/grounding.py`) and the aurora hues. Only lightness and chroma move.
- Do not lighten the dark ground towards Nord's `#2e3440` without re-running the trade:
  the selected task needs OKLCH L ≤ 0.19.
- Keep light chrome at or below L 0.942, so the document stays the brightest thing on
  Grounding.
- Do not swap either accent for a teal. On a dark ground it lands on `#2dd4bf`, which *is*
  `PALETTE[6]`.
- Check contrast on resolved colours, not config values, because Streamlit derives most
  rendered colours. A change meant for one half must leave the other's resolved colours
  unchanged.
- Check both modes in a real browser. After a live mode switch the sidebar `divider="gray"`
  rules keep the old mode's colour, so reload before judging them.
- Restart the server after editing the file. Theme changes do not hot-reload.
- A pick is stored per URL path, so a reload on `/grounding` not seeing a choice made after
  loading `/` is Streamlit, not an app bug.

For any non-trivial Streamlit work, invoke the `developing-with-streamlit` skill — it
routes to version-matched reference docs bundled inside the installed package.

## Measured baselines

- **M2 Pro 16 GB, 8-bit NMV and 4-bit R7B:** both loaded 7.66 GB · peak 11.16 GB · a ceiling
  page takes 31.2 s (prefill 267 tok/s, decode 59 tok/s).
- **M2 Max 32 GB, bf16 NMV, before the move to 8-bit:** load 3.4 s · prefill ~590 tok/s ·
  decode 72–88 tok/s · peak 6.17 GB.

The full figures, and how to reproduce them, are in
[`docs/performance.md`](docs/performance.md).

## Documentation map

Every fact has one home. Edit it there and link to it from anywhere else.

| File | Owns |
|---|---|
| `README.md` | What the app is, quick start, the licence table |
| `docs/privacy.md` | What is stored, logged and sent |
| `docs/troubleshooting.md` | On-screen errors. Headings reuse the titles in `nmv/errors.py`. |
| `docs/architecture.md` | How it works and why, the model's limits, accessibility |
| `docs/performance.md` | Measured quality, memory and speed |
| `evals/README.md` | How the quality numbers are made |
| `CONTRIBUTING.md`, `RELEASING.md` | Development, tests, CI, and the steps before a bump |
| `SECURITY.md` | How to report a vulnerability, and the trust model |
| `.streamlit/config.toml` | Theme numbers and trades, in comments |
| `CLAUDE.md` | Rules and invariants, each with its reason |
| `docs/plans/` | Records, not current truth |
