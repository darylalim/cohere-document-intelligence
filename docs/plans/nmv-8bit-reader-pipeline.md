# Plan: NMV 8-bit + Command R7B document pipeline (16 GB Macs)

Drafted 2026-10-18 for implementation in a fresh session. Target machine:
MacBook Pro M2 Pro, 16 GB. Read `CLAUDE.md` first — every invariant there still
applies, and step 7 lists the parts of it this plan changes.

## Goal

Add a **read → reason → locate** pipeline that keeps both models loaded on 16 GB:

1. **Read** — North Micro Vision transcribes each page image to Markdown.
2. **Reason** — Command R7B answers questions over those transcripts, citing
   the pages it used (its `rag` chat template).
3. **Locate** — for a cited span, NMV grounds it on the page and the app draws
   the box (reuses `nmv/grounding.py`).

## Decisions (already made — do not relitigate)

| Decision | Choice | Why |
|---|---|---|
| Vision model | `mlx-community/North-Micro-Vision-Instruct-8bit` @ `26e6b3aef507899fbf86057593d99fc12418e26f` | 3.16 GB vs ~5 GB for bf16, which frees room for the text model. The user explicitly asked for 8-bit, which lifts the bf16-only rule |
| Text model | `mlx-community/c4ai-command-r7b-12-2024-4bit` @ `2ae601b3d780410caacd2148c6c96e620144f22d` | 4.54 GB. Only small Cohere model whose template has document injection plus `<co>` citations |
| Replace or add | **Replace** bf16 with 8-bit. No variant picker | Keeps one pinned NMV checkpoint, so `is_cached`/prefetch stay simple |
| Dependency upgrade | Upgrade everything to latest first (step 0), as its own commit | Lands the pipeline on a current stack, and keeps upgrade breakage apart from feature bugs |
| New dependency | `mlx-lm==0.32.2` (exact pin, like `mlx-vlm`) | Requires `mlx>=0.32.2` and `transformers>=5.7`, both met after step 0 (mlx 0.32.4, transformers 5.20.1). mlx-vlm does not depend on it |
| Input | Page images only (PNG/JPEG/WebP, existing `IMAGE_TYPES`) | No PDF rasteriser dependency. PDF is a follow-up |

Verified on 2026-10-18 at those exact commits:
- No `.py` files in any of the three repos.
- No `auto_map` or `model_file` key in any `.json` file.

**Fallback text model** if memory measurement fails (step 8):
`mlx-community/tiny-aya-global-8bit-mlx` @ `5d5dc68b477abd7d583563a515bed98604d45d52`.
- 3.58 GB, also clean.
- No `rag` template, so citations would have to be prompted for and parsed
  loosely.
- Keep the text-model repo/revision in constants so this swap is a two-line change.

**Licence:** NMV is Apache-2.0. Command R7B and Tiny Aya are CC-BY-NC-4.0. State
this in the README and in the sidebar caption.

## Memory budget (estimate — step 8 replaces it with measurements)

macOS caps GPU-wired memory at about 10.7 GB on a 16 GB machine.

| Item | Estimate |
|---|---|
| NMV 8-bit weights | 3.2 GB |
| NMV activations + KV at the 3.87 MP ceiling | ~1.2 GB (bf16 peak 6.17 − ~5 GB weights) |
| R7B 4-bit weights | 4.5 GB |
| R7B KV at 8K context | ~0.65 GB |
| **Total** | **~9.6 GB** — fits, but with little headroom |

How the R7B KV figure is derived:
- `sliding_window_pattern: 4` gives 8 global layers × 32 KB/token and 24 sliding
  layers capped at 4096 tokens.
- The single MLX worker serialises the two models, so only one model's
  activations are live at a time. The weights of both stay resident.

**Context ceiling:** this conversion's `config.json` says
`max_position_embeddings: 8192` with no `rope_scaling`. Treat 8K as the limit until
it is tested:
- Cap the joined transcripts at ~6K tokens.
- When the transcripts exceed that, show a warning naming the pages that were
  dropped. Never truncate silently.

## Steps

### 0. Upgrade dependencies to latest — its own commit, before anything else

Commit this before any feature work. If CI goes red, you then know the upgrade
caused it.

Snapshot from 2026-10-18, taken with `uv tree --outdated --depth 1` and
`uv lock --upgrade --dry-run` on uv 0.12.22. Re-run both at implementation time,
since newer releases may have shipped.

| Package | Locked | Latest | Notes |
|---|---|---|---|
| `mlx-vlm` | 0.7.3 | **0.7.8** | Exact pin, so `--upgrade` will not move it. Needs `uv add`. Requires `mlx>=0.32.3`, `transformers>=5.14` |
| `mlx` / `mlx-metal` | 0.32.2 | 0.32.4 | Transitive |
| `transformers` | 5.17.0 | 5.20.1 | Transitive |
| `huggingface-hub` | 1.33.0 | **2.2.0** | **Major.** Moves from `httpx` to `httpx2` |
| `streamlit` | 1.64.0 | 1.66.0 | `CLAUDE.md` says "targeting 1.64" |
| `ruff` (dev) | 0.16.9 | 0.17.1 | Minor; the formatter style may change |
| `ty` (dev) | 0.0.84 | 0.0.89 | New diagnostics are possible |
| `pillow` | 12.3.0 | 13.0.0 | Expect it to **stay** on 12.x: the dry run did not move it, so a dependency caps it below 13. Leave that alone |

Commands:

```bash
uv add mlx-vlm==0.7.8            # moves the exact pin (pyproject + lock)
uv lock --upgrade                # everything else to the latest allowed
uv sync --locked
```

- Keep the existing floors in `pyproject.toml`. `streamlit>=1.62` and
  `ty>=0.0.82` are deliberate minimums, not stale pins (see `CLAUDE.md`).
  Upgrading the lock is the goal.
- **Do not touch `version`.**

What to re-verify, and where each one bites:

- **mlx-vlm 0.7.3 → 0.7.8**
  - Run `tests/test_resize_parity.py` first. `nmv/imaging.py` mirrors upstream
    `smart_resize`, so a change there must be copied across.
  - Re-check the `Sampling` comment in `nmv/model.py`. It cites
    `mlx_vlm/generate/ar.py:296-309` and 0.7.3 seed behaviour, so update the
    version and the line numbers, or the claim itself.
  - Re-check both `cast`s in `stream_reply`. Keep them while the annotations
    are still narrower than the runtime behaviour.
  - Confirm `load(..., revision=)` and `load_config(..., revision=)` still exist.
  - Re-read the `_inv_freq` story in `nmv/runtime.py`. If 0.7.8 now evaluates
    those tables, the worker is still required (generation is not reentrant),
    but the docstring's explanation must be corrected.
  - Run grounding on `assets/sample-invoice.png` and check the PAID stamp.
- **huggingface-hub 2.x**
  - Confirm `try_to_load_from_cache` and `snapshot_download` keep their
    signatures. The plan's `reader.py` uses `snapshot_download`.
  - Confirm `uv run hf download <repo> --revision <sha>` still works, because the
    README and `CLAUDE.md` prefetch commands depend on it.
- **Streamlit 1.66**
  - Update "Targeting Streamlit 1.64" in `CLAUDE.md` and keep the 1.62 floor.
  - In a real browser, recheck page-switch persistence and both themes. Theme
    config does not hot-reload, so restart the server first.
- **ty 0.0.89**
  - If ty no longer flags `st.navigation`, it now *warns* about the unused
    `ty: ignore[call-non-callable]`. Per `CLAUDE.md`, delete the ignore and its
    comment rather than silencing the warning. Fix any new diagnostics properly.
- **ruff 0.17**
  - Run `uv run ruff format .`. If it reformats files, keep that in the same
    upgrade commit and mention it in the message.

Done when the full verification checklist below passes on the upgraded lock,
before any code from steps 1–8 exists.

### 1. Switch NMV to 8-bit — `nmv/model.py`, `nmv/ui.py`

- Set `MODEL_REPO` and `MODEL_REVISION` to the 8-bit values above.
- Keep the provenance comment; it still holds for the new commit.
- Rewrite the module docstring: 8-bit, ~3.2 GB, chosen so a second model fits
  in 16 GB.
- `ui.ensure_studio`:
  - Download warning: "about 3.2 GB".
  - Spinner: "(8-bit)".
- Sidebar caption: `2.4B · 8-bit · Apache 2.0 · native resolution`.
- Check: `assets/sample-invoice.png` on Grounding still hits the PAID stamp at
  `(820, 1180, 1140, 1400)` within a few pixels. If it misses by much more, stop
  and report before continuing.
- **Reclaim the bf16 weights (about 5 GB of disk)** once the 8-bit check passes.
  - Deleting cannot be undone, so **ask the user before running it**. Do not
    script it into the app.
  - Find the exact subcommand with `uv run hf cache --help` on the upgraded
    huggingface-hub 2.x. It is likely `hf cache rm mlx-community/North-Micro-Vision-Instruct-bf16`.
  - In the README, mention the same cleanup for anyone upgrading from an
    earlier version of the app.

### 2. Text model module — new `nmv/reader.py`

This module mirrors `nmv/model.py`. **Every `mlx_lm` import goes inside a function
body that runs on the worker.**

- Constants: `READER_REPO`, `READER_REVISION`, plus a provenance comment like
  `MODEL_REVISION`'s, and `READER_TEMPERATURE = 0.3`.
- `_load(repo, revision)`, which runs on the worker:
  - `huggingface_hub.snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.jinja"])`.
    The allowlist means no `.py` is ever fetched.
  - Then `mlx_lm.load(local_path)`.
  - Return a `Reader(model, tokenizer)` dataclass.
- `load_reader()`: `@st.cache_resource`, calls `runtime.call(_load, ...)`.
- `is_reader_cached()`: same pattern as `is_cached()`.
- `stream_answer(reader, history, pages, sampling, stats)`:
  - `pages` is `list[str]`.
  - Builds `documents=[{"title": f"Page {i+1}", "text": t} for i, t in enumerate(pages)]`.
  - Calls `tokenizer.apply_chat_template(history, documents=documents, chat_template="rag", enable_citations=True, add_generation_prompt=True, tokenize=False)`.
  - Streams `mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=..., sampler=make_sampler(temp, top_p, top_k=...))` through `runtime.iterate()`.
  - Checks `emit()`'s return value, exactly like `model.stream_reply`.
  - `RunStats.absorb` works unchanged: mlx-lm's `GenerationResponse` has the same
    field names. Confirm that against the installed version.
- Command R7B **does** accept a system preamble. "Never send a system prompt" is
  an NMV rule, so add a comment saying so where the reader builds messages.
  Even so, start with no preamble; the `rag` template already instructs the model.
- Check the installed `mlx_lm.sample_utils.make_sampler` signature before writing
  `as_kwargs`-style glue. Do not guess the parameter names.
- Find out how mlx-lm decodes the control tokens (`<|START_RESPONSE|>`,
  `<|END_RESPONSE|>`, `<|START_THINKING|>`…). If they reach the stream as text,
  strip them in the citation parser, not in page code.

### 3. Boot both libraries on the worker — `nmv/runtime.py`

- In `_import_mlx`, also `import mlx_lm  # noqa: F401`.
- Update the docstring to say this covers both libraries.
- Rerun `tests/test_runtime_abandon.py`.

### 4. Citation parsing — new `nmv/citations.py` + `tests/test_citations.py`

- Format, taken from the template: `<co>span</co: 0:[1,2]>`.
  - `0` is the tool-call id. Always `0` for directly injected documents.
  - `[1,2]` are 0-based document (page) indices.
  - A closing tag can list several groups joined by `,`.
- `parse(text) -> tuple[str, list[Citation]]`:
  - Returns the clean text plus `Citation(start, end, span, pages)`.
  - Tolerant of a stream cut off mid-tag. Leave an unclosed `<co>` as plain text.
- Pure Python, no MLX import.
- Test as a plain script in the same style as the existing tests:
  - nested and multiple groups
  - a truncated tag
  - no citations at all
  - stray control tokens

### 4b. Transcript store — new `nmv/transcripts.py` + `tests/test_transcripts.py`

Transcription is the slow stage: seconds per page, and minutes for a long
document. Losing it on every refresh or restart is the worst cost in the
pipeline. Persist transcripts to disk:

- **Location:** `~/.cache/cohere-document-intelligence/transcripts/`.
  - Outside the repo, so documents can never be committed by accident.
  - Use an env var `NMV_TRANSCRIPT_DIR` to override it.
- **Key:** `sha256(` the following, joined `)`, one JSON file per key:
  - the upload's raw bytes
  - the effective budget (`ImagePlan`'s resolved pixels, not the slider value)
  - `MODEL_REVISION`
  - a `TRANSCRIBE_PROMPT_VERSION` constant
- **Why this key, and not Streamlit's `file_id`:** `file_id` changes on every
  re-upload of the same file, so it would never get a cache hit across sessions.
- **Self-invalidating:** a new checkpoint, a new prompt, or a different budget
  produces a different key, so a stale transcript is never served. Never add
  manual invalidation logic.
- **File contents:** transcript text, `finish_reason`, the resolved size, a
  `created` timestamp, plus `model_revision` and `prompt_version` in plain form
  (the key is a hash, so they cannot be read back from the filename).
- **Retention: no time-based expiry.** A valid transcript is never deleted
  behind the user's back. Re-running NMV costs seconds per page, while keeping
  the result costs a few KB. Staleness is already impossible because of the key.
  Two ways a transcript leaves the store:
  1. **Unreachable entries are pruned automatically.**
     - `prune(model_revision, prompt_version) -> int` deletes every file whose
       stored revision or prompt version differs from the current ones. No
       future key can match those files.
     - Run it once per server process, from an `st.cache_resource`-wrapped call
       on the first visit to Ask documents. Never on every rerun.
     - Different budgets are **not** pruned. They are still reachable by moving
       the slider back.
  2. **The user clears everything** with the button below.
- **Truncated transcripts:** store them, but show the truncation warning again
  when one is loaded.
- **API:** `get(key) -> Transcript | None`, `put(key, Transcript)`,
  `clear() -> int` (files removed), `usage() -> (count, bytes)`.
  - Write to a temp file and `os.replace` it into place, so a crash mid-write
    never leaves half a JSON file.
  - Treat an unreadable file as a miss.
- **No MLX import** in this module.
- **Tests** (plain script, using a temp dir via the env var):
  - put/get round trip
  - a different revision, budget or prompt version misses
  - a corrupt file is treated as a miss
  - `clear()` empties the directory
  - `prune()` removes only old-revision and old-prompt files, and keeps
    current-revision files at other budgets

Privacy: these are the contents of the user's documents, so make the store
visible and easy to clear. **All storage controls live in the main area of Ask
documents, grouped beside the uploader and the Transcribe button, not in the
sidebar.**
- Why not the sidebar: `ui.sidebar()` ends with the stats slot, so anything the
  page appends lands below "Last run". And whether to save is decided per file,
  just before transcribing it.
- The group (e.g. one `st.container(horizontal=True)`) holds:
  - a caption: "N transcripts saved · X KB"
  - a "Clear saved transcripts" button (`st.button`, with a confirm step via
    `st.popover`, never a JS dialog)
  - the "Save transcripts" toggle described below
- A README section saying where transcripts live and how to wipe them.
- A **"Save transcripts"** toggle, so a confidential file can be processed
  without anything reaching disk:
  - Settings: `st.toggle`, `key="save_transcripts"`, `value=True`,
    `persist_state="session"`.
    - Session scope is still needed off the sidebar. A widget that is not
      rendered on a run is dropped, so leaving the page and coming back would
      otherwise reset it to On.
  - Render it only on Ask documents. Never in the shared `ui.sidebar()`, because
    Chat and Grounding never save anything.
  - **A refresh resets it to On.** Streamlit 1.64/1.66 offer only
    `persist_state` = `"page"`, `"session"` or `None`, and a refresh starts a new
    session. Do not work around this with browser storage, and do not flip the
    default to Off (that would re-transcribe every normal document after a
    refresh). Instead, put the mode **on the action itself**:
    - The Transcribe button label is "Transcribe · saving to disk" or
      "Transcribe · not saving".
    - So the current mode is visible at the moment of every transcription,
      including the first one after a refresh.
  - **Off means no writes.** Transcribe as usual, but skip `put` and hold that
    run's transcripts in `st.session_state` instead, keyed by the same hash.
    They live as long as the session, which is the trade the user chose.
  - **Reads still happen when off.** A transcript already on disk is no
    additional exposure, and reusing it saves the work.
  - Turning it off does **not** delete anything already saved. The Clear button
    does that. A caption under the toggle states both facts.
    - When off: "Not saving — this session's transcripts are lost on refresh.
      Already-saved ones stay until cleared."
    - When on: the "N transcripts saved · X KB" usage line.
  - Pass the flag into the page's transcribe loop explicitly. Do not read
    `st.session_state` from inside `nmv/transcripts.py`; the module stays free of
    Streamlit.
  - Browser check:
    1. Toggle off, transcribe a page, and confirm the transcript directory's file
       count did not change.
    2. Refresh, and confirm the transcript is gone.
    3. Switch to Chat and back, and confirm the toggle is still off.
    4. Refresh, and confirm the toggle is back On and the button label reads
       "saving to disk".

### 5. The page — new `app_pages/documents.py`, registered in `streamlit_app.py`

Title "Ask documents", icon `:material/description:`. Direct script, no `main()`,
same shape as the two existing pages.

1. `st.file_uploader(accept_multiple_files=True, type=IMAGE_TYPES)`.
2. **Transcribe** button.
   - For each page, run `normalize` → `encode` → `model.stream_reply` with a
     user-turn instruction: "Transcribe this page to Markdown. Preserve tables.
     Output only the transcription."
   - Use `greedy`-style sampling for this step whatever the sidebar says. Say so
     in a caption.
   - Look each page up in `nmv/transcripts.py` first, and transcribe only the
     misses.
     - Stream each miss as it is transcribed, then `put` it.
     - Hits render instantly, with a caption saying "saved".
     - Show each transcript in an expander per page.
   - Hold only the chat history in `st.session_state`; transcripts come from the
     store.
   - **Page cap:** `MAX_PAGES = 10`.
     - Above that, transcribe the first 10 and say which pages were skipped.
       Never drop pages silently.
     - The cap also keeps the joined transcripts near the 6K-token context limit.
   - **Budget-change cost:** under the Transcribe button, add a caption saying
     transcripts depend on the image budget. Changing the budget re-transcribes
     every page that has no saved transcript at the new budget.
3. `st.chat_input`, then `reader.stream_answer` over the transcripts.
   - Render the clean text.
   - Then render citations as `st.pills`/buttons labelled "p.2: “Total due”".
4. Clicking a citation runs NMV grounding on that page with
   `Locate "<span>". Return a JSON list of {"bbox_2d": [x1, y1, x2, y2]}.`.
   - Parse with `grounding.parse_boxes`.
   - Draw with `grounding.draw_boxes` on the full-resolution `normalize()`d page.
   - Labels are optional; do not assume them.

Rules that apply to this page:
- Load the reader lazily, on the first question rather than at page render.
  This means Chat/Grounding users never pay 4.5 GB.
- Add `ui.ensure_reader()` alongside `ensure_studio()`, with the same cached /
  download warning.
- Reuse `ui.sidebar()` as is. Any new sidebar widget needs
  `persist_state="session"`.
- Fill the stats slot after each stage, and show which model ran it.
- `max_tokens` from the sidebar applies to both stages. Transcription of a dense
  A4 page can exceed 512, so surface `RunStats.truncated` as a warning per page.

### 6. Dependency

```bash
uv add mlx-lm==0.32.2   # or the latest release at implementation time, if newer
```

0.32.2 was the latest release on 2026-10-18. If a newer one exists, use it,
provided its `mlx` requirement is satisfied by the lock from step 0.

This updates `pyproject.toml` and `uv.lock`. **Do not touch `version`** — that is
a publish action.

### 7. Docs, guards and CI

**Import-guard regex** — change it everywhere from `mlx(_vlm)?` to `mlx(_vlm|_lm)?`:
- `CLAUDE.md`
- `.github/workflows/ci.yml:61`
- `.claude/settings.json:13` (the edit hook)

**`CLAUDE.md`:**
- Replace the bf16-only rule with: 8-bit chosen for 16 GB; don't add other
  variants without being asked.
- Describe both pinned checkpoints.
- Add a second prefetch command.
- PLC0415 count: six becomes six + the new `mlx_lm` imports + `huggingface_hub`
  in `reader.py`.
- Add `nmv/reader.py`, `nmv/citations.py`, `nmv/transcripts.py` and
  `app_pages/documents.py` to the module table.
- Document where transcripts are stored and how to clear them.
- Add the new test command.
- Make the baseline section name both machines.

**`README.md`:**
- 8-bit throughout.
- Prefetch commands for both repos with `--revision`.
- Licence note for R7B.
- Requirements line: 16 GB with both models resident, with the peak from step 8.
- Screenshot of the new page, if you take one.

**Elsewhere:**
- `pyproject.toml` `description`.
- `.github/workflows/ci.yml:39` comment ("5 GB").

### 8. Measure on the M2 Pro 16 GB (acceptance gate)

Use `assets/sample-invoice.png`, plus one dense multi-page document of 3–5 pages.
Record:

1. NMV-only load time, and peak memory at the ceiling budget.
2. The same numbers after the reader loads: resident memory
   (`mx.get_active_memory()` via `runtime.call`) and the peak during a question.
3. Prefill and decode tok/s for each model.
4. Whether macOS swaps. Check Activity Monitor → Memory → Swap Used, or
   `vm_stat` before and after.

**Pass:** combined peak ≤ ~10 GB, with no sustained swap. **Fail:** switch to the
Tiny Aya fallback, or add an "unload the idle model" path. Either way, report the
numbers before choosing.

Put the measured figures into the README and the `CLAUDE.md` baseline.

**Note on the peak-memory metric:** `mx.get_peak_memory()` is process-wide. Once
both models are loaded, the sidebar's peak figure includes both models' weights.
- Call `mx.reset_peak_memory()` at the start of each run (on the worker), so the
  figure means "this run".
- Label the figure accordingly.

## Verification checklist (run before calling it done)

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run python tests/test_resize_parity.py
uv run python tests/test_runtime_abandon.py
uv run python tests/test_citations.py
uv run python tests/test_transcripts.py
grep -rnE --include='*.py' '^(from|import) mlx(_vlm|_lm)?\b' nmv/ app_pages/ streamlit_app.py   # must print nothing
```

Then check in a real browser, restarting the server after the `nmv/` edits:

- Grounding still hits the PAID stamp.
- Chat works.
- On Ask documents, transcribe → ask → click a citation → a box appears on the
  right page.
- Switching pages keeps the sidebar values.
- Navigating away mid-answer does not wedge the next request.

## Out of scope

- PDF input.
- Embeddings and reranking. Cohere's are API-only; their HF repos hold
  tokenizers only.
- Keeping bf16 as an option.
- A version bump or release.
