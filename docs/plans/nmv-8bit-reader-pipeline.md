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

**Licence:**
- **This repo's code stays Apache-2.0. Do not change `LICENSE`.**
  - The weights are not vendored, so the R7B licence covers the weights only,
    not the code that loads them.
  - Every dependency is permissive (Apache-2.0, MIT, BSD), and SQLite is public
    domain.
  - Apache's §6 statement, that no trademark rights come with the code, matters
    more now that the project name contains "Cohere".
- **NMV** is Apache-2.0.
- **Command R7B and Tiny Aya** are CC-BY-NC-4.0 plus
  [Cohere's C4AI licence agreement](https://cohere.com/c4ai-cc-by-nc-license).
  - That agreement is linked from the mlx-community card's `extra_gated_prompt`.
  - The Hub reports that repo as `gated=False`, so the download needs no form.
    The terms still apply.
- So **Ask documents is non-commercial, while Chat and Grounding stay fully
  Apache-2.0.** No Cohere text model with an MLX build is permissively licensed,
  so making the pipeline commercial-safe would mean a non-Cohere reader.
- Where these facts appear: step 5 (first load), step 7 (README, `pyproject.toml`)
  and the sidebar caption.

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

### Before step 0 — work on the branch, land it through a PR

Do all of this on `plan/nmv-8bit-reader-pipeline`, which already carries this
plan and the hooks. Land it on `main` through a pull request, not by direct
commits.

- CI runs `check` on every `pull_request`, and runs `release` only on `main`.
  So a PR means the first CI run on the upgraded lock happens off `main`, and a
  red run never sits on the default branch.
- Push the branch, then open the PR. **Ask the user before pushing or opening
  it**, since both are outward-facing.
- Push after each step's commit, so CI checks every step separately.

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
pipeline. Persist transcripts in **SQLite**, using Python's built-in `sqlite3`.

Why SQLite: it adds no dependency and no server process, and needs no RAM
beyond the models. It provides transactions, single-query `prune`/`usage`, and
**FTS5** full-text search. Checked in this venv on 2026-10-18: SQLite 3.49.1,
with FTS5 and JSON functions available. Postgres, MySQL, MongoDB and Redis
would each run a server process competing with the models for 16 GB. DuckDB is
built for analytics. A vector store waits for a local embedding model (see
step 9).

- **Location:** `~/.cache/cohere-document-intelligence/transcripts.db`.
  - Outside the repo, so documents can never be committed by accident.
  - Use an env var `NMV_TRANSCRIPT_DB` to override it.
  - Create the parent directory on first use.
- **`.gitignore`, in the same commit as `nmv/transcripts.py`.** It is a second
  layer: the `~/.cache` default is the first, and these rules cover the case
  where `NMV_TRANSCRIPT_DB` is pointed into the repo.
  - Keep the existing hand-written file. Do not swap in GitHub's Python
    template; its build and framework rules do not apply here.
  - Append:

    ```gitignore
    # Transcript store (nmv/transcripts.py). It lives in ~/.cache by default, but
    # NMV_TRANSCRIPT_DB can point anywhere, and these files hold document text.
    # The -wal/-shm files are SQLite's WAL sidecars and contain the same data.
    *.db
    *.db-wal
    *.db-shm

    # Private test documents, e.g. the multi-page document for step 8. Keep real
    # invoices and scans here, never in assets/, which is tracked and published.
    /local/
    ```

  - Do **not** ignore `uv.lock`, `.python-version`, `.claude/settings.json` or
    `.claude/hooks/`. CI and the shared hooks need them tracked.
  - Ignoring weights is unnecessary; they go to the HF cache in `~`.
  - Before committing:
    - Check that `git ls-files '*.db' '*.db-wal' '*.db-shm' 'local/*'` prints
      nothing, so nothing is tracked that the new rules would hide.
    - Run the `git check-ignore` line from the verification checklist.
- **Schema.** Created on first connect; versioned with `PRAGMA user_version = 1`
  so later changes can migrate:

  ```sql
  CREATE TABLE IF NOT EXISTS transcripts (
      key            TEXT PRIMARY KEY,   -- sha256 hex, see below
      text           TEXT NOT NULL,
      finish_reason  TEXT,
      width          INTEGER NOT NULL,   -- resolved size NMV actually saw
      height         INTEGER NOT NULL,
      model_revision TEXT NOT NULL,
      prompt_version INTEGER NOT NULL,
      filename       TEXT,               -- display only, never part of the key
      created        TEXT NOT NULL       -- ISO-8601 UTC
  );
  CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
      text, content='transcripts', content_rowid='rowid'
  );
  -- plus AFTER INSERT / AFTER DELETE / AFTER UPDATE triggers that keep
  -- transcripts_fts in sync (the standard external-content pattern from the
  -- SQLite FTS5 docs)
  ```

  Create the FTS index now, even though search is optional (step 9). The data
  then never needs a backfill migration.
- **Connections:** open one per call (`contextlib.closing(sqlite3.connect(path))`
  plus `with db:` for the transaction). **Never share a connection across
  threads.**
  - `sqlite3` refuses by default (`check_same_thread`).
  - Streamlit runs each rerun on a fresh thread, the same problem `nmv/runtime.py`
    solves for MLX.
  - Opening a connection costs microseconds, so no worker is needed.
  - On each connection, set `PRAGMA journal_mode=WAL` (two tabs can read while
    one writes), `PRAGMA busy_timeout=5000` and `PRAGMA secure_delete=ON`
    (deleted text is overwritten, not left in free pages).
- **Key:** `sha256(` the following, joined `)`:
  - the upload's raw bytes
  - the effective budget (`ImagePlan`'s resolved pixels, not the slider value)
  - `MODEL_REVISION`
  - a `TRANSCRIBE_PROMPT_VERSION` constant
- **Why this key, and not Streamlit's `file_id`:** `file_id` changes on every
  re-upload of the same file, so it would never get a cache hit across sessions.
- **Self-invalidating:** a new checkpoint, a new prompt, or a different budget
  produces a different key, so a stale transcript is never served. Never add
  manual invalidation logic.
- **Plain columns too:** `model_revision` and `prompt_version` are stored as
  their own columns, because the key is a hash and cannot be read back. This is
  what makes `prune` a single `DELETE`.
- **Retention: no time-based expiry.** A valid transcript is never deleted
  behind the user's back. Re-running NMV costs seconds per page, while keeping
  the result costs a few KB. Staleness is already impossible because of the key.
  Two ways a transcript leaves the store:
  1. **Unreachable entries are pruned automatically.**
     - `prune(model_revision, prompt_version) -> int` runs
       `DELETE FROM transcripts WHERE model_revision != ? OR prompt_version != ?`
       and returns the row count. No future key can match those rows.
     - Run it once per server process, from an `st.cache_resource`-wrapped call
       on the first visit to Ask documents. Never on every rerun.
     - Different budgets are **not** pruned. They are still reachable by moving
       the slider back.
  2. **The user clears everything** with the button below.
- **Truncated transcripts:** store them, but show the truncation warning again
  when one is loaded.
- **API:**
  - `get(key) -> Transcript | None`
  - `put(key, Transcript)` (`INSERT OR REPLACE`)
  - `prune(...) -> int`
  - `clear() -> int` (rows removed)
  - `usage() -> (count, bytes)` (`SELECT count(*), coalesce(sum(length(text)), 0)`)
  - `search(query, limit) -> list[Hit]`, for step 9
  - `Transcript` is a frozen dataclass. Never pass raw rows to page code.
  - Atomicity comes from the transaction; no temp-file dance is needed.
  - **Every query is parameterised (`?`).** Never build SQL with f-strings, and
    that includes FTS `MATCH` input from the search box (step 9).
- **`clear()`:** run `DELETE FROM transcripts` (the triggers empty the FTS index),
  then `INSERT INTO transcripts_fts(transcripts_fts) VALUES('optimize')`, then
  `VACUUM` (outside the transaction), so the file actually shrinks and no
  document text survives in free pages.
- **Failure handling:** if the database cannot be opened (corrupt file, disk
  full, permissions):
  - `get` returns a miss and `put` is skipped.
  - Log the error with `logging`, and show one `st.warning` from the page saying
    transcripts are not being saved and naming the path.
  - Never crash the pipeline over the cache.
  - Never auto-delete a database that failed to open; the user may want it.
- **No MLX import** in this module, and no Streamlit import either.
- **Tests** (plain script, using a temp database via the env var):
  - put/get round trip
  - a different revision, budget or prompt version misses
  - a file that is not a database is treated as a miss, and is not deleted
  - `clear()` empties both tables, and the file size drops after `VACUUM`
  - `prune()` removes only old-revision and old-prompt rows, and keeps
    current-revision rows at other budgets
  - FTS stays in sync: after `put`, `search` finds a word from the text, and
    after `prune`/`clear` it does not
  - threads: `put` from one thread and `get` from another both succeed (guards
    against a shared connection creeping in)

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
    1. Toggle off, transcribe a page, and confirm the database's row count did not
       change: `sqlite3 ~/.cache/cohere-document-intelligence/transcripts.db 'select count(*) from transcripts'`
       before and after.
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
  - When it is about to load the reader for the first time (not cached yet),
    also show a one-line `st.info`: "Command R7B is licensed CC-BY-NC-4.0 under
    Cohere's C4AI licence agreement — non-commercial use only", with the
    agreement link.
  - Show it every time the reader is not yet cached, not just once per session.
    This is the moment someone starts using non-commercial weights.
- Reader sidebar caption: `8B · 4-bit · CC-BY-NC-4.0 · non-commercial`. Keep
  NMV's caption saying Apache 2.0.
- Reuse `ui.sidebar()` as is. Any new sidebar widget needs
  `persist_state="session"`.
- Fill the stats slot after each stage, and show which model ran it.
- `max_tokens` from the sidebar applies to both stages. Transcription of a dense
  A4 page can exceed 512, so surface `RunStats.truncated` as a warning per page.

#### Layout that adapts to the display

**One layout, adapted in the browser. Never branch on the server.**
- In Streamlit 1.64, `st.context` exposes cookies, headers, ip_address,
  is_embedded, locale, theme, timezone and url, but **no viewport size**. Do not
  sniff the user agent, and do not add custom CSS (`CLAUDE.md`: prefer native
  elements).
- Adaptation comes from four built-in behaviours:
  - columns stack at ≤ 640 px
  - horizontal containers wrap
  - the sidebar collapses on narrow screens
  - an **integer `width` acts as a maximum**: per the 1.64 docstring, if it is
    wider than the parent, the element matches the parent

Page structure, top to bottom:

1. Title and a one-line caption.
2. **Storage group:** `st.container(horizontal=True)` holding the uploader, the
   "Save transcripts" toggle, the usage caption and the Clear popover. It wraps
   by itself on narrow screens.
   - Keeping it in the main area, as step 4b already requires, matters most on
     phones. There the sidebar is a hidden drawer, so a sidebar privacy toggle
     would be invisible.
3. The **Transcribe** button, labelled with the save mode.
4. **Body:** `conversation, page = st.columns(2, gap="medium")`.
   - **Create the conversation column first.** Columns stack in creation order,
     so this gives phones the answer, then the page, and a tapped citation shows
     its box just below.
   - Use equal widths, because both the answer and the page need room.
5. **Conversation column:**
   - Put the messages in `st.container(width=760)`, about 80 characters at the
     theme's 15 px base size.
     - On a 2560 px display under `layout="wide"`, unconstrained lines run to
       ~150 characters.
     - On smaller screens the cap does nothing.
   - Citation pills use `st.pills(..., wrap=True)`.
6. **Page column:**
   - A page picker, `st.pills(..., wrap=True)`.
     - **`wrap=True` is required.** Controls placed directly in a column default
       to no wrap (the bundled `references/layouts.md`): labels get cut off and
       ten page pills would scroll sideways on a phone.
   - Then `st.image(page_with_boxes, width=880)`.
     - Above ~880 px, an A4 page is taller than a 1440p screen, so the located
       box falls below the fold.
     - Smaller screens get the full column. Streamlit's built-in fullscreen
       button covers zooming in.
   - Then **one** `st.expander("Transcript")`, for the selected page only, not
     one per page.
7. `st.chat_input` at the **top level** of the script, never inside a column,
   so Streamlit pins it to the bottom at every width.

Avoid:
- fixed `height=` on containers: it scrolls inside itself on short screens and
  leaves empty space on tall ones
- more than two columns in the body
- `use_container_width`: use `width=` (`CLAUDE.md`)

Put `760` and `880` in named constants, `READING_WIDTH` and `PAGE_IMAGE_WIDTH`,
in `nmv/ui.py`. Step 5c reuses them, so all three pages share one measure.

### 5b. Theme contrast guard — new `tests/test_theme_contrast.py`

**Do not redesign the theme.** The Polar Indigo pair in `.streamlit/config.toml`
already fits this app, and its two halves were measured as complementary on
2026-10-18. Ask documents needs **no config changes**: its new components
pass in both modes.

| Component | Dark | Light |
|---|---|---|
| Citation pills (`st.pills`, selected) | 5.33 | 6.65 |
| `st.info` licence notice | 8.85 | 5.81 |
| `st.warning` | 7.60 | 5.83 |

Two existing behaviours cover the rest:
- **Greyed-out Locate buttons** use Streamlit's disabled style, which is
  intentional for something that cannot be clicked, and they have a caption.
- **The dark toggle knob** is only 2.26:1 on its "on" track, a known residual,
  but the Transcribe label states the save mode in full-contrast text. **Keep
  that label for this reason too.**

What is missing is enforcement. Today only comments hold the halves together, so
an edit to one half can silently break the pairing. Add a plain-script test (no
MLX, same style as the others) that reads the config with `tomllib`:

- **Maths:**
  - WCAG 2.x relative luminance and contrast ratio.
  - sRGB → OKLab/OKLCH, using Ottosson's published matrices.
  - Alpha-mixing a foreground over a ground.
- **Derived colours,** computed the way Streamlit derives them:
  - caption = text at 60% over the ground
  - selected pill tint = primary at 10% over the ground
  - focus ring = primary at 50%
  - a dark callout tint with no explicit `*BackgroundColor` = its colour at 20%
    over the ground
- **Floors, both modes:**
  - body text, captions, links, code text, and `st.info`/`st.warning` text on
    their tints: ≥ 4.5
  - `borderColor` on the ground and on the sidebar: ≥ 3.0
  - selected pill: ≥ 4.5
  - white label on primary: ≥ 4.5 in light, **≥ 3.0 in dark** (the documented
    trade; comment it as such)
- **Pairing:**
  - accent hue equal across modes within 0.5°
  - ground hue within 2°
  - sidebar step off the ground within 0.02 of each other
  - focus rings within 0.1
- **Design rules taken from the config comments:**
  - dark ground OKLCH L ≤ 0.19 (the dark trade needs it)
  - every light chrome surface — ground, secondary, code, sidebar, dataframe
    header — at L ≤ 0.942, so the document stays the brightest thing on screen
  - accent OKLab distance ≥ 0.1 from every `PALETTE` colour in
    `nmv/grounding.py`, so the accent never matches a box colour (the teal
    trap). Import `PALETTE`; `nmv.grounding` has no MLX import.
- **Failure output:** each failure prints the measured value, the floor and the
  config comment that justifies the floor.
- **Write it to pass on today's config first:** run it before touching anything,
  and confirm the numbers match the table above.
- **Prove it can fail:** temporarily set `[theme.light] primaryColor` to a teal
  like `#2dd4bf`. The hue-pairing and palette checks should both go red. Then
  revert.

**Caveat, written in the test's docstring:** the test checks colours computed
from config. Streamlit's rendered colours (hover darkening, its own tints) can
differ, so this is a floor, and the real-browser check in both modes remains
the final word (`CLAUDE.md`).

Wire it in:
- **CI** (step 7): add a "Theme contrast" step in `check`, in the same commit as
  the test.
- **Stop hook** (`.claude/hooks/stop-checks.sh`):
  - Add the test to the fast-test loop.
  - Widen the changed-file filter to also match `.streamlit/config.toml`, so a
    theme-only edit triggers it.
  - Re-test the hook with a piped payload after editing it.

### 5c. Adaptive layout touch-ups for Chat and Grounding

These are small and separate from the pipeline, so give them their own commit.
They use the same two constants from step 5.

- **`app_pages/chat.py`:**
  - Wrap the message history (both the replayed history and the new turn's
    messages) in `st.container(width=READING_WIDTH)`. It has the same
    long-line problem on wide displays.
  - Leave `st.chat_input` at the top level, outside that container, so it stays
    pinned and full width.
  - Leave the starter pills alone; at the top level they already wrap.
- **`app_pages/grounding.py`:**
  - Give the overlay `st.image` calls `width=PAGE_IMAGE_WIDTH` in place of
    `"stretch"`. That is the "Ready" preview and the annotated image in
    `_overlay_panel`.
  - Keep `st.columns([3, 2], gap="medium")`. It already stacks correctly on
    phones, overlay first and box table second.
- **Re-check after the edit:** the PAID stamp is still hit, and the overlay
  fragment still reruns into its slot.

### 5d. Accessibility — text equivalents for a visual app

The theme and layout steps cover the visual minimums: contrast, borders,
reflow and zoom. What they miss is the app's core. **Boxes drawn on an image are
invisible to a screen-reader user and partly ambiguous to a colour-blind one.**
Use native Streamlit only, with no CSS. Give each item its own commit, so a
regression is easy to bisect.

1. **A text description for every box** (WCAG 1.1.1, text alternatives).
   - Add a pure function to `nmv/grounding.py` (no MLX):
     `describe(box) -> str`, e.g.
     `"upper right · about 30% of the page wide, 20% tall"`.
   - **Position:** where the box centre falls in a 3×3 grid, split at 333 and
     667 on the 0–1000 axes: upper / middle / lower × left / centre / right. Use
     `"centre"` for the middle cell.
   - **Size:** width and height as percentages of the page, rounded to the
     nearest 5. Show `"under 5%"` below that, and drop the size clause for a
     box covering ≥ 90% of both axes (`"most of the page"`).
   - **Normalise swapped coordinates** (x2 < x1) the same way `parse_boxes`
     does, never differently. Labels stay optional (`CLAUDE.md`), so
     `describe` never reads them.
   - **Where it shows:**
     - Grounding's box table gets a "where" column, placed after "label".
     - Ask documents captions the located page with
       `"Page N · “<span>” · <describe>"`.
   - **Test:** new `tests/test_grounding_describe.py`, a plain script. It
     covers the four corners, the exact centre, both threshold edges
     (332/333/667/668), a full-page box, a tiny box, and swapped coordinates.
   - **Real-fixture case:** the sample invoice's PAID stamp. In pixels it is
     `(820, 1180, 1140, 1400)` on a 1240×1754 page, which is
     `(661, 673, 919, 798)` on the 0–1000 grid. Its centre is (790, 735), so it
     must describe as `"lower right · about 25% of the page wide, 15% tall"`
     (26% and 13%, rounded to 5). Verified 2026-10-18.
2. **Never use colour as the only link between a box and its row** (WCAG
   1.4.1).
   - `PALETTE` holds red `(255, 92, 92)` and green `(74, 222, 128)`, the pair
     red-green colour-blind viewers confuse.
   - Today `draw_boxes` skips the whole caption when labels are off (the
     `if not show_labels: continue` branch), so colour is then the only link.
   - **Fix:** always draw `#N`. Show the label *text* only when the toggle is on
     and a label exists, as `#N label`.
   - Rename the toggle to **"Show label text"**. Keep its key
     `grounding_labels` so existing sessions keep their value.
   - The table's `#` column already matches `#N`. Ask documents' boxes follow
     the same rule.
3. **"Stream responses" toggle** in the shared sidebar.
   - Settings: `key="stream_responses"`, `value=True`,
     `persist_state="session"`. The last is mandatory for sidebar widgets.
   - **Off:** show `st.spinner("Answering…")`, then render the complete text
     once. This helps screen-reader users and anyone who finds moving text
     distracting.
   - Add one helper in `nmv/ui.py`, `show_reply(chunks, streaming: bool) -> str`.
     It either calls `st.write_stream` or joins the chunks under the spinner,
     and returns the text either way.
     - Chat, Grounding and Ask documents call it instead of `st.write_stream`.
     - Generation still goes through `runtime.iterate()` and is consumed fully
       in both modes, so stats and navigating away mid-answer behave the same.
   - Do **not** apply it to transcription progress. There, per-page progress is
     the useful signal; show a plain "Page 2 of 5" caption instead of the
     token stream.
4. **Labels and text equivalents, as rules for every page:**
   - Every widget keeps a visible `label`. No `label_visibility="collapsed"`
     without a comment explaining why. A `st.chat_input` placeholder is not a
     label, so the caption above each chat input says what it is for.
   - `st.image` in Streamlit 1.64 has **no `alt` parameter**, so every image gets
     a `caption=` as its text equivalent ("Ready", "Page 2 of 5 — box around
     'Total due'"). Chat's uploaded thumbnails already show file names.
   - Citations name the page in text ("p.2: 'Total due'"), never by colour or
     position alone.
   - Every `st.warning`/`st.error` states the remedy, following Grounding's
     three-way zero-boxes message.
5. **README "Accessibility" section.** List what the app does: box
   descriptions, numbered boxes, the stream toggle, keyboard operability, the
   contrast floors enforced by `tests/test_theme_contrast.py`, and reflow down to
   320 px. Also list the residuals honestly:
   - **The focus ring is 2.43:1 in both modes** (WCAG 2.4.7 recommends 3:1). No
     single config value fixes it without undoing the accent trade, and CSS is
     ruled out.
   - **The dark toggle knob is 2.26:1.** It is mitigated by the Transcribe
     button's text label.
   - **Streamlit-controlled behaviour** (widget ARIA, live regions) is outside
     this app's control. Name whatever the axe pass in the checklist reports
     there.

Wire `tests/test_grounding_describe.py` into CI (a "Box descriptions" step, in
the same commit as the test) and into `stop-checks.sh`'s fast-test loop.

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
- `.claude/settings.json`: **already done** on the plan branch (see "Claude Code
  hooks" below).

**ruff:** add `S608` (hardcoded SQL expression) to `[tool.ruff.lint] select`.
This catches f-string SQL in `nmv/transcripts.py` in the edit hook, CI and the
editor alike. Run `uv run ruff check .` after adding it. If it flags anything,
fix the code; do not ignore the rule.

**`CLAUDE.md`:**
- Replace the bf16-only rule with: 8-bit chosen for 16 GB; don't add other
  variants without being asked.
- Describe both pinned checkpoints.
- Add a second prefetch command.
- PLC0415 count: six becomes six + the new `mlx_lm` imports + `huggingface_hub`
  in `reader.py`.
- Add `nmv/reader.py`, `nmv/citations.py`, `nmv/transcripts.py` and
  `app_pages/documents.py` to the module table.
- Document where transcripts are stored (`transcripts.db`) and how to clear
  them.
- Add the per-call connection rule to the "constraint" section as a sibling of
  the MLX one: never share a `sqlite3` connection across threads.
- Add a short "Claude Code hooks" section listing the hooks below, so a future
  session knows why an edit was refused.
- Add the new test command.
- Make the baseline section name both machines.

**`README.md`:**
- 8-bit throughout.
- Prefetch commands for both repos with `--revision`.
- **Rewrite the `## License` section.** The current text says both NMV checkpoints
  are Apache-2.0 "so the two halves line up", which stops being true with R7B.
  - Keep the reasons for choosing Apache: the §3 patent grant, and §6 (no
    trademark rights), plus the "unaffiliated project" sentence.
  - Say that no weights are vendored.
  - Replace the two-halves paragraph with a three-row table:

    | Part | Licence | Commercial use |
    |---|---|---|
    | App code (this repo) | Apache-2.0 | Yes |
    | North Micro Vision 8-bit weights | Apache-2.0 | Yes |
    | Command R7B 4-bit weights | CC-BY-NC-4.0 + [C4AI licence agreement](https://cohere.com/c4ai-cc-by-nc-license) | No |

  - Then one paragraph covering three points:
    - Ask documents uses R7B, so it is non-commercial; Chat and Grounding use
      only NMV.
    - The R7B download is not gated on the Hub, but its terms still apply.
    - Each checkpoint's model card governs its weights, not this `LICENSE`.
  - Link both pinned conversions. Drop the bf16 link.
- Add the bf16-cache cleanup note from step 1.
- Requirements line: 16 GB with both models resident, with the peak from step 8.
- Screenshot of the new page, if you take one.

**Elsewhere:**
- `pyproject.toml` `description`.
- `pyproject.toml`: add `license = "Apache-2.0"` and `license-files = ["LICENSE"]`
  to `[project]`, which gives SPDX metadata (PEP 639).
  - Harmless under `package = false`.
  - The version guard allows it, because the `version` line is untouched.
  - Run `uv lock` afterwards in case uv records it, then commit both files.
- `.github/workflows/ci.yml:39` comment ("5 GB").

**`.github/workflows/ci.yml` — the `check` job.** Keep every existing step;
each one guards part of this plan:
- `uv sync --locked`: steps 0 and 6.
- ruff: the formatter change and `S608`.
- ty: the upgraded diagnostics.
- the import grep: step 2.
- resize parity: the mlx-vlm upgrade.
- abandon test: step 3.

Leave the `release` job untouched. Its `needs: check` is what stops a broken
upgrade from ever publishing. Change or add, inside `check`:

1. **Import grep regex** (`ci.yml:61`): `mlx(_vlm)?` → `mlx(_vlm|_lm)?`. Update the
   step's comment to name mlx-lm.
2. **New test steps**, after the two existing tests, in the same style
   (`uv run python tests/...`):
   - "Citation parsing" → `tests/test_citations.py`
   - "Transcript store" → `tests/test_transcripts.py`. This step also proves
     FTS5 exists in the runner's uv-managed Python.

   - "Theme contrast" → `tests/test_theme_contrast.py` (step 5b)
   - "Box descriptions" → `tests/test_grounding_describe.py` (step 5d)

   Add them in the same commit as the test files they run, never before.
   Otherwise CI goes red on a missing file.
3. **Checkpoint pin check**, reusing the local hook so there is one
   implementation:

   ```yaml
   # A checkpoint can run code here (CLAUDE.md), so every *_REVISION must be a
   # 40-hex commit and no download may fetch *.py. Same script as the edit hook.
   - name: Checkpoint pins
     run: |
       rc=0   # not `status`: read-only in zsh, so a local paste would break
       for f in nmv/*.py; do
         jq -n --arg f "$PWD/$f" '{tool_input: {file_path: $f}}' \
           | .claude/hooks/pin-guard.sh || rc=1
       done
       exit $rc
   ```

   `jq` is preinstalled on GitHub's macOS images, but confirm on the first run.
   The script matches `*/nmv/*.py`, so pass an absolute path, as above. Before
   pushing, prove the step can fail: temporarily set a revision to `"main"`
   locally and run the loop.

**Deliberately not in CI** (write this as a comment above the test steps, so
nobody "fixes" it later):
- **Step 8 and the PAID-stamp grounding check.** These need ~7.7 GB of weights
  per run, and both models resident (~9.6 GB). The standard `macos-latest`
  runner has 7 GB of RAM (3-core M1), so they cannot run there. They stay manual
  on the M2 Pro.
- **Browser checks.** Page-switch persistence, both themes and the save toggle.
  `CLAUDE.md` explains why AppTest cannot stand in for a real browser.

### 8. Measure on the M2 Pro 16 GB (acceptance gate)

Use `assets/sample-invoice.png`, plus one dense multi-page document of 3–5 pages.
Keep the multi-page document in the git-ignored `local/` folder, never in
`assets/`, which is tracked and published.
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

### 9. Optional follow-up — search saved transcripts (FTS5)

Do this only after step 8 passes, and as its own commit. It is the local stand-in
for retrieval, since Cohere's Embed and Rerank models are API-only. It uses no
model and no extra memory.

- On Ask documents, add a search box above the uploader, labelled "Search saved
  documents".
- Hits come from `transcripts.search(query, limit=20)`:
  - `SELECT key, filename, snippet(transcripts_fts, 0, '**', '**', '…', 12), bm25(transcripts_fts) … ORDER BY bm25 LIMIT ?`.
  - Show a list with the filename, the snippet and the date.
- **Sanitise the query for FTS5 syntax:** quote each user term as a phrase, so
  input like `AT&T "net 30` cannot raise `sqlite3.OperationalError`. On any
  remaining syntax error, show "no results", not a traceback.
- Selecting hits adds their transcripts to the reader's documents for the next
  question. They use the same 6K-token cap as uploads, and pages that do not fit
  are named, never silently dropped.
- **Locate is unavailable for search hits.** The store keeps text only, never
  page images; that is deliberate, for privacy and size. Grey out the citation
  buttons for those pages, with the caption "Re-upload this page to locate it."
- Respect the save toggle: search reads the database, and reading is allowed
  when saving is off (same rule as step 4b).
- Tests: add search cases to `tests/test_transcripts.py`:
  - ranking order
  - snippet contains the term
  - hostile query strings return a list, not an exception

### 10. After the PR merges — GitHub repo details

The description and topics describe `main`, so change them only once the
pipeline is on `main`, never earlier. **Ask the user before running this**:
repo details are public and outward-facing.

The current description, set on 2026-10-18, covers the pre-plan app. It already
starts with "Unofficial" because the name contains "Cohere"; keep that word.
Replace it, and add five topics (14 → 19):

```bash
gh repo edit darylalim/cohere-document-intelligence \
  --description "Unofficial, local document intelligence on Apple Silicon. Cohere North Micro Vision reads pages and grounds boxes; Command R7B answers with page citations. Runs on-device with MLX in a Streamlit app, with no cloud API." \
  --add-topic mlx-lm,command-r,rag,document-question-answering,python
```

- If step 8 fell back to Tiny Aya, swap `Command R7B` and `command-r` for the
  model actually shipped.
- Leave the 20th topic slot free. Use `sqlite` or `full-text-search` only if
  step 9 shipped.
- Keep the homepage on the North Micro Vision model card.
- Afterwards, confirm with
  `gh repo view --json description,repositoryTopics`.

## Claude Code hooks (already installed on the plan branch)

These live in `.claude/settings.json`, with scripts in `.claude/hooks/`, and are
active from the first prompt of the implementation session. They enforce the
plan's hardest rules mechanically, so treat a hook refusal as the plan speaking.
Do not work around one.

| Event | Hook | What it does |
|---|---|---|
| PreToolUse `Edit\|Write\|MultiEdit\|Bash` | `version-guard.sh` | **Denies** any change to `version` in `pyproject.toml`, `uv version --bump`/`uv version X.Y`, and in-place `sed`/`perl` rewrites of it. A version line passed through unchanged (e.g. by `uv add`) is allowed |
| PostToolUse `Edit\|Write` | ruff fix + format | Unchanged |
| PostToolUse `Edit\|Write` | MLX import guard | Regex now `mlx(_vlm\|_lm)?`, so a module-scope `mlx_lm` import in step 2 is caught |
| PostToolUse `Edit\|Write` | `pin-guard.sh` | Blocks a `nmv/*.py` edit that leaves any `*_REVISION` not a 40-hex commit, or puts `"*.py"` in a download allowlist |
| PostToolUse `Edit\|Write` | `restart-reminder.sh` | Once per session, after the first `nmv/` edit: restart Streamlit before checking in a browser |
| Stop | `stop-checks.sh` | When Python under the app is changed or untracked: runs `ty check` plus `tests/test_citations.py` / `tests/test_transcripts.py` (each once it exists). On failure, sends the output back. Skips itself on a hook-triggered re-stop, so it cannot loop |

**Permission rules** (`permissions.ask`): `hf cache rm`/`delete` (with or
without `uv run`) and `rm *transcripts.db*` always prompt. The bf16 cleanup in
step 1 therefore asks the user even in auto mode.

**If a guard is wrong, fix the script and re-test it with a piped payload.**
Never delete the hook to get past it.

## Verification checklist (run before calling it done)

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run python tests/test_resize_parity.py
uv run python tests/test_runtime_abandon.py
uv run python tests/test_citations.py
uv run python tests/test_transcripts.py
uv run python tests/test_theme_contrast.py
uv run python tests/test_grounding_describe.py
grep -rnE --include='*.py' '^(from|import) mlx(_vlm|_lm)?\b' nmv/ app_pages/ streamlit_app.py   # must print nothing
for f in nmv/*.py; do jq -n --arg f "$PWD/$f" '{tool_input:{file_path:$f}}' | .claude/hooks/pin-guard.sh; done   # must print nothing
git check-ignore -q transcripts.db transcripts.db-wal transcripts.db-shm local/scan.png && echo ignored   # must print "ignored"
git check-ignore -q uv.lock .claude/settings.json .claude/hooks/pin-guard.sh assets/sample-invoice.png || echo tracked   # must print "tracked"
```

Then confirm the PR's CI run is green, including the three new steps.

Then check in a real browser, restarting the server after the `nmv/` edits:

- Grounding still hits the PAID stamp.
- Chat works.
- On Ask documents, transcribe → ask → click a citation → a box appears on the
  right page.
- Switching pages keeps the sidebar values.
- Navigating away mid-answer does not wedge the next request.
- **Ask documents in both themes.** Pick Dark, then Light, from the app menu.
  Reload after each switch, because sidebar `divider="gray"` rules keep the old
  mode's colour until a reload. In each mode, check:
  - Citation pills: the selected one is clearly distinct from the rest.
  - The licence `st.info` and any `st.warning` are readable on their tints.
  - The "Save transcripts" toggle: confirm on/off from the button label as well
    as from the track.
  - Greyed Locate buttons read as unavailable, with their caption.
  - Transcript expanders, including a Markdown table, show their borders.
  - The page image under a drawn box. In Light it stays the brightest surface;
    in Dark the box colours stay distinct from the accent.
- **All three pages at five widths.** Use Chrome DevTools' device toolbar in a
  real browser; AppTest never renders the page.
  - **320 px (WCAG 1.4.10 reflow):** this is the width a 1280 px screen at 400%
    zoom produces. Nothing scrolls sideways, pills and the storage group wrap,
    and every control can still be reached.
  - **390 px (phone):**
    - The body columns stack, conversation above page.
    - The storage group wraps.
    - The toggle and the Transcribe label are visible without opening the
      sidebar drawer.
  - **820 px (tablet):** above the stacking threshold, so both columns must
    still be usable side by side.
  - **1512 px (M2 Pro 14" default, sidebar open):** a located box is visible
    without scrolling.
  - **2560 px (wide):**
    - Answer and Chat lines stop at the 760 px measure.
    - The page image stops at 880 px.
    - The located box is visible without scrolling.
  - **At every width:**
    - no horizontal page scroll
    - page and citation pills wrap rather than scroll
    - `st.chat_input` is pinned at the bottom
  - **Then 1512 px at 200% browser zoom.** That is how a low-vision reader sees
    the page, and it behaves like a narrow screen.
- **Accessibility (step 5d):**
  - **VoiceOver (⌘F5), about 10 minutes:**
    1. On Grounding, run "Stamps and signatures" on `assets/sample-invoice.png`.
       Confirm the box's "where" text is read from the table as "lower right ·
       about 25% of the page wide, 15% tall", the test's fixture case. A few
       pixels of model drift will not change the rounded wording.
    2. In Ask documents, transcribe, ask a question and open a citation. Confirm
       the page caption is read.
    3. Turn "Stream responses" off, ask again, and confirm the answer is read
       once, complete.
    4. Using only the keyboard (Tab / Shift-Tab / Space / Enter), upload,
       transcribe, toggle saving, pick a page pill and open the transcript.
       Focus never gets trapped.
  - **axe DevTools** (free browser extension) on each page, in Light and in
    Dark. Fix every serious or critical issue the app controls; list the ones
    Streamlit controls in the README section.
  - **Colour-blind simulation:** DevTools → Rendering → Emulate vision
    deficiencies → deuteranopia, with "Show label text" off. Every box still
    carries `#N` and matches its table row by number.

## Out of scope

- PDF input.
- Embeddings, reranking and vector search. Cohere's Embed and Rerank are
  API-only; their HF repos hold tokenizers only. If a local MLX embedding model
  is adopted later, put the vectors in the same `transcripts.db` via
  `sqlite-vec` rather than adding a separate vector database.
- Server databases (Postgres, MySQL, MongoDB, Redis). They are not needed for
  one user on one machine, and their RAM competes with the models.
- Storing page images in the database.
- Keeping bf16 as an option.
- A version bump or release.
