# Performance

## Quality

### Quality: 8-bit against bf16

`evals/set-v1`, 21 documents, 24 pages, 57 grounding targets. Temperature 0, ceiling
image budget, M2 Pro 16 GB, 2026-11-03, code at `8caf485`. Summaries are in
`evals/results/`. The 8-bit summary says `dirty: true` only because the bf16 summary was
untracked when it ran.

| Metric | bf16 | 8-bit |
|---|---|---|
| Grounding hits (IoU ≥ 0.5) | 25 / 57 | 24 / 57 |
| Grounding, no usable box | 7 | 6 |
| Mean IoU, where a box parsed | 0.531 | 0.527 |
| CER, markup removed | 4.24% | 4.29% |
| Field recall | 98.96% | 98.96% |
| Truncated replies | 1 | 1 |
| Peak memory | 6.35 GB | 4.52 GB |
| Decode, transcription | 41 tok/s | 66 tok/s |
| Prefill, transcription | 390 tok/s | 364 tok/s |

8-bit lost one target that bf16 hit (`statement-01`'s closing balance, IoU 0.60 to 0.42)
and gained none. One page's CER moved by half a point or more (`photo-02`, 21.7% to
23.0%). 26 of the 81 raw outputs differ between the two runs, so the runs are distinct.
Neither of step 1b's flags fired: more than 2 lost targets, or a CER rise above 1 point.

### What the grounding numbers mean

The absolute hit rate understates grounding, on both checkpoints alike:

- **Stamps and signatures are strong:** IoU 0.74–0.97 on every one.
- **On text lines the model boxes the salient part.** Asked for
  `Total due: $6,877.71` it boxes the amount alone, which scores about 0.3 and is still
  on target. The box's centre is inside the truth on 37 of 57 targets at 8-bit.
- **It sometimes answers with a point**, such as `[[852, 301, 852, 301]]`: 6 of 57 at
  8-bit, including all four receipt targets. `parse_boxes` rejects these as degenerate,
  so they count as "no usable box", although three of the six land inside the truth.
- **7 of 57 boxes are off target**, for example the whole column for a column heading.

## Memory

### Both models on 16 GB

`scripts/memory_check.py` on an M2 Pro 16 GB, 2026-11-03, with North Micro Vision 8-bit
and Command R7B 4-bit both resident. The GPU working-set limit reads 12.71 GB, so the bar
(90%) is 11.44 GB. Figures are peak plus MLX's buffer cache, in GB.

| Stage | No cache limit | 1 GB cache limit |
|---|---|---|
| Both models loaded | 7.66 | 7.66 |
| Grounding at the ceiling (3,762 tokens) | 9.77 | 9.77 |
| Transcribing at the ceiling | 9.77 | 9.77 |
| Question over one transcript | 10.76 | 9.75 |
| Question at the reader's context limit (6,999 tokens) | **13.32** | 10.44 |
| Chat, one ceiling image | 9.75 | 9.75 |
| Chat, two ceiling images | 10.42 | 10.42 |
| Chat, three ceiling images (11,327 tokens) | 11.38 | **10.99** |
| Result | fails | passes, at 86.5% of the limit |

- **The cache limit decides the result.** Without it, 3.89 GB of cache sits on 9.43 GB of
  live memory at the reader's context limit. The limit cost no measurable time: the two
  slowest stages took the same with and without it. The table's run set 1 GB by hand.
  The app's own `MLX_CACHE_LIMIT_BYTES` is 1 GiB, applied when the reader loads, and a
  run through that path peaked at 11.06 GB, 87.0% of the limit, with swap flat.
- **Swap grew 2.0 GB on the first run**, 328 MB to 2,361 MB, starting at the reader's load
  with only 7.66 GB active. On the second run, with the limit, it fell to 1,952 MB. That
  second run began with idle pages already paged out, so it shows no sustained paging but
  does not show a first load without swap. Measuring that needs a fresh boot.
- **A first load near a fresh boot swaps about 1.1 GB.** On 2026-11-04, 18 minutes after
  boot with Chrome open and swap at 0, `tests/test_models_e2e.py` took it to 1,135 MB, with
  peak memory at 8.97 GB. Later runs of both models stayed flat or fell, to 936 MB. So
  the growth is a one-off at first load, not sustained paging. Step 8 repeats this.
- **Chat has the least headroom.** Each extra ceiling image adds about 0.45 GB, so a
  fourth would land near the bar and a fifth would pass it (an extrapolation, not
  measured).

## Speed

### The performance pass

`scripts/perf_pass.py` on an M2 Pro 16 GB, 2026-11-04, code at `7f94c42`, with both models
resident. The summary is `evals/results/2026-11-04-7f94c42-perf-pass.json`. It says
`dirty: true` only because the script was untracked when it ran, and its `sweep` was
rescored afterwards from the saved transcripts, because the first scoring counted a
reference transcript's table pipes as errors.

The document is 10 generated pages at A4 200 dpi, which is the ceiling, with ~10 pt prose
and ~8.5 pt tables. Every third page is a table and every second is photocopied. The
set-v1 pages cannot serve here: at about 2.17 MP they never reach the ceiling. These are
clean renders, so a real scan will probably do worse at every budget.

| Budget | Image tokens | Seconds per page | Prefill | Decode | CER against 3.87 MP | Worst page | Fields missed of 389 |
|---|---|---|---|---|---|---|---|
| 3.87 MP | 3,723 | 31.2 | 267 tok/s | 59 tok/s | 0 | 0 | 0 |
| 2.0 MP | 1,924 | 21.1 | 362 tok/s | 64 tok/s | 0.15% | 1.32% | 1 |
| 1.0 MP | 962 | 16.9 | 437 tok/s | 67 tok/s | 0.33% | 0.69% | 31 |

- **Transcription stays at the full budget (decision 3).** 2.0 MP meets step 5g.3's bar of
  about 1% CER and saves about 100 s per 10 pages, but it turned `NT-5658` into `NT-5568`
  and dropped page 3's title. The saving is paid once per page, and a misread digit looks
  correct and gets cited. The slider still lets anyone trade accuracy for speed.
- **CER hides what matters on numeric pages.** 1.0 MP loses 8% of the fields at a CER of
  0.33%, so read field recall alongside it.
- **A store hit takes about 1 ms,** against 28–36 s for a miss. Ten pages take about
  5.2 minutes the first time. The first page was no slower than the rest, so no warm-up
  generation is needed.

| Question over | Prompt tokens | First word, question 1 | First word, question 2 | Thinking tokens |
|---|---|---|---|---|
| 3 pages | 5,377 | 17.7 s | 17.9–18.5 s | 0 |
| 4 pages, 6 dropped | 6,398 | 22.3 s | 22.6 s | 0 |

- **Prefill is the whole wait,** at about 300 tok/s. A follow-up costs what the first
  question does, because nothing is reused (step 5g.2).
- **R7B wrote no thinking block** in any of 12 answers, so shortening thinking would save
  nothing on these questions.
- **Four dense pages fill the reader.** Of 10 pages, 6 were dropped to fit its 8K context.

### Cache and swap in the pass

The buffer cache stood at 0.65 GB after 10 ceiling pages and reached the 1 GiB cap only
during questions. The worst peak plus cache was 10.5 GB, about 83% of the 12.71 GB limit,
so `MLX_CACHE_LIMIT_BYTES` stays at 1 GiB. Swap went from 0 to 529 MB during the load, then
fell to 451 MB with no growth during any run, which matches the one-off above.

In the app, the reader's first load took swap from 427 MB to 947 MB with pressure Normal,
and it stayed flat on the follow-up. That used to raise "Memory is running short" on the
first question, so swap growth across a model load no longer warns. Pressure still does.

## Saving run records

Set `NMV_TRACE_FILE=<path>` and every action appends its run record to that file as one
JSON line, for medians and spread over many runs instead of reading the sidebar. It is
unset by default. A record holds numbers, IDs and enum values only, never document text,
prompts or file names, so the file may be kept anywhere. A run Streamlit interrupts
mid-stream leaves no record.
