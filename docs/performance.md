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
  slowest stages took the same with and without it. The run used 1 GB, and the plan's
  `MLX_CACHE_LIMIT_BYTES` is 1 GiB, about 0.07 GB more.
- **Swap grew 2.0 GB on the first run**, 328 MB to 2,361 MB, starting at the reader's load
  with only 7.66 GB active. On the second run, with the limit, it fell to 1,952 MB. That
  second run began with idle pages already paged out, so it shows no sustained paging but
  does not show a first load without swap. Measuring that needs a fresh boot.
- **Chat has the least headroom.** Each extra ceiling image adds about 0.45 GB, so a
  fourth would land near the bar and a fifth would pass it (an extrapolation, not
  measured).
