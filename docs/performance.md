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
