# Troubleshooting

<!-- Outline from plan step 6b. Headings reuse the titles in nmv/errors.py verbatim, so
     searching the on-screen text finds them. -->

## Ran out of GPU memory

<!-- out_of_memory, and the Metal failure that stops the server (5e) -->

## Could not reach Hugging Face

<!-- download -->

## The disk is full

<!-- disk_full, and the bf16 cache cleanup (step 1) -->

## That image could not be read

<!-- bad_image -->

## Something went wrong

<!-- internal -->

## Memory is running short

<!-- pressure and swap (5k) -->

## The port is taken

<!-- 5j -->

## Doctor failures

`scripts/doctor.py` prints one line per check, and exits non-zero if any says `FAIL`:

```bash
uv run --no-project --python 3.12 scripts/doctor.py
```

- **Apple Silicon.** MLX needs Apple Silicon and its GPU, so an Intel Mac, a Linux machine
  or a Docker container on a Mac cannot run this app.
- **macOS.** The locked `mlx` has no build for this version. Update macOS to the version
  the line names.
- **Free disk.** The weights, the environment and the transcript store need 10 GB on the
  volume holding `HF_HOME`. Free some space, or set `HF_HOME` to an external drive.
- **uv.** [Install uv](https://docs.astral.sh/uv/getting-started/installation/), then open
  a new terminal.

Two checks only warn. Under 16 GB of memory, Chat and Grounding work but Ask documents will
not fit both models. `Weights cached` lists the download commands for anything missing, and
the app also offers those downloads itself.
