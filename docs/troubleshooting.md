# Troubleshooting

Each heading is the title of a message in the app, so searching for what is on screen finds
it. Every error has a **Details** section with the underlying exception.

## Ran out of GPU memory

The image or the conversation needs more memory than is free. Lower the image budget or max
tokens in the sidebar. On Ask documents, ask about fewer pages. Quit other heavy apps too,
because they share the same memory.

MLX can also fail in a way that may stop the server instead of showing this message. The
terminal then prints `[METAL] Command buffer execution failed`, and the page stops
responding. Start the app again with a lower image budget.

## Could not reach Hugging Face

A checkpoint is not on disk and could not be downloaded. Check the connection and retry. The
pinned weights download once, then work offline.

With `HF_HUB_OFFLINE=1` set, this means a checkpoint was never fetched. Run the
[prefetch commands](../README.md#weights) with the variable unset.

## The disk is full

Free some disk space. Model weights are the largest thing the app stores, in the Hugging Face
cache at `~/.cache/huggingface`. `uv run hf cache ls` shows what is there, and `HF_HOME` moves
it to an external drive.

An earlier version used the bf16 weights; `uv run hf cache rm model/mlx-community/North-Micro-Vision-Instruct-bf16` frees that space.

## That image could not be read

The file is damaged or is not really the type its name says. Try another file. The app takes
PNG, JPEG, WebP, BMP and TIFF up to 50 MB. A PDF has to be exported to images first.

## Something went wrong

The app hit an error it does not recognise. The details are worth a
[bug report](https://github.com/darylalim/cohere-document-intelligence/issues), but read them
first: an exception's message can quote your document.

## Memory is running short

A sidebar warning, shown after a run if macOS reports memory pressure or swap grew by 64 MB or
more. **Memory is critically low** is the same warning at critical pressure. Swap growth
during a model load does not count, because a first load near a fresh boot swaps about 1 GB
once.

Answers slow down once macOS swaps. Lower the image budget or max tokens. On Ask documents you
can also unload the reader, and the next question loads it again. The sidebar's System panel
shows the current numbers.

## The port is taken

If another app holds port 8501, Streamlit moves to the next free port and prints the address
in the terminal. Open that one. If you passed `--server.port` it stops instead, with
`Port 8501 is not available`. `lsof -i :8501` shows what holds it. An app you forgot to quit in
another terminal is the usual cause, and it keeps its models in memory, so quit it first.

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
