# Privacy

Documents stay on this Mac. The app sends nothing about them anywhere, and the only network
traffic it makes is downloading model weights.

## What is stored, and where

**Transcripts** are the one thing the app writes to disk from your documents: the text read
from each page, with the file's name. They live in
`~/.cache/cohere-document-intelligence/transcripts.db`, or wherever `NMV_TRANSCRIPT_DB`
points. They are kept so a slow transcription survives a restart.

- The folder is created readable only by you (mode 0700), and the database and its `-wal`
  and `-shm` files are 0600. A folder you chose yourself through `NMV_TRANSCRIPT_DB` keeps
  its own permissions.
- **The app does not encrypt the store.** A key kept next to the database would protect
  nothing, so encryption at rest comes from FileVault. Turn it on in System Settings.
- Once per server start, transcripts over 30 days old are deleted if they came from a
  different checkpoint, prompt or token limit, because they can never be used again.

**Uploads, conversations and answers** are held in memory for the browser session and are
never written to disk by the app. Refreshing the page or stopping the server loses them.

**Model weights** live in the Hugging Face cache, `~/.cache/huggingface` unless `HF_HOME`
says otherwise. Your light or dark theme choice is kept in the browser.

## The save toggle and Clear

Both are on Ask documents.

- **Save transcripts** is on by default. Off keeps this session's transcripts in memory, so
  nothing reaches disk. Transcripts saved earlier stay until cleared.
- **Clear saved transcripts** opens **Delete every saved transcript**, and shows where the
  store is. It is unavailable while pages are being transcribed. A transcription running in
  another tab can still save a row afterwards, so the message says how many remain.
- Deleted text is really removed: the store overwrites it, compacts the database and
  empties its write-ahead log, rather than only marking rows as gone.
- **Search saved documents** reads the same store, with saving on or off, and writes
  nothing. A page added from search leaves the conversation's documents once its
  transcript is cleared.

## What is logged

The app logs to the terminal only, and keeps no log file. Lines hold metadata such as
timings, counts and IDs, never document text, file names, prompts or answers, and CI rejects
a logging call that references them. `NMV_LOG_LEVEL` sets the level.

An unexpected error prints a traceback to the terminal with the message left out, because a
message can quote what it was handling. The full exception appears only under **Details** on
the page, so read it before pasting it into a bug report. This covers the app's own logging.
An error the app does not catch is logged by Streamlit, which prints it in full, still only to
the terminal.

`NMV_TRACE_FILE=<path>` is off by default. When set, each action appends a record of numbers,
IDs and enum values to that file. See
[Saving run records](performance.md#saving-run-records).

## Network traffic

- **Hugging Face, for weights.** The app contacts it to download each pinned checkpoint the
  first time it is needed. No document content is part of that.
- **Nothing else.** The server listens on `localhost` only, so other machines on the network
  cannot reach it. Streamlit's usage statistics are off, and fonts are bundled.

To run with no Hugging Face access at all, fetch both checkpoints first (the
[prefetch commands](../README.md#weights)), then start the app with:

```bash
HF_HUB_OFFLINE=1 uv run streamlit run streamlit_app.py
```

This is not the default, because the first run needs the network. Offline, a checkpoint that
is not cached shows [Could not reach Hugging Face](troubleshooting.md#could-not-reach-hugging-face).

## Model output is untrusted

A document can carry instructions the model follows. This app gives the model no
tools, so injected instructions can only produce **wrong answers**, not actions:
treat answers about untrusted documents accordingly. The one way such text could
reach outside the app is a Markdown image, which a browser loads from any URL with
no click, so every message on the page has images replaced with `[image: alt]` text
first. Links are left alone, because they need a click.
