# Privacy

<!-- Outline from plan step 6b. Each comment names what fills its section in M4. -->

## What is stored, and where

<!-- transcripts.db, its permissions, FileVault (4b, 5i) -->

## The save toggle and Clear

<!-- 4b -->

## What is logged

<!-- metadata only (5f), and NMV_TRACE_FILE (5l) -->

## Network traffic

<!-- Hugging Face on first run, HF_HUB_OFFLINE (5i.5, 5j) -->

## Model output is untrusted

A document can carry instructions the model follows. This app gives the model no
tools, so injected instructions can only produce **wrong answers**, not actions:
treat answers about untrusted documents accordingly. The one way such text could
reach outside the app is a Markdown image, which a browser loads from any URL with
no click, so every message on the page has images replaced with `[image: alt]` text
first. Links are left alone, because they need a click.
