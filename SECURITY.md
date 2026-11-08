# Security

## Reporting a vulnerability

Please report it privately through
[GitHub's private vulnerability reporting](https://github.com/darylalim/cohere-document-intelligence/security/advisories/new).
Do not open a public issue.

## Trust model

This is a local app for one person on one Mac. What needs protecting is the content of your
documents, the Mac itself, and your network.

- **Local only.** The server listens on `localhost`, so other machines cannot reach it. It has
  no login, so do not widen `server.address`. If remote access is ever needed, put an
  authenticated private tunnel in front of it.
- **No telemetry.** `gatherUsageStats` is off, and the only network traffic is downloading
  weights. [`docs/privacy.md`](docs/privacy.md) lists every data flow, and how to run with no
  Hugging Face access at all.
- **Transcripts stay local.** They are stored in a file only you can read, with a control to
  clear them. The app does not encrypt that file, so use FileVault.
- **Pinned checkpoints.** A checkpoint can run code, so each one is pinned to a reviewed Hub
  commit with no Python files, and the reader's download leaves `*.py` out. CI and the Claude
  Code hooks reject a pin that is not a full commit hash.
- **Uploads are untrusted.** Images open only with the PNG, JPEG, WebP, BMP and TIFF
  decoders, whatever their contents. PDFs go to PDFium, Chrome's PDF engine, in the build
  pypdfium2 bundles, which has no JavaScript engine and no XFA forms. Rendering is capped at
  the encoder's ceiling of 3.9 MP a page, and only the pages used are rendered.
- **Model output is untrusted.** A document can carry instructions the model follows. The
  model has no tools, so that can produce wrong answers but not actions. Images in model
  output are replaced with text, and raw HTML rendering stays off.
- **Supply chain.** Dependencies are hash-locked in `uv.lock`, a weekly CI run audits them for
  known vulnerabilities, and every GitHub Action is pinned to a full commit SHA.
