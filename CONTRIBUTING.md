# Contributing

## Development

```bash
uv run python tests/test_resize_parity.py
uv run python tests/test_runtime_abandon.py
uv run ruff check .
uv run ruff format .
uv run ty check
```

## CI and releases

Every pull request and every push to `main` runs the checks above, plus a grep for
module-scope MLX imports, on an Apple Silicon runner. There is no Linux job: the lock
installs there, but mlx's Linux wheel carries no backend unless installed with its `cpu`
or `cuda` extra, so neither test could import it. `uv sync --locked` fails the build if
`uv.lock` has drifted from `pyproject.toml`.

**Bumping `version` in `pyproject.toml` on `main` publishes a GitHub release.** Once CI is
green the workflow tags that commit, builds notes from the commit subjects since the
previous tag, and publishes — there is no draft to approve. It refuses a version that is
not valid PEP 440 or that is not ahead of the highest existing tag, and marks a
pre-release (`0.2.0rc1`, `0.3.0.dev1`) as such so it does not take the "Latest" badge.
