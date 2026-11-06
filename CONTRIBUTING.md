# Contributing

## Setup

You need an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Restart the server after editing anything under `nmv/`. Streamlit hot-reloads
`streamlit_app.py` and `app_pages/`, but not ordinary imported modules.

## Checks

```bash
uv run python tests/run.py all
uv run ruff check .
uv run ruff format .
uv run ty check
```

Each `tests/test_*.py` is a plain script, with no pytest, and names its tier on its first
line as `# tier: <name>`. A file with no header is `fast`. Adding a test means adding a
file, because CI and the hooks all call `tests/run.py`.

| Tier | Needs | Runs in CI |
|---|---|---|
| `fast` | pure Python, no MLX import | Yes |
| `mlx` | Apple Silicon, no weights | Yes |
| `ui` | Streamlit AppTest, models stubbed | Yes |
| `weights` | the model weights | No |

`all` leaves out `weights` unless you pass `--with-weights`, so nobody downloads gigabytes
by accident. `uv run python tests/run.py fast` runs one tier.

**See each new test fail once.** Before committing a test, break the code it covers and
watch the test catch it. A test that has never failed proves nothing.

## Conventions

- **Every module has a docstring explaining why,** in the style of `nmv/runtime.py`. There
  is no generated API reference, because this is not a package.
- **Commit subjects are the changelog.** CI builds release notes from them, so keep them
  imperative and meaningful to a user. There is no `CHANGELOG.md`.
- **Each fact has one home.** A change updates the document that owns its facts in the same
  commit as the code.

## Pull requests

Branch from an updated `main` and open a pull request. Don't commit to `main` directly.
Squash-merge, so one subject reaches the release notes. The pull request keeps the
individual commits.

## CI

Every pull request and every push to `main` runs the checks above, plus static greps for
module-scope MLX imports, unsafe rendering, content in logs and checkpoint pins, on an Apple
Silicon runner. `uv sync --locked` fails the build if `uv.lock` has drifted from
`pyproject.toml`.

The tests have no Linux job: the lock installs there, but mlx's Linux wheel carries no
backend unless installed with its `cpu` or `cuda` extra, so the tests could not import it.
The one check on Linux is `audit`, which looks for known vulnerabilities in the locked
dependencies and installs nothing project-specific. `release` runs there too, but only
publishes.

## Releases

**Never bump `version` in a contribution.** Bumping `version` in `pyproject.toml` on `main`
publishes a GitHub release. Once CI is green the workflow tags that commit, builds notes
from the commit subjects since the previous tag, and publishes — there is no draft to
approve. It refuses a version that is not valid PEP 440 or that is not ahead of the highest
existing tag, and marks a pre-release (`0.2.0rc1`, `0.3.0.dev1`) as such so it does not take
the "Latest" badge. A maintainer follows [`RELEASING.md`](RELEASING.md).

## Claude Code hooks

`.claude/settings.json` enforces the hardest rules while Claude Code works here. Treat a
refusal as the rule speaking. If a guard is wrong, fix its script in `.claude/hooks/` and
re-test it with a piped payload. Never delete the hook to get past it.

| Hook | What a refusal means |
|---|---|
| `version-guard.sh` | The edit or command would change `version`, which only a person may do |
| MLX import guard | A module-scope MLX import appeared. Move it into a worker function |
| `pin-guard.sh` | A `*_REVISION` is not a 40-hex commit, or a download would fetch `*.py` |
| `stop-checks.sh` | `ty`, the fast tier or a static grep failed on changed files |

Two more hooks never refuse: one runs ruff on each edited file, and `restart-reminder.sh`
reminds you to restart Streamlit after the first `nmv/` edit. Deleting cached weights or
`transcripts.db` always asks for permission first.
