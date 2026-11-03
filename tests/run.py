"""Run the plain-script tests by tier.

    uv run python tests/run.py fast|mlx|ui|weights|all [--with-weights]

Each ``tests/test_*.py`` declares its tier on its **first line** as
``# tier: <name>``; a file with no header is ``fast``. Adding a test means
dropping in a file: CI, the Stop hook and the checklist all call this runner, so
none of them lists tests by hand.

| Tier      | Needs                                      |
|-----------|--------------------------------------------|
| `fast`    | pure Python, no MLX import                 |
| `mlx`     | Apple Silicon, no weights                  |
| `ui`      | Streamlit AppTest, models stubbed          |
| `weights` | the model weights, so local only           |

``all`` leaves out ``weights`` unless ``--with-weights`` is passed, so nobody
downloads gigabytes by accident.

Every file runs as its own process, which keeps the plain-script style and
isolates MLX state between files.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
TIERS = ("fast", "mlx", "ui", "weights")
TAIL_LINES = 40

_LOOKS_LIKE_HEADER = re.compile(r"#\s*tier\b", re.IGNORECASE)
_HEADER = re.compile(r"# tier: (\w+)")


def tier_of(path: Path) -> str:
    """Read a test's tier from its first line, refusing anything unrecognised.

    Anything that looks like a header has to name a real tier, so a typo fails
    the run instead of silently moving the file to ``fast`` or dropping it.
    """
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not _LOOKS_LIKE_HEADER.match(first):
        return "fast"
    match = _HEADER.fullmatch(first)
    if match is None or match[1] not in TIERS:
        raise ValueError(f"{path.name}: bad tier header {first!r}")
    return match[1]


def run_file(path: Path) -> bool:
    """Run one test in its own process, printing its tail if it fails."""
    done = subprocess.run(
        [sys.executable, str(path)],
        cwd=TESTS.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    passed = done.returncode == 0
    print(f"{'PASS' if passed else 'FAIL'}  {path.name}", flush=True)
    if not passed:
        print("\n".join(done.stdout.splitlines()[-TAIL_LINES:]), flush=True)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the tests by tier.")
    parser.add_argument("tier", choices=(*TIERS, "all"))
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="include the weights tier in `all`",
    )
    args = parser.parse_args()

    if args.tier == "all":
        wanted = [t for t in TIERS if t != "weights" or args.with_weights]
    else:
        wanted = [args.tier]

    try:
        tiers = {path: tier_of(path) for path in sorted(TESTS.glob("test_*.py"))}
    except ValueError as error:
        print(f"FAIL  {error}; valid tiers are {', '.join(TIERS)}")
        return 1

    selected = [path for tier in wanted for path, t in tiers.items() if t == tier]
    if not selected:
        print(f"no tests in tier {args.tier}")
        return 0

    failed = [path.name for path in selected if not run_file(path)]
    print(f"\n{len(selected) - len(failed)}/{len(selected)} test files passed")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
