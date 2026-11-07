"""The docs stay true to the repo: links, pins, paths and error titles.

Run:  uv run python tests/test_docs.py

``docs/plans/`` is left out on purpose. A plan is a record, so it names files that
never landed and quotes old pins as history.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402
from run import TIERS  # noqa: E402

from nmv import errors  # noqa: E402

README_LIMIT = 300
BF16_REPO = "North-Micro-Vision-Instruct-bf16"

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"(`+)[^\n]*?\1")
_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)")
_EXTERNAL = re.compile(r"[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_REVISION = re.compile(r"--revision\s+([0-9a-f]{40})\b")
_PATH = re.compile(r"(?<![\w./-])(?:tests|scripts|nmv)/[\w./*<>-]*")
_RUNNER_TIER = re.compile(r"tests/run\.py\s+(\w+)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def documents() -> list[Path]:
    found = [*ROOT.glob("*.md")]
    for folder in ("docs", ".github", "evals"):
        found += (ROOT / folder).rglob("*.md")
    plans = ROOT / "docs" / "plans"
    return sorted(path for path in found if not path.is_relative_to(plans))


def prose(text: str) -> str:
    """Drop fenced and inline code, where a link is an example and not a link."""
    return _INLINE_CODE.sub("", _FENCE.sub("", text))


def local_targets(text: str) -> list[str]:
    """Every relative link or image target, without its anchor."""
    targets = (match.split("#")[0] for match in _LINK.findall(prose(text)))
    return [t for t in targets if t and not _EXTERNAL.match(t)]


def read_pins(path: Path) -> dict[str, str]:
    """Each ``*_REPO`` and its ``*_REVISION``, read without importing the module."""
    values: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            match node.value:
                case ast.Constant(value=str() as value):
                    values[node.targets[0].id] = value
    return {
        value: values[name.removesuffix("_REPO") + "_REVISION"]
        for name, value in values.items()
        if name.endswith("_REPO")
    }


def prefetch(repo: str, revision: str) -> re.Pattern[str]:
    return re.compile(
        rf"hf download {re.escape(repo)}\s+(?:\\\s+)?--revision {re.escape(revision)}\b"
    )


docs = {path: path.read_text(encoding="utf-8") for path in documents()}


def name(path: Path) -> str:
    return str(path.relative_to(ROOT))


print("the checker itself")
check(
    "a link inside a fence is ignored",
    local_targets("```md\n![](missing.png)\n```\n") == [],
)
check(
    "a link inside an indented fence is ignored",
    local_targets("1. Step\n\n   ```md\n   [a](missing.md)\n   ```\n") == [],
)
check("a link inside inline code is ignored", local_targets("`![](url)`") == [])
check(
    "a real link and image are found, without the anchor",
    local_targets("[a](b.md#c) ![d](e.png)") == ["b.md", "e.png"],
)
check(
    "external links and bare anchors are skipped",
    local_targets("[a](https://x.y/z) [b](#c) [d](mailto:e@f.g)") == [],
)
check("documents are found", ROOT / "README.md" in docs and ROOT / "CLAUDE.md" in docs)
check(
    "plans are left out",
    not any("plans" in path.relative_to(ROOT).parts for path in docs),
)

print("links resolve")
broken = [
    f"{name(path)} -> {target}"
    for path, text in docs.items()
    for target in local_targets(text)
    if not (path.parent / target).exists()
]
check("every relative link and image exists", not broken, "; ".join(broken))

print("pins match the code")
pins = read_pins(ROOT / "nmv/model.py") | read_pins(ROOT / "nmv/reader.py")
check("both checkpoints are read from the code", len(pins) == 2, str(sorted(pins)))
stale = [
    f"{name(path)}: {revision}"
    for path, text in docs.items()
    for revision in _REVISION.findall(text)
    if revision not in pins.values()
]
check("every --revision is a current pin", not stale, "; ".join(stale))
for filename in ("README.md", "CLAUDE.md"):
    for repo, revision in pins.items():
        check(
            f"{filename} prefetches {repo} at its pin",
            prefetch(repo, revision).search(docs[ROOT / filename]) is not None,
        )

print("referenced paths exist")
missing = sorted(
    {
        f"{name(path)}: {found}"
        for path, text in docs.items()
        for found in (match.rstrip(".,") for match in _PATH.findall(text))
        if not set("*<>") & set(found) and not (ROOT / found).exists()
    }
)
check("every tests/, scripts/ and nmv/ path exists", not missing, "; ".join(missing))
unknown = [
    f"{name(path)}: {tier}"
    for path, text in docs.items()
    for tier in _RUNNER_TIER.findall(text)
    if tier not in (*TIERS, "all")
]
check("every tests/run.py tier is real", not unknown, "; ".join(unknown))

print("no bf16 left behind")
leftover = [
    f"{name(path)}:{number}"
    for path, text in docs.items()
    for number, line in enumerate(text.splitlines(), 1)
    if BF16_REPO in line and "hf cache rm" not in line
]
check(
    "the bf16 repo appears only in a cleanup command", not leftover, "; ".join(leftover)
)

print("README length")
lines = len(docs[ROOT / "README.md"].splitlines())
check(
    f"README is at most {README_LIMIT} lines",
    lines <= README_LIMIT,
    f"{lines} lines: move detail to docs/",
)

print("troubleshooting is complete")
failures = [value for value in vars(errors).values() if isinstance(value, errors.Failure)]
check("every failure is found", len(failures) >= 5, str(len(failures)))
headings = _HEADING.findall(prose(docs[ROOT / "docs/troubleshooting.md"]))
for failure in failures:
    check(
        f"{failure.kind} has a heading",
        any(failure.title in heading for heading in headings),
        failure.title,
    )

finish()
