"""Decide whether a Bash command would change `version` in pyproject.toml.

Called by version-guard.sh with the command on stdin. Exit codes:
  0  allow
  2  deny: a real `uv version` bump, or an in-place sed/perl edit of the
     version line in pyproject.toml
  3  could not parse (e.g. unbalanced quotes): the caller falls back to its
     broad regex, so an unparseable command is still checked, never waved
     through

Only command positions count. The first version matched the raw command text,
so a heredoc or quoted string that merely *mentions* `uv version --bump` (a PR
description, a commit message) was denied. Here heredoc bodies are dropped and
shlex tokenises the rest, so quoted text is one word and never a command.
"""

import re
import shlex
import sys

OPERATORS = {";", "&&", "||", "|", "&", "(", ")", "\n"}
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
VERSION_ARG = re.compile(r"^\d+\.\d")


def strip_heredocs(command: str) -> str:
    """Drop every heredoc body, keeping the line that opens it."""
    kept: list[str] = []
    terminator: str | None = None
    for line in command.split("\n"):
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        kept.append(line)
        match = HEREDOC.search(line)
        if match:
            terminator = match.group(2)
    return "\n".join(kept)


def segments(command: str) -> list[list[str]]:
    """Split into simple commands, each a list of words."""
    # Newline is punctuation so it separates commands; quoting still wins, so a
    # newline inside a quoted string stays part of that one word.
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    result: list[list[str]] = [[]]
    for token in lexer:
        # `token and`: an empty quoted argument (macOS `sed -i ''`) is a real
        # word, and an empty set is a subset of anything.
        if token in OPERATORS or (token and set(token) <= set(";&|()\n")):
            result.append([])
        else:
            result[-1].append(token)
    return [words for words in result if words]


def denies(words: list[str]) -> bool:
    while words and ASSIGNMENT.match(words[0]):  # FOO=1 uv version 1.2
        words = words[1:]
    if not words:
        return False
    head = words[0].rsplit("/", 1)[-1]
    rest = words[1:]
    if head == "uv" and rest[:1] == ["version"]:
        args = rest[1:]
        return any(a.startswith("--bump") or VERSION_ARG.match(a) for a in args)
    if head in {"sed", "gsed", "perl"}:
        in_place = any(re.match(r"^-[A-Za-z]*i", a) for a in rest)
        targets_file = any(a.endswith("pyproject.toml") for a in rest)
        touches_version = any("version" in a for a in rest)
        return in_place and targets_file and touches_version
    return False


def main() -> int:
    command = sys.stdin.read()
    try:
        parsed = segments(strip_heredocs(command))
    except ValueError:
        return 3
    return 2 if any(denies(words) for words in parsed) else 0


if __name__ == "__main__":
    sys.exit(main())
