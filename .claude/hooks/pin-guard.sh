#!/usr/bin/env bash
# PostToolUse: keep every checkpoint pinned to a commit, and never fetch `.py`.
#
# A checkpoint can run code here (CLAUDE.md, "The checkpoint is pinned to a Hub
# commit"), so each *_REVISION must be a 40-hex commit, not a branch or tag, and
# no download allowlist may include Python files.
set -uo pipefail

f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */nmv/*.py) ;;
  *) exit 0 ;;
esac
[[ -f "$f" ]] || exit 0

problems=$(
  grep -nE '^[A-Z_]*_REVISION[[:space:]]*(:[^=]*)?=' "$f" |
    grep -vE '=[[:space:]]*"[0-9a-f]{40}"[[:space:]]*(#.*)?$'
  grep -nE "[\"']\*\.py[\"']" "$f"
)
[[ -z "$problems" ]] && exit 0

printf '%s\n%s\n' \
  "Checkpoint pin guard ($f): every *_REVISION must be a 40-hex Hub commit, and no allow_patterns may fetch *.py, because a checkpoint can run code here (CLAUDE.md, \"The checkpoint is pinned to a Hub commit\"). Offending lines:" \
  "$problems" >&2
exit 2
