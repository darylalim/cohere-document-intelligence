#!/usr/bin/env bash
# PreToolUse: refuse any change to `version` in pyproject.toml.
#
# Editing it is a publish action. On main, a green CI run tags the commit and
# publishes a release with no draft and no undo (CLAUDE.md, "CI and releases").
# The user bumps the version by hand when they mean to ship.
#
# Edits that carry the version line through unchanged (e.g. as context around a
# dependency change) are allowed; only a changed or removed line is denied.
set -uo pipefail

input=$(cat)
tool=$(jq -r '.tool_name // empty' <<<"$input")

reason='Changing `version` in pyproject.toml is a publish action: on main, a green CI run tags and releases it with no undo (CLAUDE.md, "CI and releases"). Leave the version line untouched; the user bumps it by hand when they mean to ship.'

deny() {
  jq -n --arg r "$reason" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  exit 0
}

# The [project] version line. Anchored, so `target-version` and
# `python-version` never match.
version_of() { grep -E '^version[[:space:]]*=' | head -1 || true; }

case "$tool" in
  Edit | Write | MultiEdit)
    f=$(jq -r '.tool_input.file_path // empty' <<<"$input")
    [[ "$f" == */pyproject.toml || "$f" == pyproject.toml ]] || exit 0
    if [[ "$tool" == Write ]]; then
      new=$(jq -r '.tool_input.content // ""' <<<"$input" | version_of)
      old=""
      [[ -f "$f" ]] && old=$(version_of <"$f")
      [[ "$new" == "$old" ]] || deny
    else
      while IFS= read -r pair; do
        o=$(jq -r '.old_string // ""' <<<"$pair" | version_of)
        n=$(jq -r '.new_string // ""' <<<"$pair" | version_of)
        [[ "$o" == "$n" ]] || deny
      done < <(jq -c 'if .tool_input.edits then .tool_input.edits[] else .tool_input end' <<<"$input")
    fi
    ;;
  Bash)
    cmd=$(jq -r '.tool_input.command // empty' <<<"$input")
    # `uv version` on its own only prints; a bump or an explicit number writes.
    if grep -qE '(^|[^[:alnum:]_-])uv[[:space:]]+(run[[:space:]]+)?version([[:space:]].*)?(--bump|[[:space:]][0-9]+\.[0-9])' <<<"$cmd"; then
      deny
    fi
    # In-place rewrites of the version line from the shell.
    if grep -qE 'pyproject\.toml' <<<"$cmd" &&
      grep -qE '(sed|perl)[[:space:]]+(-[[:alnum:]]*[[:space:]]+)*-[[:alnum:]]*i' <<<"$cmd" &&
      grep -qE '(^|[^[:alnum:]_-])version' <<<"$cmd"; then
      deny
    fi
    ;;
esac
exit 0
