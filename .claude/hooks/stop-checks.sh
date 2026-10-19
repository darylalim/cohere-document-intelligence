#!/usr/bin/env bash
# Stop: run the fast checks before Claude finishes a turn that touched Python.
#
# Only ty and the pure-Python tests. The two MLX tests are slower and stay in
# the plan's final checklist. Skipped when no Python under the app is changed
# or untracked, and when this stop was already triggered by this hook (so a
# failure Claude cannot fix cannot loop forever).
set -uo pipefail

input=$(cat)
[[ $(jq -r '.stop_hook_active // false' <<<"$input") == true ]] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

changed=$(
  { git diff --name-only HEAD; git ls-files --others --exclude-standard; } 2>/dev/null |
    grep -E '^(nmv|app_pages|tests)/.*\.py$|^streamlit_app\.py$'
)
[[ -z "$changed" ]] && exit 0

failures=""
if ! out=$(uv run ty check 2>&1); then
  failures+=$'\n--- uv run ty check ---\n'"$(tail -n 40 <<<"$out")"
fi
for t in tests/test_citations.py tests/test_transcripts.py; do
  [[ -f "$t" ]] || continue
  if ! out=$(uv run python "$t" 2>&1); then
    failures+=$'\n--- uv run python '"$t"$' ---\n'"$(tail -n 40 <<<"$out")"
  fi
done
[[ -z "$failures" ]] && exit 0

printf 'Fast checks failed on the changed Python files. Fix them before finishing (or say plainly that they still fail):%s\n' \
  "$failures" >&2
exit 2
