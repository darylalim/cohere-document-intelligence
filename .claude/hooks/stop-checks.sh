#!/usr/bin/env bash
# Stop: run the fast checks before Claude finishes a turn that touched Python or docs.
#
# Only ty and the fast test tier. The mlx, ui and weights tiers are slower and
# stay in CI and the plan's final checklist. Skipped when nothing under the app,
# the theme config, the test fixtures or the docs is changed or untracked, and when this
# stop was already triggered by this hook (so a failure Claude cannot fix cannot
# loop forever).
set -uo pipefail

input=$(cat)
[[ $(jq -r '.stop_hook_active // false' <<<"$input") == true ]] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

changed=$(
  { git diff --name-only HEAD; git ls-files --others --exclude-standard; } 2>/dev/null |
    grep -E '^(nmv|app_pages|tests)/.*\.py$|^streamlit_app\.py$|^\.streamlit/config\.toml$|^tests/fixtures/|\.md$|^docs/'
)
[[ -z "$changed" ]] && exit 0

failures=""
if out=$(grep -rnE --include='*.py' '^[^#]*(unsafe_allow_html=True|trust_remote_code=True)' \
  nmv/ app_pages/ streamlit_app.py); then
  failures+=$'\n--- unsafe_allow_html or trust_remote_code ---\n'"$out"
fi
if out=$(grep -rnE 'log(ger)?\.(debug|info|warning|error|exception|critical)\(.*\b(text|transcript|prompt|messages|reply|raw|span|query|label|name)\b' \
  nmv/ app_pages/); then
  failures+=$'\n--- a logging call references document content ---\n'"$out"
fi
if ! out=$(uv run ty check 2>&1); then
  failures+=$'\n--- uv run ty check ---\n'"$(tail -n 40 <<<"$out")"
fi
if ! out=$(uv run python tests/run.py fast 2>&1); then
  failures+=$'\n--- uv run python tests/run.py fast ---\n'"$(tail -n 40 <<<"$out")"
fi
[[ -z "$failures" ]] && exit 0

printf 'Fast checks failed on the changed files. Fix them before finishing (or say plainly that they still fail):%s\n' \
  "$failures" >&2
exit 2
