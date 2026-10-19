#!/usr/bin/env bash
# PostToolUse: after the first edit under nmv/ in a session, remind Claude that
# Streamlit does not hot-reload those modules (CLAUDE.md, "Restart the server
# after editing anything under nmv/"). Once per session, so it does not repeat
# on every edit.
set -uo pipefail

input=$(cat)
f=$(jq -r '.tool_input.file_path // empty' <<<"$input")
case "$f" in
  */nmv/*.py) ;;
  *) exit 0 ;;
esac

sid=$(jq -r '.session_id // "unknown"' <<<"$input")
marker="${TMPDIR:-/tmp}/claude-nmv-restart-${sid}"
[[ -e "$marker" ]] && exit 0
touch "$marker"

jq -n '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext:
  "You edited a module under nmv/. Streamlit hot-reloads streamlit_app.py and app_pages/* but not nmv/* (cached in sys.modules), so restart the Streamlit server before checking anything in a browser; otherwise a stale ImportError or old behaviour will look like a bug in the new code."}}'
exit 0
