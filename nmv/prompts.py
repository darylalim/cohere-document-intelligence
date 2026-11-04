"""Every fixed model prompt and its version. Pure: no MLX, no Streamlit.

Rule: any change to a prompt's text bumps its ``_VERSION``. The versions feed
every ``Derivation`` key, so a reworded prompt never reuses old results.
``tests/test_prompts.py`` pins a hash of each prompt beside its version, so
changing the text alone fails the test.

The app and the evals both import from here, because a different prompt would
measure a different system.
"""

TRANSCRIBE_PROMPT = (
    "Transcribe this page to Markdown. Preserve tables. Output only the transcription."
)
TRANSCRIBE_PROMPT_VERSION = 1
TRANSCRIBE_MAX_TOKENS = 2048

# Filled with .format(text=...), hence the doubled braces.
LOCATE_PROMPT = 'Locate "{text}". Return a JSON list of {{"bbox_2d": [x1, y1, x2, y2]}}.'
LOCATE_PROMPT_VERSION = 1
LOCATE_MAX_TOKENS = 512
