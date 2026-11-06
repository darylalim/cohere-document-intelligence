"""Terminal logging for the ``nmv`` loggers. Metadata only, never content.

Streamlit configures only its own loggers, so without this ``nmv`` records fall through
to Python's fallback handler, which drops INFO. There is no log file on purpose: a file
would keep document metadata with no Clear button.

``threadName`` is the project's diagnostic. Worker lines say ``mlx-worker_0`` and page
lines ``ScriptRunner``, so an MLX call logged from a page thread is the thread-affinity
bug ``nmv/runtime.py`` exists to prevent.
"""

from __future__ import annotations

import logging
import os

from nmv import trace

LEVEL_ENV = "NMV_LOG_LEVEL"
DEFAULT_LEVEL = "INFO"
FORMAT = (
    "%(asctime)s %(levelname)-7s %(threadName)-14s %(name)s "
    "run=%(run)s session=%(session)s · %(message)s"
)
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_MARKER = "_nmv_handler"


def setup() -> None:
    """Attach the handler once. ``streamlit_app.py`` reruns on every interaction, and
    a second handler would print every line twice."""
    logger = logging.getLogger("nmv")
    if any(hasattr(handler, _MARKER) for handler in logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    # On the handler, because a logger's own filters skip its children's records.
    handler.addFilter(trace.RunFilter())
    setattr(handler, _MARKER, True)
    logger.addHandler(handler)
    logger.propagate = False

    wanted = os.environ.get(LEVEL_ENV, DEFAULT_LEVEL).upper()
    logger.setLevel(wanted if wanted in _LEVELS else DEFAULT_LEVEL)
    if wanted not in _LEVELS:
        logger.warning("unknown %s, using %s", LEVEL_ENV, DEFAULT_LEVEL)
