"""The transcript store: SQLite, so a slow transcription survives a restart.

Pure: no MLX and no Streamlit. These rows are the contents of the user's documents.

- **Keys come from ``Derivation.key()``,** so a new checkpoint, prompt or budget is
  a different key and a stale transcript cannot be served. There is no invalidation.
- **One connection per call.** Streamlit runs each rerun on a fresh thread, and
  ``sqlite3`` connections stay on the thread that opened them.
- **Deleting really deletes.** FTS5 keeps deleted terms in its segments until a
  merge, and WAL keeps old pages until a checkpoint, so every path that removes
  text ends in :func:`_scrub`.
- **The cache must never take the pipeline down.** Reads and writes that fail
  count as a miss and are logged; the page asks :func:`available` and warns. The
  exception is :func:`clear`, which raises ``STORE_ERRORS``: reporting a failed
  wipe as "0 removed" would be a false promise. A database that will not open is
  never deleted, since the user may want it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

from nmv.domain import Derivation, PageId, Transcript

logger = logging.getLogger(__name__)

DB_ENV = "NMV_TRANSCRIPT_DB"
DEFAULT_DB = Path("~/.cache/cohere-document-intelligence/transcripts.db")
SCHEMA_VERSION = 1
PRUNE_AFTER_DAYS = 30

STORE_ERRORS: tuple[type[Exception], ...] = (sqlite3.Error, OSError)

# ``id`` is explicit because the FTS index refers to rows by rowid, and VACUUM may
# renumber the rowids of a table that has no INTEGER PRIMARY KEY. Once version 1 has
# shipped, never edit this: bump SCHEMA_VERSION and add a numbered migration.
_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS transcripts (
    id             INTEGER PRIMARY KEY,
    key            TEXT NOT NULL UNIQUE,  -- Derivation.key()
    page_id        TEXT NOT NULL,
    text           TEXT NOT NULL,
    finish_reason  TEXT,
    width          INTEGER NOT NULL,      -- resolved size NMV actually saw
    height         INTEGER NOT NULL,
    model_revision TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    resolved_px    INTEGER NOT NULL,
    max_tokens     INTEGER NOT NULL,
    filename       TEXT,                  -- display only, never part of the key
    created        TEXT NOT NULL          -- ISO-8601 UTC, fixed width
);
CREATE INDEX IF NOT EXISTS transcripts_page ON transcripts(page_id);
CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    text, content='transcripts', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO transcripts_fts(rowid, text) VALUES (new.id, new.text);
END;
PRAGMA user_version = 1;
COMMIT;
"""

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class Hit:
    """A search result. ``snippet`` is model output with ``**`` around the match."""

    key: str
    page: PageId
    filename: str | None
    snippet: str
    created: str


def db_path() -> Path:
    return Path(os.environ.get(DB_ENV) or DEFAULT_DB).expanduser()


def _timestamp(moment: datetime) -> str:
    """Fixed width, so comparing the stored strings compares the times."""
    return moment.isoformat(timespec="microseconds")


def _prepare_files(path: Path) -> None:
    """Create the file before SQLite does, so it is never readable by others.

    Only a directory made here is restricted: ``NMV_TRANSCRIPT_DB`` can point into
    one the user owns for other reasons. SQLite gives its ``-wal`` and ``-shm``
    files the database's permissions.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)


@contextmanager
def _connect() -> Generator[sqlite3.Connection]:
    path = db_path()
    _prepare_files(path)
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout = 5000")
        db.execute("PRAGMA secure_delete = ON")
        db.execute("PRAGMA journal_mode = WAL")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            # Every checkout shares this file, so a newer one may have migrated it.
            raise sqlite3.DatabaseError(f"schema version {version} is newer than this")
        if version < SCHEMA_VERSION:
            db.executescript(_SCHEMA)
        yield db


_last_failure: str | None = None


def _or_else(miss: Callable[[], R]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log a store failure and return ``miss()``, so the pipeline carries on.

    Every rerun reads the store several times, so a broken one would repeat the same
    line a dozen times a rerun. It logs when the failure changes, and again if the
    store recovers and then fails.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
            global _last_failure
            try:
                result = function(*args, **kwargs)
            except STORE_ERRORS as error:
                failure = f"the store at {db_path()} failed: {error}"
                if failure != _last_failure:
                    logger.warning(failure)
                _last_failure = failure
                return miss()
            _last_failure = None
            return result

        return guarded

    return decorate


def _scrub(db: sqlite3.Connection) -> None:
    """Remove deleted text from the FTS segments, then from the WAL."""
    with db:
        db.execute("INSERT INTO transcripts_fts(transcripts_fts) VALUES ('optimize')")
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()


def _transcript(row: sqlite3.Row) -> Transcript:
    derivation = Derivation(
        PageId(row["page_id"]),
        row["resolved_px"],
        row["model_revision"],
        row["prompt_version"],
        row["max_tokens"],
    )
    return Transcript(
        derivation, row["text"], row["finish_reason"], row["width"], row["height"]
    )


@_or_else(lambda: False)
def available() -> bool:
    """Whether transcripts are being saved. False means the page should warn."""
    with _connect():
        return True


def _no_transcript() -> Transcript | None:
    return None


@_or_else(_no_transcript)
def get(key: str) -> Transcript | None:
    with _connect() as db:
        row = db.execute("SELECT * FROM transcripts WHERE key = ?", (key,)).fetchone()
    return _transcript(row) if row else None


@_or_else(list)
def by_page(page: PageId) -> list[Transcript]:
    """Every saved transcript of a page, at any budget, newest first."""
    with _connect() as db:
        rows = db.execute(
            "SELECT * FROM transcripts WHERE page_id = ? ORDER BY created DESC, id DESC",
            (page,),
        ).fetchall()
    return [_transcript(row) for row in rows]


@_or_else(lambda: False)
def put(transcript: Transcript, filename: str | None = None) -> bool:
    """Save a transcript under its derivation's key. False if it was not saved.

    Never ``INSERT OR REPLACE``: REPLACE deletes the old row without firing the
    delete trigger, which leaves stale entries in the FTS index and corrupts it.
    """
    made = transcript.derivation
    key = made.key()
    with _connect() as db:
        with db:
            replaced = db.execute(
                "SELECT 1 FROM transcripts WHERE key = ?", (key,)
            ).fetchone()
            db.execute(
                """
                INSERT INTO transcripts (
                    key, page_id, text, finish_reason, width, height, model_revision,
                    prompt_version, resolved_px, max_tokens, filename, created
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    text = excluded.text,
                    finish_reason = excluded.finish_reason,
                    width = excluded.width,
                    height = excluded.height,
                    filename = excluded.filename,
                    created = excluded.created
                """,
                (
                    key,
                    made.page,
                    transcript.text,
                    transcript.finish_reason,
                    transcript.width,
                    transcript.height,
                    made.model_revision,
                    made.prompt_version,
                    made.resolved_px,
                    made.max_tokens,
                    filename,
                    _timestamp(datetime.now(UTC)),
                ),
            )
        if replaced:
            _scrub(db)
    return True


@_or_else(int)
def other_versions(model_revision: str, prompt_version: int, max_tokens: int) -> int:
    """How many rows the explicit "other versions" action would remove."""
    with _connect() as db:
        return db.execute(
            """
            SELECT count(*) FROM transcripts
            WHERE model_revision != ? OR prompt_version != ? OR max_tokens != ?
            """,
            (model_revision, prompt_version, max_tokens),
        ).fetchone()[0]


@_or_else(int)
def prune(
    model_revision: str,
    prompt_version: int,
    max_tokens: int,
    older_than_days: int | None = PRUNE_AFTER_DAYS,
) -> int:
    """Remove rows made by another version, and return how many went.

    Age matters because every checkout shares this file, so a row from another
    version may be valid in the tag the user just switched from. ``None`` ignores
    age, for the explicit action. Different budgets are not other versions: they
    stay reachable by moving the slider back.
    """
    before = (
        "9999"  # later than any timestamp
        if older_than_days is None
        else _timestamp(datetime.now(UTC) - timedelta(days=older_than_days))
    )
    with _connect() as db:
        with db:
            removed = db.execute(
                """
                DELETE FROM transcripts
                WHERE (model_revision != ? OR prompt_version != ? OR max_tokens != ?)
                  AND created < ?
                """,
                (model_revision, prompt_version, max_tokens, before),
            ).rowcount
        if removed:
            _scrub(db)
    if removed:
        logger.info("pruned %d rows", removed)
    return removed


def clear() -> int:
    """Delete every transcript and return how many went. Raises ``STORE_ERRORS``.

    In WAL mode VACUUM does not shrink the main file until a checkpoint, so the
    checkpoint comes last. A row written by another tab can still land afterwards,
    which is why the page shows the remaining count.
    """
    with _connect() as db:
        with db:
            removed = db.execute("DELETE FROM transcripts").rowcount
        _scrub(db)
        db.execute("VACUUM")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    logger.info("cleared %d rows", removed)
    return removed


@_or_else(lambda: (0, 0))
def usage() -> tuple[int, int]:
    """How many transcripts are saved, and how many bytes of text they hold."""
    with _connect() as db:
        count, size = db.execute(
            "SELECT count(*), coalesce(sum(length(CAST(text AS BLOB))), 0) "
            "FROM transcripts"
        ).fetchone()
    return count, size


@_or_else(list)
def search(query: str, limit: int = 20) -> list[Hit]:
    """Full-text search. Each term is quoted, so FTS5 syntax in it means nothing."""
    terms = " ".join('"' + term.replace('"', '""') + '"' for term in query.split())
    if not terms:
        return []
    with _connect() as db:
        rows = db.execute(
            """
            SELECT t.key, t.page_id, t.filename, t.created,
                   snippet(transcripts_fts, 0, '**', '**', '…', 12) AS snippet
            FROM transcripts_fts
            JOIN transcripts AS t ON t.id = transcripts_fts.rowid
            WHERE transcripts_fts MATCH ?
            ORDER BY bm25(transcripts_fts)
            LIMIT ?
            """,
            (terms, limit),
        ).fetchall()
    return [
        Hit(
            row["key"],
            PageId(row["page_id"]),
            row["filename"],
            row["snippet"],
            row["created"],
        )
        for row in rows
    ]
