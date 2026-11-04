"""The transcript store, on a temporary database.

Run:  uv run python tests/test_transcripts.py
"""

import logging
import os
import sqlite3
import stat
import sys
import tempfile
import threading
from contextlib import closing
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import transcripts  # noqa: E402
from nmv.domain import Derivation, Transcript, page_id  # noqa: E402

logging.disable(logging.CRITICAL)  # the failure cases log by design

REVISION = "a" * 40
PROMPT = 1
TOKENS = 2048
PAGE = page_id(b"page")
BASE = Derivation(PAGE, 1_000_000, REVISION, PROMPT, TOKENS)

temp = tempfile.TemporaryDirectory()
root = Path(temp.name)
db_number = 0


def fresh_db() -> Path:
    """Point the store at a new file in a directory that does not exist yet."""
    global db_number
    db_number += 1
    path = root / f"store-{db_number}" / "transcripts.db"
    os.environ[transcripts.DB_ENV] = str(path)
    return path


def saved(text: str = "Total due", **changes) -> Transcript:
    return Transcript(replace(BASE, **changes), text, "stop", 800, 1100)


def raw(sql: str, *args) -> list:
    with closing(sqlite3.connect(transcripts.db_path())) as db, db:
        return db.execute(sql, args).fetchall()


def age(days: int) -> None:
    moment = transcripts.datetime.now(transcripts.UTC) - transcripts.timedelta(days=days)
    raw("UPDATE transcripts SET created = ?", transcripts._timestamp(moment))


def fts_is_sound() -> bool:
    try:
        raw(
            "INSERT INTO transcripts_fts(transcripts_fts, rank) "
            "VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError:
        return False
    return True


def bytes_on_disk(path: Path) -> bytes:
    return b"".join(
        file.read_bytes()
        for file in path.parent.iterdir()
        if file.name.startswith(path.name)
    )


print("round trip")
fresh_db()
original = saved()
check("a miss before anything is saved", transcripts.get(BASE.key()) is None)
check("put reports success", transcripts.put(original, "invoice.png"))
check("get returns it", transcripts.get(BASE.key()) == original)
check(
    "it stores the file name",
    raw("SELECT filename FROM transcripts") == [("invoice.png",)],
)
check("the schema is versioned", raw("PRAGMA user_version") == [(1,)])
check("the store says it is available", transcripts.available())

print("every derivation field misses")
for name, value in {
    "model_revision": "b" * 40,
    "resolved_px": 999_999,
    "prompt_version": 2,
    "max_tokens": 4096,
    "page": page_id(b"other"),
}.items():
    check(
        f"a different {name}",
        transcripts.get(replace(BASE, **{name: value}).key()) is None,
    )

print("by_page")
fresh_db()
transcripts.put(saved("low", resolved_px=500_000))
transcripts.put(saved("high", resolved_px=2_000_000))
transcripts.put(saved("elsewhere", page=page_id(b"other")))
check(
    "every budget, newest first",
    [t.text for t in transcripts.by_page(PAGE)] == ["high", "low"],
)

print("put twice")
fresh_db()
transcripts.put(saved("first draft"))
transcripts.put(saved("second draft"))
check("the later text wins", transcripts.get(BASE.key()) == saved("second draft"))
check("one row", raw("SELECT count(*) FROM transcripts") == [(1,)])
check("FTS integrity-check passes", fts_is_sound())
check("search finds the new text", len(transcripts.search("second")) == 1)
check("search forgets the old text", transcripts.search("first") == [])

print("a file that is not a database")
path = fresh_db()
path.parent.mkdir()
path.write_bytes(b"this is not a database, " * 100)
before = path.read_bytes()
check("the store is unavailable", not transcripts.available())
check("get is a miss", transcripts.get(BASE.key()) is None)
check("by_page is a miss", transcripts.by_page(PAGE) == [])
check("put is skipped", not transcripts.put(saved()))
check("usage is empty", transcripts.usage() == (0, 0))
check("search is empty", transcripts.search("total") == [])
check("the file is untouched", path.read_bytes() == before)
try:
    transcripts.clear()
except transcripts.STORE_ERRORS:
    check("clear refuses to report success", True)
else:
    check("clear refuses to report success", False)

print("a database from a newer checkout")
path = fresh_db()
transcripts.put(saved())
raw("PRAGMA user_version = 2")
check("it is left alone", not transcripts.available())
check("the row is still there", raw("SELECT count(*) FROM transcripts") == [(1,)])

print("a parent that cannot be a directory")
blocker = root / "a-file"
blocker.write_text("x")
os.environ[transcripts.DB_ENV] = str(blocker / "transcripts.db")
check("it is a miss, not a crash", transcripts.get(BASE.key()) is None)

print("clear")
path = fresh_db()
for number in range(200):
    transcripts.put(saved(f"filler text number {number} " * 200, resolved_px=number + 1))
full = path.stat().st_size
check("it reports the rows removed", transcripts.clear() == 200)
check("the table is empty", transcripts.usage() == (0, 0))
check("the index is empty", transcripts.search("filler") == [])
check("FTS integrity-check passes", fts_is_sound())
check(
    "the file shrank", path.stat().st_size < full / 2, f"{full} -> {path.stat().st_size}"
)
check("clearing nothing is fine", transcripts.clear() == 0)

print("prune")
path = fresh_db()
transcripts.put(saved("zanzibarquartz", model_revision="b" * 40))
transcripts.put(saved("kept current"))
transcripts.put(saved("kept other budget", resolved_px=5))
check("the pruned term starts on disk", b"zanzibarquartz" in bytes_on_disk(path))
current = (REVISION, PROMPT, TOKENS)
check("a new other-version row survives", transcripts.prune(*current) == 0)
check("it is counted as another version", transcripts.other_versions(*current) == 1)
age(31)
check("an old one goes", transcripts.prune(*current) == 1)
check("search forgets it", transcripts.search("zanzibarquartz") == [])
check("its term leaves the file", b"zanzibarquartz" not in bytes_on_disk(path))
check("current rows stay at any age and budget", transcripts.usage()[0] == 2)
check("FTS integrity-check passes", fts_is_sound())

fresh_db()
transcripts.put(saved("a", prompt_version=2))
transcripts.put(saved("b", max_tokens=1))
transcripts.put(saved("c"))
check(
    "the explicit action ignores age",
    transcripts.prune(*current, older_than_days=None) == 2,
)
check("and keeps current rows", [t.text for t in transcripts.by_page(PAGE)] == ["c"])

print("overwriting removes the old text from disk")
path = fresh_db()
transcripts.put(saved("mozambiquejasper"))
transcripts.put(saved("replacement"))
check("its term leaves the file", b"mozambiquejasper" not in bytes_on_disk(path))

print("usage")
fresh_db()
transcripts.put(saved("héllo"))
check("it counts bytes, not characters", transcripts.usage() == (1, 6))

print("search")
fresh_db()
transcripts.put(saved("Invoice total due from AT&T"), "bill.png")
hits = transcripts.search("total")
check("it finds a word", len(hits) == 1)
check(
    "the hit reconnects to its page", hits[0].page == PAGE and hits[0].key == BASE.key()
)
check("the hit carries the file name", hits[0].filename == "bill.png")
check("the match is marked", "**total**" in hits[0].snippet, hits[0].snippet)
check("no match is empty", transcripts.search("absent") == [])
check("a blank query is empty", transcripts.search("   ") == [])
for query in ('AT&T "net 30', "total*", "a AND", "NEAR(", "col:x", "'; DROP TABLE x"):
    transcripts.search(query)
    check(f"{query!r} is not syntax", raw("SELECT count(*) FROM transcripts") == [(1,)])
check("a quoted term still matches", len(transcripts.search("AT&T")) == 1)

print("permissions")
path = fresh_db()
transcripts.put(saved())
mode = stat.S_IMODE
with closing(sqlite3.connect(path)) as held:  # keeps the WAL files after put's close
    held.execute("PRAGMA journal_mode = WAL")
    held.execute("SELECT * FROM transcripts").fetchall()
    transcripts.put(saved("while another connection is open"))
    check("the directory is 0700", mode(path.parent.stat().st_mode) == 0o700)
    for suffix in ("", "-wal", "-shm"):
        file = Path(f"{path}{suffix}")
        check(f"{file.name} exists", file.exists())
        check(f"{file.name} is 0600", mode(file.stat().st_mode) == 0o600)

existing = root / "existing-folder"
existing.mkdir(mode=0o755)
os.environ[transcripts.DB_ENV] = str(existing / "transcripts.db")
transcripts.put(saved())
check("a folder it did not make is left alone", mode(existing.stat().st_mode) == 0o755)

print("threads")
fresh_db()
errors: list[BaseException] = []


def from_thread(job) -> object:
    box = []

    def run() -> None:
        try:
            box.append(job())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return box[0] if box else None


from_thread(lambda: transcripts.put(saved()))
check(
    "put on one, get on another",
    from_thread(lambda: transcripts.get(BASE.key())) == saved(),
)
check("no thread raised", not errors, str(errors))

print("location")
del os.environ[transcripts.DB_ENV]
check(
    "the default is outside the repo",
    transcripts.db_path()
    == Path.home() / ".cache/cohere-document-intelligence/transcripts.db",
)

finish()
