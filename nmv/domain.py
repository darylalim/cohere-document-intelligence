"""The document pipeline's data model: pure types, no MLX and no Streamlit.

Three rules these types exist to enforce:

1. **Identity comes from content; position is not identity.** A page is the hash
   of its upload bytes, never its upload order or file name, and ``page_id`` is
   the only thing that hashes them, so the store, the caches and the logs agree.
2. **Every derived result records how it was made.** A result without a
   ``Derivation`` may not be cached or compared.
3. **A citation number is not a page number.** The model cites positions in the
   document list it was given, and pages can be dropped from that list, so a
   citation only means something through the ``sent_pages`` of its ``Turn``.

Boxes stay on the 0-1000 grid here; pixels are computed when drawing.

Existing types (``Box``, ``ImagePlan``, ``Sampling``, ``RunStats``) stay in their
own modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import NewType

from PIL import Image

from nmv.grounding import Box

PageId = NewType("PageId", str)


def page_id(data: bytes) -> PageId:
    """The one identity function: the sha256 of the original upload bytes."""
    return PageId(hashlib.sha256(data).hexdigest())


@dataclass(frozen=True)
class Page:
    """An uploaded page. ``image`` is None for a search hit without its upload,
    which is exactly what decides whether Locate is available."""

    id: PageId
    display_name: str  # never part of a key, never logged
    source_size: tuple[int, int]
    image: Image.Image | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Derivation:
    """How a page-derived result was made, so caches invalidate themselves.

    ``max_tokens`` belongs here because a transcript cut off at one limit is a
    different result from a complete one.
    """

    page: PageId
    resolved_px: int
    model_revision: str
    prompt_version: int
    max_tokens: int

    def key(self) -> str:
        """sha256 of the fields in the order declared above.

        A JSON array rather than a joined string, so that no two different
        field values can run together into the same input. Changing the order
        or the encoding orphans every stored transcript.
        """
        fields = [
            self.page,
            self.resolved_px,
            self.model_revision,
            self.prompt_version,
            self.max_tokens,
        ]
        return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


@dataclass(frozen=True)
class Transcript:
    derivation: Derivation
    text: str
    finish_reason: str | None
    width: int  # the resolved size NMV actually saw
    height: int

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def empty(self) -> bool:
        """An empty transcript is kept but never sent to the reader."""
        return not self.text.strip()


@dataclass(frozen=True)
class Grounding:
    derivation: Derivation
    instruction: str
    boxes: tuple[Box, ...]
    truncated: bool


@dataclass(frozen=True)
class Citation:
    """A cited span. ``doc_indices`` are positions in the list **as sent**, never
    page numbers; ``start`` and ``end`` are offsets in the clean text."""

    span: str
    doc_indices: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class Turn:
    """One message in an Ask documents conversation.

    ``history_text`` feeds the chat template and ``display_text`` feeds the page.
    They differ, so they are never interchangeable: the first is the response
    body with its citation tags, the second has been through ``safe_markdown``.
    """

    role: str
    display_text: str
    history_text: str
    citations: tuple[Citation, ...] = ()
    sent_pages: tuple[PageId, ...] = ()
    run_id: str = ""

    def page_for(self, index: int) -> PageId | None:
        """Map a citation's document index to the page that was sent there.

        Returns None rather than raising for an index the model invented,
        because this runs while the page renders. Negative indices are invented
        too; Python would otherwise count them from the end of the list.
        """
        if 0 <= index < len(self.sent_pages):
            return self.sent_pages[index]
        return None

    def message(self) -> dict[str, str]:
        """This turn as a chat-template message, built from ``history_text``."""
        return {"role": self.role, "content": self.history_text}
