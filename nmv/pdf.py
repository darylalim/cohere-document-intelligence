"""Render PDF pages to images, so Ask documents can read a PDF like page images.

No MLX, no Streamlit. North Micro Vision reads pixels, so a PDF is rasterised and
each page then takes the same path as an uploaded image. Any text layer is ignored:
a transcript has to come from the page as it looks, or Locate's boxes would not
match it.

PDFium is not thread-safe, and Streamlit runs each rerun on a fresh thread, so every
call into it happens under ``_LOCK``, and no document object outlives the call that
opened it. The bundled build has no V8 and no XFA, so a PDF's JavaScript never runs.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from nmv.domain import PageId, page_id
from nmv.imaging import MAX_PIXELS

PDF_TYPES = ["pdf"]

# Pages are rendered at 200 dpi, the resolution North Micro Vision's ceiling was
# set from (an A4 page at 200 dpi is the encoder's maximum), and never above that
# ceiling's area, so a poster-sized page costs no more memory than an A4 one. The
# image budget slider then scales the rendering down like any upload.
RENDER_DPI = 200
POINTS_PER_INCH = 72

# A page is treated as a scan when one image covers at least this share of it. A
# scan holds no detail beyond its own pixels, so it is rendered at their density,
# never upsampled to 200 dpi: that costs tokens and, on the sample invoice, moved a
# Locate box onto the wrong part of the page.
SCAN_COVERAGE = 0.9

# Part of every PDF page's identity. Change it whenever a change here would render
# a page differently, so transcripts of the old rendering are not served for it.
RENDER_VERSION = 1

# What PDFium raises for a corrupt, encrypted or unsupported file.
PDF_ERRORS: tuple[type[Exception], ...] = (pdfium.PdfiumError, ValueError, OSError)

PDF_REMEDY = "Remove any password and save it again, or export its pages as PNG."

_LOCK = threading.Lock()


@dataclass(frozen=True)
class RenderedPage:
    id: PageId
    number: int  # 1-based, as a PDF viewer numbers it
    image: Image.Image


def is_pdf(name: str) -> bool:
    """By extension, as the uploader filters. A file under the wrong extension
    reaches the wrong decoder, which refuses it, so it never gets past an error."""
    return name.lower().endswith(".pdf")


def scale_for(width_pt: float, height_pt: float, native: float | None = None) -> float:
    """Pixels per point: 200 dpi, a scan's own density if lower, and never past the
    encoder's ceiling."""
    if width_pt <= 0 or height_pt <= 0:
        raise ValueError(f"Page has a zero dimension: {width_pt} x {height_pt} pt")
    at_dpi = RENDER_DPI / POINTS_PER_INCH
    at_ceiling = math.sqrt(MAX_PIXELS / (width_pt * height_pt))
    # Sizes in points are float32 in PDFium, so a 1240 px scan can measure 1240.0001
    # px wide, which pypdfium2's ceil() turns into 1241. Shaving a millionth stops it.
    scale = min(at_dpi, at_ceiling, native or at_dpi) * (1 - 1e-6)
    if math.ceil(width_pt * scale) * math.ceil(height_pt * scale) <= MAX_PIXELS:
        return scale
    # pypdfium2 sizes the bitmap with ceil(), which can add a row and a column and
    # pass the ceiling. Snapping to whole pixels first keeps it under.
    width_px = max(1, math.floor(width_pt * scale))
    height_px = max(1, math.floor(height_pt * scale))
    scale = min(width_px / width_pt, height_px / height_pt)
    while math.ceil(width_pt * scale) * math.ceil(height_pt * scale) > MAX_PIXELS:
        scale = math.nextafter(scale, 0)  # float error only: a step or two at most
    return scale


def scan_density(page: pdfium.PdfPage, width_pt: float, height_pt: float) -> float | None:
    """Pixels per point of the image that makes up a scanned page, or None.

    Only top-level objects count, and only an image covering ``SCAN_COVERAGE`` of
    the page. A scanner's invisible text layer sits beside the image, so a scan
    with OCR text still counts.
    """
    best = None
    for item in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=1):
        if not isinstance(item, pdfium.PdfImage):
            continue
        left, bottom, right, top = item.get_bounds()
        shown_w, shown_h = right - left, top - bottom
        if shown_w * shown_h < SCAN_COVERAGE * width_pt * height_pt:
            continue
        px_w, px_h = item.get_px_size()
        # A rotated image swaps which pixel edge lies along which page edge.
        if (px_w > px_h) != (shown_w > shown_h):
            px_w, px_h = px_h, px_w
        density = min(px_w / shown_w, px_h / shown_h)
        best = density if best is None else max(best, density)
    return best


def pdf_page_id(data: bytes, number: int) -> PageId:
    return page_id(data, f"pdf-page:{number}:render-v{RENDER_VERSION}")


def render(data: bytes, limit: int) -> tuple[list[RenderedPage], int]:
    """The first ``limit`` pages as RGB images, and how many pages the file has.

    Only those pages are rendered, so a long PDF costs no more than the pages used.
    """
    with _LOCK:
        document = pdfium.PdfDocument(data)
        try:
            total = len(document)
            rendered = []
            for index in range(min(limit, total)):
                page = document[index]
                try:
                    width_pt, height_pt = page.get_size()
                    native = scan_density(page, width_pt, height_pt)
                    bitmap = page.render(scale=scale_for(width_pt, height_pt, native))
                    image = bitmap.to_pil().convert("RGB")
                    bitmap.close()
                finally:
                    page.close()
                number = index + 1
                rendered.append(RenderedPage(pdf_page_id(data, number), number, image))
        finally:
            document.close()
    return rendered, total
