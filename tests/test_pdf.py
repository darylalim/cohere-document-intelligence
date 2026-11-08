"""PDF pages render at 200 dpi under the encoder's ceiling, with stable identities.

Run:  uv run python tests/test_pdf.py
"""

import hashlib
import io
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdfium2 as pdfium  # noqa: E402
from _check import check, finish  # noqa: E402
from PIL import Image  # noqa: E402

from nmv.domain import page_id  # noqa: E402
from nmv.imaging import MAX_PIXELS, plan  # noqa: E402
from nmv.pdf import PDF_ERRORS, is_pdf, pdf_page_id, render, scale_for  # noqa: E402

A4_PT = (595.276, 841.89)


def pdf_of(*images: Image.Image, dpi: float = 200) -> bytes:
    """A PDF with one page per image, each page sized so the image fills it at dpi."""
    out = io.BytesIO()
    first, *rest = images
    first.save(out, "PDF", resolution=dpi, save_all=True, append_images=rest)
    return out.getvalue()


def near(pixel: object, rgb: tuple[int, int, int]) -> bool:
    """Pillow stores a PDF's RGB images as JPEG, so colours come back a step off."""
    assert isinstance(pixel, tuple)
    return all(abs(a - b) <= 8 for a, b in zip(pixel, rgb, strict=True))


def a4_page(colour: str = "white") -> Image.Image:
    return Image.new("RGB", (1654, 2339), colour)


print("page_id")
check(
    "without a part is the plain sha256, so saved image ids hold",
    page_id(b"abc") == hashlib.sha256(b"abc").hexdigest(),
)
check("a part changes it", page_id(b"abc", "x") != page_id(b"abc"))
check("different parts differ", page_id(b"abc", "x") != page_id(b"abc", "y"))

print("is_pdf")
check("by extension, any case", is_pdf("Report.PDF") and is_pdf("a.pdf"))
check("an image is not", not is_pdf("page.png"))

print("scale_for")
check("A4 is 200 dpi, to the pixel", abs(scale_for(*A4_PT) / (200 / 72) - 1) < 1e-3)
a0 = (2383.94, 3370.39)
check(
    "A0 stays under the ceiling",
    a0[0] * a0[1] * scale_for(*a0) ** 2 <= MAX_PIXELS + 1,
)
try:
    scale_for(0, 100)
except ValueError:
    check("a zero-size page raises ValueError", True)
else:
    check("a zero-size page raises ValueError", False)

print("render")
red = a4_page()
red.paste((255, 0, 0), (0, 0, 827, 1169))  # top-left quarter
three = pdf_of(red, a4_page("blue"), a4_page("green"))
pages, total = render(three, 10)
check("counts every page", total == 3, str(total))
check("renders every page under the limit", len(pages) == 3)
check("numbers from 1", [p.number for p in pages] == [1, 2, 3])
first = pages[0].image
check("is RGB", first.mode == "RGB", first.mode)
w, h = first.size
check("A4 at 200 dpi is about 1654 x 2339", abs(w - 1654) <= 2 and abs(h - 2339) <= 2)
check("fits the encoder's ceiling", w * h <= MAX_PIXELS, f"{w * h:,}")
check("keeps orientation: red is top left", near(first.getpixel((100, 100)), (255, 0, 0)))
check("and only there", near(first.getpixel((w - 100, h - 100)), (255, 255, 255)))
check("pages stay in order", near(pages[1].image.getpixel((10, 10)), (0, 0, 255)))
plan(w, h, MAX_PIXELS)  # raises if the encoder could not take it

limited, total = render(three, 2)
check("renders only up to the limit", len(limited) == 2 and total == 3)

print("identity")
ids = [p.id for p in pages]
check("each page has its own id", len(set(ids)) == 3)
check("stable across renders", ids == [p.id for p in render(three, 10)[0]])
check("is pdf_page_id", ids[0] == pdf_page_id(three, 1))
check("differs from the file's own hash", page_id(three) not in ids)
other = pdf_of(red, a4_page("blue"))
check(
    "the same page in another file is another page", render(other, 1)[0][0].id != ids[0]
)

print("large and odd pages")
poster = pdf_of(Image.new("RGB", (4000, 6000), "white"), dpi=50)  # 80 x 120 in
big, _ = render(poster, 1)
bw, bh = big[0].image.size
check("a poster renders under the ceiling", bw * bh <= MAX_PIXELS, f"{bw}x{bh}")

print("scans render at their own density")
scan = Image.new("RGB", (1240, 1754), "white")  # A4 scanned at 150 dpi
scan_pages, _ = render(pdf_of(scan, dpi=150), 1)
check(
    "a 150 dpi scan is not upsampled to 200 dpi",
    scan_pages[0].image.size == (1240, 1754),
    str(scan_pages[0].image.size),
)
vector = pdfium.PdfDocument.new()
vector.new_page(*A4_PT)
blank = io.BytesIO()
vector.save(blank)
vector.close()
vw, vh = render(blank.getvalue(), 1)[0][0].image.size
check("a page with no image renders at 200 dpi", abs(vw - 1653) <= 2, f"{vw}x{vh}")
small = Image.new("RGB", (400, 300), "white")
framed = Image.new("RGB", (1654, 2339), "white")
framed.paste(small, (100, 100))
check(
    "an image that only covers part of the page does not count as a scan",
    render(pdf_of(framed), 1)[0][0].image.size[0] >= 1650,
)

print("errors")
for name, data in (
    ("garbage", b"not a pdf"),
    ("empty", b""),
    ("truncated", three[: len(three) // 3]),
    ("an image", (lambda b: (red.save(b, "PNG"), b.getvalue())[1])(io.BytesIO())),
):
    try:
        pages_, _ = render(data, 10)
    except PDF_ERRORS:
        check(f"{name} raises a PDF error", True)
    except Exception as error:  # noqa: BLE001
        check(f"{name} raises a PDF error", False, type(error).__name__)
    else:
        # PDFium recovers some damaged files; that is fine as long as it renders.
        check(f"{name} raises a PDF error or renders", all(p.image for p in pages_))

print("threads")
results: list[list[str]] = []
errors: list[BaseException] = []


def worker() -> None:
    try:
        results.append([p.id for p in render(three, 10)[0]])
    except BaseException as error:  # noqa: BLE001
        errors.append(error)


threads = [threading.Thread(target=worker) for _ in range(6)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
check("concurrent renders all succeed", not errors, repr(errors[:1]))
check("and agree", all(r == ids for r in results) and len(results) == 6)

finish()
