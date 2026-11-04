"""Generate the synthetic eval set: every document, and its ground truth.

    uv run python evals/make_set.py [--out evals/set-v1]

The generator knows every string it draws and where, so the truth needs no hand
labelling. All content is fictional. Changing the set means creating a new
``set-vN``; this refuses to write over an existing one, so a comparison never
shifts underneath a metric.

Text uses Pillow's bundled font, so rendering does not depend on the fonts
installed on the Mac. That font is ASCII-only: accents would come out as tofu,
so ``Sheet.text`` draws the ASCII base string and adds the marks as shapes, and
amounts are written ``EUR`` rather than with a euro sign.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from itertools import cycle
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmv.imaging import normalize  # noqa: E402

A4 = (1240, 1754)
INK = (24, 24, 28)
RULE = (150, 150, 158)
STAMP_RED = (196, 30, 42)
SIGN_BLUE = (28, 42, 130)
EXIF_ORIENTATION = 0x0112

Grid = list[int]


@cache
def font_at(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def ascii_base(text: str) -> str:
    """Strip combining marks. Refuses anything the bundled font cannot draw."""
    base = "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )
    if not base.isascii():
        raise ValueError(f"no glyph for {text!r}: the bundled font is ASCII-only")
    return base


class Sheet:
    """One page. Records what it draws, in reading order, for the reference."""

    def __init__(self, size: tuple[int, int] = A4, paper=(255, 255, 255)):
        self.image = Image.new("RGB", size, paper)
        self.draw = ImageDraw.Draw(self.image)
        self.lines: list[str] = []

    def grid(self, box) -> Grid:
        """Pixels to the 0-1000 grid, per axis."""
        width, height = self.image.size
        x1, y1, x2, y2 = box
        return [
            round(x1 / width * 1000),
            round(y1 / height * 1000),
            round(x2 / width * 1000),
            round(y2 / height * 1000),
        ]

    def _marks(self, x: float, y: float, text: str, size: int, fill) -> float:
        """Draw each diacritic over its base letter, returning their top edge."""
        font = font_at(size)
        base = ascii_base(text)
        top = float("inf")
        for at, char in enumerate(unicodedata.normalize("NFC", text)):
            parts = unicodedata.normalize("NFD", char)
            letter = parts[0]
            if len(parts) > 1:
                left = x + font.getlength(base[:at])
                width = font.getlength(letter)
                box = self.draw.textbbox((x, y), letter, font=font)
                mid = left + width / 2
                peak = box[1]
                line = max(1, size // 12)
                gap, tall = size * 0.08, size * 0.2
                low, high = peak - gap, peak - gap - tall
                if "́" in parts:  # acute
                    self.draw.line(
                        [(mid - tall / 3, low), (mid + tall / 2, high)], fill, line
                    )
                elif "̀" in parts:  # grave
                    self.draw.line(
                        [(mid + tall / 3, low), (mid - tall / 2, high)], fill, line
                    )
                elif "̂" in parts:  # circumflex
                    self.draw.line(
                        [(mid - tall / 2, low), (mid, high), (mid + tall / 2, low)],
                        fill,
                        line,
                    )
                elif "̃" in parts:  # tilde
                    q = tall / 2
                    self.draw.line(
                        [
                            (mid - q * 1.2, low),
                            (mid - q / 2, high),
                            (mid + q / 2, low),
                            (mid + q * 1.2, high),
                        ],
                        fill,
                        line,
                    )
                elif "̈" in parts:  # diaeresis
                    for dx in (-tall / 2, tall / 2):
                        self.draw.ellipse(
                            [mid + dx - line, low - line * 2, mid + dx + line, low],
                            fill,
                        )
                elif "̧" in parts:  # cedilla
                    foot = box[3]
                    self.draw.line(
                        [
                            (mid, foot),
                            (mid + tall / 3, foot + tall),
                            (mid - tall / 3, foot + tall),
                        ],
                        fill,
                        line,
                    )
                else:
                    raise ValueError(f"no mark for {char!r}")
                top = min(top, high)
        return top

    def _place(self, x: float, y: float, text: str, size: int, fill) -> tuple:
        font = font_at(size)
        base = ascii_base(text)
        self.draw.text((x, y), base, font=font, fill=fill)
        box = self.draw.textbbox((x, y), base, font=font)
        if base != unicodedata.normalize("NFC", text):
            top = self._marks(x, y, text, size, fill)
            box = (box[0], min(box[1], top), box[2], box[3])
        return box

    def text(self, x: float, y: float, text: str, size: int = 22, fill=INK) -> Grid:
        """Draw one line, record it, and return its box on the grid."""
        box = self._place(x, y, text, size, fill)
        self.lines.append(text)
        return self.grid(box)

    def right(self, x: float, y: float, text: str, size: int = 22, fill=INK) -> Grid:
        """Like ``text``, with ``x`` as the right edge."""
        return self.text(
            x - font_at(size).getlength(ascii_base(text)), y, text, size, fill
        )

    def row(self, y: float, cells: list[tuple[float, str]], size: int = 20) -> list[Grid]:
        """A table row: one reference line, one box per cell. A negative x
        right-aligns the cell to ``-x``."""
        boxes = []
        for x, text in cells:
            width = font_at(size).getlength(ascii_base(text))
            left = -x - width if x < 0 else x
            boxes.append(self.grid(self._place(left, y, text, size, INK)))
        self.lines.append(" ".join(text for _, text in cells))
        return boxes

    def rule(self, x1: float, y: float, x2: float) -> None:
        self.draw.line([(x1, y), (x2, y)], RULE, 2)

    def wrap(
        self, x: float, y: float, width: float, text: str, size: int = 20, leading=1.4
    ) -> float:
        """Draw a paragraph, returning the y below it."""
        font = font_at(size)
        line = ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if font.getlength(ascii_base(trial)) > width and line:
                self.text(x, y, line, size)
                y += size * leading
                line = word
            else:
                line = trial
        if line:
            self.text(x, y, line, size)
            y += size * leading
        return y

    def stamp(self, x: int, y: int, word: str, angle: float, size: int = 64) -> Grid:
        """A boxed, slightly rotated rubber stamp. The truth is its alpha bounds."""
        font = font_at(size)
        base = ascii_base(word)
        width = int(font.getlength(base)) + size
        tile = Image.new("RGBA", (width, size * 2), (0, 0, 0, 0))
        pen = ImageDraw.Draw(tile)
        pen.rectangle(
            [4, 4, width - 5, size * 2 - 5], outline=STAMP_RED + (255,), width=6
        )
        pen.text((size // 2, size // 3), base, font=font, fill=STAMP_RED + (255,))
        turned = tile.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        self.image.paste(turned, (x, y), turned)
        bounds = turned.getchannel("A").getbbox()
        assert bounds is not None
        self.lines.append(word)
        return self.grid((x + bounds[0], y + bounds[1], x + bounds[2], y + bounds[3]))

    def signature(self, x: float, y: float, width: int, height: int, rng) -> Grid:
        """A seeded scribble. The truth is the bounds of its points."""
        points = [(x, y + height * 0.6)]
        for step in range(1, 15):
            points.append(
                (
                    x + width * step / 14,
                    y + height * (rng.random() if step % 2 else rng.random() * 0.5 + 0.5),
                )
            )
        self.draw.line(points, SIGN_BLUE, 3, joint="curve")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return self.grid((min(xs), min(ys), max(xs), max(ys)))

    def checkbox(self, x: int, y: int, ticked: bool, side: int = 24) -> Grid:
        self.draw.rectangle([x, y, x + side, y + side], outline=INK, width=2)
        if ticked:
            self.draw.line([(x + 4, y + 4), (x + side - 4, y + side - 4)], INK, 3)
            self.draw.line([(x + side - 4, y + 4), (x + 4, y + side - 4)], INK, 3)
        return self.grid((x, y, x + side, y + side))


@dataclass
class Doc:
    id: str
    sheets: list[Sheet]
    boxes: list[dict] = field(default_factory=list)
    fields: dict[int, list[str]] = field(default_factory=dict)
    qa: list[dict] = field(default_factory=list)
    # Applied to each page's pixels after drawing, and the orientation tag to save.
    finish: Callable[[Image.Image], Image.Image] | None = None
    exif_orientation: int | None = None

    def target(self, what: str, bbox: Grid, page: int = 0) -> None:
        self.boxes.append({"page": page, "target": what, "bbox": bbox})

    def ask(self, q: str, answer: str, kind: str, bbox: Grid, page: int = 0) -> None:
        self.qa.append(
            {
                "q": q,
                "answer": answer,
                "kind": kind,
                "pages": [page],
                "answer_bbox": {"page": page, "bbox": bbox},
            }
        )

    def unanswerable(self, q: str) -> None:
        self.qa.append({"q": q, "answer": None})

    def truth(self) -> dict:
        return {
            "id": self.id,
            "pages": [f"page-{i + 1}.png" for i in range(len(self.sheets))],
            "boxes": self.boxes,
            "text": [
                {
                    "page": i,
                    "reference": "\n".join(sheet.lines),
                    "fields": self.fields.get(i, []),
                }
                for i, sheet in enumerate(self.sheets)
            ],
            "qa": self.qa,
        }


# --- Invoices -----------------------------------------------------------------

EN = {
    "title": "INVOICE",
    "number": "Invoice no.",
    "date": "Date",
    "bill": "Bill to",
    "item": "Description",
    "qty": "Qty",
    "unit": "Unit price",
    "amount": "Amount",
    "subtotal": "Subtotal",
    "tax": "Tax",
    "total": "Total due",
    "stamp": "PAID",
    "month": ("January", "March", "June", "September", "November"),
}
FR = {
    "title": "FACTURE",
    "number": "Facture n.",
    "date": "Date",
    "bill": "Facturé à",
    "item": "Désignation",
    "qty": "Qté",
    "unit": "Prix unitaire",
    "amount": "Montant",
    "subtotal": "Sous-total",
    "tax": "TVA",
    "total": "Total à payer",
    "stamp": "PAYE",
    "month": ("janvier", "mars", "juin", "septembre", "novembre"),
}
ES = {
    "title": "FACTURA",
    "number": "Factura n.",
    "date": "Fecha",
    "bill": "Facturar a",
    "item": "Descripción",
    "qty": "Cant.",
    "unit": "Precio unitario",
    "amount": "Importe",
    "subtotal": "Subtotal",
    "tax": "IVA",
    "total": "Total a pagar",
    "stamp": "PAGADO",
    "month": ("enero", "marzo", "junio", "septiembre", "noviembre"),
}

SELLERS = (
    "Brackenfold Freight Ltd",
    "Quillmere Print Works",
    "Tarnwick Marine Supply",
    "Osprey Lantern Studio",
    "Vellacott Tool & Die",
    "Ferncombe Office Goods",
    "Dunmarrow Cold Storage",
    "Hollowbrook Instruments",
    "Atelier Beaumarais",
    "Talleres Montecalvo",
)
BUYERS = (
    "Corvane Holdings",
    "Pellworth Bakery",
    "Linden & Ashgrove",
    "Marrowgate Cycles",
    "Sundial Row Clinic",
    "Thistlebank Farm",
    "Northwick Glassworks",
    "Elmhurst Tutors",
    "Maison Ravenel",
    "Hostal Peñaverde",
)
GOODS_EN = (
    "Pallet wrap, 500 mm",
    "Courier service, zone 3",
    "Stainless fittings, box",
    "Label roll, thermal",
    "Inspection visit",
    "Replacement filter set",
    "Packing crates, large",
    "Calibration certificate",
)
GOODS_FR = (
    "Rouleau d'étiquettes",
    "Livraison express",
    "Caisses en pin",
    "Entretien préventif",
    "Câble électrique, 20 m",
    "Forfait déplacement",
)
GOODS_ES = (
    "Rollo de etiquetas",
    "Envío urgente",
    "Cajas de cartón",
    "Revisión técnica",
    "Tubería de cobre",
    "Mano de obra, jornada",
)


def money(cents: int, style: str = "en") -> str:
    whole, part = divmod(cents, 100)
    digits = f"{whole:,}"
    if style == "fr":
        return f"{digits.replace(',', ' ')},{part:02d} EUR"
    if style == "es":
        return f"{digits.replace(',', '.')},{part:02d} EUR"
    return f"${digits}.{part:02d}"


@dataclass
class Layout:
    """Where the moving parts of an invoice go."""

    header_right: bool
    table_y: int
    total_top: bool  # the total sits in a summary box above the table
    stamp_at: tuple[int, int]
    stamp_angle: float


LAYOUTS = (
    Layout(False, 520, False, (820, 1180), 12),
    Layout(True, 560, False, (120, 1240), -9),
    Layout(False, 640, True, (760, 300), 8),
    Layout(True, 640, True, (130, 1420), 14),
    Layout(False, 470, False, (500, 1460), -6),
)


def invoice(
    doc_id: str,
    seed: int,
    layout: Layout,
    labels: dict = EN,
    goods: tuple = GOODS_EN,
    style: str = "en",
    paper=(255, 255, 255),
) -> Doc:
    rng = random.Random(seed)
    sheet = Sheet(paper=paper)
    doc = Doc(doc_id, [sheet])
    seller = SELLERS[seed % len(SELLERS)]
    buyer = BUYERS[seed % len(BUYERS)]
    number = f"{rng.randint(1000, 9899)}-{rng.randint(10, 99)}"
    day = rng.randint(3, 27)
    date = f"{day} {rng.choice(labels['month'])} 2026"

    left = 90
    head_x = 760 if layout.header_right else left
    meta_x = left if layout.header_right else 760
    seller_box = sheet.text(head_x, 80, seller, 34)
    sheet.text(head_x, 130, f"{rng.randint(2, 98)} Harrow Quay, Port Selwyn", 20)
    sheet.text(meta_x, 80, labels["title"], 40)
    sheet.text(meta_x, 140, f"{labels['number']} {number}", 22)
    date_box = sheet.text(meta_x, 174, f"{labels['date']}: {date}", 22)
    sheet.text(left, 280, f"{labels['bill']}:", 20)
    buyer_box = sheet.text(left, 312, buyer, 26)

    rows = rng.sample(goods, k=rng.randint(4, 6))
    items = [(name, rng.randint(1, 12), rng.randint(1500, 48000)) for name in rows]
    subtotal = sum(qty * price for _, qty, price in items)
    tax = subtotal // 5
    total = subtotal + tax
    total_line = f"{labels['total']}: {money(total, style)}"

    if layout.total_top:
        total_box = sheet.text(left, 420, total_line, 30)

    y = layout.table_y
    sheet.row(
        y,
        [
            (left, labels["item"]),
            (640, labels["qty"]),
            (-930, labels["unit"]),
            (-1150, labels["amount"]),
        ],
    )
    sheet.rule(left, y + 34, 1150)
    y += 50
    for name, qty, price in items:
        sheet.row(
            y,
            [
                (left, name),
                (640, str(qty)),
                (-930, money(price, style)),
                (-1150, money(qty * price, style)),
            ],
        )
        y += 42
    sheet.rule(left, y + 4, 1150)
    y += 24
    sheet.right(1150, y, f"{labels['subtotal']}: {money(subtotal, style)}", 22)
    y += 36
    tax_box = sheet.right(1150, y, f"{labels['tax']}: {money(tax, style)}", 22)
    y += 44
    if not layout.total_top:
        total_box = sheet.right(1150, y, total_line, 30)

    stamp_box = sheet.stamp(*layout.stamp_at, labels["stamp"], layout.stamp_angle)

    doc.target(total_line, total_box)
    doc.target("the stamp" if style != "en" else "the PAID stamp", stamp_box)
    doc.target(seller, seller_box)
    amount = money(total, style).removesuffix(" EUR").removeprefix("$")
    doc.fields[0] = [seller, buyer, number, amount]
    doc.ask("What is the total due?", amount, "number", total_box)
    doc.ask(f"Which company was billed on invoice {number}?", buyer, "text", buyer_box)
    doc.ask(
        "How much tax was charged?",
        money(tax, style).removesuffix(" EUR").removeprefix("$"),
        "number",
        tax_box,
    )
    doc.ask("What is the date of the invoice?", date, "date", date_box)
    doc.unanswerable("What is the purchase order number?")
    return doc


# --- Bank statements ----------------------------------------------------------

PAYEES = (
    "Pellworth Bakery",
    "Sundial Row Clinic",
    "Tarnwick Water Board",
    "Rent, Flat 4 Moorgate",
    "Linden Grocers",
    "Port Selwyn Transit",
    "Marrowgate Cycles",
    "Ferncombe Energy",
    "Salary, Corvane Holdings",
    "Thistlebank Farm Shop",
)


def statement(doc_id: str, seed: int) -> Doc:
    rng = random.Random(seed)
    sheet = Sheet()
    doc = Doc(doc_id, [sheet])
    account = f"{rng.randint(10, 99)}-{rng.randint(1000, 9999)}-{rng.randint(100, 999)}"
    holder = ("Maren Oskalde", "Tobin Vreeland")[seed % 2]
    sheet.text(90, 80, "Wrenfield Mutual Savings", 34)
    sheet.text(90, 134, "Current account statement, April 2026", 22)
    holder_box = sheet.text(90, 200, f"Account holder: {holder}", 22)
    sheet.text(90, 234, f"Account number: {account}", 22)

    balance = rng.randint(1200000, 1900000)
    opening = balance
    opening_box = sheet.text(90, 290, f"Opening balance: {money(opening)}", 22)

    y = 360
    columns = (90, 230, -830, -990, -1150)
    sheet.row(
        y,
        list(
            zip(
                columns,
                ("Date", "Details", "Paid out", "Paid in", "Balance"),
                strict=True,
            )
        ),
    )
    sheet.rule(90, y + 32, 1150)
    y += 46

    payee_boxes = {}
    days = sorted(rng.sample(range(1, 30), k=len(PAYEES) + 8))
    payees = [*PAYEES, *rng.choices(PAYEES[:8], k=8)]
    rng.shuffle(payees)
    for day, payee in zip(days, payees, strict=True):
        pay_in = payee.startswith("Salary")
        amount = rng.randint(310000, 390000) if pay_in else rng.randint(450, 92000)
        balance += amount if pay_in else -amount
        cells = (
            f"{day:02d} Apr",
            payee,
            "" if pay_in else money(amount),
            money(amount) if pay_in else "",
            money(balance),
        )
        boxes = sheet.row(y, list(zip(columns, cells, strict=True)))
        payee_boxes.setdefault(payee, (amount, boxes, cells))
        y += 40

    assert balance > 0, "money() has no negative form"
    sheet.rule(90, y + 2, 1150)
    closing_line = f"Closing balance: {money(balance)}"
    closing_box = sheet.right(1150, y + 20, closing_line, 26)

    name = "Sundial Row Clinic"
    amount, boxes, cells = payee_boxes[name]
    doc.target(closing_line, closing_box)
    doc.target(name, boxes[1])
    doc.fields[0] = [holder, account, money(opening), money(balance)]
    doc.ask("What is the closing balance?", money(balance)[1:], "number", closing_box)
    doc.ask("What is the opening balance?", money(opening)[1:], "number", opening_box)
    doc.ask(
        f"How much was paid to {name} the first time it appears?",
        money(amount)[1:],
        "number",
        boxes[2],
    )
    doc.ask("Who holds this account?", holder, "text", holder_box)
    doc.unanswerable("What is the overdraft limit on this account?")
    return doc


# --- Contracts ----------------------------------------------------------------

CLAUSES = (
    (
        "Standard of service",
        "The Supplier shall carry out the Services with reasonable care and skill and "
        "in accordance with the Schedule, and shall keep the Client informed of any "
        "matter likely to delay the Services by more than five working days.",
    ),
    (
        "Confidentiality",
        "Each party shall keep confidential all information of a commercial nature "
        "which it receives from the other, and shall use that information only for "
        "the purpose of performing its obligations under this Agreement.",
    ),
    (
        "Events beyond control",
        "Neither party shall be liable for any failure to perform caused by events "
        "beyond its reasonable control, provided that it gives prompt written notice "
        "and takes all reasonable steps to resume performance.",
    ),
    (
        "Intellectual property",
        "All intellectual property created by the Supplier in the course of the "
        "Services shall remain with the Supplier, and the Client is granted a "
        "non-exclusive licence to use it for its internal business purposes.",
    ),
    (
        "Access and co-operation",
        "The Client shall provide access to its premises and records at such times as "
        "the Supplier may reasonably request, and shall ensure that its staff give "
        "all co-operation needed for the Services to proceed on schedule.",
    ),
    (
        "Variation",
        "No variation of this Agreement shall be effective unless it is made in "
        "writing and signed by an authorised representative of each party, and no "
        "failure to exercise a right shall operate as a waiver of that right.",
    ),
    (
        "Notices",
        "Any notice under this Agreement shall be delivered by hand or sent by "
        "recorded post to the address stated above, and shall be deemed received on "
        "the second working day after posting.",
    ),
    (
        "Insurance",
        "The Supplier shall maintain adequate insurance against its liabilities under "
        "this Agreement and shall produce evidence of that insurance on request, not "
        "more than once in any period of twelve months.",
    ),
)
OFFICES = ("Port Selwyn", "Tarnwick", "Harrowfield", "Moorgate", "Saltmarsh", "Linden")


def particulars(rng: random.Random, number: int) -> str:
    """A sentence unique to one clause, so no page repeats itself: looping on
    duplicated text would measure the decoder rather than the reading."""
    return (
        f"For the purposes of clause {number} the reference period is "
        f"{rng.randint(2, 40)} weeks, the responsible office is "
        f"{rng.choice(OFFICES)}, and the file reference is "
        f"{rng.choice('ABCDEFGH')}{rng.randint(100, 999)}/{rng.randint(10, 99)}."
    )


def contract(doc_id: str, seed: int, pages: int) -> Doc:
    rng = random.Random(seed)
    client = BUYERS[seed % 8]
    supplier = SELLERS[seed % 8]
    fee = rng.randint(180000, 640000)
    notice = rng.choice((30, 45, 60, 90))
    term = rng.choice((12, 18, 24, 36))
    sheets = [Sheet() for _ in range(pages)]
    doc = Doc(doc_id, sheets)

    first = sheets[0]
    title_box = first.text(90, 80, "SERVICES AGREEMENT", 36)
    y = first.wrap(
        90,
        150,
        1060,
        f"This Agreement is made between {client} (the Client) and {supplier} (the "
        "Supplier), and takes effect on 1 February 2026.",
        20,
    )
    y += 20
    fee_line = f"Monthly fee: {money(fee)}"
    term_line = f"Initial term: {term} months"
    fee_box = first.text(90, y, fee_line, 20)
    term_box = first.text(90, y + 32, term_line, 20)
    y += 90

    clauses = rng.sample(CLAUSES, k=len(CLAUSES))
    numbered = enumerate(cycle(clauses), 1)
    last = sheets[-1]
    for index, sheet in enumerate(sheets):
        bottom = 1000 if sheet is last else 1640
        if index:
            y = 90
        while y < bottom - 150:
            number, (heading, clause) = next(numbered)
            sheet.text(90, y, f"{number}. {heading}", 20)
            body = f"{clause} {particulars(rng, number)}"
            y = sheet.wrap(90, y + 30, 1060, body, 18) + 16

    y = 1020
    termination = last.text(90, y, "Termination", 22)
    notice_line = f"Either party may terminate on {notice} days written notice."
    notice_box = last.text(90, y + 36, notice_line, 18)
    y += 120
    last.text(90, y, f"Signed for {supplier}:", 20)
    signature = last.signature(90, y + 40, 300, 80, rng)
    signer_box = last.text(90, y + 140, "Name: Ilse Warrender, Director", 18)

    page = pages - 1
    doc.target("the handwritten signature", signature, page)
    doc.target("Termination", termination, page)
    doc.target("SERVICES AGREEMENT", title_box)
    doc.fields[0] = [client, supplier, money(fee)]
    doc.fields[page] = [notice_line]
    doc.ask("What is the monthly fee?", money(fee)[1:], "number", fee_box)
    doc.ask(
        "How many days of notice does termination require?",
        str(notice),
        "number",
        notice_box,
        page,
    )
    doc.ask("How long is the initial term, in months?", str(term), "number", term_box)
    doc.ask("Who signed for the Supplier?", "Ilse Warrender", "text", signer_box, page)
    doc.unanswerable("Which court has jurisdiction over disputes?")
    return doc


# --- Receipts -----------------------------------------------------------------

SHOP_ITEMS = (
    "RYE LOAF",
    "OAT MILK 1L",
    "PEARS 1KG",
    "COFFEE BEANS",
    "DISH SOAP",
    "TINNED TOMS",
    "BATTERIES AA",
    "SEA SALT",
    "HONEY 340G",
    "NOTEBOOK A5",
)


def receipt(doc_id: str, seed: int) -> Doc:
    rng = random.Random(seed)
    sheet = Sheet((576, 1400), (250, 249, 244))
    doc = Doc(doc_id, [sheet])
    shop = ("Larkspur Corner Stores", "The Gantry Pantry")[seed % 2]
    shop_box = sheet.text(40, 40, shop, 28)
    sheet.text(40, 84, "14 Saltmarsh Lane", 16)
    time_line = f"12/05/2026 {rng.randint(9, 19)}:{rng.randint(10, 59)}"
    sheet.text(40, 108, time_line, 16)
    sheet.rule(40, 140, 536)

    y = 156
    total = 0
    picks = rng.sample(SHOP_ITEMS, k=8)
    first_price = 0
    first_box: Grid = []
    for name in picks:
        price = rng.randint(95, 1450)
        total += price
        boxes = sheet.row(y, [(40, name), (-536, money(price))], 17)
        if not first_box:
            first_price, first_box = price, boxes[1]
        y += 30
    sheet.rule(40, y + 4, 536)
    total_line = f"TOTAL {money(total)}"
    total_box = sheet.right(536, y + 18, total_line, 26)
    card = rng.randint(1000, 9999)
    card_line = f"CARD ENDING {card}"
    card_box = sheet.text(40, y + 66, card_line, 16)
    sheet.text(40, y + 92, "THANK YOU, CALL AGAIN", 16)

    doc.target(total_line, total_box)
    doc.target(card_line, card_box)
    doc.fields[0] = [shop, money(total), str(card)]
    doc.ask("What was the total paid?", money(total)[1:], "number", total_box)
    doc.ask("Which shop issued this receipt?", shop, "text", shop_box)
    doc.ask("What are the last four digits of the card?", str(card), "number", card_box)
    doc.ask(f"How much was {picks[0]}?", money(first_price)[1:], "number", first_box)
    doc.unanswerable("What is the name of the cashier?")
    return doc


# --- Forms --------------------------------------------------------------------


def form(doc_id: str, seed: int) -> Doc:
    rng = random.Random(seed)
    sheet = Sheet()
    doc = Doc(doc_id, [sheet])
    person = ("Anneke Dorrow", "Caspian Hult")[seed % 2]
    town = ("Port Selwyn", "Tarnwick")[seed % 2]
    sheet.text(90, 80, "Allotment Membership Application", 34)
    sheet.text(90, 134, "Harrowfield Growers Association", 22)
    name_box = sheet.text(90, 220, f"Full name: {person}", 24)
    town_box = sheet.text(90, 270, f"Town: {town}", 24)
    plot = rng.randint(11, 94)
    plot_box = sheet.text(90, 320, f"Preferred plot number: {plot}", 24)

    sheet.text(90, 410, "How should we contact you?", 24)
    ticked = rng.choice(("Email", "Post", "Telephone"))
    tick_box: Grid = []
    y = 460
    for choice in ("Email", "Post", "Telephone"):
        box = sheet.checkbox(110, y, choice == ticked)
        label = sheet.text(150, y, choice, 22)
        if choice == ticked:
            tick_box = box
            tick_label = label
        y += 46

    sheet.text(90, 640, "Do you need a shed?", 24)
    shed = rng.random() < 0.5
    y = 690
    for choice, value in (("Yes", True), ("No", False)):
        sheet.checkbox(110, y, value == shed)
        sheet.text(150, y, choice, 22)
        y += 46

    y = sheet.wrap(
        90,
        830,
        1060,
        "I agree to keep my plot cultivated and to follow the rules of the Association, "
        "a copy of which I have read.",
        20,
    )
    sheet.text(90, y + 30, "Signature:", 22)
    signature = sheet.signature(230, y + 10, 340, 90, rng)

    doc.target("the handwritten signature", signature)
    doc.target(f'the ticked box next to "{ticked}"', tick_box)
    doc.target(f"Full name: {person}", name_box)
    doc.fields[0] = [person, town, str(plot)]
    doc.ask("What is the applicant's full name?", person, "text", name_box)
    doc.ask("Which plot number is preferred?", str(plot), "number", plot_box)
    doc.ask("Which contact method is ticked?", ticked, "text", tick_label)
    doc.ask("Which town does the applicant live in?", town, "text", town_box)
    doc.unanswerable("What is the applicant's date of birth?")
    return doc


# --- Two-column articles ------------------------------------------------------

ARTICLE = (
    "The harbour commission met on Tuesday to weigh the future of the old grain "
    "pier, which has stood idle since the last barge left. Members heard from "
    "engineers, fishermen and a delegation of residents from the terraces above "
    "the quay, most of whom wanted the timber kept.",
    "Repairs would take the better part of two summers. The surveyor told the "
    "meeting that the piles nearest the channel were sound, but that the deck had "
    "softened along much of its length and could not safely carry a vehicle.",
    "Traders along the waterfront were divided. Some saw a market hall on the pier "
    "as the only way to bring visitors down the hill, while others feared that a "
    "long closure of the quay road would cost them a season of custom.",
    "A smaller group argued for doing nothing at all. The tide, they said, would "
    "settle the question eventually, and the money would be better spent on the "
    "lifeboat station, whose slipway has needed attention for some years.",
    "The commission will publish its findings after a further round of soundings, "
    "and has promised a public meeting before any contract is signed. Until then "
    "the gates on the pier will stay locked and the warning notices will remain.",
)


ARTICLE_REPLY = (
    "Letters to this paper have run strongly in favour of keeping the pier, though "
    "few of the writers offer to pay for it. One retired pilot recalled tying up "
    "alongside it in fog, guided only by the smell of the malt sheds.",
    "The quay has changed hands four times since the war, and each owner has left "
    "behind a plan that came to nothing. A folder of drawings for a ferry terminal "
    "still sits in the commission's basement, its corners nibbled by mice.",
    "What the town lacks is not ideas but a season long enough to pay for them. "
    "The hotels on the front open at Easter and shutter by the first gales, and "
    "their owners are wary of promises about year-round trade.",
    "Younger residents see it differently. A petition from the sixth-form college "
    "asks for a rowing club and a small boatyard, arguing that a working pier would "
    "keep skills in the town that a market stall never could.",
    "Whatever is decided, the engineers agree on one thing: the longer the timber "
    "sits untreated, the less there will be to argue over. Each winter takes a "
    "little more of the outer deck, and the gulls have already moved in.",
)


def article(doc_id: str, seed: int) -> Doc:
    rng = random.Random(seed)
    sheet = Sheet()
    doc = Doc(doc_id, [sheet])
    author = ("Rosalind Tiernan", "Emeric Halloway")[seed % 2]
    cost = rng.randint(3, 9) * 100 + rng.choice((25, 50, 75))
    title = ("Grain Pier Faces a Long Wait", "Commission Weighs the Old Pier")[seed % 2]
    title_box = sheet.text(90, 80, title, 44)
    author_box = sheet.text(90, 150, f"By {author}", 22)
    sheet.rule(90, 190, 1150)

    width = 500
    y = 215
    columns = (ARTICLE, ARTICLE_REPLY) if seed % 2 else (ARTICLE_REPLY, ARTICLE)
    for paragraph in rng.sample(columns[0], k=len(columns[0])):
        y = sheet.wrap(90, y, width, paragraph, 20) + 16
    cost_line = f"Estimated cost: {cost},000 pounds"
    cost_box = sheet.text(90, y + 10, cost_line, 22)
    y = 215
    heading = sheet.text(650, y, "Opinion on the quay", 26)
    y += 46
    for paragraph in rng.sample(columns[1], k=len(columns[1])):
        y = sheet.wrap(650, y, width, paragraph, 20) + 16

    doc.target(title, title_box)
    doc.target("Opinion on the quay", heading)
    doc.fields[0] = [title, author, cost_line]
    doc.ask("Who wrote this article?", author, "text", author_box)
    doc.ask(
        "What is the estimated cost, in thousands of pounds?",
        str(cost),
        "number",
        cost_box,
    )
    doc.ask("What is the headline?", title, "text", title_box)
    doc.ask(
        "What is the second column's heading?", "Opinion on the quay", "text", heading
    )
    doc.unanswerable("On what date will the public meeting be held?")
    return doc


# --- Degraded copies ----------------------------------------------------------


def photocopy(image: Image.Image) -> Image.Image:
    """Lower the contrast onto a grey ground, as a tired copier would."""
    return (
        ImageEnhance.Contrast(image).enhance(0.55).filter(ImageFilter.GaussianBlur(0.5))
    )


def phone_photo(image: Image.Image) -> Image.Image:
    """Soften it, then store it sideways: orientation 6 means the viewer turns
    it 90 degrees clockwise, so the file holds the page turned the other way."""
    soft = (
        ImageEnhance.Brightness(image).enhance(0.93).filter(ImageFilter.GaussianBlur(0.7))
    )
    return soft.transpose(Image.Transpose.ROTATE_90)


def build() -> list[Doc]:
    docs = [invoice(f"invoice-{i + 1:02d}", i, LAYOUTS[i]) for i in range(5)]
    docs += [statement("statement-01", 21), statement("statement-02", 22)]
    docs += [contract("contract-01", 31, 2), contract("contract-02", 32, 3)]
    docs += [receipt("receipt-01", 41), receipt("receipt-02", 42)]
    docs += [form("form-01", 51), form("form-02", 52)]
    docs += [article("article-01", 61), article("article-02", 62)]

    for i, seed in enumerate((6, 7)):
        doc = invoice(
            f"photocopy-{i + 1:02d}", seed, LAYOUTS[i + 1], paper=(226, 226, 222)
        )
        doc.finish = photocopy
        docs.append(doc)
    for i, seed in enumerate((3, 4)):
        doc = invoice(
            f"photo-{i + 1:02d}", seed + 70, LAYOUTS[i + 2], paper=(246, 243, 236)
        )
        doc.finish = phone_photo
        doc.exif_orientation = 6
        docs.append(doc)
    docs.append(invoice("invoice-fr", 8, LAYOUTS[0], FR, GOODS_FR, "fr"))
    docs.append(invoice("invoice-es", 9, LAYOUTS[1], ES, GOODS_ES, "es"))

    for doc in docs:
        check_doc(doc)
    return docs


def check_doc(doc: Doc) -> None:
    """Every document has targets, and one question in five is unanswerable."""
    answerable = [q for q in doc.qa if q["answer"] is not None]
    assert doc.boxes, doc.id
    assert len(answerable) == 4 and len(doc.qa) == 5, doc.id
    every = [b["bbox"] for b in doc.boxes]
    every += [q["answer_bbox"]["bbox"] for q in answerable]
    for x1, y1, x2, y2 in every:
        assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000, (doc.id, x1, y1, x2, y2)
    for page, wanted in doc.fields.items():
        reference = "\n".join(doc.sheets[page].lines)
        for text in wanted:
            assert text in reference, (doc.id, text)


def save(doc: Doc, root: Path) -> None:
    folder = root / doc.id
    folder.mkdir(parents=True)
    for i, sheet in enumerate(doc.sheets):
        image = doc.finish(sheet.image) if doc.finish else sheet.image
        path = folder / f"page-{i + 1}.png"
        if doc.exif_orientation is None:
            image.save(path, optimize=True)
            continue
        exif = Image.Exif()
        exif[EXIF_ORIENTATION] = doc.exif_orientation
        image.save(path, optimize=True, exif=exif)
        shown = normalize(Image.open(path))
        assert shown.size == sheet.image.size, (doc.id, shown.size)
    text = json.dumps(doc.truth(), indent=2, ensure_ascii=False)
    (folder / "truth.json").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/set-v1"))
    args = parser.parse_args()
    if args.out.exists():
        print(f"{args.out} exists: a set is never edited in place. Make a new one.")
        return 1
    docs = build()
    args.out.mkdir(parents=True)
    for doc in docs:
        save(doc, args.out)
    pages = sum(len(doc.sheets) for doc in docs)
    targets = sum(len(doc.boxes) for doc in docs)
    print(f"{len(docs)} documents, {pages} pages, {targets} grounding targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
