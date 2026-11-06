"""The two halves of the theme, held to the numbers in ``.streamlit/config.toml``.

Run:  uv run python tests/test_theme_contrast.py

This checks colours computed from the config. Streamlit's rendered colours (hover
darkening, its own tints) can differ, so this is a floor: the real-browser check in
both modes stays the final word (``CLAUDE.md``). When a check fails, the reason beside
it is the config comment the limit comes from. Read that before moving the limit.
"""

import math
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from nmv.grounding import PALETTE  # noqa: E402

RGB = tuple[float, float, float]
WHITE: RGB = (255.0, 255.0, 255.0)

CAPTION_ALPHA = 0.6
PILL_TINT_ALPHA = 0.1
FOCUS_RING_ALPHA = 0.5
DARK_CALLOUT_ALPHA = 0.2

TEXT_FLOOR = 4.5
UI_FLOOR = 3.0
DARK_LABEL_FLOOR = 3.0
DARK_GROUND_MAX_L = 0.19
LIGHT_CHROME_MAX_L = 0.942
ACCENT_HUE_DRIFT = 0.5
GROUND_HUE_DRIFT = 2.0
SIDEBAR_STEP_DRIFT = 0.02
FOCUS_RING_DRIFT = 0.1
PALETTE_DISTANCE = 0.1


def rgb(value: str) -> RGB:
    digits = value.removeprefix("#")
    return (
        float(int(digits[0:2], 16)),
        float(int(digits[2:4], 16)),
        float(int(digits[4:6], 16)),
    )


def linear(channel: float) -> float:
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(colour: RGB) -> float:
    red, green, blue = (linear(channel) for channel in colour)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(a: RGB, b: RGB) -> float:
    """The WCAG 2.x ratio."""
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def mix(colour: RGB, alpha: float, ground: RGB) -> RGB:
    """``colour`` at ``alpha`` over ``ground``, as the browser blends sRGB."""
    red, green, blue = (
        alpha * top + (1 - alpha) * under
        for top, under in zip(colour, ground, strict=True)
    )
    return (red, green, blue)


def oklab(colour: RGB) -> RGB:
    """Ottosson's published matrices."""
    red, green, blue = (linear(channel) for channel in colour)
    long = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (long, medium, short))
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def lightness(colour: RGB) -> float:
    return oklab(colour)[0]


def hue(colour: RGB) -> float:
    _, a, b = oklab(colour)
    return math.degrees(math.atan2(b, a)) % 360


def hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def distance(a: RGB, b: RGB) -> float:
    return math.dist(oklab(a), oklab(b))


def at_least(name: str, value: float, floor: float, why: str) -> None:
    check(name, value >= floor, f"{value:.3f}, floor {floor}: {why}")


def at_most(name: str, value: float, ceiling: float, why: str) -> None:
    check(name, value <= ceiling, f"{value:.3f}, ceiling {ceiling}: {why}")


print("the maths")
check("black on white is 21", round(contrast((0, 0, 0), WHITE), 2) == 21.0)
check("a colour on itself is 1", contrast(WHITE, WHITE) == 1.0)
check("white is L 1", round(lightness(WHITE), 3) == 1.0)
check("black is L 0", round(lightness((0, 0, 0)), 3) == 0.0)
check("a mix lands halfway", mix(WHITE, 0.5, (0, 0, 0)) == (127.5, 127.5, 127.5))
check("hues wrap", hue_gap(359.0, 1.0) == 2.0)

config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
theme = config["theme"]
dark, light = theme["dark"], theme["light"]
modes = {"dark": dark, "light": light}


def callout(mode: dict, name: str) -> tuple[RGB, RGB]:
    """A callout's text and its tint: explicit if set, else Streamlit's derived one."""
    ground = rgb(mode["backgroundColor"])
    tint = mode.get(f"{name}BackgroundColor")
    derived = mix(rgb(mode[f"{name}Color"]), DARK_CALLOUT_ALPHA, ground)
    return rgb(mode[f"{name}TextColor"]), rgb(tint) if tint else derived


def focus_ring(mode: dict) -> float:
    ground = rgb(mode["backgroundColor"])
    return contrast(mix(rgb(mode["primaryColor"]), FOCUS_RING_ALPHA, ground), ground)


def sidebar_step(mode: dict) -> float:
    return contrast(rgb(mode["backgroundColor"]), rgb(mode["sidebar"]["backgroundColor"]))


for label, mode in modes.items():
    print(f"floors, {label}")
    ground = rgb(mode["backgroundColor"])
    sidebar = rgb(mode["sidebar"]["backgroundColor"])
    text = rgb(mode["textColor"])
    primary = rgb(mode["primaryColor"])

    at_least("body text", contrast(text, ground), TEXT_FLOOR, "WCAG AA")
    at_least("body text, sidebar", contrast(text, sidebar), TEXT_FLOOR, "WCAG AA")
    why = "captions are text at 60% opacity, and they set the ink"
    caption = contrast(mix(text, CAPTION_ALPHA, ground), ground)
    at_least("caption", caption, TEXT_FLOOR, why)
    caption = contrast(mix(text, CAPTION_ALPHA, sidebar), sidebar)
    at_least("caption, sidebar", caption, TEXT_FLOOR, why)
    link = contrast(rgb(mode["linkColor"]), ground)
    at_least("link", link, TEXT_FLOOR, "answers autolink OCR'd URLs")
    code = contrast(rgb(mode["codeTextColor"]), rgb(mode["codeBackgroundColor"]))
    at_least("code text", code, TEXT_FLOOR, "WCAG AA")
    for name, callout_name in (("st.info", "blue"), ("st.warning", "yellow")):
        ink, tint = callout(mode, callout_name)
        at_least(f"{name} text", contrast(ink, tint), TEXT_FLOOR, "WCAG AA")

    why = "input and uploader edges meet WCAG 1.4.11"
    at_least("border", contrast(rgb(mode["borderColor"]), ground), UI_FLOOR, why)
    why = 'the "Deterministic" toggle\'s off track, on screen by default'
    edge = rgb(mode["sidebar"]["borderColor"])
    at_least("border, sidebar", contrast(edge, sidebar), UI_FLOOR, why)

    why = "the selected task is on screen the whole time"
    pill = contrast(primary, mix(primary, PILL_TINT_ALPHA, ground))
    at_least("selected pill", pill, TEXT_FLOOR, why)

    if mode is dark:
        why = "the documented trade: dark gives up this label to keep the selected task"
        at_least(
            "white label on primary", contrast(WHITE, primary), DARK_LABEL_FLOOR, why
        )
    else:
        why = "on a light ground one dark primary passes both"
        at_least("white label on primary", contrast(WHITE, primary), TEXT_FLOOR, why)

    for index, colour in enumerate(PALETTE):
        at_least(
            f"accent against PALETTE[{index}]",
            distance(primary, (float(colour[0]), float(colour[1]), float(colour[2]))),
            PALETTE_DISTANCE,
            "the teal trap: the accent must never match a box colour",
        )

print("pairing")
why = "the halves are one system: both accents sit in PALETTE's widest hue gap"
gap = hue_gap(hue(rgb(dark["primaryColor"])), hue(rgb(light["primaryColor"])))
at_most("accent hue", gap, ACCENT_HUE_DRIFT, why)
why = "the halves are one system: neutrals sit at OKLCH hue ~266"
gap = hue_gap(hue(rgb(dark["backgroundColor"])), hue(rgb(light["backgroundColor"])))
at_most("ground hue", gap, GROUND_HUE_DRIFT, why)
why = "the sidebar steps off the ground by the same amount in both halves"
gap = abs(sidebar_step(dark) - sidebar_step(light))
at_most("sidebar step", gap, SIDEBAR_STEP_DRIFT, why)
why = "the light primary is as light as it can be while its ring matches the dark one"
at_most("focus ring", abs(focus_ring(dark) - focus_ring(light)), FOCUS_RING_DRIFT, why)

print("design rules")
why = "the dark trade needs a ground at OKLCH L 0.19 or darker"
at_most("dark ground L", lightness(rgb(dark["backgroundColor"])), DARK_GROUND_MAX_L, why)
why = "no chrome rises above L 0.942, so the document stays the brightest thing"
surfaces = {
    "ground": light["backgroundColor"],
    "secondary": light["secondaryBackgroundColor"],
    "code": light["codeBackgroundColor"],
    "sidebar": light["sidebar"]["backgroundColor"],
    "sidebar secondary": light["sidebar"]["secondaryBackgroundColor"],
    "sidebar code": light["sidebar"]["codeBackgroundColor"],
    "dataframe header": light["dataframeHeaderBackgroundColor"],
}
for name, value in surfaces.items():
    # To three places, as the config states it: fields sit exactly on the ceiling.
    at_most(f"light {name} L", round(lightness(rgb(value)), 3), LIGHT_CHROME_MAX_L, why)

finish()
