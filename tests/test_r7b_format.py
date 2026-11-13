"""Splitting R7B's stream into thinking and response.

Run:  uv run python tests/test_r7b_format.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv.r7b_format import (  # noqa: E402
    START_RESPONSE,
    Phase,
    PhaseParser,
    render,
    stream_phases,
)

NORMAL = (
    "<|START_THINKING|>I will check page 1.<|END_THINKING|>\n"
    "<|START_RESPONSE|>The total is <co>$5</co: 0:[0]>.<|END_RESPONSE|>"
)


def text_of(chunks: list[str], phase: Phase) -> str:
    return "".join(p.text for p in stream_phases(chunks) if p.phase is phase)


def split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


print("normal stream")
check("thinking", text_of([NORMAL], Phase.THINKING) == "I will check page 1.")
response = "The total is <co>$5</co: 0:[0]>."
check("response keeps its citation tags", text_of([NORMAL], Phase.RESPONSE) == response)
for size in (1, 2, 3, 7, 19):
    check(
        f"markers survive {size}-character chunks",
        text_of(split(NORMAL, size), Phase.RESPONSE) == response
        and text_of(split(NORMAL, size), Phase.THINKING) == "I will check page 1.",
    )

print("captured stream")
# R7B at temperature 0 on two short pages. It went straight to the response, with
# no thinking block, so that path is the common one and not an edge case.
fixture = Path(__file__).parent / "fixtures" / "r7b" / "response-only.json"
captured = json.loads(fixture.read_text(encoding="utf-8"))
check("it arrived in many chunks", len(captured) > 10, str(len(captured)))
check("no thinking", text_of(captured, Phase.THINKING) == "")
reply = text_of(captured, Phase.RESPONSE)
check(
    "the response is exact",
    reply == "The total due is <co>$4,182.55</co: 0:[0]> and "
    "<co>Brackenfold Freight Ltd</co: 0:[1]> carried the parcels.",
    reply,
)

print("cut off in thinking")
parser = PhaseParser()
pieces = parser.feed("<|START_THINKING|>I will") + parser.close()
check("ends in thinking", parser.phase is Phase.THINKING)
check("no response", not [p for p in pieces if p.phase is Phase.RESPONSE])

print("no thinking block")
bare = "<|START_RESPONSE|>Just this.<|END_RESPONSE|>"
check("markers without a plan", text_of([bare], Phase.RESPONSE) == "Just this.")
check(
    "no markers at all is still the reply",
    text_of(["\n Plain."], Phase.RESPONSE) == "Plain.",
)

print("the prompt opens the response")


class Template:
    def apply_chat_template(self, history, **options) -> str:
        return "<|CHATBOT_TOKEN|>"


check("render ends at the reply", render(Template(), [], []).endswith(START_RESPONSE))
check(
    "a reply with no opening marker is the reply",
    text_of(["The total.<|END_RESPONSE|>"], Phase.RESPONSE) == "The total.",
)

print("an action block, which the reader has no tools for")
ACTION = (
    "<|START_THINKING|>I will look.<|END_THINKING|><|START_ACTION|>[\n"
    '    {"tool_call_id": "1", "tool_name": "direct-injected-document"}\n'
    "]<|END_ACTION|>"
)
check("alone, it is no reply", text_of([ACTION], Phase.RESPONSE) == "")
check("split, it is no reply", text_of(split(ACTION, 3), Phase.RESPONSE) == "")
check(
    "after an answer, only the answer stays",
    text_of(["Half an answer.<|START_ACTION|>[]<|END_ACTION|>"], Phase.RESPONSE)
    == "Half an answer.",
)

print("<co> inside the thinking")
tagged = (
    "<|START_THINKING|><co>draft</co: 0:[3]><|END_THINKING|>"
    "<|START_RESPONSE|>Clean.<|END_RESPONSE|>"
)
check("stays out of the response", text_of([tagged], Phase.RESPONSE) == "Clean.")

print("edges")
check(
    "text after the response is dropped",
    text_of([bare + "tail"], Phase.RESPONSE) == "Just this.",
)
check(
    "a held-back near-marker is flushed",
    text_of(["Cost <|"], Phase.RESPONSE) == "Cost <|",
)
check("a < in prose survives", text_of(["a < b and c"], Phase.RESPONSE) == "a < b and c")
cut = PhaseParser()
cut.feed("<|START_RESPONSE|>Half")
check("cut off in the response ends there", cut.phase is Phase.RESPONSE)
check("an empty stream yields nothing", list(stream_phases([])) == [])

print("a prompt that opens the plan")
import nmv.r7b_format as r7b_format  # noqa: E402

r7b_format.OPENING = r7b_format.START_THINKING
planned = ["Add the", " two.<|END_THINKING|><|START_RESPONSE|>It is 20.<|END_RESPONSE|>"]
check(
    "the unmarked start is thinking", text_of(planned, Phase.THINKING) == "Add the two."
)
check("the response follows it", text_of(planned, Phase.RESPONSE) == "It is 20.")
acted = ["Plan.<|END_THINKING|><|START_ACTION|>[{}]<|END_ACTION|>"]
check("an action after the plan shows nothing", text_of(acted, Phase.RESPONSE) == "")
r7b_format.OPENING = r7b_format.START_RESPONSE
check("the default opens the response", PhaseParser().phase is Phase.IDLE)

print("tool results answer their own call")
result = (
    '<|START_TOOL_RESULT|>[{{"tool_call_id": "{}", "results": {{}}}}]<|END_TOOL_RESULT|>'
)
prompt = (
    "Preamble quoting <|START_TOOL_RESULT|> and <|END_TOOL_RESULT|>."
    + r7b_format._DOCUMENT_CALL
    + result.format(0)  # the documents
    + "call 1"
    + result.format(0)  # the template's number for call 1
    + "call 2"
    + result.format(1)
)
fixed = r7b_format._renumber_results(prompt)
check("the documents stay 0", result.format(0) in fixed.split("call 1")[0])
check("the first result is call 1", fixed.split("call 1")[1].startswith(result.format(1)))
check("the second result is call 2", fixed.endswith(result.format(2)))
check("the preamble is untouched", fixed.startswith("Preamble quoting <|START_TOOL"))
check(
    "no documents, no change",
    r7b_format._renumber_results("x" + result.format(0)) == "x" + result.format(0),
)

finish()
