"""The calculator R7B may call: arithmetic only, never Python's ``eval``.

The expression is model output, so it is untrusted. It is parsed with ``ast`` and
only numbers, ``+ - * /``, unary minus and brackets are evaluated, in ``Decimal`` so
money adds up exactly. Powers are refused because ``9**9**9`` would never finish.
Nothing here raises on any input: a bad expression is an error message, which goes
back to the model as the tool's result.

The result is pasted into the template unescaped, inside a JSON string, so it may
only ever be digits, a point, a minus sign or one of the fixed messages below.
"""

from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

MAX_LENGTH = 200
NAME = "calculate"
TOOL = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Evaluates an arithmetic expression exactly, for sums, differences, "
            "products and quotients of numbers from the documents. Numbers and "
            "+ - * / and brackets only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "For example (2563.72 * 36) or 828.97 + 18.32",
                }
            },
            "required": ["expression"],
        },
    },
}

REFUSED = "error: only numbers, + - * / and brackets are allowed"
TOO_LONG = "error: the expression is too long"
DIVIDE_BY_ZERO = "error: division by zero"
TOO_BIG = "error: the result is too large"

# Money and multiplication as people write them: $1,234.56, 3 x 4, 3 × 4, 6 ÷ 2.
_CURRENCY = re.compile(r"[$€£]|\b(?:USD|EUR|GBP)\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TIMES = re.compile(r"(?<=[\d)\s])[x×](?=[\s\d(])")


class _Refused(Exception):
    pass


def _value(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _value(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        # repr keeps what was written: 0.1 stays 0.1, not its binary neighbour.
        return Decimal(repr(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        inner = _value(node.operand)
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise _Refused


def _written(value: Decimal) -> str:
    """Plain digits: no exponent, no trailing zeros, at most 6 decimal places."""
    value = value.quantize(Decimal("0.000001")).normalize()
    text = format(value, "f")
    return "0" if text == "-0" else text


def calculate(expression: object) -> str:
    """The value of ``expression`` as plain digits, or a fixed error message."""
    if not isinstance(expression, str):
        return REFUSED
    if len(expression) > MAX_LENGTH:
        return TOO_LONG
    text = _CURRENCY.sub("", expression)
    text = _THOUSANDS.sub("", text)
    text = _TIMES.sub("*", text).replace("÷", "/").strip().rstrip("=").strip()
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return REFUSED
    with localcontext() as context:
        context.prec = 34
        context.traps[DivisionByZero] = True
        try:
            return _written(_value(tree))
        except _Refused:
            return REFUSED
        except (DivisionByZero, ZeroDivisionError):
            return DIVIDE_BY_ZERO
        except InvalidOperation:
            # 0/0, or a result past 28 whole digits that cannot be written out.
            return TOO_BIG
        except (RecursionError, OverflowError):
            return REFUSED
