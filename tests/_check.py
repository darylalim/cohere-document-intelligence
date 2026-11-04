"""The ``check`` / ``finish`` pair every plain-script test shares.

The leading underscore keeps ``tests/run.py`` from treating this as a test. A
test script runs with ``tests/`` as ``sys.path[0]``, so ``from _check import
check, finish`` resolves without any path setup.
"""

import sys

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}{f' — {detail}' if detail else ''}"
    )
    if not condition:
        failures.append(name)


def finish() -> None:
    """Report the run and exit: 0 if every check passed, 1 otherwise."""
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed")
    sys.exit(0)
