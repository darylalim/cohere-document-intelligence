"""What macOS says about memory. No MLX, no Streamlit.

Everything here feeds a status panel, so nothing here raises: any failure or
unexpected output returns ``None``, which the panel shows as "Unknown".
"""

from __future__ import annotations

import re
import subprocess

NORMAL, WARNING, CRITICAL = 1, 2, 4
PRESSURE_LABELS = {NORMAL: "Normal", WARNING: "Warning", CRITICAL: "Critical"}
TIMEOUT_S = 1

_SWAP_USED = re.compile(r"\bused = (\d+(?:\.\d+)?)([KMG])\b")
_MB_PER_UNIT = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}


def _sysctl(name: str) -> str | None:
    try:
        done = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def parse_pressure(output: str | None) -> int | None:
    text = (output or "").strip()
    level = int(text) if text.isascii() and text.isdigit() else None
    return level if level in PRESSURE_LABELS else None


def parse_swap_used(output: str | None) -> float | None:
    match = _SWAP_USED.search(output or "")
    return float(match[1]) * _MB_PER_UNIT[match[2]] if match else None


def pressure_level() -> int | None:
    """macOS memory pressure: 1 normal, 2 warning, 4 critical."""
    return parse_pressure(_sysctl("kern.memorystatus_vm_pressure_level"))


def swap_used_mb() -> float | None:
    return parse_swap_used(_sysctl("vm.swapusage"))
