"""Check this Mac can run the app, before anything is installed.

    uv run --no-project --python 3.12 scripts/doctor.py

Plain ``uv run`` would sync the project first, so an unsupported Mac would fail inside
``uv sync`` and never see the reason. That is why this file imports only the standard
library at module level, and why ``huggingface_hub`` is optional: it exists only after
``uv sync``.

The pins come from ``nmv/`` and the macOS floor from ``uv.lock``, read rather than
imported or copied, so neither can drift from what the app really uses.
"""

from __future__ import annotations

import ast
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN_FILES = (ROOT / "nmv" / "model.py", ROOT / "nmv" / "reader.py")
LOCK = ROOT / "uv.lock"

GIB = 2**30
GB = 1e9
MIN_MEMORY_BYTES = 16 * GIB
# ~7.7 GB of weights, ~0.9 GB of environment, and room for the transcript store.
MIN_FREE_BYTES = 10 * GB

OK, WARN, FAIL = "OK", "WARN", "FAIL"

_MAC_WHEEL = re.compile(r"macosx_(\d+)_(\d+)_arm64")


@dataclass(frozen=True)
class Result:
    name: str
    status: str
    detail: str
    fix: str = ""


def parse_version(version: str) -> tuple[int, ...]:
    """Padded, because ``(14,)`` sorts below ``(14, 0)`` and "14" is a real answer."""
    parts = [int(part) for part in re.findall(r"\d+", version)]
    return (*parts, 0) if len(parts) == 1 else tuple(parts)


def check_machine(machine: str) -> Result:
    if machine == "arm64":
        return Result("Apple Silicon", OK, machine)
    return Result(
        "Apple Silicon",
        FAIL,
        machine or "unknown machine",
        "Not supported: MLX needs Apple Silicon.",
    )


def check_macos(version: str, floor: tuple[int, int]) -> Result:
    wanted = ".".join(map(str, floor))
    if version and parse_version(version) >= floor:
        return Result("macOS", OK, f"{version} (needs {wanted})")
    return Result(
        "macOS", FAIL, f"{version or 'not macOS'} (needs {wanted})", "Update macOS."
    )


def check_memory(memory_bytes: int | None) -> Result:
    if memory_bytes is None:
        return Result("Memory", WARN, "could not be read")
    amount = f"{memory_bytes / GIB:.1f} GB"
    if memory_bytes >= MIN_MEMORY_BYTES:
        return Result("Memory", OK, amount)
    return Result(
        "Memory",
        WARN,
        f"{amount}: Chat and Grounding work, but Ask documents will not fit both models",
    )


def check_disk(free_bytes: int, location: Path) -> Result:
    amount = f"{free_bytes / GB:.1f} GB free at {location}"
    if free_bytes >= MIN_FREE_BYTES:
        return Result("Free disk", OK, amount)
    return Result(
        "Free disk",
        FAIL,
        f"{amount}, needs {MIN_FREE_BYTES / GB:.0f} GB",
        "Free some space, or set HF_HOME to an external drive.",
    )


def check_uv(path: str | None) -> Result:
    if path:
        return Result("uv", OK, path)
    return Result(
        "uv",
        FAIL,
        "not found",
        "Install it: https://docs.astral.sh/uv/getting-started/installation/",
    )


def prefetch_command(repo: str, revision: str) -> str:
    return f"uv run hf download {repo} --revision {revision}"


def check_weights(pins: dict[str, str], missing: list[str] | None) -> Result:
    """``missing`` is None when ``huggingface_hub`` is not installed yet."""
    if missing == []:
        return Result("Weights cached", OK, f"{len(pins)} of {len(pins)}")
    detail = "not checked before uv sync" if missing is None else "not all cached"
    wanted = pins if missing is None else missing
    commands = "\n".join(prefetch_command(repo, pins[repo]) for repo in wanted)
    return Result(
        "Weights cached", WARN, detail, f"After uv sync, fetch them with:\n{commands}"
    )


def read_pins(paths: tuple[Path, ...] = PIN_FILES) -> dict[str, str]:
    """Each ``*_REPO`` and its ``*_REVISION``, read without importing the module."""
    pins: dict[str, str] = {}
    for path in paths:
        values: dict[str, str] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                match node.value:
                    case ast.Constant(value=str() as value):
                        values[node.targets[0].id] = value
        for name, value in values.items():
            if name.endswith("_REPO"):
                pins[value] = values[name.removesuffix("_REPO") + "_REVISION"]
    return pins


def macos_floor(lock: Path = LOCK) -> tuple[int, int]:
    """The lowest macOS the locked ``mlx`` publishes an arm64 wheel for."""
    with lock.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    urls = [
        wheel["url"]
        for package in packages
        if package["name"] == "mlx"
        for wheel in package.get("wheels", [])
    ]
    found = [_MAC_WHEEL.search(url) for url in urls]
    return min((int(m[1]), int(m[2])) for m in found if m)


def missing_weights(pins: dict[str, str]) -> list[str] | None:
    """The pinned repos not on disk, or None before ``uv sync``."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    return [
        repo
        for repo, revision in pins.items()
        if not isinstance(
            try_to_load_from_cache(repo, "config.json", revision=revision), str
        )
    ]


def read_memory() -> int | None:
    try:
        done = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        )
        return int(done.stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def weights_volume() -> Path:
    """The closest existing folder to ``HF_HOME``, which may not exist yet."""
    home = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    path = Path(home).expanduser()
    while not path.exists():
        path = path.parent
    return path


def report(results: list[Result]) -> int:
    """Print one line per check, each with its fix, and return the exit code."""
    for result in results:
        print(f"{result.status:<4}  {result.name}: {result.detail}")
        for line in result.fix.splitlines():
            print(f"      {line}")
    return 1 if any(result.status == FAIL for result in results) else 0


def main() -> int:
    pins = read_pins()
    volume = weights_volume()
    return report(
        [
            check_machine(platform.machine()),
            check_macos(platform.mac_ver()[0], macos_floor()),
            check_memory(read_memory()),
            check_disk(shutil.disk_usage(volume).free, volume),
            check_uv(shutil.which("uv")),
            check_weights(pins, missing_weights(pins)),
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
