# tier: mlx
"""Loading the reader must cap MLX's buffer cache.

Without the cap, both models on 16 GB go past the working-set limit, so this is
what the memory check's pass rests on. Stubs mlx-lm's load, so it needs no weights.

Run:  uv run python tests/test_cache_limit.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import reader, runtime  # noqa: E402


def stub_load(path: str) -> tuple[str, str]:
    return f"model at {path}", "tokenizer"


def load_with_stub() -> reader.Reader:
    with patch("mlx_lm.load", stub_load):
        return reader._load("somewhere")


def swap_limit(value: int) -> int:
    """Set the limit, returning what it was. MLX has no plain getter."""
    import mlx.core as mx

    return mx.set_cache_limit(value)


unset = runtime.call(swap_limit, 123)
runtime.call(swap_limit, unset)
print("before the reader loads")
check("no cap is in place", unset != runtime.MLX_CACHE_LIMIT_BYTES, str(unset))

print("after the reader loads")
loaded = runtime.call(load_with_stub)
check("it still returns the reader", loaded.model == "model at somewhere")
applied = runtime.call(swap_limit, unset)
check("the cap is 1 GiB", applied == 1024**3, str(applied))

finish()
