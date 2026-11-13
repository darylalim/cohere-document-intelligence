# tier: mlx
"""Reusing R7B's KV cache: which tokens are kept, and when it starts over.

Drives ``reader._reuse`` and ``reader._settle`` with mlx-lm's own cache classes on
a stand-in model whose layers mirror R7B's mix (a sliding window and a full one),
with the window shrunk to 8 tokens so the untrimmable case is reachable. Needs no
weights. Everything MLX runs on the worker, as in the app.

Run:  uv run python tests/test_reader_cache.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, finish  # noqa: E402

from nmv import reader, runtime  # noqa: E402

WINDOW = 8


class Model:
    def make_cache(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        return [RotatingKVCache(max_size=WINDOW, keep=0), KVCache()]


def feed(cache, count: int) -> None:
    """Advance every layer by ``count`` tokens, as a forward pass would."""
    import mlx.core as mx

    for layer in cache:
        keys = mx.zeros((1, 1, count, 4))
        layer.update_and_fetch(keys, keys)
    mx.eval([layer.state for layer in cache])


def offsets(cache) -> list[int]:
    return [layer.offset for layer in cache]


def generate(loaded: reader.Reader, ids: list[int], output: list[int]):
    """What ``produce`` does around one generation: reuse, prefill, decode, settle."""
    cache, kept = reader._reuse(loaded, ids)
    feed(cache, len(ids) - kept)
    feed(cache, len(output))
    reader._settle(loaded, ids + output)
    return kept, offsets(cache), list(loaded.cached_ids)


def scenario():
    loaded = reader.Reader(model=Model(), tokenizer=None)
    results = {"first": generate(loaded, [1, 2, 3], [9])}
    first = loaded.prompt_cache
    results["extends"] = generate(loaded, [1, 2, 3, 9, 4], [8])
    results["same object"] = loaded.prompt_cache is first
    results["diverges"] = generate(loaded, [1, 2, 5], [7])
    results["same prompt"] = generate(loaded, [1, 2, 5], [7])
    results["unrelated"] = generate(loaded, [6, 6], [6])
    long = list(range(100, 100 + WINDOW + 2))
    generate(loaded, long, [1])
    before = loaded.prompt_cache
    results["past the window"] = generate(loaded, [*long[:3], 0], [2])
    results["fresh cache"] = loaded.prompt_cache is not before
    return results


r = runtime.call(scenario)
print("reusing the reader's KV cache")
check("a first call prefills everything", r["first"] == (0, [4, 4], [1, 2, 3, 9]))
check(
    "a prompt that extends the last prefills only the new token",
    r["extends"] == (4, [6, 6], [1, 2, 3, 9, 4, 8]),
)
check("and keeps the same cache", r["same object"])
check(
    "a prompt that parts ways trims back to the shared prefix",
    r["diverges"] == (2, [4, 4], [1, 2, 5, 7]),
)
check(
    "the same prompt again still prefills its last token",
    r["same prompt"] == (2, [4, 4], [1, 2, 5, 7]),
)
check("nothing shared starts afresh", r["unrelated"] == (0, [3, 3], [6, 6, 6]))
check(
    "a sliding window it cannot trim starts afresh",
    r["past the window"][0] == 0 and r["fresh cache"],
)
check("and holds only the new prompt", r["past the window"][2] == [100, 101, 102, 0, 2])

finish()
