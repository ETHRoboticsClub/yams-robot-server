#!/usr/bin/env python3
"""Write a T5 prompt-embedding cache entry from a precomputed zarr episode.

Some episodes already ship the T5 ``language_embedding`` (shape [1, 512, 1024],
f16) and the raw ``instruction`` baked into the zarr. This pulls that embedding
out and saves it in the same on-disk format cosmos_worker.py expects (see
_t5_cache.save), so the worker hits the T5 cache for that prompt and skips the
~80 s T5-11B load — no need to re-run T5 via scripts/precompute_prompt.py.

The cache stores a (num_tokens, embed_dim) embedding plus a bool padding mask.
The zarr has no mask, so we derive it: padded positions are exact-zero rows.

Usage:
    scripts/cache_embedding_from_zarr.py /path/to/episode_0001.zarr

Run with the cosmos venv python (it has zarr + the right torch):
    /home/ethrc/Desktop/mimic-video/model/.venv/bin/python \
        scripts/cache_embedding_from_zarr.py /home/ethrc/Downloads/emb/episode_0001.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import zarr

# Make _t5_cache (repo root) importable from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _t5_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zarr_path", help="Path to an episode .zarr with language_embedding + instruction")
    ap.add_argument(
        "--prompt",
        default=None,
        help="Override the prompt string (default: read the zarr's `instruction` attr)",
    )
    args = ap.parse_args()

    g = zarr.open(args.zarr_path, mode="r")
    prompt = args.prompt if args.prompt is not None else g.attrs["instruction"]

    emb = torch.from_numpy(g["language_embedding"][:])  # [1, 512, 1024], f16
    if emb.dim() == 3:
        emb = emb.squeeze(0)  # -> [512, 1024], matching the cache format
    if emb.dim() != 2:
        raise ValueError(f"expected a 2D embedding after squeeze, got shape {tuple(emb.shape)}")
    num_tokens = emb.shape[0]

    # Padded positions are exact-zero rows; everything else is a real token.
    mask = emb.to(torch.float32).abs().sum(dim=-1) > 0  # [512], bool

    # The pipeline runs in bf16; match the dtype the live T5 path writes.
    emb = emb.to(torch.bfloat16)

    path = _t5_cache.save(prompt, emb, mask, max_length=num_tokens)
    print(
        f"cached: {path.name} shape={tuple(emb.shape)} dtype={emb.dtype} "
        f"real_tokens={int(mask.sum())}/{num_tokens} -- {prompt!r}"
    )
    print(f"  -> {path}")
    print(f"  cache dir: {_t5_cache.CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
