"""T5 prompt embedding cache — shared between cosmos_worker.py and precompute_prompt.py.

Stores the output of ``CosmosT5TextEncoder.encode_prompts`` so cosmos_worker.py
can skip the ~80 s T5-11B load when the configured prompt is already known.

The cache lives in ``~/.cache/mimic-yams/t5_embeddings/`` (override with
``MIMIC_T5_CACHE_DIR``). One entry per unique prompt string, keyed by
``sha256(prompt)[:12]``. Each entry is a small torch.save'd dict (~100 KB at
max_length=512, embed_dim=1024 in bf16) plus a companion .txt file with the
raw prompt for debugging. Clearing the cache: ``rm -rf $MIMIC_T5_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch


CACHE_DIR = Path(
    os.environ.get(
        "MIMIC_T5_CACHE_DIR",
        str(Path.home() / ".cache" / "mimic-yams" / "t5_embeddings"),
    )
)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def cache_path(prompt: str) -> Path:
    return CACHE_DIR / f"{prompt_hash(prompt)}.pt"


def load(prompt: str, device: str | torch.device = "cpu") -> dict | None:
    """Return ``{"embedding": ..., "mask": ..., "max_length": ...}`` or None on miss/corrupt."""
    path = cache_path(prompt)
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location=device)
    except Exception:
        return None
    if not isinstance(data, dict) or "embedding" not in data:
        return None
    return data


def save(
    prompt: str,
    embedding: torch.Tensor,
    mask: torch.Tensor | None,
    max_length: int,
) -> Path:
    """Atomically write the embedding for ``prompt`` to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(prompt)
    tmp = path.with_suffix(".pt.tmp")
    payload = {
        "embedding": embedding.detach().to("cpu"),
        "mask": mask.detach().to("cpu") if mask is not None else None,
        "max_length": max_length,
        "prompt": prompt,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)
    (CACHE_DIR / f"{prompt_hash(prompt)}.txt").write_text(prompt + "\n")
    return path
