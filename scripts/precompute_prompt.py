#!/usr/bin/env python3
"""Precompute the T5-11B embedding for a task prompt and cache it to disk.

Run this once per new task prompt. Loads T5-11B (~80 s), encodes the prompt,
saves a ~100 KB tensor to ``~/.cache/mimic-yams/t5_embeddings/<hash>.pt``.
After that, cosmos_worker.py skips the T5 load whenever the same prompt is
configured, dropping cold start from ~105 s to ~25 s.

Usage:
    ./scripts/precompute_prompt.py "Pick up the item and place it in the box"

Must be run with the cosmos venv's python interpreter (the same one
mimic_adapter spawns the worker with). The bash wrapper handles that — see
the bottom of this file for a one-liner if you invoke it manually.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Mirror cosmos_worker.py's pre-import setup so behaviour matches the worker.
sys.modules.setdefault("apex.normalization", None)  # type: ignore[assignment]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Make _t5_cache (which lives at the repo root) importable from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cosmos_predict2.configs.config import make_config  # noqa: E402
from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder  # noqa: E402
from imaginaire.utils.config_helper import override  # noqa: E402

import _t5_cache  # noqa: E402


# Verbatim from cosmos_worker.py. Kept inline so this script doesn't drag in
# the worker's heavy imports (it only needs T5, not the full cosmos pipeline).
_COSMOS_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor "
    "color balance, washed out colors, choppy sequences, jerky movements, low frame "
    "rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual "
    "noise, and flickering. Overall, the video is of poor quality."
)

# Default experiment name from mimic_adapter.MimicVideoConfig — drives the T5
# config the worker will use at inference time.
_DEFAULT_EXPERIMENT = (
    "w2a_bi_yams_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
    "_lr1.000e-04_layer20_bsz256"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="Task prompt to encode")
    ap.add_argument(
        "--experiment-name",
        default=_DEFAULT_EXPERIMENT,
        help="Cosmos experiment name (drives T5 config; the default matches mimic_adapter)",
    )
    ap.add_argument(
        "--skip-negative",
        action="store_true",
        help="Skip caching the negative prompt used by the DUMP_VIDEO debug path",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-encode and overwrite existing cache entries",
    )
    args = ap.parse_args()

    targets = [args.prompt]
    if not args.skip_negative:
        targets.append(_COSMOS_DEFAULT_NEGATIVE_PROMPT)

    if not args.force:
        if all(_t5_cache.load(p) is not None for p in targets):
            print("All target prompts already cached. Use --force to recompute.")
            for p in targets:
                short = p if len(p) < 60 else p[:60] + "..."
                print(f"  hit: {_t5_cache.prompt_hash(p)} = {short}")
            return 0

    print("Loading T5-11B (~80 s)...", file=sys.stderr, flush=True)
    cfg = make_config()
    cfg = override(cfg, ["--", f"experiment={args.experiment_name}"])
    t5_cfg = cfg.model.config.video_pipe_config.text_encoder.t5

    # Match the worker's runtime path: encoder lives on CUDA when encode runs.
    # (The worker uses offload_text_encoder=True so it shuttles CPU->GPU per
    # call; the actual encode is on CUDA, so doing the precompute on CUDA gives
    # numerically identical outputs.)
    encoder = CosmosT5TextEncoder(
        config=t5_cfg,
        device="cuda",
        torch_dtype=None,
    )
    print("T5-11B loaded — encoding prompts.", file=sys.stderr, flush=True)

    for prompt in targets:
        if not args.force and _t5_cache.load(prompt) is not None:
            short = prompt if len(prompt) < 60 else prompt[:60] + "..."
            print(f"skip (already cached): {_t5_cache.prompt_hash(prompt)} = {short}")
            continue
        emb, mask = encoder.encode_prompts(prompt, return_mask=True)
        path = _t5_cache.save(prompt, emb, mask, max_length=t5_cfg.num_tokens)
        short = prompt if len(prompt) < 60 else prompt[:60] + "..."
        print(
            f"cached: {path.name} shape={tuple(emb.shape)} dtype={emb.dtype} -- {short}"
        )

    print("\nDone. cosmos_worker.py will now skip T5 for these prompts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
