"""Fuse the YAMS LoRA video backbone (cosmos_ethrc_7000it.pt) into a flat checkpoint.

The raw training checkpoint stores LoRA adapters as separate tensors
(lora_A/lora_B/base_layer) plus an EMA copy. The inference pipeline expects
pre-fused weights with only net.* keys. This script mirrors the training-time
fuse steps from yams_train_setup_notes.md:

  python model/scripts/fuse_lora_ckpt.py iter_000007000.pt --alpha 16
  # then filter to net.* keys only

Output: <input>_fused.pt (alpha=16, net.* only).
"""
import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/ethrc/Desktop/mimic-yams/checkpoints/cosmos_ethrc_7000it.pt")
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()

    in_path = Path(args.ckpt)
    out_path = in_path.with_name(in_path.stem + "_fused.pt")
    print(f"Loading {in_path} (alpha={args.alpha}) ...", flush=True)
    ckpt = torch.load(str(in_path), map_location="cpu", weights_only=False)
    print(f"  loaded {len(ckpt)} keys", flush=True)

    lora_rank: int | None = None
    fused_count = 0
    for key in list(ckpt.keys()):
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        base_key = key.replace("lora_A.default", "base_layer")
        this_rank = ckpt[key].shape[0]
        if lora_rank is None:
            lora_rank = this_rank
        elif lora_rank != this_rank:
            raise RuntimeError(f"inconsistent lora rank: {lora_rank} vs {this_rank} at {key}")

        a = ckpt[key].float()
        b = ckpt[b_key].float()
        adapter = b @ a
        fused = ckpt[base_key].float() + (args.alpha / this_rank) * adapter
        fused = fused.to(ckpt[base_key].dtype)

        del ckpt[key], ckpt[b_key], ckpt[base_key]
        ckpt[base_key.replace(".base_layer", "")] = fused
        fused_count += 1

    print(f"  fused {fused_count} LoRA pairs, rank={lora_rank}, alpha={args.alpha}, "
          f"scale={args.alpha / (lora_rank or 1):.4f}", flush=True)

    # Filter to net.* only (drops net_ema.* — matches upstream fused format).
    pre = len(ckpt)
    ckpt = {k: v for k, v in ckpt.items() if k.startswith("net.")}
    print(f"  filtered to net.*: {len(ckpt)} keys (was {pre})", flush=True)

    print(f"Saving to {out_path} ...", flush=True)
    torch.save(ckpt, str(out_path))
    print(f"Done: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
