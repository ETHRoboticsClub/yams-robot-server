"""Standalone video generation diagnostic.

Loads the fused fine-tuned video backbone, conditions on a single PNG, runs
full 35-step denoising + VAE decode, saves MP4. No action decoder. No IPC.

Mirrors cosmos_worker.py's startup workarounds (apex.normalization block for
sm_120) before importing cosmos. Run from the mimic-video venv:

  cd /home/ethrc/Desktop/mimic-video/model
  .venv/bin/python /home/ethrc/Desktop/mimic-yams/scripts/standalone_video_gen.py \
    --dit-path /home/ethrc/Desktop/mimic-yams/checkpoints/cosmos_ethrc_7000it_fused.pt \
    --input /home/ethrc/Desktop/mimic-yams/test_inputs/topdown_live.png \
    --prompt "Pick up the item and place it in the box" \
    --guidance 7.0 \
    --output /home/ethrc/Desktop/mimic-yams/logs/standalone_test/fused_g7.mp4
"""
import os
import sys

# Apex's pre-built CUDA extensions are not compiled for sm_120 (RTX 50xx).
# Block the import so cosmos doesn't replace T5LayerNorm with FusedRMSNorm.
sys.modules.setdefault("apex.normalization", None)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse

import torch

from cosmos_predict2.configs.config_video2world import (
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-path", required=True)
    parser.add_argument("--input", required=True, help="conditioning image or video")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--num-sampling-step", type=int, default=35)
    parser.add_argument("--num-conditional-frames", type=int, default=1, choices=[1, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = get_cosmos_predict2_video2world_pipeline(
        model_size="2B", resolution="480", fps=10
    )
    config.guardrail_config.enabled = False

    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=args.dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        offload_text_encoder=True,
    )

    negative_prompt = args.negative_prompt
    if negative_prompt is None:
        # Verbatim from mimic-video/model/scripts/run_video2world.py.
        negative_prompt = (
            "The video captures a series of frames showing ugly scenes, static with no motion, "
            "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
            "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor "
            "color balance, washed out colors, choppy sequences, jerky movements, low frame "
            "rate, artifacting, color banding, unnatural transitions, outdated special effects, "
            "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual "
            "noise, and flickering. Overall, the video is of poor quality."
        )

    print(f"[standalone] input={args.input}")
    print(f"[standalone] prompt={args.prompt!r}")
    print(f"[standalone] guidance={args.guidance} steps={args.num_sampling_step} "
          f"cond_frames={args.num_conditional_frames}")
    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            aspect_ratio="4:3",
            input_path=args.input,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            use_cuda_graphs=False,
        )

    if video is None:
        print("[standalone] pipe returned None (guardrail or input rejected)")
        sys.exit(1)

    fps = 10 if getattr(pipe.config, "state_t", 16) == 16 else 16
    from imaginaire.utils.io import save_image_or_video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image_or_video(video, args.output, fps=fps)
    print(f"[standalone] saved {args.output}  fps={fps}")


if __name__ == "__main__":
    main()
