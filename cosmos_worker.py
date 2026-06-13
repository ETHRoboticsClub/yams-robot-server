"""Cosmos Video2World2Action worker.

Runs in the isolated mimic-video venv (Python 3.10 + cosmos deps) and exchanges
length-prefixed pickled blobs with a parent process (the lerobot policy adapter)
over stdin/stdout. The parent stays in the YAMS venv (Python 3.12, lerobot).

Protocol — every message is `[4-byte big-endian length][pickle bytes]`:

  parent -> worker: {"type": "infer",
                     "video": np.ndarray (1, C, T, H, W) float32 in [-1, 1],
                     "state": np.ndarray (1, H_O, 10) float32,
                     "prompt": str,
                     "num_sampling_step": int,
                     "stop_after_step": int | None,
                     "use_cuda_graphs": bool,
                     "future_video_dump_path": str | None}  # optional debug MP4 path

  worker -> parent: {"type": "actions", "actions": np.ndarray (1, 15, 10) float32}

  parent -> worker: {"type": "exit"}                       # graceful shutdown

  worker -> parent: {"type": "ready"}                      # one-shot, after init
  worker -> parent: {"type": "error", "msg": "<traceback>"}  # on infer failure
"""

from __future__ import annotations

import os
import sys

# Apex's pre-built CUDA extensions are not compiled for sm_120 (RTX 50xx).
# Block the import so cosmos doesn't replace T5LayerNorm with FusedRMSNorm.
sys.modules.setdefault("apex.normalization", None)  # type: ignore[assignment]

# Cosmos sits at ~29 GiB on a 32 GiB card; expandable_segments avoids the
# fragmentation that otherwise wedges the VAE decoder at the end of each query.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# Reserve raw stdin/stdout for the binary IPC protocol BEFORE importing cosmos
# (which writes to sys.stdout in places). Any subsequent print()/stdout writes
# go to stderr instead, where the parent has them in /tmp/cosmos_worker.stderr.
_PROTOCOL_OUT = os.fdopen(os.dup(1), "wb", buffering=0)
_PROTOCOL_IN = os.fdopen(os.dup(0), "rb", buffering=0)
sys.stdout = sys.stderr
sys.stdin = open(os.devnull, "r")

import argparse
import json
import pickle
import struct
import traceback

import numpy as np
import torch


_ARRAY_TAG = "__ndarray__"

# Verbatim from mimic-video/model/scripts/run_video2world.py:_DEFAULT_NEGATIVE_PROMPT.
# CFG with guidance>1 needs a meaningful negative prompt to steer away from;
# an empty string degrades to unconditional and tends to produce noise.
_COSMOS_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor "
    "color balance, washed out colors, choppy sequences, jerky movements, low frame "
    "rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual "
    "noise, and flickering. Overall, the video is of poor quality."
)

# Default CFG scale for the cosmos predict2 video backbone. Matches the value
# used in mimic-video/model/scripts/run_video2world.py. If the live action
# pipeline ends up using a different guidance internally, set this to match
# so the debug dump visualizes the same latent distribution the action
# decoder is actually consuming.
_COSMOS_DEFAULT_GUIDANCE = 7.0


def _pack_array(arr: np.ndarray) -> tuple:
    """Mirror of mimic_adapter._pack_array — see there for rationale."""
    arr = np.ascontiguousarray(arr)
    return (_ARRAY_TAG, arr.tobytes(), tuple(arr.shape), str(arr.dtype))


def _unpack_array(obj):
    if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == _ARRAY_TAG:
        _, data, shape, dtype_str = obj
        return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    return obj

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

# T5 prompt embedding cache lives next to this file at the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _t5_cache  # noqa: E402


def _save_predicted_video(video: torch.Tensor, path: str, fps: int) -> None:
    """Write a (B, C, T, H, W) bf16 tensor in [-1, 1] as an MP4 at `path`.

    Mirrors the encoding used by the video guardrail path in video2world.py
    (clamp -> [-1,1], shift to [0,1], scale to uint8, permute to T,H,W,C).
    """
    import os as _os
    import imageio.v2 as imageio

    frames = video.clamp(-1.0, 1.0)[0]                     # (C, T, H, W)
    frames = ((frames + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()      # (T, H, W, C) RGB

    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def _read_blob(stream):
    header = stream.read(4)
    if not header or len(header) < 4:
        return None
    n = struct.unpack(">I", header)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(bytes(buf))


def _write_blob(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


class _CachedTextEncoder(torch.nn.Module):
    """Drop-in replacement for CosmosT5TextEncoder backed by the on-disk cache.

    Returns precomputed embeddings instead of running T5-11B. The pipeline's
    encode_prompt() at video2world.py:464 iterates parameters() to decide whether
    to shuttle the encoder between CPU and GPU; we report no parameters so the
    shuttle is skipped (cached tensors are tiny and already on the right device).
    """

    def __init__(self, num_tokens: int, embed_dim: int, device: str = "cuda"):
        super().__init__()
        self.device = device

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.num_tokens = num_tokens
        self.config.embed_dim = embed_dim

    def parameters(self, recurse: bool = True):
        return iter([])

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device", self.device)
        if isinstance(target, torch.device):
            self.device = str(target)
        elif isinstance(target, str):
            self.device = target
        return self

    @torch.inference_mode()
    def encode_prompts(self, prompts, max_length=None, return_mask: bool = False):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("The input prompt list is empty.")

        emb_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        for p in prompts:
            entry = _t5_cache.load(p, device=self.device)
            if entry is None:
                raise RuntimeError(
                    f"T5 cache miss inside _CachedTextEncoder for prompt: {p!r}. "
                    f"Run scripts/precompute_prompt.py first, or restart the worker "
                    f"without --task-prompt to fall back to T5."
                )
            emb = entry["embedding"]
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            emb_list.append(emb.to(self.device))
            if return_mask:
                m = entry.get("mask")
                if m is None:
                    raise RuntimeError(
                        f"T5 cache entry missing mask for prompt: {p!r}. "
                        f"Re-run scripts/precompute_prompt.py --force to regenerate."
                    )
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                mask_list.append(m.to(self.device))

        emb_out = torch.cat(emb_list, dim=0)
        if return_mask:
            return emb_out, torch.cat(mask_list, dim=0).bool()
        return emb_out


def _t5_cache_targets(task_prompt: str) -> list[str]:
    """Prompts whose embeddings must be cached to skip the T5 load.

    Just the task prompt. Both the live action path and the DUMP_VIDEO debug
    dump run with negative_prompt="" (no negative prompt, guidance=0.0), so the
    cached text encoder is never asked to encode anything but the task prompt.
    """
    return [task_prompt]


def load_pipeline(args):
    cfg = make_config()
    cfg = override(cfg, ["--", f"experiment={args.experiment_name}"])
    cfg.model.config.video_pipe_config.guardrail_config.enabled = False
    t5_cfg = cfg.model.config.video_pipe_config.text_encoder.t5

    # T5 prompt embedding cache decision: if every prompt we might encode is
    # already on disk, skip the ~80 s T5-11B load entirely.
    cache_targets = _t5_cache_targets(args.task_prompt) if args.task_prompt else []
    t5_cache_hit = bool(cache_targets) and all(
        _t5_cache.load(p) is not None for p in cache_targets
    )
    if t5_cache_hit:
        hit_files = ", ".join(_t5_cache.cache_path(p).name for p in cache_targets)
        print(
            f"[cosmos_worker] T5 cache HIT — skipping T5-11B load "
            f"(using cached embeddings in {_t5_cache.CACHE_DIR}: {hit_files})",
            file=sys.stderr, flush=True,
        )
    elif args.task_prompt:
        missing = [p for p in cache_targets if _t5_cache.load(p) is None]
        print(
            f"[cosmos_worker] T5 cache MISS ({len(missing)}/{len(cache_targets)} prompt(s) "
            f"uncached) — loading T5-11B and will populate cache for next run",
            file=sys.stderr, flush=True,
        )
    else:
        print(
            "[cosmos_worker] no --task-prompt provided — loading T5-11B (cache disabled)",
            file=sys.stderr, flush=True,
        )

    video_pipe = Video2WorldPipeline.from_config(
        config=cfg.model.config.video_pipe_config,
        dit_path=args.video_backbone_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        offload_text_encoder=True,
        use_text_encoder=not t5_cache_hit,
    )

    if t5_cache_hit:
        # Pipeline got text_encoder=None back from from_config; install the stub.
        video_pipe.text_encoder = _CachedTextEncoder(
            num_tokens=t5_cfg.num_tokens,
            embed_dim=t5_cfg.embed_dim,
        )
    elif args.task_prompt:
        # T5 is loaded — encode the prompts we want cached and write them out
        # so the next worker restart hits the fast path. Each encode is fast
        # (~100 ms); the expensive work is the T5 load we already paid for.
        for prompt in cache_targets:
            if _t5_cache.load(prompt) is not None:
                continue
            emb, mask = video_pipe.text_encoder.encode_prompts(
                prompt, return_mask=True,
            )
            path = _t5_cache.save(prompt, emb, mask, max_length=t5_cfg.num_tokens)
            print(
                f"[cosmos_worker] cached T5 embedding {path.name} "
                f"(shape={tuple(emb.shape)}) for prompt: {prompt[:60]}"
                f"{'...' if len(prompt) > 60 else ''}",
                file=sys.stderr, flush=True,
            )

    action_pipe = World2ActionPipeline.from_config(
        cfg.model.config.pipe_config,
        dit_path=args.action_decoder_path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    data_cfg = instantiate(cfg.data_config)
    with open(args.dataset_statistics_path, "rb") as f:
        stats = json.load(f)
    action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_cfg.policy_io.policy_io),
        concat_groups=data_cfg.policy_io.concat_groups,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pipeline = Video2World2ActionPipeline(video_pipe, action_pipe).cuda()
    pipeline.eval()
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-backbone-path", required=True)
    parser.add_argument("--action-decoder-path", required=True)
    parser.add_argument("--dataset-statistics-path", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="If set, run a warmup pass with this stop step. If unset, skip warmup.",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default="",
        help=(
            "Task prompt this worker will encode. When set, the worker checks "
            "the T5 embedding cache (~/.cache/mimic-yams/t5_embeddings/) and "
            "skips the ~80 s T5-11B load if the prompt is already cached. "
            "When empty (default), the cache is bypassed and T5 loads normally."
        ),
    )
    args = parser.parse_args()

    pipeline = load_pipeline(args)

    if args.stop_after_step is not None:
        # Warmup pass — runs the inductor compile / cuda-graph capture once so
        # the parent's first real query doesn't pay the 30-90s compile cost.
        print("[cosmos_worker] warming up pipeline...", file=sys.stderr, flush=True)
        dummy_video = torch.zeros((1, 3, 5, 480, 640), dtype=torch.bfloat16, device="cuda")
        dummy_state = torch.zeros((1, 1, 14), dtype=torch.bfloat16, device="cuda")
        warmup_prompt = args.task_prompt or "warmup"
        with torch.no_grad():
            pipeline(
                input_vid=dummy_video,
                state_B_HO_O=dummy_state,
                prompt=warmup_prompt,
                num_sampling_step=args.num_sampling_steps,
                stop_after_step=args.stop_after_step,
                use_cuda_graphs=args.use_cuda_graphs,
            )
        print("[cosmos_worker] warmup done, signaling ready", file=sys.stderr, flush=True)
    else:
        print(
            "[cosmos_worker] no --stop-after-step, skipping warmup, signaling ready",
            file=sys.stderr, flush=True,
        )

    _write_blob(_PROTOCOL_OUT, {"type": "ready"})

    while True:
        msg = _read_blob(_PROTOCOL_IN)
        if msg is None or msg.get("type") == "exit":
            break
        if msg.get("type") != "infer":
            _write_blob(
                _PROTOCOL_OUT,
                {"type": "error", "msg": f"unknown message type {msg.get('type')!r}"},
            )
            continue

        try:
            video_np = _unpack_array(msg["video"])
            state_np = _unpack_array(msg["state"])
            video = torch.from_numpy(video_np).cuda().to(torch.bfloat16)
            state = torch.from_numpy(state_np).cuda().to(torch.bfloat16)
            with torch.no_grad():
                pred = pipeline(
                    input_vid=video,
                    state_B_HO_O=state,
                    prompt=msg["prompt"],
                    num_sampling_step=msg["num_sampling_step"],
                    stop_after_step=msg.get("stop_after_step"),
                    use_cuda_graphs=msg.get("use_cuda_graphs", True),
                )
            actions = pred.float().cpu().numpy().astype(np.float32)

            # Send actions to the parent FIRST so the live policy doesn't wait
            # on the debug video dump. The dump (full denoising + VAE decode)
            # runs after the response is on the wire; the parent has ~3 s of
            # buffered actions to execute before it asks for the next batch.
            _write_blob(
                _PROTOCOL_OUT,
                {"type": "actions", "actions": _pack_array(actions)},
            )

            future_path = msg.get("future_video_dump_path")
            if future_path:
                # Mirror the live action call exactly — same T=5 input, same
                # CFG settings — only difference is return_context_at_step=None
                # (run all denoising steps) so we can VAE-decode the final
                # latent into pixels.
                #
                # Mirror the live action path exactly: it conditions on the
                # task prompt only, with negative_prompt="" and guidance=0.0
                # (see video2world2action.py). We use NO negative prompt; with
                # negative_prompt="" the pipeline skips the neg-embedding encode
                # (_get_data_batch_input gates it on `if negative_prompt:`), so
                # the cached text encoder is never asked for an uncached prompt.
                T = video.shape[2]
                with torch.no_grad():
                    decoded = pipeline.video2world_pipeline.generate_video(
                        vid_input=video,
                        num_latent_conditional_frames=1 if T == 1 else 2,
                        prompt=msg["prompt"],
                        negative_prompt="",
                        guidance=0.0,
                        num_sampling_step=msg["num_sampling_step"],
                        return_context_at_step=None,
                        use_cuda_graphs=msg.get("use_cuda_graphs", True),
                    )
                _save_predicted_video(decoded, future_path, fps=5)
                print(
                    f"[cosmos_worker] wrote predicted future video to {future_path}",
                    file=sys.stderr, flush=True,
                )
        except Exception:
            _write_blob(
                _PROTOCOL_OUT,
                {"type": "error", "msg": traceback.format_exc()},
            )


if __name__ == "__main__":
    main()
