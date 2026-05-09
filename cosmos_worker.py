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
                     "use_cuda_graphs": bool}
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


def load_pipeline(args):
    cfg = make_config()
    cfg = override(cfg, ["--", f"experiment={args.experiment_name}"])
    cfg.model.config.video_pipe_config.guardrail_config.enabled = False

    video_pipe = Video2WorldPipeline.from_config(
        config=cfg.model.config.video_pipe_config,
        dit_path=args.video_backbone_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        offload_text_encoder=True,
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
    args = parser.parse_args()

    pipeline = load_pipeline(args)

    if args.stop_after_step is not None:
        # Warmup pass — runs the inductor compile / cuda-graph capture once so
        # the parent's first real query doesn't pay the 30-90s compile cost.
        print("[cosmos_worker] warming up pipeline...", file=sys.stderr, flush=True)
        dummy_video = torch.zeros((1, 3, 5, 480, 640), dtype=torch.bfloat16, device="cuda")
        dummy_state = torch.zeros((1, 1, 10), dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            pipeline(
                input_vid=dummy_video,
                state_B_HO_O=dummy_state,
                prompt="warmup",
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
            _write_blob(
                _PROTOCOL_OUT,
                {"type": "actions", "actions": _pack_array(actions)},
            )
        except Exception:
            _write_blob(
                _PROTOCOL_OUT,
                {"type": "error", "msg": traceback.format_exc()},
            )


if __name__ == "__main__":
    main()
