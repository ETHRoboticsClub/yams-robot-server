"""LeRobot adapter for the mimic-video (Cosmos VAM) policy.

Wires the mimic-video Video2World2ActionPipeline into LeRobot as a policy named
``mimic_video``. The cosmos pipeline runs in a subprocess (cosmos_worker.py)
backed by the mimic-video venv (Python 3.10 + cosmos deps); this module talksto it over stdin/stdout. The bi_yams checkpoint outputs absolute 14-dim joint
positions (both arms); the adapter linearly interpolates from the current
observed joints to the target over action_hold_steps ticks.
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


_ARRAY_TAG = "__ndarray__"


def _pack_array(arr: np.ndarray) -> tuple:
    """Serialize a numpy array as (tag, bytes, shape, dtype_str).

    Avoids pickling numpy's internal module references — the parent is on
    numpy 2.x (yams venv) and the worker is on numpy 1.26 (cosmos venv); a
    raw `pickle.dumps(arr)` from the parent embeds `numpy._core.numeric`
    paths the worker can't import.
    """
    arr = np.ascontiguousarray(arr)
    return (_ARRAY_TAG, arr.tobytes(), tuple(arr.shape), str(arr.dtype))


def _unpack_array(obj):
    if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == _ARRAY_TAG:
        _, data, shape, dtype_str = obj
        return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    return obj

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import IdentityProcessorStep, PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


_HERE = Path(__file__).parent
_DEFAULT_CKPT_DIR = _HERE / "checkpoints"
_DEFAULT_STATS_DIR = _HERE / "dataset_statistics"
_DEFAULT_WORKER = _HERE / "cosmos_worker.py"
_DEFAULT_COSMOS_VENV = Path("/home/ethrc/Desktop/mimic-video/model/.venv")
_DEFAULT_COSMOS_PYTHON = _DEFAULT_COSMOS_VENV / "bin" / "python"

_VIDEO_BACKBONE = "cosmos_ethrc_7000it_fused-16.pt"
_ACTION_DECODER = (
    "action_decoder.pt"
)
_EXPERIMENT = (
    "w2a_bi_yams_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
    "_lr1.000e-04_layer20_bsz256"
)


@PreTrainedConfig.register_subclass("mimic_video")
@dataclass
class MimicVideoConfig(PreTrainedConfig):
    type: str = "mimic_video"

    video_backbone_path: str = str(_DEFAULT_CKPT_DIR / _VIDEO_BACKBONE)
    action_decoder_path: str = str(_DEFAULT_CKPT_DIR / _ACTION_DECODER)
    dataset_statistics_path: str = str(_DEFAULT_STATS_DIR / "bi_yams_carton.json")
    experiment_name: str = _EXPERIMENT

    cosmos_python: str = str(_DEFAULT_COSMOS_PYTHON)
    cosmos_worker_script: str = str(_DEFAULT_WORKER)

    img_horizon: int = 5
    lowdim_horizon: int = 1
    camera_fps: int = 5
    target_fps: int = 5
    num_sampling_steps: int = 35
    stop_video_denoising_step: int | None = None
    num_execute_actions: int = 15
    # Training emits 30 actions at 10 Hz (3 s); lerobot ticks at 5 Hz, so
    # consume every other action to keep the wall-clock playback rate matched.
    action_stride: int = 2

    # Cuda graphs allocate ~1 GiB of private pool. On a 32 GiB card this
    # bumps the VAE decoder over the edge during the first decode → OOM.
    # Leave off until we have headroom (or run on a 48 GiB+ GPU).
    use_cuda_graphs: bool = False
    skip_warmup: bool = False

    action_hold_steps: int = 1

    task_prompt: str = ""
    image_obs_key: str = "observation.images.topdown"
    state_obs_key: str = "observation.state"

    # Debug: if set, dump the cosmos predicted-future video (full denoising +
    # VAE decode) as MP4 under this directory. Runs a second forward pass per
    # dump tick — roughly doubles per-batch inference time, so use
    # future_video_dump_every_n to throttle.
    future_video_debug_dir: str = ""
    future_video_dump_every_n: int = 1

    optimizer_lr: float = 1e-4

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        return None


def make_mimic_video_pre_post_processors(
    config: MimicVideoConfig,
    dataset_stats: dict | None = None,
    **kwargs: Any,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Identity pre/post processors — all normalization is internal to the pipeline."""
    pre = PolicyProcessorPipeline(
        steps=[IdentityProcessorStep()],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    post = PolicyProcessorPipeline(
        steps=[IdentityProcessorStep()],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post


def _cosmos_subprocess_env(cosmos_python: Path) -> dict[str, str]:
    """Env vars so transformer-engine and torch can find the venv-bundled CUDA libs."""
    env = os.environ.copy()
    nvidia_root = cosmos_python.parent.parent / "lib" / "python3.10" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        # transformer-engine globs CUDA_HOME for libnvrtc.
        env["CUDA_HOME"] = str(nvidia_root)
        # And the bundled libs need to be loadable at runtime.
        ld_dirs = sorted(str(p) for p in nvidia_root.glob("*/lib") if p.is_dir())
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            ld_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    # Don't leak the YAMS venv into the cosmos subprocess.
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    return env


class _CosmosClient:
    """Spawns and talks to cosmos_worker.py over stdin/stdout pickled blobs."""

    def __init__(self, config: MimicVideoConfig) -> None:
        cosmos_python = Path(config.cosmos_python)
        if not cosmos_python.exists():
            raise FileNotFoundError(
                f"Cosmos python interpreter not found at {cosmos_python}. "
                "Set MimicVideoConfig.cosmos_python or run "
                "`cd /home/ethrc/Desktop/mimic-video/model && uv sync --extra cu128`."
            )

        cmd = [
            str(cosmos_python),
            str(Path(config.cosmos_worker_script)),
            "--video-backbone-path", config.video_backbone_path,
            "--action-decoder-path", config.action_decoder_path,
            "--dataset-statistics-path", config.dataset_statistics_path,
            "--experiment-name", config.experiment_name,
            "--num-sampling-steps", str(config.num_sampling_steps),
        ]
        if config.stop_video_denoising_step is not None and not config.skip_warmup:
            cmd += ["--stop-after-step", str(config.stop_video_denoising_step)]
        if config.use_cuda_graphs:
            cmd.append("--use-cuda-graphs")
        env = _cosmos_subprocess_env(cosmos_python)

        logging.info("Spawning cosmos worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit; user sees cosmos init logs in their terminal
            env=env,
        )

        ready = self._read()
        if ready is None or ready.get("type") != "ready":
            self.close()
            raise RuntimeError(
                f"Cosmos worker did not signal ready (got {ready!r}). "
                "Check the worker stderr above for the failure."
            )

    def _read(self) -> dict | None:
        assert self._proc.stdout is not None
        header = self._proc.stdout.read(4)
        if not header or len(header) < 4:
            return None
        n = struct.unpack(">I", header)[0]
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return pickle.loads(bytes(buf))

    def _write(self, obj: dict) -> None:
        assert self._proc.stdin is not None
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack(">I", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def infer(
        self,
        video: np.ndarray,
        state: np.ndarray,
        prompt: str,
        num_sampling_step: int,
        stop_after_step: int | None,
        use_cuda_graphs: bool,
        future_video_dump_path: str | None = None,
    ) -> np.ndarray:
        self._write(
            {
                "type": "infer",
                "video": _pack_array(video),
                "state": _pack_array(state),
                "prompt": prompt,
                "num_sampling_step": num_sampling_step,
                "stop_after_step": stop_after_step,
                "use_cuda_graphs": use_cuda_graphs,
                "future_video_dump_path": future_video_dump_path,
            }
        )
        msg = self._read()
        if msg is None:
            raise RuntimeError("Cosmos worker exited unexpectedly.")
        if msg.get("type") == "error":
            raise RuntimeError(f"Cosmos worker error:\n{msg.get('msg')}")
        if msg.get("type") != "actions":
            raise RuntimeError(f"Cosmos worker sent unexpected message: {msg}")
        return _unpack_array(msg["actions"])

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        try:
            self._write({"type": "exit"})
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None  # type: ignore[assignment]



class MimicVideoPolicy(PreTrainedPolicy):
    name = "mimic_video"
    config_class = MimicVideoConfig

    def __init__(self, config: MimicVideoConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config)
        self.cfg = config
        self._client = _CosmosClient(config)

        if not config.task_prompt:
            logging.warning(
                "MimicVideoConfig.task_prompt is empty. The bridge model is "
                "language-conditioned; pass --policy.task_prompt=\"...\"."
            )

        self._stride = max(1, config.camera_fps // config.target_fps)
        hist_len = (config.img_horizon - 1) * self._stride + 1
        self._img_history: deque[np.ndarray] = deque(maxlen=hist_len)
        self._lowdim_history: deque[np.ndarray] = deque(maxlen=config.lowdim_horizon)
        self._action_buf: list[np.ndarray] = []
        self._hold_remaining = 0
        self._pending_start_joints: np.ndarray | None = None
        self._pending_target_joints: np.ndarray | None = None

        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: list[np.ndarray] | None = None
        self._prefetch_exc: BaseException | None = None

        self._tick_count = 0
        self._logged_first_frame = False
        self._inference_count = 0

    def reset(self) -> None:
        # Join any in-flight prefetch so the cosmos IPC channel is free.
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
        self._prefetch_thread = None
        self._prefetch_result = None
        self._prefetch_exc = None
        self._img_history.clear()
        self._lowdim_history.clear()
        self._action_buf = []
        self._hold_remaining = 0
        self._pending_start_joints = None
        self._pending_target_joints = None

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict | None]:
        raise NotImplementedError("mimic_video policy is inference-only.")

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("mimic_video policy is inference-only.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        img = batch[self.cfg.image_obs_key]
        state = batch[self.cfg.state_obs_key]
        self._tick_count += 1

        if not self._logged_first_frame:
            x = img[0].detach().cpu().numpy()
            logging.warning(
                "[mimic_video] FIRST observation: image[%s]=%s "
                "(min=%.3f mean=%.3f max=%.3f std=%.4f) state=%s",
                self.cfg.image_obs_key,
                tuple(x.shape),
                float(x.min()),
                float(x.mean()),
                float(x.max()),
                float(x.std()),
                state[0].detach().cpu().numpy().round(2).tolist(),
            )
            self._logged_first_frame = True

        proc_img = self._process_image(img)
        self._img_history.append(proc_img)
        while len(self._img_history) < (self._img_history.maxlen or 0):
            self._img_history.append(proc_img.copy())

        state_vec = self._state_from_obs(state)
        self._lowdim_history.append(state_vec)
        while len(self._lowdim_history) < (self._lowdim_history.maxlen or 0):
            self._lowdim_history.append(state_vec.copy())

        if self._hold_remaining == 0:
            if not self._action_buf:
                # Buffer empty: collect prefetch (may block briefly if not done).
                if self._prefetch_thread is not None:
                    self._action_buf = self._collect_prefetch()
                else:
                    self._action_buf = self._query_pipeline()
                # Batch just refilled — launch next inference immediately so it
                # runs during this batch's execution window (~3 s) rather than
                # blocking after the last action is consumed.
                if self._prefetch_thread is None:
                    self._launch_prefetch()
            self._latch_next_action(self._action_buf.pop(0), state)
            self._hold_remaining = max(1, self.cfg.action_hold_steps)

        self._hold_remaining -= 1
        return self._to_joint_tensor(state)

    def _latch_next_action(
        self, action: np.ndarray, obs_state: torch.Tensor
    ) -> None:
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            logging.error("[mimic_video] NaN/Inf in action — holding current position")
            observed = obs_state[0].detach().cpu().numpy().astype(np.float64)
            self._pending_start_joints = observed
            self._pending_target_joints = observed
            return
        self._pending_start_joints = obs_state[0].detach().cpu().numpy().astype(np.float64)
        self._pending_target_joints = action.astype(np.float64)

    def _process_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        # img_tensor: (1, C, H, W) float32 in [0, 1] from prepare_observation_for_inference.
        x = img_tensor[0].detach().cpu().numpy().astype(np.float32)
        x = 2.0 * x - 1.0  # bridge expects [-1, 1]
        return x[:, None, :, :]  # (C, 1, H, W) — T-axis has length 1 per frame

    def _state_from_obs(self, obs_state: torch.Tensor) -> np.ndarray:
        # obs_state: (1, 14) — [left_j1..j6, left_grip, right_j1..j6, right_grip] in radians.
        return obs_state[0].detach().cpu().numpy().astype(np.float32)

    def _snapshot_obs(self) -> tuple[np.ndarray, np.ndarray]:
        sampled = list(self._img_history)[:: self._stride]
        images = np.concatenate(sampled, axis=1).astype(np.float32)
        lowdims = np.stack(list(self._lowdim_history), axis=0).astype(np.float32)
        return images, lowdims

    def _run_inference(self, images: np.ndarray, lowdims: np.ndarray) -> list[np.ndarray]:
        logging.info(
            "[mimic_video] query: video=%s [min=%.3f mean=%.3f max=%.3f] state=%s",
            tuple(images.shape),
            float(images.min()),
            float(images.mean()),
            float(images.max()),
            lowdims[-1].round(3).tolist(),
        )
        future_video_dump_path = self._next_future_video_dump_path()
        actions = self._client.infer(
            video=images[None],
            state=lowdims[None],
            prompt=self.cfg.task_prompt,
            num_sampling_step=self.cfg.num_sampling_steps,
            stop_after_step=self.cfg.stop_video_denoising_step,
            use_cuda_graphs=self.cfg.use_cuda_graphs,
            future_video_dump_path=future_video_dump_path,
        )
        actions = actions[0]  # (action_horizon, 14)
        logging.info(
            "[mimic_video] actions[0]=%s actions[-1]=%s",
            actions[0].round(3).tolist(),
            actions[-1].round(3).tolist(),
        )
        strided = actions[:: self.cfg.action_stride]
        n = min(self.cfg.num_execute_actions, strided.shape[0])
        return [strided[i] for i in range(n)]

    def _query_pipeline(self) -> list[np.ndarray]:
        images, lowdims = self._snapshot_obs()
        return self._run_inference(images, lowdims)

    def _next_future_video_dump_path(self) -> str | None:
        self._inference_count += 1
        if not self.cfg.future_video_debug_dir:
            return None
        n = max(1, self.cfg.future_video_dump_every_n)
        if (self._inference_count - 1) % n != 0:
            return None
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"pred_{self._inference_count:05d}_{ts}.mp4"
        return str(Path(self.cfg.future_video_debug_dir) / fname)

    def _launch_prefetch(self) -> None:
        """Snapshot observations now and run inference in a background thread."""
        images, lowdims = self._snapshot_obs()

        def _run() -> None:
            try:
                self._prefetch_result = self._run_inference(images, lowdims)
            except BaseException as exc:
                self._prefetch_exc = exc

        self._prefetch_result = None
        self._prefetch_exc = None
        self._prefetch_thread = threading.Thread(target=_run, daemon=True)
        self._prefetch_thread.start()

    def _collect_prefetch(self) -> list[np.ndarray]:
        """Wait for background inference and return its result."""
        assert self._prefetch_thread is not None
        self._prefetch_thread.join()
        self._prefetch_thread = None
        exc = self._prefetch_exc
        result = self._prefetch_result
        self._prefetch_exc = None
        self._prefetch_result = None
        if exc is not None:
            raise exc
        return result  # type: ignore[return-value]

    def _to_joint_tensor(self, obs_state: torch.Tensor) -> torch.Tensor:
        hold_steps = max(1, self.cfg.action_hold_steps)
        elapsed = hold_steps - self._hold_remaining  # 1..hold_steps after decrement
        progress = elapsed / hold_steps              # 1/N, 2/N, ..., 1.0
        joints = self._pending_start_joints + (self._pending_target_joints - self._pending_start_joints) * progress
        return torch.from_numpy(joints.astype(np.float32)).unsqueeze(0)

    def __del__(self) -> None:
        try:
            client = getattr(self, "_client", None)
            if client is not None:
                client.close()
        except Exception:
            pass


# Allow lerobot's dynamic plugin importer to resolve modeling_mimic_video -> this module.
sys.modules.setdefault("modeling_mimic_video", sys.modules[__name__])
