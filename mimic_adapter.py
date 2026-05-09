"""LeRobot adapter for the mimic-video (Cosmos VAM) policy.

Wires the mimic-video Video2World2ActionPipeline into LeRobot as a policy named
``mimic_video``. The cosmos pipeline runs in a subprocess (cosmos_worker.py)
backed by the mimic-video venv (Python 3.10 + cosmos deps); this module talks
to it over stdin/stdout. The bridge-benchmark single-arm checkpoints (EEF
deltas + 6D rotation + gripper) are zero-shot transferred to the YAMS right
arm via numerical Jacobian IK; the left arm is held at its current pose.
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import sys
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

from lerobot_robot_yams.forward_kinematics import arm_fk

_HERE = Path(__file__).parent
_DEFAULT_CKPT_DIR = _HERE / "checkpoints"
_DEFAULT_STATS_DIR = _HERE / "dataset_statistics"
_DEFAULT_WORKER = _HERE / "cosmos_worker.py"
_DEFAULT_COSMOS_VENV = Path("/home/ethrc/Desktop/mimic-video/model/.venv")
_DEFAULT_COSMOS_PYTHON = _DEFAULT_COSMOS_VENV / "bin" / "python"

_VIDEO_BACKBONE = "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
_ACTION_DECODER = (
    "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
    "_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
)
_EXPERIMENT = (
    "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
    "_lr1.000e-04_layer20_bsz256"
)


@PreTrainedConfig.register_subclass("mimic_video")
@dataclass
class MimicVideoConfig(PreTrainedConfig):
    type: str = "mimic_video"

    video_backbone_path: str = str(_DEFAULT_CKPT_DIR / _VIDEO_BACKBONE)
    action_decoder_path: str = str(_DEFAULT_CKPT_DIR / _ACTION_DECODER)
    dataset_statistics_path: str = str(_DEFAULT_STATS_DIR / "bridge.json")
    experiment_name: str = _EXPERIMENT

    cosmos_python: str = str(_DEFAULT_COSMOS_PYTHON)
    cosmos_worker_script: str = str(_DEFAULT_WORKER)

    img_horizon: int = 5
    lowdim_horizon: int = 1
    camera_fps: int = 30
    target_fps: int = 5
    num_sampling_steps: int = 35
    stop_video_denoising_step: int | None = None
    num_execute_actions: int = 8

    # Cuda graphs allocate ~1 GiB of private pool. On a 32 GiB card this
    # bumps the VAE decoder over the edge during the first decode → OOM.
    # Leave off until we have headroom (or run on a 48 GiB+ GPU).
    use_cuda_graphs: bool = False
    skip_warmup: bool = False

    # Hold each model action for this many lerobot ticks before popping the
    # next. The bridge model is trained at 5 fps; at 30 Hz cameras one delta
    # would execute 6× faster than trained, so default = camera_fps/target_fps.
    action_hold_steps: int = 6

    # If False, IK ignores the model's absolute rotation and tracks only
    # position (current EEF rotation is held). Bridge's rot6d lives in the
    # WidowX base frame and won't transfer cleanly to YAMS — set False if the
    # arm rotates wildly on first roll.
    use_action_rotation: bool = True

    task_prompt: str = ""
    image_obs_key: str = "observation.images.topdown"
    state_obs_key: str = "observation.state"

    # Right gripper joint position (radians, YAMS i2rt convention) for bridge
    # gripper signs -1 (open) and +1 (closed). Calibrate to your gripper's
    # mechanical range — read off a teleop session.
    gripper_open_rad: float = 0.0
    gripper_close_rad: float = 0.785  # ~45 degrees

    ik_max_iter: int = 30
    ik_damping: float = 1e-2

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


def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Decode bridge's 6D rotation (first two rows of R) back into a 3x3 matrix."""
    r1 = r6[:3]
    r2 = r6[3:]
    r1 = r1 / (np.linalg.norm(r1) + 1e-9)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-9)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=0)


def _matrix_to_6d(R: np.ndarray) -> np.ndarray:
    return R[:2].reshape(6).astype(np.float32)


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if abs(theta) < 1e-6:
        return np.zeros(3)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )
    return axis * theta / (2.0 * np.sin(theta))


def _eef_pose(q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, T = arm_fk(q_rad)
    return T[:3, 3].astype(np.float64).copy(), T[:3, :3].astype(np.float64).copy()


def _eef_jacobian(q_rad: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    pos0, R0 = _eef_pose(q_rad)
    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        dq = q_rad.copy()
        dq[i] += eps
        pos1, R1 = _eef_pose(dq)
        dR = R1 @ R0.T
        J[:3, i] = (pos1 - pos0) / eps
        J[3:, i] = _matrix_to_axis_angle(dR) / eps
    return J


def _ik(
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    target_R: np.ndarray,
    max_iter: int,
    lam: float,
) -> np.ndarray:
    q = q_seed.astype(np.float64).copy()
    for _ in range(max_iter):
        cur_pos, cur_R = _eef_pose(q)
        pos_err = target_pos - cur_pos
        rot_err = _matrix_to_axis_angle(target_R @ cur_R.T)
        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err) < 1e-4:
            break
        J = _eef_jacobian(q)
        dq = np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ err)
        q = q + dq
    return q


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
        self._last_q_rad: np.ndarray | None = None

        self._hold_remaining = 0
        self._pending_start_pos: np.ndarray | None = None
        self._pending_delta_pos: np.ndarray | None = None
        self._pending_action_R: np.ndarray | None = None
        self._pending_gripper_rad: float = 0.0

        # Diagnostics: log a one-time summary of the first frame the policy
        # receives, then a stat line on every model query, so we can tell
        # at a glance whether the topdown camera is feeding real pixels vs
        # black/garbage frames.
        self._tick_count = 0
        self._logged_first_frame = False

    def reset(self) -> None:
        self._img_history.clear()
        self._lowdim_history.clear()
        self._action_buf = []
        self._last_q_rad = None
        self._hold_remaining = 0
        self._pending_start_pos = None
        self._pending_delta_pos = None
        self._pending_action_R = None

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
                self._action_buf = self._query_pipeline()
            self._latch_next_action(self._action_buf.pop(0), state)
            self._hold_remaining = max(1, self.cfg.action_hold_steps)

        self._hold_remaining -= 1
        return self._to_joint_tensor(state)

    def _latch_next_action(
        self, action: np.ndarray, obs_state: torch.Tensor
    ) -> None:
        # Snapshot the EEF position at latch time. _to_joint_tensor marches
        # the IK target linearly from snapshot_pos toward snapshot_pos+Δ
        # over `action_hold_steps` ticks. Re-anchoring to obs each tick
        # (the previous strategy) leaks motion when the arm follows slowly:
        # the target only ever sits Δ/N ahead of the current pose so the
        # cumulative motion is bounded by tracking error, not by Δ.
        full = obs_state[0].detach().cpu().numpy().astype(np.float64)
        cur_q_rad = full[7:13]
        cur_pos, cur_R = _eef_pose(cur_q_rad)
        self._pending_start_pos = cur_pos
        self._pending_delta_pos = action[:3].astype(np.float64)
        self._pending_action_R = (
            _rot6d_to_matrix(action[3:9].astype(np.float64))
            if self.cfg.use_action_rotation
            else cur_R
        )
        self._pending_gripper_rad = float(
            self.cfg.gripper_close_rad
            if float(action[9]) > 0
            else self.cfg.gripper_open_rad
        )

    def _process_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        # img_tensor: (1, C, H, W) float32 in [0, 1] from prepare_observation_for_inference.
        x = img_tensor[0].detach().cpu().numpy().astype(np.float32)
        x = 2.0 * x - 1.0  # bridge expects [-1, 1]
        return x[:, None, :, :]  # (C, 1, H, W) — T-axis has length 1 per frame

    def _state_from_obs(self, obs_state: torch.Tensor) -> np.ndarray:
        # obs_state: (1, 14) joint positions in *radians* (YAMS / i2rt
        # convention), layout [left_j1..left_j6, left_grip, right_j1..right_j6,
        # right_grip].
        full = obs_state[0].detach().cpu().numpy().astype(np.float64)
        right_q_rad = full[7:13]
        right_gripper_rad = full[13]
        pos, R = _eef_pose(right_q_rad)
        rot6d = _matrix_to_6d(R)

        center = 0.5 * (self.cfg.gripper_close_rad + self.cfg.gripper_open_rad)
        half_span = 0.5 * (self.cfg.gripper_close_rad - self.cfg.gripper_open_rad)
        gripper_signed = float((right_gripper_rad - center) / (half_span + 1e-9))
        return np.concatenate([pos, rot6d, [gripper_signed]]).astype(np.float32)

    def _query_pipeline(self) -> list[np.ndarray]:
        sampled_frames = list(self._img_history)[:: self._stride]
        # Each frame is (C, 1, H, W); concat along T -> (C, T, H, W).
        images = np.concatenate(sampled_frames, axis=1).astype(np.float32)
        lowdims = np.stack(list(self._lowdim_history), axis=0).astype(np.float32)

        logging.info(
            "[mimic_video] query #%d: video=%s [min=%.3f mean=%.3f max=%.3f] "
            "state=%s",
            self._tick_count // max(1, self.cfg.action_hold_steps * self.cfg.num_execute_actions) + 1,
            tuple(images.shape),
            float(images.min()),
            float(images.mean()),
            float(images.max()),
            lowdims[-1].round(3).tolist(),
        )

        # Add batch axis -> (1, C, T, H, W) and (1, H_O, 10).
        actions = self._client.infer(
            video=images[None],
            state=lowdims[None],
            prompt=self.cfg.task_prompt,
            num_sampling_step=self.cfg.num_sampling_steps,
            stop_after_step=self.cfg.stop_video_denoising_step,
            use_cuda_graphs=self.cfg.use_cuda_graphs,
        )
        actions = actions[0]  # (15, 10)
        logging.info(
            "[mimic_video] actions[0]=%s actions[7]=%s actions[14]=%s",
            actions[0].round(3).tolist(),
            actions[min(7, actions.shape[0] - 1)].round(3).tolist(),
            actions[-1].round(3).tolist(),
        )
        n = min(self.cfg.num_execute_actions, actions.shape[0])
        return [actions[i] for i in range(n)]

    def _to_joint_tensor(self, obs_state: torch.Tensor) -> torch.Tensor:
        assert self._pending_start_pos is not None
        assert self._pending_delta_pos is not None
        assert self._pending_action_R is not None

        full = obs_state[0].detach().cpu().numpy().astype(np.float32)
        # IK seed must be the observed joints (not the last computed
        # solution) to avoid drifting onto fictional inverse branches.
        cur_right_q_rad = full[7:13].astype(np.float64)

        # March the IK target linearly along the snapshotted trajectory:
        # at tick k of N, target = start + Δ * (k/N). Reaches start+Δ on
        # the final tick, regardless of how slowly the arm tracks.
        hold_steps = max(1, self.cfg.action_hold_steps)
        elapsed = hold_steps - self._hold_remaining  # 0..hold_steps-1 after decrement
        progress = (elapsed + 1) / hold_steps        # 1/N, 2/N, ..., 1.0
        target_pos = self._pending_start_pos + self._pending_delta_pos * progress
        target_R = self._pending_action_R

        q_rad = _ik(
            cur_right_q_rad,
            target_pos,
            target_R,
            max_iter=self.cfg.ik_max_iter,
            lam=self.cfg.ik_damping,
        )
        self._last_q_rad = q_rad

        out = full.copy()
        out[7:13] = q_rad.astype(np.float32)
        out[13] = self._pending_gripper_rad
        # Left arm (out[0:7]) stays at currently observed positions.
        return torch.from_numpy(out).unsqueeze(0)

    def __del__(self) -> None:
        try:
            client = getattr(self, "_client", None)
            if client is not None:
                client.close()
        except Exception:
            pass


# Allow lerobot's dynamic plugin importer to resolve modeling_mimic_video -> this module.
sys.modules.setdefault("modeling_mimic_video", sys.modules[__name__])
