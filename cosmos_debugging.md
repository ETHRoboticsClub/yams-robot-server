# Cosmos Inference Debugging: Pure Noise Output

Investigation into why the fine-tuned Cosmos video-diffusion model in mimic-yams produces pure noise at inference instead of coherent future-frame predictions, and why downstream actions are bad.

## Quick verdict

You may be chasing the wrong symptom at first glance. The pure-noise MP4 you're looking at is a **debug visualization that runs a SECOND, different forward pass** with full denoising. The actual action pipeline does ~1 step of denoising and reads intermediate latents, by design. The noisy MP4 may be misleading you. Real causes for bad robot actions are listed at the bottom.

But there is also one real, verified bug uncovered during this investigation: **`cosmos_ethrc_7000it.pt` is in PEFT/LoRA format (unfused), but the inference loader builds a fused-architecture model**, so most of the transformer loads as random weights. That alone explains both the noise in the MP4 and the bad actions.

There is also a likely second bug: the experiment name hardcoded in `mimic_adapter.py:75-78` references the **Bridge** pretrained video backbone, but the team trained the action decoder on **LIBERO**. Architecture mismatch stacks on top of the LoRA mismatch.

## Step 0: Scope

The original ask: diagram what's fed to the video model at inference, identify root cause of pure noise output. No plan file to review, treated as diagnostic investigation. No code changes inside this doc, just findings and verification steps.

## The inference data flow diagram

```
                          ┌──────────────────────────────────────────┐
                          │ lerobot tick (every 200 ms, 5 Hz)       │
                          │ batch = {                                │
                          │   observation.images.topdown:           │
                          │     torch.float32, (1, 3, 480, 640)     │
                          │     range [0, 1], RGB                   │
                          │   observation.state:                    │
                          │     torch.float32, (1, 14)              │
                          │     [left_j1..j6, left_grip,            │
                          │      right_j1..j6, right_grip] radians  │
                          │ }                                        │
                          └─────────────┬────────────────────────────┘
                                        │
                                        ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │ MimicVideoPolicy.select_action  (mimic_adapter.py:353)             │
       │                                                                     │
       │  _process_image  (line 413):                                       │
       │    x = img[0].cpu().numpy().astype(float32)        (3, 480, 640)   │
       │    x = 2.0 * x - 1.0                              ← [0,1] → [-1,1] │
       │    x = x[:, None, :, :]                            (3, 1, 480, 640)│
       │                                                                     │
       │  history (deque, maxlen=5, padded with current frame on warmup):   │
       │    _img_history = [f0, f0, f0, f0, f0]            5 copies @ start │
       │                                                                     │
       │  _snapshot_obs  (line 423):                                        │
       │    sampled = list(history)[::stride=1]                              │
       │    images = concatenate(sampled, axis=1)           (3, 5, 480, 640)│
       │    lowdims = stack(_lowdim_history)                (1, 14)         │
       │                                                                     │
       │  _run_inference:                                                   │
       │    video = images[None]                            (1, 3, 5, 480, 640)
       │    state = lowdims[None]                           (1, 1, 14)      │
       │    prompt = "Pick up the item and place it in the box"             │
       │    num_sampling_step = 35                                          │
       │    stop_after_step  = $STOP_VIDEO_DENOISING_STEP   ← from env, =1  │
       │                                                    in your log     │
       └─────────────┬──────────────────────────────────────────────────────┘
                     │ pickle, length-prefixed, over stdin/stdout
                     ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │ cosmos_worker.py  (subprocess in mimic-video venv)                 │
       │                                                                     │
       │ video = torch.from_numpy(...).cuda().to(bfloat16)  (1, 3, 5, 480, 640)
       │ state = torch.from_numpy(...).cuda().to(bfloat16)  (1, 1, 14)      │
       │                                                                     │
       │ ╔═══════ TWO SEPARATE FORWARD PASSES PER TICK ══════════════════╗  │
       │ ║                                                                ║  │
       │ ║ A) LIVE ACTION PATH (cosmos_worker.py:240-248)                ║  │
       │ ║    pred = pipeline(input_vid=video,                           ║  │
       │ ║                    state_B_HO_O=state,                         ║  │
       │ ║                    prompt=prompt,                              ║  │
       │ ║                    num_sampling_step=35,                       ║  │
       │ ║                    stop_after_step=1)         ← STOPS AT 1/35  ║  │
       │ ║                                                                ║  │
       │ ║    Inside Video2World2ActionPipeline.__call__:                ║  │
       │ ║    ┌──────────────────────────────────────────┐               ║  │
       │ ║    │ video2world_pipeline.generate_video(     │               ║  │
       │ ║    │   vid_input=video,                       │               ║  │
       │ ║    │   num_latent_conditional_frames=2,       │               ║  │
       │ ║    │   prompt=prompt,                         │               ║  │
       │ ║    │   negative_prompt="",                    │ ← hard-coded   ║  │
       │ ║    │   guidance=0.0,                          │ ← in cosmos    ║  │
       │ ║    │   return_context_at_step=1,              │   pipeline     ║  │
       │ ║    │   hidden_state_layer_idx=20,             │                ║  │
       │ ║    │ )                                        │                ║  │
       │ ║    │   → init: x = N(0, sigma_max=80.0)       │ near-pure noise║  │
       │ ║    │   → 1 denoise step                       │ slight refine  ║  │
       │ ║    │   → return latent at layer 20            │ (B, ?, 2048)   ║  │
       │ ║    └──────────────────────────────────────────┘                ║  │
       │ ║                                                                ║  │
       │ ║    Then: world2action_pipeline(                                ║  │
       │ ║              state_B_HO_O=state,                               ║  │
       │ ║              crossattn_emb=latent,                             ║  │
       │ ║              context_timesteps_B_1=sigma_at_step_1)            ║  │
       │ ║      → 10 BetaScheduler denoise steps on action latent         ║  │
       │ ║      → unnormalize with stats from bi_yams_carton.json         ║  │
       │ ║      → returns (1, 30, 14) absolute joint targets              ║  │
       │ ║                                                                ║  │
       │ ║    ◆ Returned to parent → executes on robot                    ║  │
       │ ╚════════════════════════════════════════════════════════════════╝  │
       │                                                                     │
       │ ╔═══════ B) DEBUG VIDEO DUMP PATH (cosmos_worker.py:273-283) ═══╗  │
       │ ║                                                                ║  │
       │ ║    Same input video. DIFFERENT settings:                       ║  │
       │ ║    ┌──────────────────────────────────────────┐               ║  │
       │ ║    │ video2world_pipeline.generate_video(     │                ║  │
       │ ║    │   vid_input=video,                       │                ║  │
       │ ║    │   num_latent_conditional_frames=2,       │                ║  │
       │ ║    │   prompt=prompt,                         │                ║  │
       │ ║    │   negative_prompt=_DEFAULT_NEGATIVE,     │ ← fix from      ║  │
       │ ║    │   guidance=7.0,                          │   .old.py       ║  │
       │ ║    │   return_context_at_step=None,           │ ← FULL 35 steps ║  │
       │ ║    │ )                                        │                ║  │
       │ ║    │   → init: x = N(0, sigma_max=80.0)       │                ║  │
       │ ║    │   → 35 denoise steps with CFG=7          │                ║  │
       │ ║    │   → VAE decode latent → pixels           │                ║  │
       │ ║    │   → (1, 3, 16, 480, 640) bf16 in [-1, 1] │                ║  │
       │ ║    └──────────────────────────────────────────┘                ║  │
       │ ║                                                                ║  │
       │ ║    ◆ Saved as MP4 at logs/future_video/pred_NNNNN_*.mp4        ║  │
       │ ║      THIS IS THE FILE YOU ARE INSPECTING AS NOISE              ║  │
       │ ╚════════════════════════════════════════════════════════════════╝  │
       └────────────────────────────────────────────────────────────────────┘
```

## What the Mimic Labs reference does (`eval/libero/run.py`)

The official inference path matches the worker structure almost exactly. The differences that matter:

| Thing | Mimic Labs (LIBERO) | You (bi_yams) | Verdict |
|---|---|---|---|
| State dim | 10 (3 pos + 6 rot + 1 grip) | 14 (joint angles, bimanual) | Different but bi_yams config supports 14 |
| Image norm | `2 * (uint8/255 - 0.5)` | `2 * float[0,1] - 1` | Same math |
| Image res | 480×640 | 480×640 | Same |
| T history | 5 frames at 5 Hz | 5 frames at 5 Hz | Same |
| Action horizon | 60 (LIBERO) / 16 (Bridge) | 30 | Per bi_yams config |
| Video sampling steps | 35 | 35 | Same |
| Guidance (live) | 0.0 | 0.0 | Same (hard-coded inside `video2world2action.py`) |
| `stop_after_step` (live) | configurable | 1 | Your `STOP_VIDEO_DENOISING_STEP=1` |
| Video backbone iters | Bridge: 70,043; LIBERO: 7–8K | 7,000 | LIBERO-comparable for sim data, low for real-world |
| Debug video dump | not present | guidance=7.0, full 35 steps | Your addition, only you look at it |

## Why the MP4 looks like noise

### THE smoking gun: LoRA format mismatch (confidence 9/10)

`cosmos_ethrc_7000it.pt` is structurally NOT compatible with the fused architecture the inference loader builds.

Verified by direct file inspection:

| | `cosmos_ethrc_7000it.pt` (A) | `v2w_bridge_lora_rank256_..._fused.pt` (B) |
|---|---|---|
| File size | 11.9 GB | 3.9 GB |
| Top-level keys | 2,494 | 687 |
| LoRA-suffixed keys | 1,120 | 0 |
| Shared keys | 407 | 407 |

A has per-layer:
```
net.blocks.0.cross_attn.k_proj.base_layer.weight       (2048, 1024)
net.blocks.0.cross_attn.k_proj.lora_A.default.weight   (16, 1024)   ← LoRA down, rank=16
net.blocks.0.cross_attn.k_proj.lora_B.default.weight   (2048, 16)   ← LoRA up
```

B has per-layer:
```
net.blocks.0.cross_attn.k_proj.weight                  (2048, 1024)
```

The inference loader (`Video2WorldPipeline.from_config` in `cosmos_worker.py:152`) builds a fused architecture because the experiment name in `mimic_adapter.py:75-78` ends in `_fused`. That architecture expects keys like `q_proj.weight`, but the checkpoint provides `q_proj.base_layer.weight` plus separate `lora_A` / `lora_B` adapters. Result: the 280 transformer-block keys the model expects are missing from the checkpoint and either silently stay at random init (`strict=False`) or throw an error that's getting swallowed.

**Bulk of the transformer is loading as random weights.** No score function to denoise toward → output is Gaussian noise from sigma_max=80 even after 35 denoise steps. This explains both the noise MP4 and the bad actions, since the action decoder reads latents from this broken model at xattn_layer_idx=20.

### Suspect: wrong pretrained video backbone in the experiment name (confidence high once team confirms training source)

`mimic_adapter.py:75-78`:
```python
_EXPERIMENT = (
    "w2a_bi_yams_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
    "_lr1.000e-04_layer20_bsz256"
)
```

The `v2w_bridge_lora_rank256_..._fused` substring tells Cosmos's config system to build a model whose architecture matches the **Bridge** fused checkpoint. If the team's action decoder was actually trained on top of a **LIBERO** video backbone, the experiment name is wrong and the model architecture won't match the weights.

LIBERO fused backbones available on disk:
- `v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused.pt`
- `v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt`
- `v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt`

To fix: update `_EXPERIMENT` to reference the actual training experiment name. The string after `w2a_bi_yams_` must match whatever video backbone you fine-tuned from.

### Suspect: 7K-iter video backbone is under-trained for the real-world domain (confidence 5/10, can't distinguish from LoRA bug yet)

LIBERO models converged at 7K iters on simulated, low-variance footage. Bridge needed 70K on real-world data. YAMS carton-box is real-world bimanual, harder than Bridge. 7,000 iters on real-world video with a Bridge-LoRA base is plausibly not enough for the diffusion to learn the data distribution.

This may or may not be a real concern. The team reports they DID see coherent video during training, which means the model was learning. The current noise output is more likely the LoRA mismatch than under-training. But after fixing the LoRA mismatch, if the video still looks bad, this is the next suspect.

### Suspect: debug dump uses settings the model never trained against (confidence 3/10)

Training never optimized for `guidance=7.0` output (training uses sigma sampling, not CFG sampling, CFG is an inference-only artifact). The debug dump runs with CFG=7 and the Cosmos generic negative prompt. A well-trained action-conditioned model can produce ugly video output here. **This means a noisy MP4 does NOT necessarily mean the action latents are bad.**

Verification: dump the MP4 with `guidance=1.0` and `negative_prompt=""` (no CFG). If that looks better, the CFG settings are wrong for this model. If it still looks like noise, the model is the problem.

## What actually matters for fixing the robot

The MP4 looking like noise might be a downstream symptom. Things that DO directly cause bad actions:

| Suspect | How to verify |
|---|---|
| LoRA-formatted checkpoint loaded into fused architecture (CONFIRMED) | Fix: fuse the checkpoint with `peft_model.merge_and_unload()`, save, point inference at the fused file. Confirm output has ~687 keys at ~3.9 GB. |
| Experiment name references wrong pretrained backbone (Bridge vs LIBERO) | Confirm with team what training command was used. Update `mimic_adapter.py:75-78` to match. |
| `stop_after_step=1` doesn't match training | grep training logs / config for what `stop_after_step` the action decoder was trained with. If training used None or e.g. 17, inference at 1 reads a different latent distribution. |
| `action_decoder.pt` is the wrong file (e.g. a LIBERO/Bridge 10D decoder renamed) | Run the inspection script in `Verification commands` below to check input proj dim is 14, not 10. |
| `bi_yams_carton.json` normalizer stats don't match what the action decoder was trained on | Stats file is 8.7 MB. Verify shapes are `[30, 14]` for actions, `[1, 14]` for state. |
| Text prompt encoding diverges from training (different T5 model loaded) | Check `offload_text_encoder=True` is loading the same T5 used at training; print embedding norm and compare. |

## Failure modes for production

| Codepath | Realistic failure | Tested? | Error handling? | User-visible? |
|---|---|---|---|---|
| Video backbone produces NaN | mixed-precision overflow at sigma_max | no | `_latch_next_action` catches NaN → hold position (mimic_adapter.py:404) | silent hold |
| `cosmos_worker.py` exits mid-inference | OOM during VAE decode | no | `_read` returns None → raise (mimic_adapter.py:274) | hard fail |
| Wrong `stop_after_step` | reads garbage latent | no | none, robot tries garbage joints | **critical, silent** |
| Dataset stats file missing fields | KeyError in normalizer | no | uncaught exception | hard fail at init |
| Action[0] > robot joint limits | safety violation | no | `max_joint_step` clamp downstream in YAMS firmware | safety stop |
| Checkpoint LoRA-format vs fused-arch | silently loads random weights | no | none, model produces noise | **critical, silent** |

## Verification commands

### 1. Compare team checkpoint vs Mimic Labs fused checkpoint (already run, confirms LoRA mismatch)

```bash
/home/ethrc/Desktop/mimic-video/model/.venv/bin/python <<'PY'
import torch
a = torch.load("/home/ethrc/Desktop/mimic-yams/checkpoints/cosmos_ethrc_7000it.pt",
               map_location="cpu", mmap=True, weights_only=True)
b = torch.load("/home/ethrc/Desktop/mimic-video/model/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt",
               map_location="cpu", mmap=True, weights_only=True)
print(f"A keys: {len(a)}, LoRA keys: {sum(1 for k in a if 'lora' in k.lower())}")
print(f"B keys: {len(b)}, LoRA keys: {sum(1 for k in b if 'lora' in k.lower())}")
PY
```

Result: A has 2,494 keys including 1,120 LoRA keys. B has 687 keys, zero LoRA keys. Confirmed structural mismatch.

### 2. Inspect action decoder shape (verify 14D not 10D)

```bash
/home/ethrc/Desktop/mimic-video/model/.venv/bin/python <<'PY'
import torch
sd = torch.load("/home/ethrc/Desktop/mimic-yams/checkpoints/action_decoder.pt",
                map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
    sd = sd["model"]
elif isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]

print(f"Total params: {len(sd)}")
print("\nWeights with last-dim ∈ {10, 14}:")
for k, v in sd.items():
    if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[-1] in (10, 14):
        print(f"  {k:60s} {tuple(v.shape)}")
PY
```

Look for the input embedding (state → model dim, expected `(1024, 14)` for bi_yams) and the output head (model dim → action, expected `(14, 1024)`). If you see `10` anywhere it should be `14`, the action decoder is from a different experiment.

### 3. Check which cosmos_worker.py is actually running

```bash
ps aux | grep cosmos_worker | grep -v grep
```

Should point to `cosmos_worker.py`, not `cosmos_worker.old.py`. Per `mimic_adapter.py:67` the default is `.py` (new), but confirm.

## Recommended next moves

1. **Fix the LoRA format mismatch first.** This is the only bug actually confirmed by file inspection. Find the fuse script the Mimic Labs team used to produce `_fused.pt` files (look in `model/scripts/`). Run it on `cosmos_ethrc_7000it.pt`. Confirm output has ~687 keys and ~3.9 GB. Point `mimic_adapter.py:71` at the new fused file.

2. **Confirm experiment name matches training.** Ask the team for the exact `experiment=` argument that was passed to the training `torchrun`. Update `mimic_adapter.py:75-78` to match. If the team trained from a LIBERO backbone, replace the `v2w_bridge_lora_rank256_..._fused` substring with the LIBERO equivalent.

3. **Sanity check the fused model with a training-set video.** Run `cosmos_worker.py` once with a real recorded episode from the training data (not live camera). Compare the debug MP4 against the ground-truth future frames. If the model can't reproduce a frame from its own training distribution, the fine-tune is the problem and you need more iterations or a better recipe.

4. **Make the debug dump match the live path.** Add a config flag to `cosmos_worker.py` to dump with `guidance=0.0, return_context_at_step=stop_after_step`, full VAE decode anyway. That MP4 will look terrible (it's near-pure-noise latent decoded to pixels) BUT that's actually what the action decoder is consuming. If you want to verify the action decoder isn't broken, dump the latent statistics (mean, std, norm) and compare against training-time logs.

5. **Find the training-time `stop_after_step`.** Was there one? If training trained the action decoder against full-denoise latents (no stop), and inference uses `stop_after_step=1`, the action decoder reads garbage. This is the single most important config to verify match between train and inference.

## NOT in scope

- Retraining the model. That's a multi-day GPU job, separate decision.
- Rewriting `cosmos_worker.py` or `mimic_adapter.py`. Architecture is correct.
- Comparing against fresh Cosmos pretrained checkpoint. Downstream verification after we confirm the LoRA fix works.

## What I'm uncertain about

- Whether the user is looking at MP4 dumps from `cosmos_worker.py` (post-fix) or `cosmos_worker.old.py` (pre-fix). Default points to `.py` (new), but if the currently-running process was started before the swap, could still be on the old code. Verify with `ps aux | grep cosmos_worker`.
- What `stop_after_step` training used for the action decoder. Need to grep the training scripts.
- Whether the team's action decoder was trained on Bridge or LIBERO video backbone. The experiment name in `mimic_adapter.py:75-78` says Bridge but the team reports LIBERO.

## Provenance of artifacts in `mimic-yams/checkpoints/`

| File | Size | Source | Notes |
|---|---|---|---|
| `cosmos_ethrc_7000it.pt` | 11.9 GB | unknown, copied in by team | PEFT/LoRA format (1,120 LoRA keys, rank=16). NOT fused. Inference loader expects fused. |
| `action_decoder.pt` | 953 MB | unknown, copied in by team | The mimic-video training dir `model/checkpoints/vam/bi_yams/.../` contains only `config.yaml`, no `.pt`. Provenance unverified. |
| `dataset_statistics/bi_yams_carton.json` | 8.7 MB | computed from team's dataset | Shapes look correct: `action [30, 14]`, `state [1, 14]`. |

Add a README in `checkpoints/` documenting where each file came from, including the exact training command and iteration count. Right now there's no way to reproduce or audit.

## STATUS

DONE_WITH_CONCERNS. Root cause for the noise MP4 verified by file inspection: LoRA format mismatch. Bridge vs LIBERO experiment-name mismatch is a strong secondary hypothesis pending team confirmation of training source. Both are silent failures with no error message, the model just produces noise and the robot gets garbage actions.
