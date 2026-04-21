# Carton Box Dataset Overview

## Datasets

There are two operator recording sessions (Tom & Matt), both stored in the HuggingFace LeRobot cache:

| Dataset | Path | Episodes | Total Frames | Avg Duration |
|---|---|---|---|---|
| `yams-carton-box-closing-mon-tom-mat` | `~/.cache/huggingface/lerobot/ETHRC/yams-carton-box-closing-mon-tom-mat` | 52 | 48,971 | ~31.4s |
| `yams-carton-box-closing-tue-tom-mat-2` | `~/.cache/huggingface/lerobot/ETHRC/yams-carton-box-closing-tue-tom-mat-2` | 20 | 20,124 | ~33.5s |
| **Combined** | | **72** | **69,095** | |

A third smaller dataset (`recordings/yams-carton-box-closing`, 11 episodes) also exists locally on the Desktop but appears to be an earlier or separate collection.

---

## Format

All datasets use **LeRobot v3.0** format. Each dataset folder contains:

```
meta/
  info.json                          # dataset-level metadata
  stats.json                         # global feature statistics
  tasks.parquet                      # task index → task string
  episodes/chunk-000/file-NNN.parquet  # per-episode metadata (length, timestamps, per-ep stats)
data/
  chunk-000/file-000.parquet         # frame-level trajectory data
videos/
  observation.images.left_wrist/     # left wrist camera (640×480, H.264, 30fps)
  observation.images.right_wrist/    # right wrist camera (640×480, H.264, 30fps)
  observation.images.topdown/        # top-down camera (640×480, H.264, 30fps)
```

---

## Robot

- **Type:** `bi_yams_follower` — bimanual robot, two 6-DOF arms + 2 grippers = **14 DoF total**
- **Recording rate:** 30 fps

---

## Task

**"Pick & Place and Closing a Box"** — a single task across all episodes. The robot picks up flaps of a carton box and closes them.

---

## Data Schema

Each row in the trajectory parquet is one timestep (one frame).

| Column | Type | Description |
|---|---|---|
| `action` | float32[14] | Commanded joint positions sent to the robot |
| `observation.state` | float32[14] | Actual measured joint positions |
| `timestamp` | float32 | Time in seconds from episode start |
| `frame_index` | int64 | Frame counter within the episode |
| `episode_index` | int64 | Episode identifier |
| `index` | int64 | Global frame index across the dataset |
| `task_index` | int64 | Task identifier (always 0 here) |

### Joint layout (same for both `action` and `observation.state`)

| Index | Name |
|---|---|
| 0–5 | `left_joint_1` … `left_joint_6` (radians) |
| 6 | `left_gripper` (0 = closed, ~1 = open) |
| 7–12 | `right_joint_1` … `right_joint_6` (radians) |
| 13 | `right_gripper` (0 = closed, ~1 = open) |

---

## Episode Metadata

The `meta/episodes/` parquet files contain per-episode statistics precomputed at record time:

- `length` — number of frames
- `tasks` — list of task strings
- `stats/action/min`, `max`, `mean`, `std`, `q01`, `q10`, `q50`, `q90`, `q99` — per-joint action statistics
- Same stats for `observation.state`, `timestamp`, `frame_index`
- Video timestamp ranges and chunk/file indices for each camera stream

---

## Episode Duration Distribution

### Monday session (`mon-tom-mat`, 52 episodes)

- Min: **5.0s** (episode 21 — likely aborted)
- Max: **37.8s**
- Mean: **31.4s**

### Tuesday session (`tue-tom-mat-2`, 20 episodes)

- Min: **3.2s** (episode 19 — likely aborted)
- Max: **43.1s**
- Mean: **33.5s**

---

## Known Quality Issues

| Dataset | Episode | Duration | Notes |
|---|---|---|---|
| `mon-tom-mat` | 21 | 5.0s (151 frames) | Likely aborted/failed demo |
| `tue-tom-mat-2` | 19 | 3.2s (95 frames) | Likely aborted/failed demo |

---

## Operator Labels

Tom and Matt's episodes are **not labeled** in the data files — both operators' recordings are mixed within each dataset. The split between operators is not currently recoverable from the data alone without external notes.

---

## Data Analysis Goals

Potential quality metrics to investigate:

1. **Smoothness / jerk** — joint velocity and acceleration per episode; high jerk indicates hesitation or instability
2. **Episode consistency** — DTW or per-joint variance across episodes from the same operator
3. **Duration analysis** — where is time spent in the task? Are there phases that vary significantly?
4. **Gripper event analysis** — open/close events as a proxy for grasp quality and timing
5. **Action vs state residual** — difference between commanded `action` and measured `observation.state` as a signal for tracking quality
6. **Cross-operator comparison** — style differences between Tom and Matt once episodes are labeled
7. **Outlier detection** — flag short/anomalous episodes automatically
