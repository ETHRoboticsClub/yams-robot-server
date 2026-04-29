[ -z "$BASH" ] && exec bash "$0" "$@"
echo "working directory $PWD"

LOG=false
for arg in "$@"; do [ "$arg" = "--log" ] && LOG=true; done
if $LOG; then
    mkdir -p logs
    LOGFILE="logs/$(date +%Y%m%d_%H%M%S).out"
    echo "Logging to $LOGFILE"
    exec > >(tee "$LOGFILE") 2>&1
fi

pgrep -f "lerobot-record|lerobot-teleoperate|yams_server.py" | grep -vx "$$" | xargs -r kill

YAML=configs/arms.yaml
RESUME=${RESUME:-false}
PUSH_TO_HUB=${PUSH_TO_HUB:-true}
NEW_REPO=${NEW_REPO:-false}
MIN_CAMERA_FPS=$(yq '[.cameras.configs[].fps] | min' "$YAML")
DATASET_FPS=${DATASET_FPS:-$MIN_CAMERA_FPS}
NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
RESET_TIME_S=${RESET_TIME_S:-0}
VCODEC=${VCODEC:-auto}

# =============================================================================
# TASK: TOWEL FOLDING
# Dataset: ETHRC/towelspring26_2
# Recommended: EPISODE_TIME_S=20  RESET_TIME_S=10
# To activate: uncomment the REPO, TASK and one POLICY_PATH below,
#              comment out the CARTON BOX section.
# =============================================================================
# REPO=${REPO:-ETHRC/eval_towelspring26_test}
# TASK=${TASK:-Fold the towel.}
#
# -- Baraq / previous runs --
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2/checkpoints/last} # WORKS WELL at night with light ON, NOT in daylight
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/realsense_1/checkpoints/last} # trained on towelspring26_3-trimmed
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/realsense_1_notrim/checkpoints/last}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/run1/pretrained_model}
#
# -- April 17/18 augmentation batch (towelspring26_2) --
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_dark_noise_20260417_224504_74152/checkpoints/last}   # WORKS WELL - daylight tested
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_dark_shadow_20260417_224504_74152/checkpoints/last}  # WORKS WELL - daylight tested
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_dark_blur_20260417_224504_74152/checkpoints/last}    # NOT WORKING
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_no_aug_20260417_224504_74152/checkpoints/last}       # NOT WORKING
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/run2_augmented_20260417_224504_74152/checkpoints/last}    # NOT WORKING

# =============================================================================
# TASK: CARTON BOX CLOSING
# Dataset: ETHRC/yams-carton-box-closing-mon-tom-mat  |  EPISODE_TIME_S=120  |  RESET_TIME_S=10
# To activate: uncomment the REPO, TASK, EPISODE_TIME_S, RESET_TIME_S and one POLICY_PATH below,
#              comment out the TOWEL FOLDING section.
# =============================================================================
REPO=${REPO:-ETHRC/eval_carton_box_test}
TASK=${TASK:-Pick & Place and Closing a Box}
EPISODE_TIME_S=${EPISODE_TIME_S:-120}
RESET_TIME_S=${RESET_TIME_S:-10}
#
# -- April 20 carton box runs --
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_no_aug/checkpoints/last}  # NOT WORKING - doesn't pick up object, lifts too low
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_no_aug_full_20260420_221222_217313/checkpoints/last} # Doesn't work (but seems the best so far)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_dark_noise_strong_full_20260420_221222_217313/checkpoints/last} # Doesn't work - way to jerky (worse than the two previous ones)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_dark_noise_full_20260420_221222_217313/checkpoints/last} # Doesn't work - same as last one
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/carton_dark_shadow_full_20260420_221222_217313/checkpoints/last} # Doesn't work
#
# -- April 22 suspicion-bucket sweep (RUN_ID 20260422_023919_440097) --
# Dataset: ETHRC/yams-carton-box-closing-combined; each bucket uses a different subset:
#   S1=22 eps (strictest), S2=38, S3=53, S4=65, S5=70 (most permissive). x {no_aug, kitchen_sink}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sweep_s5_kitchen_sink_20260422_023919_440097/checkpoints/last}  # 70 eps # doesn't work
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sweep_s4_no_aug_20260422_023919_440097/checkpoints/last}        # 65 eps # doesn't work
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sweep_s4_kitchen_sink_20260422_023919_440097/checkpoints/last}  # 65 eps # doesn't work
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sweep_s3_no_aug_20260422_023919_440097/checkpoints/last}        # 53 eps # doesn't work
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sweep_s3_kitchen_sink_20260422_023919_440097/checkpoints/last}  # 53 eps # doesn't work
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sweep_s2_no_aug_20260422_023919_440097/checkpoints/last}  # 38 eps — most promising from the S1..S5 sweep
#
# -- April 23 mtw120 (RUN_ID 20260422_235757_131115) --
# Dataset: ETHRC/yams-carton-box-closing-mon-tue-wed, 120 eps (122 merged minus aborts 21+71), 15K steps
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw120_no_aug_20260422_235757_131115/checkpoints/last}  # no_aug on full 120-ep dataset
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw120_dark_noise_20260423_014430_153499/checkpoints/last}   #  
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw120_dark_shadow_20260423_014430_153499/checkpoints/last}  # 
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw120_heavy_aug_20260423_014430_153499/checkpoints/last}    # 25k training steps heavy augm
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw120_kitchen_sink_20260423_014430_153499/checkpoints/last} #
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw120_max_aug_20260423_014430_153499/checkpoints/last}    #
#
# -- April 24 mtw95 S3 sweep (RUN_ID 20260424_085516_305845, slow/video path) --
# Dataset: ETHRC/yams-carton-box-closing-mon-tue-wed, 95 eps (122 minus 27 outliers @ S3 threshold suspicion>6.5), 15K steps
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw95_s3_no_aug_20260424_085516_305845/checkpoints/last}        # no_aug — DONE (15000 steps, finished ~09:58)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw95_s3_kitchen_sink_20260424_085516_305845/checkpoints/last}   # kitchen_sink — PARTIAL (10800/15000 = 72%, interrupted ~11:10) — active test target
#
# -- April 24 fast-path sweep (predecoded JPG cache, RUN_ID 20260424_111724_350350) --
# Dataset: ETHRC/yams-carton-box-closing-mon-tue-wed, 95 eps (S3 cut), dark_noise aug, 15K steps
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw95_s3_dark_noise_fast_20260424_111724_350350/checkpoints/last}  # dark_noise — PARTIAL (14400/15000 steps, stopped ~12:09 before final checkpoint)
#
# -- April 24 fast-path continuation (RUN_ID 20260424_fast_continuation, watchdog-launched after above) --
# All use dark_noise aug, 15K steps, predecoded path
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/wed50_dark_noise_fast_20260424_fast_continuation/checkpoints/last}      # 50 eps (wed-tom-elias only, camera not shifted) — DONE (15000 steps, finished 13:06)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/tw69_dark_noise_fast_20260424_fast_continuation/checkpoints/last}       # 69 eps (tue+wed, no aborts) — DONE (15000 steps, finished 14:03)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/m40tw81_dark_noise_fast_20260424_fast_continuation/checkpoints/last}    # 81 eps (mon[40:]+tue+wed) — DONE (15000 steps, finished 15:00)
#
# -- April 24 sebastian_aug (RUN_ID 20260424_163346_sebastian_aug, fast path) --
# Dataset: ETHRC/yams-carton-box-closing-mon-tue-wed, 50 eps (wed-only, eps 72–121), 15K steps.
# NEW AUG "sebastian_aug": 3-transform random-order mix of RandomAffine (±3°, 10% translate)
# + ColorJitter (brightness/contrast/hue/saturation, no sharpness). First run that adds a
# GEOMETRIC transform (affine) on top of the color-jitter profiles used in dark_noise.
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/wed50_sebastian_aug_20260424_163346_sebastian_aug/checkpoints/last}    # wed50 + sebastian_aug (affine+color) — DONE (15000 steps, finished 17:38)
#
# -- April 24 translation_sweep (follow-up to wed50_sebastian_aug, NOT YET TRAINED as of this edit) --
# 3-run sweep varying one axis against wed50_sebastian_aug. Paths will exist once the sweep runs.
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mw_sebastian_aug_<RUN_ID>/checkpoints/last}        # mon+wed, 10% translate — does adding mon help?
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/mtw_sebastian_aug_<RUN_ID>/checkpoints/last}       # all 3 days, 10% translate — can aug bridge tue outlier?
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/wed50_sebastian_aug_t05_<RUN_ID>/checkpoints/last} # wed only, 5% translate — is 10% overkill?

# -- April 25-26 30-run sweep, Phase 1 (RUN_ID 20260425_194751_231046, fast/predecoded path) --
# Dataset: ETHRC/yams-carton-box-closing-sat-michael-mat-varing-fan-position-25-04-2025
# Saturday-only suspicion-tier ablation. 5 tiers (T1=strictest, T5=lenient) × {no_aug, kitchen_sink}.
# 15K steps each. Cell 10 (T5+kitchen_sink) skipped — covered by sat_michael_mat_fan_v1_fast below.
# Plan: ~/.gstack/projects/Desktop/tommaso-no-branch-eng-review-plan-20260425-184351.md
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T1_strictest_no_aug_20260425_194751_231046/checkpoints/last}         # not working # cell 1: T1=130 eps (drops top-6 suspicion: 12,2,17,5,76,3), no_aug — strictest filter, no regularization
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T1_strictest_kitchen_sink_20260425_194751_231046/checkpoints/last}  # cell 2: T1=130 eps, kitchen_sink (4/9) — aug-on-clean delta
#POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T2_strict_no_aug_20260425_194751_231046/checkpoints/last}            # cell 3: T2=131 eps (drops top-5: 12,2,17,5,76), no_aug — direct comparator to p2_11
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T2_strict_kitchen_sink_20260425_194751_231046/checkpoints/last}     # cell 4: T2=131 eps, kitchen_sink
#POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T3_balanced_kitchen_sink_20260425_194751_231046/checkpoints/last}   # BEST SO FAR # cell 6: T3=132 eps, kitchen_sink — likely best Phase-1 candidate # BEST SO FAR
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T4_lenient_no_aug_20260425_194751_231046/checkpoints/last}          # BEST SO FAR # cell 7: T4=133 eps (drops top-3: 12,2,17), no_aug
#POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T4_lenient_kitchen_sink_20260425_194751_231046/checkpoints/last}    # cell 8: T4=133 eps, kitchen_sink — does aug compensate for noisier data?
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_T5_only_abort_no_aug_20260425_194751_231046/checkpoints/last}       # cell 9: T5=135 eps (drops only manual-abort ep 12), no_aug — max-data Saturday baseline

# -- April 26 30-run sweep, Phase 2 (RUN_ID 20260425_195046_232159, fast/predecoded path) --
# Dataset: ETHRC/yams-carton-box-closing-mtw-sat (258 eps merged: Sat 0-135, Mon 136-187, Tue 188-207, Wed 208-257)
# Orthogonal layout — size arm (cells 11-15, no_aug) and aug arm (cells 16-20, full ~251-ep set).
# Cells 14 + 16 share data (full no_aug vs full kitchen_sink) and form the no-aug-vs-aug delta.
# Cell 20 (max_aug, 35K steps) STILL TRAINING as of 2026-04-26 18:00 — path will exist when finished.
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_11_sat_only_no_aug_20260425_195046_232159/checkpoints/last}        # cell 11: sat T2 only (131 eps), no_aug — size-arm baseline; same data as Phase-1 cell 3
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_12_sat_mon_no_aug_20260425_195046_232159/checkpoints/last}         # cell 12: sat T2 + mon (182 eps), no_aug — Sat→Sat+Mon size benefit
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_13_sat_mon_tue_no_aug_20260425_195046_232159/checkpoints/last}    # cell 13: sat T2 + mon + tue (201 eps), no_aug — adds 19 Tue eps
#POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_14_full_no_aug_20260425_195046_232159/checkpoints/last}            # cell 14: full mtw-sat (251 eps), no_aug — multi-day no-aug baseline; PAIRS with cell 16
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_15_susp_filtered_no_aug_20260425_195046_232159/checkpoints/last}   # cell 15: full minus suspicion top-quartile (~190 eps), no_aug — multi-day filter generalization
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_16_full_kitchen_sink_20260425_195046_232159/checkpoints/last}      # cell 16: full mtw-sat (251 eps), kitchen_sink (4/9) — aug-arm baseline; PAIRS with cell 14
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_17_full_dark_noise_20260425_195046_232159/checkpoints/last}        # cell 17: full mtw-sat (251 eps), dark_noise — lighting-focused
#POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_18_full_dark_shadow_20260425_195046_232159/checkpoints/last}       # cell 18: full mtw-sat (251 eps), dark_shadow — occlusion-focused
#POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_19_full_heavy_aug_20260425_195046_232159/checkpoints/last}         # cell 19: full mtw-sat (251 eps), heavy_aug (6/9), 25K steps — heavier regularization
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p2_20_full_max_aug_20260425_195046_232159/checkpoints/last}            # cell 20: full mtw-sat (251 eps), max_aug (8/9), 35K steps — TRAINING IN PROGRESS

# -- April 27 30-run sweep, Phase 3 (RUN_ID 20260425_195053_232215, fast/predecoded, NOT YET TRAINED) --
# Dataset: ETHRC/yams-carton-box-closing-mtw-sat with explicit train/val episode-list splits.
# Day-combination ablation. Each cell trains on a _train list; matching _val list is on disk at
# analytics/output/sweep_lists/p3_*.val.episodes.txt — reserved for real-robot rollout eval (TODO 3).
# All cells use kitchen_sink aug, 15K steps. Will exist after Phase 2 finishes (~05:00 2026-04-27).
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p3_21_sat_train_mon_val_20260425_195053_232215/checkpoints/last}              # cell 21: train sat T2 (131), val mon (51) — Sat→Mon transfer
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p3_22_mon_train_sat_val_20260425_195053_232215/checkpoints/last}              # cell 22: train mon (51), val sat T2 (131) — Mon→Sat transfer (smallest train; watch issue #2853)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p3_23_sat_mon_train_tue_val_20260425_195053_232215/checkpoints/last}          # cell 23: train sat+mon (182), val tue (19) — combined vs Tue holdout
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p3_24_sat_mon_tue_train_wed_val_20260425_195053_232215/checkpoints/last}      # cell 24: train sat+mon+tue (201), val wed (50) — operator generalization to Wed
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p3_25_random80_seed42_20260425_195053_232215/checkpoints/last}                # cell 25: random 80% (204), val 20% (51) — within-distribution generalization
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p3_26_sat_train_rest_val_20260425_195053_232215/checkpoints/last}             # cell 26: train sat T2 (131), val mon+tue+wed (120) — Sat alone vs other days

# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/sat_michael_mat_fan_v1_fast_20260425_155404_183070/checkpoints/last}   # carton box saturday — Phase-1 cell-10 stand-in (T5 + kitchen_sink-style aug) FIRST NEARLY WORKING VERSION!!!!!!!!!!

# -- April 27 Phase 4 steps-curve (RUN_ID 20260427_115142_420656, fast/predecoded path) --
# Dataset: ETHRC/yams-carton-box-closing-sat-michael-mat-varing-fan-position-25-04-2025
# T4 lenient (drops eps 2,12,17) + kitchen_sink aug, 90/10 holdout split (seed=42),
# 120 train eps / 13 held-out eps {1,10,19,24,29,34,50,52,75,101,107,110,126}, 35K steps.
# Held-out list: analytics/output/sweep_lists/sat_T4_lenient_holdout10_seed42.episodes.txt
# Checkpoints: 5K, 10K, 15K, 20K, 25K, 30K, 35K (==last) — for steps-vs-success-rate sweep.
POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/last}   # 35K steps (final) — primary eval target on the 13 holdout eps
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/030000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/025000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/020000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/015000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/010000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_steps_curve_T4_kitchen_sink_holdout_20260427_115142_420656/checkpoints/005000}

# -- April 29 Phase 4 aug sweep (RUN_ID 20260429_224030_52752, fast/predecoded path) --
# Same dataset + T4 lenient holdout split as 04-27 run. 4 runs trained sequentially overnight:
#   1) kitchen_sink   60K  (120 holdout-train eps, save every 5K)
#   2) dark_noise     60K  (120 holdout-train eps, save every 10K)
#   3) dark_shadow    70K  (120 holdout-train eps, save every 10K)
#   4) affine_dark   100K  (133 eps incl. holdout — production candidate, save every 10K)
# Eval runs 1-3 on the 13 held-out eps {1,10,19,24,29,34,50,52,75,101,107,110,126}.
# Run 4 trained on those — pick a different eval set.
# Run 1: kitchen_sink 60K (longer-train follow-up to current best)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_kitchen_sink_60k_holdout_20260429_224030_52752/checkpoints/last}     # 60K (final)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_kitchen_sink_60k_holdout_20260429_224030_52752/checkpoints/055000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_kitchen_sink_60k_holdout_20260429_224030_52752/checkpoints/050000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_kitchen_sink_60k_holdout_20260429_224030_52752/checkpoints/045000}
# Run 2: dark_noise 60K
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_dark_noise_60k_holdout_20260429_224030_52752/checkpoints/last}      # 60K
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_dark_noise_60k_holdout_20260429_224030_52752/checkpoints/050000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_dark_noise_60k_holdout_20260429_224030_52752/checkpoints/040000}
# Run 3: dark_shadow 70K
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_dark_shadow_70k_holdout_20260429_224030_52752/checkpoints/last}     # 70K
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_dark_shadow_70k_holdout_20260429_224030_52752/checkpoints/060000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_dark_shadow_70k_holdout_20260429_224030_52752/checkpoints/050000}
# Run 4: affine_dark 100K — FULL T4-lenient (133 eps, holdout reincluded)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_affine_dark_100k_FULL_20260429_224030_52752/checkpoints/last}       # 100K (production candidate)
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_affine_dark_100k_FULL_20260429_224030_52752/checkpoints/090000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_affine_dark_100k_FULL_20260429_224030_52752/checkpoints/080000}
# POLICY_PATH=${POLICY_PATH:-/home/ethrc/Desktop/training/checkpoints/act/p4_aug_sweep_affine_dark_100k_FULL_20260429_224030_52752/checkpoints/070000}

LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
LEFT_CAN=$(yq '.follower.left_arm.can_port' "$YAML")
RIGHT_CAN=$(yq '.follower.right_arm.can_port' "$YAML")
LEFT_SERVER=$(yq '.follower.left_arm.server_port' "$YAML")
RIGHT_SERVER=$(yq '.follower.right_arm.server_port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")
CAMERA_PATHS=$(yq -r '.cameras.configs[] | select(has("index_or_path")) | .index_or_path' "$YAML")
INTERRUPTED=false

if [ "$NEW_REPO" = "true" ]; then
    RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
    REPO="${REPO}_$RUN_ID"
    echo "NEW_REPO=true: writing this eval to $REPO"
fi
DATASET_BASE_DIR=${DATASET_BASE_DIR:-"$HOME/.cache/huggingface/lerobot"}
DATASET_ROOT=${DATASET_ROOT:-"$DATASET_BASE_DIR/$REPO"}

cleanup_zero() {
    echo "Signal received: moving follower arms to zero"
    pgrep -f "lerobot-record|lerobot-teleoperate" | grep -vx "$$" | xargs -r kill
    PYTHONPATH=src uv run python -m utils.move_arms_zero
}

trap 'INTERRUPTED=true; [ -n "${LEROBOT_PID:-}" ] && kill -INT "$LEROBOT_PID" 2>/dev/null || true' INT TERM

[ -d "$POLICY_PATH/pretrained_model" ] && POLICY_PATH="$POLICY_PATH/pretrained_model"

PYTHONPATH=src uv run python -c "from utils.connection import _free_port; _free_port('$LEFT_PORT'); _free_port('$RIGHT_PORT'); _free_port(int('$LEFT_SERVER')); _free_port(int('$RIGHT_SERVER'))"
bash third_party/i2rt/scripts/reset_all_can.sh
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

for camera in $CAMERA_PATHS; do
    dev="$(readlink -f "$camera")"
    current_ae=$(v4l2-ctl -d "$dev" --get-ctrl=auto_exposure 2>/dev/null | awk -F: '{print $2}' | tr -d ' ')
    if [ "$current_ae" != "1" ]; then
        ./scripts/set_camera_profile.sh "$dev"
    else
        echo "Camera profile already applied to $dev, skipping"
    fi
done

if [ "$RESUME" != "true" ] && [ -d "$DATASET_ROOT" ]; then
    read -r -p "ATTENTION: You set resume to false. DELETE YOUR ENTIRE DATASET at $DATASET_ROOT?? [y/N] " confirm
    [ "$confirm" = "y" ] || [ "$confirm" = "Y" ] || exit 1
    rm -rf "$DATASET_ROOT"
fi

export PYNPUT_BACKEND_KEYBOARD=uinput
export PYNPUT_BACKEND_MOUSE=dummy
uv run lerobot-record \
    --robot.type=bi_yams_follower \
    --teleop.type=bi_yams_leader \
    --teleop.left_arm_port="$LEFT_PORT" \
    --teleop.right_arm_port="$RIGHT_PORT" \
    --robot.left_arm_can_port="$LEFT_CAN" \
    --robot.right_arm_can_port="$RIGHT_CAN" \
    --display_data=false \
    --dataset.fps="$DATASET_FPS" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.single_task="$TASK" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --dataset.vcodec="$VCODEC" \
    --robot.cameras="$cameras" \
    --dataset.streaming_encoding=true \
    --policy.path="$POLICY_PATH" \
    --play_sounds=false &
LEROBOT_PID=$!
wait "$LEROBOT_PID"
status=$?
trap - INT TERM

if $INTERRUPTED || [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then
    trap '' INT TERM
    cleanup_zero
fi

exit "$status"