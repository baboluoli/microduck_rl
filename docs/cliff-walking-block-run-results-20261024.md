# Walking-block pilot results, 2026-10-24

Current objective: [drop, land, walk one metre; revised scores and height progression](cliff-distance-goal-and-height-plan-20261024.md). Walking speed is unrestricted. The old 12-second completion requirement below is historical; the revised training task is not yet installed.

One approved seed-42 continuation completed 250 PPO updates with 2048 environments. Smoke and pilot independently loaded the untouched source-500 checkpoint. The five 8-point intermediate bonuses and 40-point strict final bonus were the only reward revision. The 55 cm start, 2 cm drop and strict 12-second test stayed fixed.

The registered ranking retained **model_100.pt** for review. Selection used strict success, capped clean-prefix mean across all attempts, fewer impact terminations, then old landing rate. Exact ties retained the earlier checkpoint. Only fully evaluated candidates were eligible.

| Evaluated checkpoint | Strict /72 | Clean seconds (capped at 5) | Impacts /72 | Old landing /72 | Boundary /72 | Nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| Frozen source 500 | 0 | 2.431 | 60 | 61 | 1 | 0 |
| model_100 | 1 | 2.319 | 13 | 57 | 56 | 0 |
| model_249 | 1 | 2.014 | 5 | 53 | 63 | 0 |

These are matched RTX A6000 development evaluations, seeds 20–22, 24 first episodes per seed. Historical RTX A5000 source counts (2.25 capped seconds, 54/72 impacts, 0/72 strict) remain separate. The new source scored 2.431 seconds and 60/72 impacts before PPO updates; device/backend differences and rollout variability cannot be attributed to training.

| Checkpoint | Seed | Strict /24 | Capped mean s | Impact /24 | Old landing /24 | Boundary /24 | Nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen source 500 | 20 | 0 | 2.500 | 20 | 20 | 1 | 0 |
| Frozen source 500 | 21 | 0 | 2.375 | 19 | 21 | 0 | 0 |
| Frozen source 500 | 22 | 0 | 2.417 | 21 | 20 | 0 | 0 |
| model_100 | 20 | 0 | 2.542 | 4 | 21 | 20 | 0 |
| model_100 | 21 | 0 | 2.333 | 5 | 18 | 17 | 0 |
| model_100 | 22 | 1 | 2.083 | 4 | 18 | 19 | 0 |
| model_249 | 20 | 0 | 1.708 | 1 | 19 | 23 | 0 |
| model_249 | 21 | 0 | 1.833 | 2 | 16 | 19 | 0 |
| model_249 | 22 | 1 | 2.500 | 2 | 18 | 21 | 0 |

Cumulative clean-walking rates use all attempts, assigning zero to episodes without a clean block. Each cell shows count and percentage; later failure preserves earlier completed blocks.

| Checkpoint | Seed / pooled | ≥1 s | ≥2 s | ≥3 s | ≥4 s | ≥5 s |
|---|---|---:|---:|---:|---:|---:|
| Frozen source 500 | 20 | 19/24 (79.2%) | 18/24 (75.0%) | 14/24 (58.3%) | 7/24 (29.2%) | 2/24 (8.3%) |
| Frozen source 500 | 21 | 19/24 (79.2%) | 17/24 (70.8%) | 15/24 (62.5%) | 4/24 (16.7%) | 2/24 (8.3%) |
| Frozen source 500 | 22 | 19/24 (79.2%) | 17/24 (70.8%) | 14/24 (58.3%) | 6/24 (25.0%) | 2/24 (8.3%) |
| Frozen source 500 | pooled | 57/72 (79.2%) | 52/72 (72.2%) | 43/72 (59.7%) | 17/72 (23.6%) | 6/72 (8.3%) |
| model_100 | 20 | 19/24 (79.2%) | 17/24 (70.8%) | 14/24 (58.3%) | 8/24 (33.3%) | 3/24 (12.5%) |
| model_100 | 21 | 17/24 (70.8%) | 14/24 (58.3%) | 12/24 (50.0%) | 9/24 (37.5%) | 4/24 (16.7%) |
| model_100 | 22 | 16/24 (66.7%) | 13/24 (54.2%) | 11/24 (45.8%) | 7/24 (29.2%) | 3/24 (12.5%) |
| model_100 | pooled | 52/72 (72.2%) | 44/72 (61.1%) | 37/72 (51.4%) | 24/72 (33.3%) | 10/72 (13.9%) |
| model_249 | 20 | 15/24 (62.5%) | 11/24 (45.8%) | 7/24 (29.2%) | 5/24 (20.8%) | 3/24 (12.5%) |
| model_249 | 21 | 14/24 (58.3%) | 11/24 (45.8%) | 8/24 (33.3%) | 7/24 (29.2%) | 4/24 (16.7%) |
| model_249 | 22 | 17/24 (70.8%) | 16/24 (66.7%) | 12/24 (50.0%) | 9/24 (37.5%) | 6/24 (25.0%) |
| model_249 | pooled | 46/72 (63.9%) | 38/72 (52.8%) | 27/72 (37.5%) | 21/72 (29.2%) | 13/72 (18.1%) |

Disjoint capped-prefix histograms (0, 1, 2, 3, 4, ≥5 completed clean seconds):

| Checkpoint | Seed / pooled | 0 | 1 | 2 | 3 | 4 | ≥5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen source 500 | 20 | 5 | 1 | 4 | 7 | 5 | 2 |
| Frozen source 500 | 21 | 5 | 2 | 2 | 11 | 2 | 2 |
| Frozen source 500 | 22 | 5 | 2 | 3 | 8 | 4 | 2 |
| Frozen source 500 | pooled | 15 | 5 | 9 | 26 | 11 | 6 |
| model_100 | 20 | 5 | 2 | 3 | 6 | 5 | 3 |
| model_100 | 21 | 7 | 3 | 2 | 3 | 5 | 4 |
| model_100 | 22 | 8 | 3 | 2 | 4 | 4 | 3 |
| model_100 | pooled | 20 | 8 | 7 | 13 | 14 | 10 |
| model_249 | 20 | 9 | 4 | 4 | 2 | 2 | 3 |
| model_249 | 21 | 10 | 3 | 3 | 1 | 3 | 4 |
| model_249 | 22 | 7 | 1 | 4 | 3 | 3 | 6 |
| model_249 | pooled | 26 | 8 | 11 | 6 | 8 | 13 |

Full, uncapped clean-prefix histograms (completed seconds: attempts):

~~~json
{
  "Frozen source 500": {
    "0": 15,
    "1": 5,
    "2": 9,
    "3": 26,
    "4": 11,
    "5": 3,
    "6": 2,
    "7": 1
  },
  "model_100": {
    "0": 20,
    "1": 8,
    "2": 7,
    "3": 13,
    "4": 14,
    "5": 6,
    "6": 2,
    "7": 1,
    "8": 1
  },
  "model_249": {
    "0": 26,
    "1": 8,
    "2": 11,
    "3": 6,
    "4": 8,
    "5": 7,
    "6": 3,
    "7": 3
  }
}
~~~

Boundary exits reconstructed from saved pre-reset root positions:

~~~json
{
  "Frozen source 500": {
    "forward X": 0,
    "backward X": 0,
    "lateral Y": 1
  },
  "model_100": {
    "forward X": 55,
    "backward X": 0,
    "lateral Y": 1
  },
  "model_249": {
    "forward X": 63,
    "backward X": 0,
    "lateral Y": 0
  }
}
~~~

A lower impact count can reflect episodes ending at the tile boundary before later impacts could occur. Forward boundary exits are censored failures under the unchanged test; they do not establish 12-second success. Checkpoint 100 had 55 forward exits and one lateral exit, while its capped mean fell slightly. Its one strict success is a development-set observation, not evidence of reliable sustained walking.

Regression decisions used the fixed same-session source for walking/impact and retained best for strict success. Each condition had an independent consecutive-check counter. Early-stop rule triggered: False. Final streaks: {'strict': 0, 'walking': 0, 'impact': 0}. The 250-update budget ended the experiment; no extension, replication or held-out evaluation ran.

The final checkpoint was evaluated after the required seven-minute interval. Held-out seeds 30–32 remain unused for checkpoint selection and are reserved for a separately reviewed finalist.

The [pre-training CUDA reward audit](cliff-walking-block-reward-audit-20261024.md) records all term means and distributions. All 54 GPU tests passed, as did the 64-environment/five-update smoke and finite normalized ONNX export. The numerical monitor was independently active during candidate evaluation; its log-pattern tests passed locally. The current local runner automatically starts that same sidecar; this session started it separately without altering PPO.

Artifacts are under logs/walking-blocks-20261024/artifacts/. result.json and progress.json record ranking and regression decisions; each evaluation contains episodes.json, reward-audit.json, per-seed raw NPZ traces and a software/checkpoint/source manifest. All source, smoke and pilot checkpoints, configs, per-update stdout and offline W&B records are retained.

Videos use seed 20/environment 0 chosen in advance, matched cameras following the robot, and show full first rollouts until physical termination or timeout. They are separately rendered CUDA replays, so their outcomes may differ from the authoritative batched evaluation. Sidecar JSON records each replay outcome. No frozen terminal frames or early-cut landing-success clips are used.

The retained best ONNX includes observation normalization and remains an experimental 63-input/14-output policy. It is not compatible with the current 61-input robot runtime.

The preselected source replay ends at 4.16 seconds with an impact; the selected-best replay reaches the forward boundary at 6.38 seconds without a head/trunk impact, with maximum recorded tilt 17.0 degrees. The latter does not record the original landing milestone (maximum stable hold 0.08 seconds; historical lower-contact classifier reports no lower contact), so it is not strict success. These observations describe separate illustrative replays and must not replace the 72-attempt evaluation or be interpreted as 12-second performance.

One secure RTX A6000 pod, 81fc2f5bkg543b, ran at $0.59/hour from 15:16:47 to 15:57:05 UTC (about 40 minutes 18 seconds, approximately $0.40 GPU compute excluding storage and billing adjustments). EGL/CUDA preflight passed with the documented image/bootstrap. Runtime details are retained in platform.json: PyTorch 2.9.1+cu128, CUDA 12.8, MuJoCo 3.10.0, Warp 1.12.0 and NVIDIA driver 595.91.07.

All 337 artifact checksums and the complete archive checksum were verified locally before shutdown. Source copies and the untouched original match SHA256 4b1eacacdab3773c8000a47416a30ed693d8d141915c258e2b0948b59702c651. The retained ONNX also passed the local ONNX checker and a finite 63-to-14 CPU inference check. RunPod confirmed desiredStatus=EXITED, runtimeStatus=stopped, reason=stopped_by_user. The volume is retained; the local session watchdog has been retired.

The next decision is review of the saved behavior and the interaction between fast forward traversal, the finite tile, and the original landing hold before authorizing further work. The current experiment does not demonstrate reliable strict completion or justify automatic training extension. Its lower impact count, lower old landing rate and increased boundary exits must be considered together.

Follow-up spawn/speed audit: all 72 first samples for every evaluated checkpoint place the root at X=0.000043–0.000105 m relative to the spawn, giving 0.549895–0.549957 m to the drop. The upper platform spans terrain X=0–1.5 m, with spawn X=0.95 m, and Y=0 at the width center. The robot is 20 cm forward of the upper platform's longitudinal midpoint, not at the ledge. The low camera projects the near side of the platform close to the feet; the gray/blue transition ahead is the actual drop.

For samples after crossing the edge while tilt stays below 45 degrees, the mean of each episode's mean forward speed is 0.356 m/s for source, 0.445 m/s for checkpoint 100, and 0.409 m/s for checkpoint 249. Evaluation commands 0.21 m/s. The progress reward saturates at commanded speed and has no overspeed decrement; the new block bonus also imposes only a minimum progress threshold. Thus it does not teach the robot to match the slower command. The source was already too fast; attributing the precise speed increase to any one reward term would require another controlled experiment. Raw per-episode speed/spawn calculations are saved in logs/walking-blocks-20261024/speed-and-spawn-audit.json.

The pilot consisted of 250 PPO updates, each collecting 24 steps from 2048 environments: 12,288,000 simulated transitions, with six saved pilot checkpoints. It was a short warm-start pilot, not a convergence study. The measured result supports investigating speed tracking and ensuring the fixed 12-second test fits on the terrain before extending training under the same incentives.
