# Cliff drop: 3,000-update RunPod report

**Date:** 2026-09-26  
**Task:** `Mjlab-CliffDrop-Flat-MicroDuck`  
**Verdict:** The run did not establish the requested skill. It produced occasional
logged landing completions, but the drop-height curriculum stayed near its
minimum. No successful upper-platform-to-lower-platform rollout was visually
verified. Do not deploy this policy or resume it unchanged for a longer run.

## What ran

- Code: `baboluoli/microduck_rl`, branch `codex/hole-descent`, training commit
  `22f49d1` (the later `2c6073a` commit only corrected documentation).
- RunPod Pod `vfu4f830huzjn3`: one RTX 4090, 2,048 training environments,
  50 GB persistent `/workspace` volume, PyTorch CUDA image. The Pod was
  stopped at 21:35 UTC, then deleted after the local artifact copies were
  verified. RunPod listed no remaining Pods or separate network volumes.
- A 64-environment, 5-iteration smoke run completed first. Then the main run
  completed 1,000 PPO updates, saved `model_999.pt`, and resumed for 2,000
  additional updates. The resumed runner's final checkpoint is
  `model_2998.pt`; the resumed iteration labels overlap the first run at 999.
- W&B ran offline. TensorBoard event files, configuration snapshots, ONNX
  files, and checkpoints were copied to ignored local
  `logs/rsl_rl/cliff_drop/` directories before the Pod stopped. The two run
  directories are `2026-09-26_20-12-42_cliff-drop-first-1000` and
  `2026-09-26_20-40-00_cliff-drop-to-3000`.
- Pod uptime was about 96 minutes at the reported $0.74/hour GPU rate, or
  roughly **$1.18 for GPU time**. This is an estimate, not a billing receipt;
  storage charges are separate.

## Evidence from saved TensorBoard events

These are *single-iteration logged values*, not measured success percentages.
The `Episode_Termination/cliff_landing` metric can be positive for training
episodes that began in midair, so it does not prove the full walk-off sequence.
`Curriculum/terrain_levels` is the **mean level across 2,048 environments**;
the generator has levels 0–9.

| Iteration label | Landing termination metric | Mean terrain level | Mean training reward |
| ---: | ---: | ---: | ---: |
| 250 | 0 | 0 | 12.27 |
| 500 | 0 | 0 | 13.15 |
| 750 | 0 | 0.0015 | 18.15 |
| 999, end of stage 1 | 0 | 0.0097 | 13.17 |
| 1,500 | 0.2083 | 0.0236 | 5.69 |
| 2,000 | 0.0417 | 0.0112 | 1.67 |
| 2,500 | 0.0417 | 0.0221 | -1.78 |
| 2,998, final | 0.1667 | 0.0286 | -0.91 |

The first nonzero landing metric occurred at iteration **667**. The mean
terrain level peaked at only **0.0682** during the whole continuation and
ended at **0.0286**. Almost all environments therefore remained on the first
curriculum row, whose drops are near the 2 cm minimum; the intended 2–10 cm
range was not learned or evaluated. The final 100 logged mean rewards averaged
**3.32**, versus **14.21** over the last 100 iterations of stage 1. The
reward curve did not show steady improvement with more training.

At the final logged iteration, `fallen_too_long` was **7.4167** and
`cliff_landing` was **0.1667** in their respective termination metrics.
Their exact units are logger-defined, so these should not be presented as
episode percentages. `nan_state` was zero at the final iteration, though it
was nonzero in some earlier windows.

## Why this does not demonstrate the goal

1. The task starts 35% of training episodes already falling beyond the edge.
   The logs do not separate those episodes from upper-platform starts. Logged
   completions may therefore reflect recovery from the injected midair state
   rather than walking off the cliff.
2. The terrain curriculum increments only after the strict success gate and
   decrements on every other reset. Rare successes did not move the population
   beyond the easiest row. This is a plausible explanation for the near-zero
   mean level, not a proven diagnosis of the robot's motion.
3. The completion gate requires progress 40 cm beyond the edge, foot contact,
   upright and settled velocity, 0.3 s of continued forward motion, and no
   head/trunk impact above thresholds. The event log does not show which part
   failed most often. The inherited walking rewards can rise even when the
   maneuver fails.
4. No per-height success evaluation or force-trace review was performed.
   Headless video capture failed first because the Pod image lacked the EGL
   loader; after installing `libegl1` and `libgles2`, playback reached a
   `viser` GUI slider assertion. The likely trigger is the play command's
   forward range `(0.18, 0.24)` excluding the GUI slider's zero initial value.
   That cause still needs verification. **No rollout video was produced.**

## Next session: suggested order

1. **Fix and inspect playback before another paid run.** Reproduce the
   `viser` slider assertion with the saved final checkpoint. Make play mode's
   forward-command GUI range accept its initial value while preserving the
   intended forward command, then view several episodes from the upper
   platform. A headless evaluator can avoid the GUI entirely. On a fresh
   RunPod image, offscreen video also needs `libegl1`, `libgles2`, and
   `MUJOCO_GL=egl`.
2. **Measure the actual failure stages.** Evaluate both `model_999.pt` and
   `model_2998.pt` from upper-platform starts, separately at 2, 4, 6, 8,
   and 10 cm drops. Record: reaches edge, leaves platform, lands feet-first,
   peak head/trunk/servo forces, stays upright, and walks after impact.
   Report denominators and sample counts. Evaluate midair starts separately.
3. **Instrument the training task.** Log the same stage outcomes and actual
   sampled drop height per episode. In particular, identify which condition
   in `cliff_landing_complete` blocks most otherwise promising landings.
4. **Revise the curriculum and reward only after that diagnosis.** The first
   candidates are less punitive terrain downshifts, a progression criterion
   based on measured success over a window, and a dedicated early landing
   stage with more recoverable midair states. Then restore upper-platform
   starts and require the whole sequence. Keep an explicit evaluation split
   across heights so a policy cannot appear to improve solely at 2 cm.
5. Run another 5-iteration smoke and a short capped pilot with evaluation
   checkpoints. Do not default to a longer continuation of this checkpoint.

## Local artifacts

The headless first-stage comparison is implemented in
[`scripts/eval_cliff_drop.py`](../scripts/eval_cliff_drop.py). On a machine with
the project's mjlab dependencies installed, run:

```bash
uv run python scripts/eval_cliff_drop.py \
  logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt \
  logs/rsl_rl/cliff_drop/2026-09-26_20-40-00_cliff-drop-to-3000/model_2998.pt \
  --num-envs 64 --seed 0 --output docs/cliff-eval-2cm-seed0
```

It writes `cliff-eval-2cm-seed0.csv` with one first episode per environment
and `cliff-eval-2cm-seed0.json` with outcome counts. These generated files
are kept locally and ignored by Git.
For paired video, add `--video-env 2` and use
`--output docs/cliff-eval-2cm-video-env2`; this renders the selected first
episode at 25 fps while evaluating the same 64 environments.
Both checkpoints use the same seed,
upper-platform starts, 2 cm drop, 0.21 m/s command, and neutral head/body
commands. `lower_contact` is a stage proxy: foot contact after the trunk has
moved 15 cm beyond the edge. The task's own completion gate determines
`complete`. The inherited `out_of_terrain_bounds` termination was disabled
for this evaluation because it truncated valid outer grid tiles at step one.
The simulator still used the cliff terrain and its physical collisions.

The local CPU run used the project's pinned Python 3.12 environment and
finished all 128 first episodes. Results:

| Checkpoint | Reached edge | Contact proxy | Completed | Median max forward travel |
| --- | ---: | ---: | ---: | ---: |
| 999 | 41/64 | 27/64 | 13/64 | 0.647 m |
| 2998 | 3/64 | 2/64 | 0/64 | 0.260 m |

The continuation appears to have lost the ability to approach the edge under
these fixed conditions. These are simulated episodes at one drop height and
one seed. A paired offscreen recording of environment 2 shows checkpoint 999
walking off the ledge, landing on the lower platform, and continuing forward;
checkpoint 2998 remains on the upper platform through the 10 s timeout:

- `cliff-eval-2cm-video-env2-model_999-env2.mp4` — 6.7 s, task completion.
- `cliff-eval-2cm-video-env2-model_2998-env2.mp4` — 10 s, no edge crossing.

These videos show one paired episode. The per-episode CSV above is the basis
for the 64-episode counts.

- Stage 1: `logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/`
  including `model_999.pt` and `events.out.tfevents.*`.
- Continuation: `logs/rsl_rl/cliff_drop/2026-09-26_20-40-00_cliff-drop-to-3000/`
  including `model_2998.pt` and `events.out.tfevents.*`.
- Both directories are ignored by Git. The setup and launch commands are in
  [`cliff-drop-runpod.md`](cliff-drop-runpod.md).
- The remote Pod and its Pod volume have been deleted. These local ignored
  directories are the remaining checkpoint copies; the report itself is
  committed to Git.
