# Handover: course-walk task, trained from scratch — 2026-10-25

**Goal:** train `Mjlab-CliffOracleCourseWalk-MicroDuck` from random weights for
3000 PPO updates at a fixed 2 cm drop. Then evaluate checkpoints under the
unchanged 12 s course rule, and answer one question: **does the robot walk
properly (commanded pace, clean steps) and still cross the cliff?**

The user is new to robotics: explain results in plain language. Per the root
`AGENTS.md`, **ask the user before creating a RunPod pod**. Expected cost is
about $1.50–2.10 on one A40 at $0.59/h, with a **hard 3.5-hour cap**. This is
longer than earlier sessions; get the cap approved explicitly.

## Why: what the hold-height session showed

Full results: `logs/hold-height-20261025/RESULTS.md` (gitignored, local).
Videos: `logs/hold-height-20261025/videos/`.

Fine-tuning at a fixed 2.25 cm still degraded the walking. The causes are in the
reward and configuration, not in the cliff height:

1. **Finishing was punished.** Standing upright earns about 3 points/s
   (`upright`, `head_pose_tracking`, `pose`, `air_time`). Success *ended* the
   episode, and that income ended with it. At 80 points/m and gamma 0.99 at
   50 Hz, the break-even is about 8 cm before the goal. The timeout episodes
   stood **6–8 cm short of the goal** and shuffled sideways for about 5 s
   (side exits 18 → 49).
2. **Rushing was free.** Capped progress paid nothing above the command, but
   charged nothing either, and the 80-point zone after touchdown paid 4x more
   per second than the approach. Speed before the edge was 0.55–0.6 m/s
   against a 0.18–0.3 m/s command. The source policy rushed too (0.58 m/s
   after landing).
3. **Walking schedules were frozen at their easiest stage.** The cliff base
   cfg removed the walking recipe's curricula so that height stages were
   unconfounded. It froze them at stage 0, so smoothing (`action_rate_l2`)
   stayed at −0.1 in every cliff run, **10x weaker** than the −1.0 the source
   walking policy was trained under. This fits the growing jank and
   action-std creep. `AGENTS.md` says to collapse inherited curricula to their
   **final** stage when warm-starting; this did the opposite.

Fine-tuning from `model_99` would carry the rushing and jitter forward. The
user chose to start from scratch.

## What the task does (RL commits after `5f3324f`)

`make_microduck_cliff_course_walk_env_cfg` in `cliff/env_cfg.py` starts from
the course task, which is the same terrain, starts, 63/78/14 interface and
failure terminations. Changes:

- **No success termination and no completion credit.** Every attempt runs to
  the timeout. The timeout is bootstrapped and the policy cannot see time, so
  there is no end to avoid.
- **8 s episodes.** That still leaves about 5 s after touchdown from the
  farthest 65 cm start at about 0.21 m/s, and fits ~1.5x more landings per
  step than 12 s.
- **Velocity tracking from the walking recipe** (weight 2, std² 0.1). It peaks
  at the commanded speed and charges overspeed and sideways motion. Capped
  forward progress stays at half weight (2) as `forward_progress`, so standing
  at the edge pays less than stepping off. At a 0.2 m/s command, per second:
  walk 4.0, rush at 0.6 m/s 2.4, stand 1.3 (locked by a test).
- **Walking schedules restored:** `action_rate_weight` (−0.1 → −1.0 by update
  1500), `com_range`, `head_com_range`, `head_pose_range`, `body_pose_range`
  and `head_pose_bias_weight`, with their stage-0 weights and ranges.
  `standing_envs` stays off, because the course commands forward walking only.
- **Entropy coefficient 0.01** (walking recipe), not the course's 0.003, which
  was a fine-tuning fix.
- **Random initial episode ages** (`MicroduckCourseWalkRunner`), as in the
  walking task. The course runner turns them off so scored attempts get the
  full 12 s. Training attempts here are not scored.
- The other cliff penalties (impacts, `servo_stall`, `servo_acc_spike`,
  `sideways_drift`, `upright_progress`) are kept. Each was under 0.25 points/s
  in the hold-height run, so none is a large tax on exploration.

Known and left as is:
- No random pushes.
- Commands are forward only. The inherited `rel_forward_envs = 0.2` makes
  mjlab raise 20% of commands to **≥ 0.3 m/s**, so the real command mix is
  0.18–0.24 m/s (80%) and 0.3 m/s (20%). Both speed rewards use the actual
  command, so this is consistent. When judging "rushing", compare speed
  against the episode's command in the trace, not against 0.24.

Evaluation is **unchanged**: `scripts/eval_cliff_course.py` builds the 12 s
course cfg with the 1 m rule, so its numbers compare directly with
RESULTS.md. Checkpoints load into either task.

## Run it step by step

Work from `tinyduck/lemming/rl`, with `S=logs/course-walk-scratch-YYYYMMDD`.
Follow steps 1–6 and 9 of `cliff-drift-penalty-run-handover-20261024.md`
(tests and clean tree, pod create, watchdog, SSH, upload, bootstrap with the
EGL preflight, retrieve and stop), including its **Gotchas**. Differences:

- **Pod name** `course-walk-scratch`. **Watchdog deadline** `+12600` s (3.5 h),
  not `+7200`.
- **There are no source inputs to upload.** Only `COMMIT` goes to
  `/workspace/rl/sources/`.
- **Tests (step 1)** also include
  `tests/test_height_curriculum.py`. Expect 114 passed and 2 CUDA-skipped
  locally.
- **Do not use the session runner.** It warm-starts from a checkpoint and
  stops after two failed 2 cm checks, which a from-scratch policy fails for a
  long time. Use plain `train` commands.

On the pod, after the preflight:

```
cd /workspace/rl; export PATH=/root/.local/bin:$PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  WANDB_MODE=offline PYTHONUNBUFFERED=1 MICRODUCK_GIT_HEAD=$(cat sources/COMMIT)
T=Mjlab-CliffOracleCourseWalk-MicroDuck
uv run --with pytest pytest tests/test_cliff_course.py tests/test_nan_forensics.py -q   # CUDA tests run here
# Smoke: builds, steps, every reward term computes, ONNX exports.
uv run train $T --env.scene.num-envs 64 --agent.max-iterations 5 --agent.run-name smoke
uv run python scripts/export.py $T --checkpoint-file logs/rsl_rl/oracle_course_walk/*_smoke/model_4.pt \
  --num-envs 1 --onnx-file /workspace/smoke.onnx
# Speed probe: 20 updates at 4096 envs. Note "Iteration time".
uv run train $T --env.scene.num-envs 4096 --agent.max-iterations 20 --agent.run-name probe
```

Choose the env count from the probe. At **≤ 3.5 s/update**, use 4096 (the
walking recipe's count; 3000 updates ≈ 2.9 h). Otherwise use **2048**. That
was about 2.4 s/update in the hold-height run, so 3000 updates ≈ 2 h. The
curricula advance per update, not per sample, so both counts see the same
schedule. Then:

```
nohup bash -c 'uv run train Mjlab-CliffOracleCourseWalk-MicroDuck --env.scene.num-envs ENVS \
  --agent.max-iterations 3000 --agent.save-interval 250 --agent.seed 42 \
  --agent.run-name course-walk-scratch --enable-nan-guard True > /workspace/train.log 2>&1; \
  echo $? > /workspace/train-exit' >/dev/null 2>&1 &
```

If the run cannot finish 3000 updates before the deadline minus 40 min (for
evaluation and retrieval), stop it at the last saved checkpoint rather than
skipping evaluation.

### Monitor (about every 2 min)

- `train-exit`, plus `grep -E 'Traceback|RuntimeError' /workspace/train.log`.
- NaN resets: `grep -c '\[NanGuard\] NaN/Inf found' /workspace/train.log`.
  `nan_state` resets the robot and training continues. **Stop** if the count
  passes 20, or if any loss or `Mean reward` is nan/inf.
- **Every penalty `Episode_Reward/*` must be ≤ 0.** That covers
  `action_rate_l2`, `sideways_drift`, impacts, `servo_*`, `foot_*`,
  `self_collisions`, `dof_pos_limits` and `body_ang_vel`.
- What healthy learning looks like:
  - `track_linear_velocity` rises.
  - `Mean episode length` climbs toward 400 (8 s), as falls stop.
  - `Episode_Termination/cliff_failed_landing` and `fallen_too_long` fall.
  - Expect dips at updates 500, 750, 1000, 1250 and 1500, where the smoothing
    weight steps up. Dips exactly at those boundaries are pacing, not
    breakage. `Mean action std` should *fall* over the run; record it at
    every 250.
- Total reward can rise on standing rewards alone. If `track_linear_velocity`
  stays flat while `upright` and `head_pose_tracking` grow, the robot is
  learning to stand. Report that rather than letting it run on.

### Evaluate on the pod (12 s course rule)

For `model_1000`, `model_2000` and the final checkpoint (`model_2999`, or the
last saved one):

```
R=logs/rsl_rl/oracle_course_walk/*_course-walk-scratch
uv run python scripts/eval_cliff_course.py $R/model_N.pt --output logs/course-walk-scratch/eval-N \
  --heights .02 .0225 --distances .45 .55 .65 --seeds 20 21 22
```

Copy `/workspace/train.log` into `logs/course-walk-scratch/`, then retrieve
with `--output-relative logs/course-walk-scratch`. Also retrieve the checkpoint
folder, `--output-relative logs/rsl_rl/oracle_course_walk`. Stop the pod and
confirm it stopped. Ask the user before deleting it.

## Analysis (local)

Outcomes and walking quality per evaluated checkpoint:

```
uv run python - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path
from summarize_height_curriculum import outcomes, walking_quality
for d in sorted(Path('logs/course-walk-scratch-YYYYMMDD/artifacts/logs/course-walk-scratch').glob('eval-*')):
    print(d.name, outcomes(d), walking_quality(d))
PY
```

Videos on the Mac CPU, with no pod needed:
`MUJOCO_GL=cgl uv run python scripts/record_course_videos.py CKPT --label scratch-N --height .02 --output $S/videos --with-success`.

Compare against the hold-height **source baseline (block 0)**: 72/71/71 at
2.25 cm, speed 0.435 before the edge and 0.582 after landing, yaw rate 1.19,
time to goal 3.26 s, action std 1.05. Report, without overclaiming (one seed):

- **Pace.** Speed before the edge and after landing, against each episode's
  command (0.18–0.3 m/s). The target is close to the command, not 0.5+.
- **Gait.** Foot air time (`Metrics/air_time_mean`; the reward pays from
  0.125 s, and hold-height block 2 was 0.085 s), action std, yaw rate, and
  what the videos show.
- **Cliff.** Successes, impacts, falls and side exits at 2 cm and 2.25 cm. The
  time to goal should be about 7–9 s at the commanded pace. Slower is
  expected and fine; timeouts are a problem only if the robot stops.
- **Learning curve.** Did 1000 → 2000 → final improve, or regress after the
  smoothing ramp ended at 1500?

The user decides what comes next: the height ladder, more updates, or reward
changes.
