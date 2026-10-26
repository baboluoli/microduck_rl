# Handover: height ladder 2 → 24 cm, wrap-up and next steps — 2026-10-26

**Where things stand:** the course-walk policy now passes the gated ladder up
to **18 cm**. **24 cm** was trained for 2 blocks and missed the gate only at
the 65 cm start. Most of those misses are **timeouts**: the robot lands fine but
walks too slowly to cover the 1 m after landing within 12 s. **The next job:**
fix the evaluation time budget (a user decision is needed, see Part A), then
finish 24 cm in one short session.

The user is new to robotics: explain results in plain language. Per the root
`AGENTS.md` "Vast.ai" section, **ask the user before renting an instance**, for
every session, and destroy it afterwards. No instances are running now (all
destroyed). Vast credit was **$7.30** on 2026-10-26.

This doc supersedes the planning parts of
`cliff-height-ladder-walk-handover-20261025.md`. The procedure there (Part B,
steps 1–10) is still the one to follow.

## The whole ladder, four sessions (all Vast A40, 2048 envs, seed 42)

| Session folder (`logs/…`) | Stages passed | Blocks | Stopped because | Cost |
|---|---|---|---|---|
| `height-ladder-walk-20261025` | 2.5, 3, 3.5, 4, 5 cm | 1 each | time budget | ~$0.95 |
| `height-ladder-walk-20261025-2` | 6, 7, 8 cm | 1 each | stopped by hand (exit 143) | ~$0.60 |
| `height-ladder-walk-20261026` | 9, 10 cm | 1 + 2 | last stage (then 10 cm) mastered | ~$0.47 |
| `height-ladder-walk-20261026-2` | 14, 18 cm; 24 cm not yet | 1 + 2 + 2 | time budget | ~$0.70 |

The source was the from-scratch course-walk `model_2000` (height input zeroed,
see the 2026-10-25 handover). Every session used `--resume-source`, so the
walking schedules stayed at their final stage. Each session continued from the
previous session's `curriculum.json` and `selection.json` `last`.

Walking quality per block (`quality` in `stages.jsonl`, development seeds):

| Top height | Action std | Speed before edge / after landing (m/s) | Yaw rate | Time to goal (s) |
|---|---|---|---|---|
| source `model_2000` (2 cm) | 0.14 | 0.25 / 0.26 | 0.18 | 6.3 |
| 5 cm | 0.23 | 0.26 / 0.28 | 0.33 | 6.2 |
| 8 cm | 0.25 | 0.23 / 0.27 | 0.35 | 6.5 |
| 10 cm | 0.25 | 0.24 / 0.28 | 0.34 | 6.4 |
| 14 cm | 0.25 | 0.21 / 0.27 | 0.36 | 6.9 |
| 18 cm | 0.24 | **0.17** / 0.25 | 0.35 | 7.9 |
| 24 cm (block 5) | 0.24 | **0.14** / 0.26 | 0.40 | 9.0 |

How to read this:
- **Action std** crept up only in the first session (0.14 → 0.23 by 5 cm), then
  stayed flat. The old course sessions crept +0.1 per block; this one did not.
- **Yaw rate** (turning wobble) has crept steadily, 0.18 → 0.40. It is the one
  walking metric still drifting. Watch it.
- **The approach got slower from 14 cm on.** The command is 0.18–0.3 m/s.
  After landing the robot is back to command pace, so this is caution before
  the edge, not a broken gait. The user watched the 24 cm videos and judged
  the landings fine.
- 2 cm regression stayed at 72/72 in every block of every session.
- NaN resets: a few per session, never more than 3 in one block (limit 5). No
  clustering at taller drops.

## 24 cm: what exactly failed

Gate per condition (unchanged): ≥ 65/72 successes, **no seed below 21**, ≤ 3
head/trunk impacts.

| Block | 45 cm start | 55 cm start | 65 cm start | 18 cm regression |
|---|---|---|---|---|
| 4 (`model_3496`) | 70, 2 impacts ✓ | 72, 0 ✓ | **63** (21/22/20), 3 impacts ✗ | 72 ✓ |
| 5 (`model_3596`) | 70, 2 ✓ | 70, 1 ✓ | 67 but **seed 20 = 20/24** ✗ | 70 ✓ |

At the 65 cm start:
- Block 4: 9 misses, of which **6 timeouts** and 3 falls.
- Block 5: 5 misses, of which 2 timeouts and 3 falls (seed 20: 3 falls +
  1 timeout).
- Even the **successes take up to 11.9 s** of the 12 s budget at 24 cm. At
  10 cm they took about 6.5 s. Successes at 18 cm already took up to 11.1 s.

The timeout in the video (`logs/height-ladder-walk-20261026-2/videos/ladder4-b05-h24.00cm-d65-s20-env05-timeout-none.mp4`)
landed cleanly and had walked **0.95 m of the required 1 m** when time ran out.

**Root cause, agreed with the user:** the policy learned a slower, careful
approach for tall drops. That is reasonable behaviour. The evaluation's 12 s
budget was sized for about 0.2 m/s. From the 65 cm start that is now too
tight: 0.65 m at 0.14 m/s is about 4.6 s before the edge, then about 4 s for
the 1 m afterwards, plus start-up and hesitation at the edge.

Two related points:
- **Training episodes are only 8 s** (`env_cfg.py:410`, course-walk task). From
  the 65 cm start the robot reaches the edge after about 5 s. It therefore
  practises only about 3 s of post-landing walking from far starts. This is
  the "Episodes are 8 s in training and 12 s in the evaluation" risk from the
  2026-10-25 handover, now materialised.
- **Slow approach is still under-tracking the command.** The walk reward
  charges it. The policy pays that cost on purpose, which is arguably right for
  a 24 cm ledge. Don't "fix" it with rewards unless the user asks.

## Part A — evaluation time budget (ask the user first)

The user's view (2026-10-26): "it learned the right thing… it just timeouts
because it approaches the edge more slowly… which is a mistake in the env."
They have **not yet chosen** the fix. The project rule is: gate numbers are
experiment choices, **never relax them silently**. So get an explicit choice
and record it. Options, in the order I'd recommend:

1. **Longer evaluation episodes:** 12 → 16 s
   (`make_microduck_cliff_course_env_cfg`, `env_cfg.py:347`, which is what
   `scripts/eval_cliff_course.py` builds. Line 330 is the unrelated walkthrough
   task). The course-walk training cfg builds on this function and then sets
   8 s itself, so training is unaffected.
   - It is one constant, keeps the 1 m-after-landing rule, and doesn't touch
     training.
   - Note that block 5 would *still* fail with it: seed 20 had 3 real falls.
     Only block 4 would have passed. So this removes the false failures but
     doesn't lower the bar.
2. **Also lengthen training episodes** 8 → 12 s (`env_cfg.py:410`), so the
   robot practises walking after landing from far starts. It changes the
   training distribution mid-ladder (fewer landings per update, about −33%).
   Consider it only if timeouts persist after option 1.
3. Not recommended: lowering `GOAL` (1 m). That weakens the "don't just stand
   after landing" rule the user asked for on 2026-10-25.

Whatever is chosen:
- Re-check the 18 cm `mastered` checkpoint under the new rule as part of the
  session baseline. The runner does the baseline automatically.
- Old results stay comparable for successes and impacts. Only timeouts change
  meaning.
- Add or adjust a CPU test that pins the evaluation episode length, and commit
  in the RL submodule (with `nihiluis`, then bump the parent).

## Part B — finish 24 cm (one short session)

Inputs are prepared, local and gitignored, in
`logs/keep/height-ladder-24cm-attempt/`:

```
model_3596.pt                  sha256 a0b47e8e56292f767a326736def26b751790a978a26769c74ae98e2f6ac2d190   (selection "last", 24 cm block 5)
curriculum.json                sha256 11a28a9079f9e19511c9a93617d526353ce31c9a65cf3eff6f3056790973738d   (upper_mm 240, blocks_at_stage 2)
model_3396-18cm-mastered.pt    sha256 343d5835ac0fb1512a01a1a3f7036e3b6e002f2cf7b846dcba00a669cae347b5   (selection "best", 18 cm)
```

- Continue from `model_3596.pt` (the user's choice between `last` and `best`;
  `last` is further along at 24 cm).
- **`blocks_at_stage` is 2**, and the stall stop triggers at 5. So the next
  session has **3 blocks** at 24 cm before it stops on stall. That should be
  plenty. If the user wants more, pass `--max-blocks-per-stage` explicitly
  (and say so) rather than editing the json.
- 24 cm is the **last stage** (`course.STAGES` ends at `.24`), so the runner
  stops with `last stage mastered` once it passes.
- Expected: baseline (~8 min) + 1–3 blocks of ~20 min each (block ~4.5 min +
  development check ~9 min + confirmation ~5 min) + GPU tests and smoke run
  (~5 min). That is **about 45–75 min, $0.30–0.50** at $0.40/h.
- Procedure: `cliff-height-ladder-walk-handover-20261025.md` Part B. The
  session command is the same, with `--source sources/model.pt --source-sha
  a0b47e…` and `--resume-source`.

### What worked smoothly this time (copy it)

- Offer `51338030` (A40, $0.40/h) on host `154.42.3.215`, three rentals in a row.
  Query: `vastai search offers 'gpu_name=A40 num_gpus=1 reliability>0.98 inet_down>200 cuda_max_good>=12.8 disk_space>=40' -o 'dph+'`.
- Image `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` with `--env '-e NVIDIA_DRIVER_CAPABILITIES=all'`, plus
  `apt-get install curl ca-certificates libegl1 libgles2 libgl1 libglx-mesa0 libglib2.0-0`.
  The EGL preflight passed first time, every time.
- Per-session folder `logs/height-ladder-walk-YYYYMMDD[-N]` containing `ssh_config` (copied from the
  previous session, with the new port), `known_hosts` from `ssh-keyscan -p PORT HOST`,
  `instance_id`, `deadline_epoch`, `local.sha`/`remote.sha`.
- Source bundle: `git archive --format=tar.gz -o $S/source.tar.gz HEAD` plus `COMMIT`.
- The watchdog runs locally:
  `uv run --no-sync python scripts/cliff_course_pod_control.py watch --provider vast --pod-id ID --deadline-epoch $D`.
  Kill it after the manual stop.
- After `retrieve`, copy `logs/rsl_rl/oracle_course_walk` with `tar` over SSH
  and diff the sha lists. Then `… stop --provider vast`. After the user
  approves, `vastai destroy instance ID -y`.
- `vastai create instance … --raw` prints an `instance_api_key`. Don't keep
  that output in the session folder.

### Gotchas (all sessions, condensed)

- zsh does not word-split `$C="-F ssh_config"`. Write `-F ssh_config` out in
  full each time.
- `pkill -f "train Mjlab…"` over ssh kills its own shell. Use `[t]rain`.
- `kill -INT` on the runner does nothing (nohup). To stop early, `kill -TERM`
  the process group of the current `train` child. The runner still writes
  `stopped.json` and `SHA256SUMS`.
- `ssh -n` breaks heredocs; use plain `ssh` for `bash -s <<EOF`.
- Pods have no `.git`. `MICRODUCK_GIT_HEAD` must be exported for evaluation.
- The `action_rate_l2` cfg table at the top of each block log prints the cfg
  default −0.1. That is **not** a counter reset. Check the first
  `Episode_Reward/action_rate_l2` values against the previous session's b01
  log (a −0.06, −0.18, −0.30 … ramp within the first iterations is normal).
  The runner also asserts the schedules are final.
- Mac CPU replays differ slightly from the GPU evaluation. A seed that failed
  on the pod may succeed locally; try other seeds/starts to find a failure
  video.

## Changes made in this round (RL commits `884b9d2`, `9bbec23`; parent `42f7c72`, `e267304`)

- `course.HEIGHTS` has **67 rows**. The first 49 (2–10 cm) are unchanged. Then
  105–140 mm at 5 mm and 150–240 mm at 10 mm. Rows are only appended, so old
  checkpoints' `course_height_counts` keep their meaning. Every row is a
  terrain tile in every world; training speed was unchanged at 67 rows (2.4–2.5 s/update logged, as at 49 rows).
- `course.STAGES` = `… .09, .10, .14, .18, .24`. **User decision 2026-10-26:**
  go to 24 cm, larger jumps allowed. Evidence for the jump sizes: the untrained
  10 cm `model_3097` on one CPU seed (55 cm start) landed 24/24 at 12 cm, 21 at
  14, 20 at 17, 17 at 20 and 17 at 24 cm. Probe files:
  `logs/height-ladder-walk-20261026/probe-above-10cm/`.
- Tests updated: `tests/test_height_curriculum.py` (bank size, top stage,
  loop-stop stage). 144 passed, 2 CUDA-skipped.
- The tile thickness is 0.3 m, so drops up to about 29 cm fit without terrain
  changes. Going beyond 24 cm needs new `HEIGHTS` rows and `STAGES`, plus the
  same tests.

## Things the next agent should keep in mind

- **Sim2real warning, already told to the user:** 24 cm is about the robot's
  own height (~25 cm). It lands at about 2.2 m/s, against 1.4 m/s at 10 cm. The
  sim counts head/trunk impacts but doesn't model XL330 gear stress. A clean
  24 cm sim policy may still be harsh on hardware. Suggest testing the 10 or
  14 cm policies on the robot first. Also, this is the **oracle** interface
  (63 inputs, edge distance + drop height given). It is not a deployable
  61-input policy; see the module docstring of `cliff/course.py`.
- **Don't re-zero the height input.** It is trained now. Its normalizer still
  has std ≈ 0 from `model_2000`, so 24 cm reads as about 22 normalized units.
  It has been fine, with no instability at stage changes so far. If 24 cm
  training shows sudden instability at stage changes, rescaling that one
  normalizer dimension is the remedy to *propose*.
- The **2-hour session cap** is hard-coded (`SESSION_CAP_S` in the runner) and
  was a user decision. Don't raise it silently.
- **After 24 cm is done, open questions for the user:** a smoothing or quality
  pass (yaw rate crept 0.18 → 0.40, action std 0.14 → 0.24), and hardware
  testing of a lower-height policy. The policy would also need to become
  deployable: no oracle inputs, ONNX export via `scripts/export.py`.
- Videos are recorded locally (`scripts/record_course_videos.py`, `MUJOCO_GL=cgl`).
  The user likes to watch one success and one failure per new top height.
  Existing ones: `logs/height-ladder-walk-20261026/videos/` (10 cm) and
  `logs/height-ladder-walk-20261026-2/videos/` (24 cm, `model_3596`).

## What to report after the next session

- Did 24 cm pass (development + confirmation)? In how many blocks? With what
  time-budget rule?
- 65 cm-start breakdown: successes, falls, impacts, timeouts, and the slowest
  success duration.
- Regression at 2 cm and 18 cm.
- Walking quality vs. the table above. In particular, did the yaw rate keep
  creeping and did the approach speed recover?
- NaN resets per block.
- Cost, and that the instance was destroyed.
