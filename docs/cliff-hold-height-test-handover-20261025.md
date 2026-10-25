# Handover: fixed-height test of the walking fixes — 2026-10-25

**Goal:** find out whether two training fixes stop the walking from degrading,
with the cliff height held at **2.25 cm** the whole time (no ladder). If the
walking stays clean over 3–4 blocks, the user decides whether to turn the height
ladder back on. This needs a small code change first (Part A), then one pod
session (Part B).

The user is new to robotics: explain results in plain language. Per the root
`AGENTS.md`, **ask the user before creating a RunPod pod**. One session costs at
most about $1.20 (A40 at $0.59/h, hard 2-hour cap).

## Why: what height-curriculum session 1 showed

Full results: `logs/height-curriculum-20261025/RESULTS.md` (gitignored, local).
Videos: `logs/height-curriculum-20261025/fail-videos/`.

- 2.25 cm passed after 2 blocks. At 2.5 cm, block 3 failed only on impacts, and
  block 4 collapsed at every height, including 2 cm. The session stopped on
  regression, as designed.
- The user watched the videos. Their judgement: **"it's not related much to the
  cliff height"**. The robot looks increasingly janky, as if it is unlearning how
  to walk, especially in block 4, and is sometimes fast. In one block-3 video it
  pushes off the edge wall with its foot, like a swimmer, and loses balance.
- The data agrees:
  - Training already got worse late in block 2, while the height was still
    2.25 cm.
  - Half of block 4's 2.5 cm failures were falls on flat ground, before the
    edge.
  - Every impact was the head; the trunk never touched.
- **Speed crept up.** The command is 0.18–0.24 m/s. Speed before the edge went
  0.44 → 0.47 → 0.50 → 0.52 → 0.55 m/s (source, then blocks 1–4). The robot also
  sped up toward the old finish line: block 2 ran 0.47 m/s before the edge and
  0.80 m/s near the finish. Wobble (|yaw rate|) jumped from 1.3 to 1.83 rad/s in
  block 4.
- **Exploration noise crept up.** Action std went 0.96 (source) → 1.05 → 1.19 →
  1.33 → 1.43, about +0.1 per block. Earlier one-block runs showed the same
  creep.

## What changed (RL commit `2828d41`, user-approved 2026-10-25)

1. **Weaker entropy bonus**, course task only: `entropy_coef` 0.01 → **0.003**
   (`MicroduckCliffCourseRlCfg`). Other tasks keep 0.01.
2. **No finish line.** Success is now **1 m of clean walking after the first
   lower-floor touchdown**: no impact, tilt, physical failure or boundary exit.
   - The user's words: "it shouldn't need to reach a finish, just walk a
     minimum distance … to avoid it just standing still after making the drop."
   - The old 3 m finish was unreachable at the commanded pace within 12 s.
   - `course.GOAL = 1.0`. `FINISH = 3.55` remains only as geometry.
   - `course_evaluation.GOAL` is a deliberate duplicate, and a test asserts the
     two agree.
3. **Pay as you go.** The 80-point completion budget is paid per metre over that
   1 m instead of as one lump:
   - Only new ground counts (a high-water mark).
   - Each control step pays at most commanded speed × dt.
   - Distance covered faster than that becomes a backlog. It is paid only if
     the robot slows down later, and it is lost when the episode succeeds.
   - Walking at the commanded pace therefore pays best.
   - The old lump made earlier arrival worth more, because of discounting, and
     the robot can see its own position along the course.
   - Measured on block 2's `model_199`: all 24 runs succeed, but the policy
     collects only about 28/80 points per run because it rushes. That is the
     slow-down signal.

**Results are not comparable with the old rule.** Episodes now end 1 m after
touchdown, much earlier than at the old 3 m finish, so there is less time to
fall or hit the head. Impact counts will drop for that reason alone. The old
`one_metre` column is what the new `success` measures. **Re-baseline the
source under the new rule (A2) before reading any trend.**

## Part A — code (local, before the pod)

Work in `tinyduck/lemming/rl`.

### A1. Hold-height mode in `scripts/run_height_curriculum_session.py`

Add `--hold-height` (flag) and `--blocks N` (default 4).

With `--hold-height`:
- Run the same block → development check → confirmation loop.
- **Never call `state.advance`.** Record what *would* have happened (`would_advance`).
- Disable the stall stop.
- Stop after `N` blocks, or on the time budget.
- **Keep the regression stop.** Two failed checks in a row of 2 cm at 55 cm
  still mean the walking broke, and the session should not keep burning time.
  - At 2.25 cm the regression set is `20/55` only.

Keep `run_blocks` pure and testable. Pass the hold behaviour in as a parameter.
Add CPU tests next to the existing loop tests in `tests/test_height_curriculum.py`:
- hold mode never advances;
- hold mode runs exactly N blocks;
- hold mode still stops on regression.

### A2. Baseline the source under the new rule

Before block 1, evaluate `--source` with the same development check: the frontier
plus regression conditions on seeds 20–22. Write it as `eval-000-*` and as a
`block: 0` line in `stages.jsonl`. Without it, no trend can be read: see
"Results are not comparable" above.

### A3. Record walking-quality numbers per block

The question is "is it unlearning how to walk", so record these in every
`stages.jsonl` entry and print them in `scripts/summarize_height_curriculum.py`:
- **Action std:** the last `Mean action std:` value in the block's training log.
- **Forward speed** (`velocity_body[0]` mean) before the edge (x < 0.5, age > 10)
  and after landing (x > 0.8), from the frontier evaluation traces.
- **Wobble:** mean |yaw rate| (`angular_velocity_body[2]`) before the edge.
- **Time to goal:** mean `duration` of successful episodes.

The definitions above are the ones used for the session-1 numbers in this doc.
Put them in the summary helper and add a small test on a synthetic trace.

### A4. Video tool

Move `logs/height-curriculum-20261025/fail-videos/fail_videos.py` into the repo
as `scripts/record_course_videos.py`.
- It runs on the Mac CPU (`MUJOCO_GL=cgl`) and needs no pod.
- Pass 1 finds failures; pass 2 re-runs the same seed and follows one failing
  env, checking that the replay outcome matches.
- Tidy the argument names. Keep the half-speed caption and the "hold the last
  real frame" fix: after `done`, the env has already auto-reset.

Then run
`uv run --with pytest pytest tests/test_nan_forensics.py tests/test_cliff_course.py tests/test_cliff_course_process.py tests/test_height_curriculum.py -q`.
Commit with `nihiluis` in the RL submodule and bump the submodule in the parent
repo.

## Part B — the session

Follow `cliff-drift-penalty-run-handover-20261024.md` steps 1–9, as session 1
did. Session 1 went smoothly with that procedure; see "Gotchas" there. Differences:

- **Session folder** `S=logs/hold-height-YYYYMMDD`, pod name `hold-height`.
- **Inputs** for `/workspace/rl/sources/`:
  - The drift-penalty `model_99.pt`, the same input as session 1:
    `logs/drift-penalty-20261024/artifacts/logs/rsl_rl/oracle_course/2026-10-24_22-01-47_drift-penalty-100/model_99.pt`,
    sha256 `010a94116935924381dd16769bb4e79f1ee4e723491f6307b97eac097e78cdea`.
    **User decision (2026-10-25):** start from this, not from block 2's
    `model_199`. It is the least degraded policy: action std 1.05 vs 1.19,
    speed before the edge 0.44 vs 0.50 m/s. It is slightly less skilled: 2.25 cm
    passed at the 45/55 cm starts and failed at 65 cm only on impacts (5 > 3).
    The test is about whether walking stays clean, so the cleanest start gives
    the clearest answer. Training from scratch was considered and deferred: it
    is worth its cost only if this session shows the rushing and noise are
    baked in.
  - `logs/course-session-20261024/artifacts/logs/course-session/curriculum.json`
    (`upper_mm` 22.5). Do **not** use session 1's `curriculum.json`: it is at 25 mm.
  - `COMMIT`.
- **Command:**
  ```
  nohup bash -c 'uv run python scripts/run_height_curriculum_session.py --source sources/model_99.pt \
    --source-sha 010a94116935924381dd16769bb4e79f1ee4e723491f6307b97eac097e78cdea \
    --curriculum sources/curriculum.json --output logs/hold-height --deadline-epoch DEADLINE \
    --max-nan-resets 5 --label hold-height --hold-height --blocks 4 > /workspace/session.log 2>&1; \
    echo $? > /workspace/session-exit' >/dev/null 2>&1 &
  ```
- **Timing:**
  - Session 1 measured about 4.5 min per 100-update block (2.6 s/update) on
    the 49-row bank, about 8 min per development check and about 5 min per
    confirmation.
  - Episodes now end sooner (about 3–9 s instead of 5–8+ s), so evaluations
    should be a little faster.
  - The baseline adds about 8 min. Expect all 4 blocks to fit.
- **Monitoring:**
  - As in session 1, plus the new per-block action std and speed.
  - `Episode_Reward/cliff_landing` now accrues per metre. It is still ≥ 0.
    Expect it to start low, because the source rushes and loses backlog, and to
    rise if the robot slows toward the command.
  - Every penalty term must stay ≤ 0.
- **Afterwards:**
  - Retrieve, stop the pod, and ask before deleting it.
  - Record videos locally with `scripts/record_course_videos.py`. Include block
    2/4 failures and one success per block, so the user can judge jankiness by
    eye.

Pods: `ytfxzw61zdoe5k` (session 1) and `823jn94ii5ko8o` (drift-penalty) are
**stopped**. The user has not decided whether to delete them.

## What to report

This is one seed, so differences are evidence, not proof. Report per block,
against the A2 baseline:
- **Action std:** does it stay flat or fall, instead of +0.1 per block?
- **Speed:** does it move toward the command (0.18–0.24 m/s)? Before the edge
  and after landing separately. Is the robot still faster near the goal?
- **Wobble** (|yaw rate|) and **time to goal**.
- **Test results** at 2.25 cm (45/55/65 cm starts) and 2 cm: successes, impacts,
  falls, side exits, timeouts. Does it hold steady, unlike session 1?
- **NaN resets.**
- **Videos:** does it still look janky to the user? Does the swimmer push-off
  off the edge wall still happen?

Then let the user decide:
- Clean walking → turn the ladder back on from the best hold-height checkpoint,
  with a fresh session-1-style run starting at 2.25 cm.
- Still degrading → the fixes were not enough. Candidates, each the user's
  call: reset or cap the action std, an even lower entropy, or stronger
  smoothness penalties.
- Still rushing although rushing no longer pays, or noise not coming down →
  the habits look baked in. Then retraining from an earlier ancestor (the
  original walking policy) becomes worth its cost of several sessions.

## Risks

- **The source's action std is already 1.05** (0.96 before its own 100 updates). A weaker entropy bonus removes
  the push upward but may not bring it down quickly. If it stays high, report
  that; do not silently add a cap.
- **Reward change shock.** The critic was trained on the lump-sum reward, so
  expect block 1 to look worse before it gets better (session 1 saw the same
  after warm starts).
- **Too slow is now possible in principle.** Walking slower than the command
  earns less per second, but the capped progress reward
  (`cliff_forward_progress`, weight 4) still pays for forward travel, and 1 m at
  0.18 m/s fits easily in 12 s. Watch the timeouts.
- **Gate numbers.** The 65/72 success rule, at most 3 impacts and at least 21
  per seed were chosen for the old 3 m course. With shorter episodes they may be
  easier to pass. Report this, and do not tune it silently.
