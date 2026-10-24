# Run plan: confirm the walking course, then expand height gradually

Date: 2026-10-24. Status: proposed future session, not authorization to start a
pod or training. Requires the completed
[implementation plan](cliff-course-curriculum-implementation-plan-20261024.md).

The objective is one policy that steps down and walks to a finish line before
the end of the lower floor, with unrestricted speed. A valid finish is success,
not a boundary failure. One-metre success is secondary; the old 12-second
survival objective is not reinstated.

## 1. Prepare and approve one bounded session

Use the RunPod CLI with the required `uv run` prefix. Quote the actually available
GPU and current rate, selected checkpoint hashes, one-run scope, and hard
wall-clock cap before requesting fresh approval. Earlier approval and prices do
not authorize this session.

Proposed budget:

- One seed-42 continuation, 2,048 training environments, at most 1,000 new PPO
  updates; save every 100 updates and retain the final checkpoint.
- At most two hours for setup, tests, confirmation, training, evaluation and
  artifact retrieval combined. Reserve retrieval time; the clock wins over the
  update target. If confirmation uses the budget, return its evidence without
  starting or extending the pilot.
- Height capped at 3 cm for this session. Advancement through intermediate
  0.25 cm increments is included only when gates pass. No replication, second
  training arm, automatic extension or higher frontier is included.

Prepare source, commands, watchdog, retrieval and shutdown before approval. Keep
account credentials local. If shutdown relies on the Mac, keep it awake/online
and disclose that dependency. Preserve retained historical volumes.

## 2. GPU preflight and fixed-start confirmation

Follow the RunPod headless EGL/CUDA preflight in RL AGENTS.md before importing
MuJoCo for tests or evaluation. Run the prepared task-specific GPU tests.

Compare unchanged continuation 249, continuation 100 and original 500 at 2 cm,
55 cm start, under the new course-completion rule. Use development seeds 20–22,
24 first episodes each. Keep software/backend/geometry identical. Checkpoint 249
is the lead candidate, not a predetermined winner.

Rank these candidates by full-course success, then fewer disqualifying impacts
before episode end, then higher one-metre success. Exact ties retain the earlier
checkpoint; source 500 precedes continuation 100, which precedes 249. Inspect
matched full-course videos and missing-contact evidence before accepting the
reward/contact implementation. Do not count an unrecorded contact as observed.

Confirm the selected candidate on development seeds 40–42, again 24 first
episodes per seed. A failed confirmation does not authorize repeatedly trying
alternatives against that batch until one passes. Preserve the result and train
at 2 cm if needed. Seeds 40–42 are development confirmation seeds, not final
held-out evidence.

## 3. Confirm the approach-distance range before raising height

Evaluate the selected base at 2 cm and fixed 45, 55 and 65 cm starts. Reuse the
55 cm results only when checkpoint and configuration are identical. Each
height/distance condition uses 72 first episodes on seeds 20–22; advancement
requires confirmation on seeds 40–42 as well.

For every required condition, the proposed gate is:

- At least 65/72 full-course successes overall.
- At least 21/24 successes in every seed.
- At most 3/72 attempts with a disqualifying head/trunk impact before episode end.

These are practical pilot thresholds, not statistical confidence guarantees.
The saved approximately 90% one-metre scores cannot satisfy this new gate.

If 2 cm fails at any planned distance, begin the one continuation at fixed 2 cm
with uniformly sampled 45–65 cm starts. Establish this capability before raising
height. If it passes, begin with heights up to 2.25 cm. Keep the 45–65 cm training
range unchanged throughout the session.

## 4. Smoke, audit rewards, and start the continuation

Run a 64-environment/five-update smoke using the selected source and intended
starting curriculum. Verify actual checkpoint-load and warm-start messages,
63/78/14 dimensions, finite observations/rewards/losses, truthful terrain inputs,
and normalizer-baked ONNX export. Keep full weights, normalizers and optimizer
state; explicitly reset iteration/curriculum counters for the new experiment.

Audit actual weighted, dt-integrated per-episode rewards on the frozen base:
each dense term, positive subtotal, negative subtotal and completion bonus.
Check penalty signs and exact once-only 80-point payout. Synthetic tests must
establish payout correctness even if the frozen policy or smoke earns none.
Record the scale comparison; do not silently change weights during the run.

Start the main pilot independently from the selected unchanged checkpoint, not
from the smoke's final weights. Keep optimizer, other rewards, robot, heading
controller and speed behavior fixed.

## 5. Expand height only after measured success

Sample 75% of training episodes from the newest 0.25 cm height band and 25% from
the admitted easier range. Initially that easier share is fixed at 2 cm. Use
the implemented terrain resolution and log actual sampled heights.

Before training at a new upper limit, evaluate the entering policy at that exact
height. At behavior checks, evaluate the current upper limit and every admitted
endpoint (2, 2.25, 2.5, 2.75, 3 cm as applicable), each at fixed 45/55/65 cm starts.
Never pool heights or distances to conceal a weak condition.

Advance the upper limit by 0.25 cm only when all required conditions pass the
development gate and the same candidate passes the separate confirmation batch.
Otherwise hold the range steady. Do not change the finish distance, impact
thresholds, speed requirements or approach range to manufacture a pass.

Behavior checks start at least seven minutes apart and always include the final
checkpoint. Evaluate completed checkpoint files only. Confirmation and retained
height checks can be expensive: prioritize completing the current gate within
the session cap rather than launching additional stages with incomplete evidence.
If a trained finalist reaches the cap without complete evaluation, mark selection
provisional and reserve held-out tests for a later approved session.

## 6. Select checkpoints and stop on regression

Within a stage, prefer candidates that preserve all mastered height/distance
gates. Among eligible candidates, rank by pooled full-course success at the
current upper limit across the three equally sized distance conditions, then
fewer impacts there, then higher one-metre success. Exact ties retain the earlier
checkpoint. Report every condition alongside the pooled ranking. Do not compare
scores from different upper limits as if they were the same task.

Retain the entering source and stage best. If no candidate preserves mastered
conditions, retain the entering policy rather than promoting a regressed one.

Stop immediately on nonfinite state or losses. For each mastered height/distance
condition, freeze its baseline to the checkpoint/evaluation that established
mastery. Stop after two consecutive completed checks where that same condition
loses at least 15 percentage points of full-course success overall and at least
10 points in two of three seeds versus that baseline. A non-breaching check
clears that condition's streak. Log impacts and failure geometry even when this
rule is not triggered. Rising training reward is never a reason to override it.

At 1,000 updates or the session deadline, stop. Failure to improve is a result,
not permission to extend. Reaching the 3 cm cap does not authorize a higher drop.

## 7. Final review, retrieval and shutdown

When a fully evaluated finalist exists and time permits, evaluate it on reserved
seeds 30–32 across the claimed mastered endpoints and fixed 45/55/65 cm starts.
These results are for final review, not another round of checkpoint selection or
curriculum tuning. Record their use; do not later describe them as unseen.

Report course and one-metre success, impacts, tilt, contact evidence, exits,
timeouts, durations and servo loads by height, distance and seed. Include matched
full-course source/finalist videos and disclose example selection. Distinguish
measured course completion from claims of indefinite walking or hardware readiness.

Download and checksum checkpoints, normalizers, optimizer states, configs, source
snapshots, curriculum history, manifests, raw traces and videos before stopping.
Confirm actual stopped runtime and retire the session watchdog. Report the achieved
range, remaining failures, measured cost/timing and whether another bounded
session is warranted. No further session is automatically authorized.
