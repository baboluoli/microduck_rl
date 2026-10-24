# Implementation plan: complete the walking course and gradually increase drop height

Date: 2026-10-24. Status: agreed direction, implementation pending. This document
authorizes no paid session or training run. Execution is described in the
[run plan](cliff-course-curriculum-run-plan-20261024.md).

These two plans supersede the proposed one-metre episode ending, discrete height
stages, and fixed-start-only training in the
[earlier distance plan](cliff-distance-goal-and-height-plan-20261024.md).
Historical results and tasks remain unchanged. One-metre success remains useful
as an intermediate measurement; completing the lower-floor course becomes the
primary objective.

## 1. Add a separate course-completion task

Keep the existing robot, actuator model, heading controller, optimizer,
normalization, other dense reward weights, and experimental 63/78/14
actor/critic/action interface. Preserve all existing user edits and source
checkpoints. Do not add a speed restriction or a new speed objective.

Define the episode as follows:

1. Start on the upper platform and approach the actual drop.
2. Record the first genuine foot contact on the lower floor, checking detected
   contact position against that environment's actual floor geometry and height.
3. Continue walking to a fixed forward finish line on the lower floor.
4. Crossing the line upright and valid ends the episode successfully. A failure
   on the same sample vetoes success. Require the recorded lower-floor contact;
   crossing without it cannot qualify.
5. Retain impact and nonfinite invalidity from reset. From first lower-floor
   contact onward, including that sample, tilt reaching 45 degrees invalidates
   the attempt. Preserve the existing head/trunk thresholds of 15/20 N and
   physical failure terminations. Failure eligibility stays latched.

Place the finish line before the physical forward edge. Determine and document
its exact position and margin during implementation by checking the actual
terrain footprint and foot positions at crossing. Leave enough supported ground
for the feet and one control-step crossing overshoot; do not require walking off
the platform. Keep this geometry identical across heights and start distances.
The course must include at least one metre after normal touchdown. Report late
first-contact cases rather than inventing contact evidence.

Keep the finish line fixed relative to the cliff, not relative to the randomized
start: every episode should require the same lower-floor course. A side exit or
an unexpected forward-boundary exit before qualification is an unqualified,
separately reported truncation, not automatically a physical fall or a success.
Keep such attempts in the all-attempt success denominator.

Use 12 seconds as the initial maximum attempt duration, not a required walking
duration. Verify that it permits the full course from the longest planned
approach. If valid walking is routinely cut short by it, revise and document the
timeout before the pilot; do not quietly impose a minimum speed through an
unreachable deadline.

Pay the existing 80-point completion budget once, on successful termination.
Remove the old block payments and strict-timeout payout from this new task.
Divide point pulses by control dt to account for RewardManager integration.
Success itself must not suppress its reward: mask competing failures, not every
terminal outcome. Use shared, once-per-control-step state and reset only the
episode rows that reset. Ending a simulated episode does not teach a physical
stop command.

## 2. Make height sampling gradual and truthful

Sample a height between episodes and keep the terrain fixed during each episode.
First inspect whether the simulator supports efficient, correct per-reset
geometry changes, including collision data. If not, pre-generate a dense set of
terrain heights and sample among them. State the actual resolution in the run
manifest; do not describe a finite terrain bank as truly continuous sampling.
Ensure the bank includes every curriculum endpoint and evaluation height.

Parameterize terrain geometry, oracle height/distance inputs, lower-floor
contact checks, reset height, evaluator, and video labels from each environment's
actual terrain. The current fixed 0.02 m assumptions cannot remain in these paths.
Check inherited observation-normalizer behavior when exposing new heights; keep
the full checkpoint state rather than silently resetting normalization.

Initial curriculum settings:

| Parameter | Proposed value |
|---|---|
| Minimum practised height | 2 cm |
| First upper limit after 2 cm confirmation | 2.25 cm |
| Upper-limit increment | 0.25 cm (2.5 mm) |
| Harder-height share | 75%, sampled uniformly within the newest 0.25 cm band |
| Easier-height share | 25%, sampled uniformly over the previously admitted range |
| First-session maximum | 3 cm |

At the first expansion, the easier share is fixed at 2 cm. The sampling weights
and increment are experiment choices, not established optima. Advance only after
explicit evaluations pass the run-plan gates. Otherwise hold the range steady.
Persist curriculum limits, stage history and sampling settings with checkpoints.
Training iteration alone must not advance difficulty.

## 3. Add a narrow range of approach distances

Implement both fixed-distance evaluation and per-episode randomized starts.
First confirm the new course at 55 cm. Then train with start distance sampled
uniformly from **45–65 cm**; keep that range fixed while expanding height.

Move the robot's reset position relative to the fixed cliff. Do not move the
edge, finish line or lower-floor course with the robot. Reset the full robot
consistently and keep the initial heading and lateral placement conventions.
Verify truthful signed edge distance immediately after reset for each start.
This should not require a new observation dimension.

Evaluate at fixed 45, 55 and 65 cm starts. Before height expansion, confirm that
2 cm remains reliable across this distance range. If it does not, train at 2 cm
with randomized starts first. Do not expand the distance range and height range
simultaneously. Random starts expose different approach times and footfall phases;
they are not evidence of generalization until the fixed-distance checks pass.

## 4. Implement evaluation, selection and curriculum state

Maintain an independent evaluator agreeing with training on synthetic traces.
Capture state before automatic reset. Report by height, distance and seed:

- Full-course success, one-metre-after-contact success, and genuine contact rate.
- Impact, tilt, nonfinite, boundary and timeout outcomes, with overlapping
  diagnostic events distinguished from terminal outcomes.
- Distance travelled after contact, duration, finish-line position, and servo loads.

Use all attempted first episodes as denominators. Preserve exact contact
found/position/force evidence for both feet, including missing-contact cases.
Keep fixed evaluation anchors independent of the random training sampler.

Implement the run plan's advancement, checkpoint ranking and regression rules.
Evaluate a candidate fully before promoting it. Keep source and best checkpoints
with their normalizers and optimizer state. Save reproducible manifests with
source/software fingerprints, checkpoint hashes, geometry, seeds, heights and
distances. The old one-metre scores do not establish course-completion capability.

## 5. Validate and prepare the executable session

Add focused tests for contact validity at multiple heights, failure-step vetoes,
same-step goal/contact handling, finish-line clearance, once-only reward, dt
integration, manager caching and staggered resets. Test randomized reset geometry,
truthful observations, curriculum sampling/support, advancement gates, ranking,
regression streaks and saved-state restoration.

Prepare CUDA runtime tests for actual train/play resets and dimensions, finite
normalized observations, and success termination. Prepare the mandatory
64-environment/five-update smoke and normalizer-baked ONNX export check. Synthetic
tests must cover terminal payout even if the short smoke never completes a course.

Prepare the runner, integrated reward audit, matched videos, artifact retrieval,
checksum verification and shutdown control before requesting GPU approval. Follow
the RL AGENTS.md headless graphics preflight on the approved pod before GPU tests.

Deliverables: reviewable implementation and local test results, exact course
geometry and curriculum semantics, runnable session commands, and a concrete
one-session approval request. No hardware deployment is included.
