# Boundary-semantics A/B and drift telemetry — plan, 2026-10-24

Follows [the failure audit](cliff-course-failure-audit-20261024.md). One pod session, at most two hours
(including a 15-minute retrieval reserve), started with `scripts/launch_boundary_ab.sh POD_ID SSH_HOST SSH_PORT`.

## What runs

1. **Telemetry evaluations** with the new evaluator fields (heading, full command, body/world angular and linear
   velocities): unchanged source 249 (`ef70ae03…`) and the old drifting `model_99` (`ce9f08bb…`). Battery is
   identical to `eval-005`/`eval-006`: heights 2/2.25 cm × starts 45/55/65 cm × seeds 20/21/22 × 24 envs.
   Question answered: does `model_99` drift toward −Y by **turning** (heading error, yaw-rate tracking) or by
   **walking sideways** while facing forward?
2. **100-update continuation** from the same source with the same staged curriculum (upper 22.5 mm, from the
   2026-10-24 `curriculum.json`), seed 42, 2048 envs, warm start — matching the run that produced `model_99`.
   The only intended behavioral change is `course_boundary.time_out=False` (side exits are terminal failures).
   Other code changes since then are logging-only (height-sample counts, evaluator telemetry, NaN forensics).
3. **Evaluation of the new checkpoint** on the same battery.

Compare new versus `eval-006`: negative-Y boundary exits (90), mean Y at the edge crossing (−12.37 cm),
course successes (332/432). One seed: a difference is evidence, not proof.

## NaN handling

Training runs with the NaN guard plus `src/mjlab_microduck/nan_forensics.py`, which also records per-step
controls, solver warm start, accelerations and actuator/constraint forces; policy and processed actions;
contacts and constraint rows of the last finite step and the offending post-step state; per-env randomized
model fields and per-env entity tensors (e.g. BAM `friction_scale`). It writes before the guard's detection
message, so the immediate-stop watcher cannot cut it off. Dumps are copied into `logs/boundary-ab/nan-dumps`.
Any NaN still stops the session.

## Validation status

`tests/test_nan_forensics.py` was written for CPU but **not yet executed**: the authoring session could not
build a Python environment. The launch script runs it, plus the course tests, locally before uploading anything to the pod, and the session runs it again on the pod before evaluation.
