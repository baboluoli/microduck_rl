# Cliff checkpoint-999 continuation — October 3, 2026

The guarded pilot stopped after **200 additional PPO updates** because the
second block lost approach and landing performance. The retained candidate is
the **100-update checkpoint**, not the last checkpoint. Its recorded replay
shows walking to the edge, stepping down, staying upright and continuing.
It still falls well short of the approach and completion advancement gates.

## Initialization and training changes

Source: `logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt`.
Both the smoke and first pilot block explicitly loaded this file. Logs confirm
counter 24000 and iteration 999 reset to zero while weights, observation
normalizers and optimizer were kept. The second block explicitly loaded the
first block's checkpoint without another counter reset. Training and logging
were offline; no W&B checkpoint download was used.

`MICRODUCK_CLIFF_PROFILE=model999` disables the new heading controller and
restores a zero yaw-rate command. Drops stay at 2 cm. Posture, CoM and smoothing
schedules are frozen immediately before the original 1000-update boundary:
action-rate weight -0.4, head-bias weight 1.0, head command ranges
±(0.17, 0.17, 0.21, 0.047), body ranges
±(0.005, 0.005, 0.005, 0.05, 0.05, 0.05), and trunk/head CoM offsets ±5 mm.
The final source reset's posture stage cannot be reconstructed exactly because
pose schedules use `>=` and other schedules use `>` at counter 24000. These
pre-boundary values were selected explicitly to avoid the simultaneous increase.
Other reward weights and the 35% synthetic midair training resets remain fixed.

A 64-environment, five-update smoke preceded the 2048-environment pilot.
The pilot ran in two 100-update blocks; the planned final 50 updates did not
run. RSL-RL repeats the saved iteration label on resume, so the block-end files
are `model_99.pt` and `model_198.pt`, representing 100 and 200 actual updates.
Their environment counters are 2400 and 4800 respectively.

## Measured results

Each upper-platform battery has 144 episodes: 16 environments × three seeds
(0, 1, 2) × three approach distances (35, 55, 75 cm), all at 2 cm. Another
48 midair-start episodes per checkpoint are reported separately and excluded
from selection. Completion requires the task gate plus an actual lower-floor
foot contact. Historical contact-proxy results are not directly comparable.

| Evaluation | Reached edge | Within corridor | Lower-floor contact | Completed |
| --- | ---: | ---: | ---: | ---: |
| Untouched source, initial baseline | 90/144 | 53/144 | 86/144 | 31/144 |
| Untouched source, first paired battery | 82/144 | 52/144 | 80/144 | 23/144 |
| +100 updates, first paired battery | 100/144 | 56/144 | 99/144 | 34/144 |
| +100 updates, second paired battery | 102/144 | 51/144 | 99/144 | 38/144 |
| +200 updates, second paired battery | 41/144 | 11/144 | 39/144 | 9/144 |

The retained checkpoint completed 23.6% and 26.4% in its two batteries. The
200-update checkpoint completed 6.3% and reached the edge in only 28.5%.
Its regression exceeded the 15-percentage-point stop threshold overall and
within individual seeds. Neither paired gate passed; height remains 2 cm.

GPU runs and rendering replays vary even with the same seed. The source's
repeated results demonstrate that variation. The small improvement at +100
over the initial baseline is not proof of a statistically reliable gain.
The large +200 regression is a reason to retain the earlier candidate.

## Failure evidence and interpretation

At +100, 54/144 episodes were impact-disqualified, and all 54 recorded head
force above 15 N; 44/144 ended before the edge. The median episode mean
heading error among those before-edge failures was 88.1°. At +200, 103/144
ended before the edge, with median episode mean heading error 86.5°; another
24/144 were impact-disqualified. Approach direction and head impacts remain
important measured failure modes.

Regression occurred with curricula frozen, so changes in the old schedules
cannot be assumed to be the sole cause of the earlier long-run regression.
The next investigation should compare PPO/reward behavior between the retained
and regressed checkpoints before another training run or reward change.

The source actor normalizer has yaw-command mean and standard deviation zero,
with 49,152,000 observation samples. RSL-RL normalization adds epsilon 0.01,
so a newly introduced yaw-rate command of 0.6 maps to 60 normalized units.
This is a concrete observation-distribution shift and a plausible explanation
for poor transfer to the newer heading controller. Any future heading-control
experiment should introduce that input gradually and evaluate the transition.
It does not explain the +200 regression here, where yaw commands stayed zero.

## Artifacts and verification

Retained checkpoint:
`logs/rsl_rl/cliff_drop/2026-10-03_19-07-56_cliff-model999-update100/model_99.pt`.

Validated export: `logs/cliff-model999-pilot/selected.onnx`. It accepts 61
observations and returns 14 actions; the graph includes normalization. Across
three input probes, the maximum absolute difference from the loaded PyTorch
actor was 2.794e-7. Details are in `export-validation.json`.

Verified upper-platform completion video:
`logs/cliff-model999-pilot/eval-update100-model_99-upper-complete-2cm-seed0-env9.mp4`.
The recorded episode lasts 4.24 seconds. Other successful and failed replays,
per-episode CSVs, per-seed gates, configs and checkpoints are in the same logs
tree and the checkpoint run directories. `pilot-result.json` records selection.

The original evaluator incorrectly named replays from the sampled outcome.
The initial baseline's video labelled `success` visibly falls after departure;
that filename must not be treated as confirmation of the recorded outcome.
The evaluator was corrected before the paired batteries to store replay
outcomes and name videos accordingly. A regression test covers this mismatch.
The early baseline and first paired CSV also label the command mode
`world_heading` despite the active zero-yaw profile; this metadata issue was
fixed for subsequent evaluations. Their actual command configuration and
episode metrics are unchanged by that label correction.

Six focused configuration/reporting tests passed. All five smoke iterations
and all 200 pilot iterations logged `nan_state = 0`. Smoke and retained-policy
ONNX exports passed checks. No height promotion, publication or deployment ran.

All remote pilot artifacts were copied and extracted locally. Archive:
`logs/cliff-model999-pilot-artifacts.tar.gz`, SHA-256
`dde4e31c26445f4e798f746c170bb0b420732d734a9bcdacc0e5f6bb065c8683`.
Remote and local hashes matched. Local export validation was added afterward.

Runpod pod `ky2i3y3b6vjn5d` used one A40 at the quoted $0.49/hour rate,
18:50:19–19:45:28 UTC, approximately $0.45 GPU time (not a billing receipt).
It is stopped; its 40 GB volume is retained and can incur storage charges.
The earlier unused staging pod was deleted with user approval.

For the reproducible launch recipe, see [cliff-drop-runpod.md](cliff-drop-runpod.md)
and `scripts/run_cliff_checkpoint_pilot.py`.
