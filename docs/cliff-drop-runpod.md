# Cliff drop: staged experiment

## Latest pilot: steering repair (completed)

The zero-yaw continuation below was stopped after 200 updates for regression.
The subsequent flat-ground diagnostic found substantial heading drift even
when the robot stayed upright. The steering repair pilot improved final 2 cm
completion from 24/144 to 89/144. It stopped at +100 updates for regression
and retained +75. See the [run report](cliff-steering-pilot-report.md) and
[recipe and diagnosis](cliff-steering-repair.md). The GPU is stopped and all
artifacts are local. Higher drops have not been validated.

The new `steering` profile **requires a prepared checkpoint**; do not load an
unmodified zero-yaw source under it. Its yaw normalizer previously had zero
variance, and its unused input weights were not trained for steering.

## Completed pilot: preserve checkpoint 999's approach

The selected source is the local September 26 checkpoint:
`logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt`.
No W&B download is required; set `WANDB_MODE=offline` for logging.

Set `MICRODUCK_CLIFF_PROFILE=model999` for both training and evaluation. This
disables heading correction and freezes the original schedules immediately
before step 24000: action-rate weight -0.4, head-bias weight 1.0, head ranges
±(0.17, 0.17, 0.21, 0.047), body ranges ±(0.005, 0.005, 0.005, 0.05, 0.05,
0.05), and trunk/head CoM offsets ±0.005 m. The checkpoint stores counter
24000; different schedules use `>` and `>=`, so the final reset's posture
stage is ambiguous. These explicitly selected values avoid crossing the
1000-update boundary. Other reward weights and synthetic midair resets stay
unchanged pending measured failure modes.

Evaluate the untouched source first using the current true-contact evaluator:

```bash
MICRODUCK_CLIFF_PROFILE=model999 uv run python scripts/eval_cliff_drop.py \
  logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt \
  --heights-cm 2 --num-envs 16 --output docs/cliff-model999-baseline
```

Then smoke-test the explicit local load:

```bash
WANDB_MODE=offline MICRODUCK_CLIFF_PROFILE=model999 \
MICRODUCK_CLIFF_MAX_HEIGHT_CM=2 MICRODUCK_WARM_START=1 \
  uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.run-name cliff-model999-smoke --agent.resume True \
  --agent.load-run 2026-09-26_20-12-42_cliff-drop-first-1000 \
  --agent.load-checkpoint model_999.pt
```

Require the explicit checkpoint-load and `Patch 5: WARM START from` messages.
The pilot starts again from the untouched source, with 2048 environments,
100 updates initially, and save interval 100. Evaluate its final checkpoint
against the baseline before another 100 updates, then at most 50 more (250
additional PPO updates total). For later blocks use `MICRODUCK_WARM_START=0`
and explicitly select the preceding block's run and final checkpoint. RSL-RL
repeats the saved iteration label on resume; block-end labels are 99, 198,
247 even though the actual update counts are 100, 200, 250. Keep these counts
in the report. Remain at 2 cm, retain all candidates, and stop if evaluation
shows an approach or completion regression. Height advancement still requires
the paired per-seed gates below. The historical 13/64 completions used a
contact proxy and are not directly comparable to today's true-contact gate.

The heading-controller/walking-checkpoint recipe below is an alternative
experiment, not the initialization selected for this pilot.

On an approved GPU pod with the source checkpoint in the path above, the
guarded runner executes the baseline, smoke and three training blocks:

```bash
uv run python scripts/run_cliff_checkpoint_pilot.py
```

Results and logs are written to `logs/cliff-model999-pilot/`. The runner stops
if approach or completion drops by more than 15 percentage points against
the baseline or preceding block, either pooled or within any seed. It retains
all checkpoints and chooses the best upper-platform result, preferring the
source on a tie. This sampling-based stop is not proof of statistical
significance or a height-advancement gate. Review its videos before accepting
the selection. EGL libraries must be installed for rendering.

The earlier 3,000-update run did not establish a reliable approach or drop.
See the [run report](cliff-drop-3000-report.md). Do not resume checkpoint 2998
unchanged.

The current task keeps the 61D actor observation and uses the existing twist
yaw-rate slot for a proportional correction toward world +X. Head and body
commands stay near neutral. Head-command, CoM, and action-rate schedules are
frozen at their initial values. Training starts at a fixed 2 cm drop.

Height buckets are 2, 4, 6, 8, and 10 cm. Set
`MICRODUCK_CLIFF_MAX_HEIGHT_CM` to one of these values before constructing a
training environment. All buckets up to that height remain in the spawn mix.
No episode automatically promotes or demotes a bucket. Change the maximum only
after the evaluator's two-checkpoint gates pass across every configured seed.

## Evaluate candidate initializations

Evaluate checkpoint 999 and a compatible walking checkpoint on the **cliff
robot model**. This battery uses upper-platform starts at every height, three
seeds, and approach distances of 35, 55, and 75 cm. It writes per-episode CSV,
per-height JSON summaries and gates, and representative success/failure videos.
Add `--include-midair` to produce a separate midair-start section. Adjust
`--num-envs` to reach the desired sample count per seed and condition.

```bash
uv run python scripts/eval_cliff_drop.py \
  logs/rsl_rl/cliff_drop/<run>/model_999.pt \
  <compatible-walking-checkpoint.pt> \
  --num-envs 16 --output docs/cliff-baseline
```

Inspect the path-efficiency, heading-error, lateral-drift, edge-time, true
lower-floor contact, and post-landing-travel columns. A completion is counted
only if the task gate fires **and** a foot contact is located on the lower
platform's top face. The contact sensor reports a contact position; the
evaluator checks it against the selected tile's lower-floor footprint and
height.

Use `--legacy-command` for a diagnostic baseline with the old zero-yaw
command. Keep its results in a separate output file; the default evaluation
tests the new heading controller. The saved checkpoint 999 has so far reached
the edge in 12/16 local 2 cm episodes under the legacy command and 0/16 under
the initial heading controller, so warm-start selection must account for that
command transition.

## Advancement gates

Evaluate every 100–250 updates during a short pilot. Supply two consecutive
checkpoint paths in chronological order. The JSON gate report requires:

- Straight approach: at least 95% reach the edge within a 15 cm corridor.
- 2 cm: at least 90% complete the full sequence.
- Higher buckets: at least 85% complete that bucket while 2 cm stays at least 90%.

Each rate must pass independently for every evaluation seed at both
checkpoints. Run the 64-environment, five-update smoke test first. Do not start
a RunPod training run without asking the user, per `AGENTS.md`.

## Load the walking weights

The October 1 pilot started from scratch (`resume: false` in its saved agent
config). Its approach failures do not establish how a pilot initialized from
the walking policy will perform.

Checkpoint loading requires `--agent.resume True` and an explicit checkpoint
source. `MICRODUCK_WARM_START=1` only resets curriculum and iteration counters
**after loading**; setting it alone still starts with random weights. The load
keeps the checkpoint's weights, observation normalizers, and optimizer.

The deployed walking checkpoint recorded in `AGENTS.md` is
`pollen-robotics/mjlab_microduck/441tzs6d`, `model_3750.pt`. Evaluate this
candidate on the cliff robot model with the current heading command before
selecting it. The following commands explicitly load that candidate; W&B
checkpoint access must be configured on the Pod.

First run the smoke with the same checkpoint initialization as the pilot:

```bash
MICRODUCK_CLIFF_MAX_HEIGHT_CM=2 MICRODUCK_WARM_START=1 \
  uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.run-name cliff-walk-warmstart-smoke \
  --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/441tzs6d \
  --wandb-checkpoint-name model_3750.pt
```

Before allowing training to continue, require both log messages:

- `[INFO]: Loading model checkpoint from: .../model_3750.pt`
- `[mdp] Patch 5: WARM START from ...` with counters reset to zero and
  `weights/normalizer/optimizer kept`.

The generic `Patch 5 active` message only confirms that the load hook was
installed. It does not confirm that any checkpoint was loaded. If the load
messages are missing, stop and correct the launch command. Evaluate the loaded
walking checkpoint before its first PPO update to establish the approach
baseline.

After selecting the initialization and obtaining permission for the RunPod
pilot, keep its checkpoint source explicit and cap the pilot at 250 updates:

```bash
MICRODUCK_CLIFF_MAX_HEIGHT_CM=2 MICRODUCK_WARM_START=1 \
  uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 250 \
  --agent.save-interval 100 \
  --agent.run-name cliff-walk-warmstart-pilot \
  --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/441tzs6d \
  --wandb-checkpoint-name model_3750.pt
```

If another walking checkpoint passes the baseline evaluation better, replace
both source flags in both commands. Select the source by straight walking and
cliff approach performance, not by its name or iteration.

The next reward and reset-state changes should follow measured approach and
landing failure modes. The current midair reset is still the earlier synthetic
state; it is reported separately and should be replaced with states collected
from real ledge departures before a landing-recovery pilot.
