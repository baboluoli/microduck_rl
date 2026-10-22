# Fixed 2 cm descent with exact terrain observations

Task: `Mjlab-CliffOracle-Flat-MicroDuck`. This is a simulation experiment to
isolate movement learning from terrain perception. It uses the steering recipe,
upper-platform starts only, and a fixed 0.02 m drop in both training and play.
Height/profile environment variables do not alter this task.

The actor has 63 inputs and 14 joint-target outputs. Its original 61 inputs
retain their ordering; the last two are:

1. Signed world-X distance to the edge in metres, measured from the robot root:
   `env_origin_x + EDGE_DISTANCE - root_x`. Positive before the edge, zero at
   the edge, negative after it. It is not distance along the robot's heading.
2. Positive upper-to-lower-floor drop height, always 0.02 m in this experiment.

Both actor and critic receive the same exact geometry without observation
noise or delay on these two values. The critic grows from 76 to 78 inputs.
The constant height input alone does not teach adaptation to other heights.
Geometry is tied to this fixed terrain; changing terrain requires updating the
observation parameters too. The evaluator rejects other heights/flat terrain
for this task so it cannot silently supply false height information.

This intentionally extends the deployment interface for a separate experimental
task. Existing policies remain 61D. Old checkpoints cannot be loaded directly
into this larger network: start a fresh run. The current hardware runtime and
publisher require 61D and cannot deploy this policy. A matching terrain-sensing
interface and retraining/transfer validation are needed for hardware.

## Training (requires approval before starting Runpod)

From `tinyduck/lemming/rl` on a Linux CUDA machine:

```bash
uv run train Mjlab-CliffOracle-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5
```

Inspect the smoke run for finite observations/rewards and confirm actor 63D,
critic 78D and 14 actions before a longer run. Do not resume a 61D checkpoint.

## Evaluation

Use the matching task ID so the evaluator constructs the larger network and
supplies terrain inputs throughout inference:

```bash
uv run python scripts/eval_cliff_drop.py path/to/model_N.pt \
  --task Mjlab-CliffOracle-Flat-MicroDuck --heights-cm 2 \
  --num-envs 16 --seeds 0 1 2 --approach-distances 0.35 0.55 0.75 \
  --output logs/cliff-oracle-eval
```

This records 144 upper-start episodes, stage failures, impact disqualifications,
and completion counts, plus representative success/failure videos. Videos need
a working offscreen renderer; use `--no-videos` for the metric-only battery.
A successful smoke check does not establish a learned descent or hardware
feasibility. Compare actual checkpoint batteries before claiming improvement.

## Local verification (2026-10-22)

- 14 focused tests passed: geometry signs/tile offsets, fixed train/play
  configuration, legacy cliff behavior, steering, and evaluator consistency.
- Four CPU simulation environments reset and stepped ten times with finite
  observations and rewards. Observed shapes were actor `(4, 63)` and critic
  `(4, 78)`; the actor's last two inputs matched live geometry at every step.
- No training run or learned-policy evaluation has been performed for this task.
  The required 64-environment, five-update CUDA training smoke remains pending.
