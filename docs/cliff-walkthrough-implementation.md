# Fixed-start oracle walk-through experiment

Task: `Mjlab-CliffOracleWalkThrough-Fixed-MicroDuck`, experiment directory
`oracle_walkthrough_fixed`. The historical oracle factory is unchanged. The
new factory inherits its robot, BAM actuators, 63/78/14 interface, dense rewards,
steering, randomization and resets; only horizon, success reward/termination and
tile-boundary censoring change. The reset remains 55 cm before a 2 cm drop.

## Episode state

`cliff/walkthrough.py` holds vectorized, independent episode rows. A control-step
cache prevents managers from advancing counters twice. Newly reset rows clear
on their first control step; the first qualifying milestone is never moved.

The original 0.30 s supported, upright, forward-moving landing hold is retained.
At the milestone there must also be a detected foot contact on the actual lower
box's top: local X [0.55, 4.05], |Y| <= 1.35, Z within 8 mm of -0.02. Coordinates
are relative to the terrain spawn, which is 0.95 m into its 5 m tile.

From the first milestone, every full one-second block must advance world X at
least 5 cm. Blocks do not depend on continuous foot support, so alternating
support and aerial gait phases do not reset the walking timer. An incomplete
final progress block is ignored. At least five complete blocks are necessary.

Tilt >=45 degrees after the milestone, head force >15 N, trunk force >20 N, nonfinite evidence or
departure from X [-0.75, 3.8] / |Y| <=1.15 permanently disqualify the episode.
Existing impact, fall and numerical termination handling remains active.
Disqualification does not itself add a speculative dense penalty. Boundary
exit is a separate truncation, rather than an impact failure.

At the 12 s timeout, the valid qualifying episode gets one pulse. Dividing the
pulse by control dt cancels RewardManager's dt multiplier, paying exactly the
unchanged weight of 80. Other terminal failures veto payment even on timeout.

## Validation and launch gates

`tests/test_cliff_walkthrough.py` covers reward timing, cached manager calls,
invalid trajectories, later failure, row-isolated resets, lower-floor contact,
and independent scalar/vectorized classification. CUDA-only tests build both
training/playback environments and check actual reset position and dimensions.
The common evaluator asserts 63/78/14 while loading the source architecture.

`eval_cliff_walkthrough.py` scores pre-reset first-episode traces independently
and asserts agreement with the training state. Development seeds are 20–22,
24 environments each; held-out seeds 30–32 remain reserved for finalists.
The original goal and clean-follow-through metrics remain in the records.
No CPU counts are pooled with CUDA counts.

`run_walkthrough_session.py` first evaluates the frozen checkpoint. Zero primary
successes stop the session before smoke/pilot, as required by the plan. A
qualifying reference permits a five-update, 64-environment warm-start smoke and
normalizer-baked 63→14 ONNX export/inference check. The single seed-42 pilot
restarts from the untouched source, runs at most 250 updates / 2048 environments,
and saves every 50. Behavior checks are at least seven minutes apart, plus final
evaluation. Two consecutive pre-registered regressions stop the pilot;
nonfinite state/loss stops immediately. No replication or automatic extension.

`render_cliff_walkthrough.py` uses seed 20/environment 0 chosen in advance, with
matching camera/starts and all frames until termination. It does not select an
early success clip or repeat frozen final frames.

The Runpod account key stays on the Mac. A local watchdog runs under caffeinate,
stops the specific pod within two hours, and retires when shutdown is confirmed.
This backstop requires the Mac to stay online. Artifacts must be downloaded and
checksummed before the normal stop, with sufficient transfer time.
