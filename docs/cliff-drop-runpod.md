# Cliff drop: staged experiment

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

```bash
MICRODUCK_CLIFF_MAX_HEIGHT_CM=2 uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5
```

For a cross-task initialization, use `MICRODUCK_WARM_START=1` so the runner
restarts curriculum and iteration counters when it loads the compatible
checkpoint. Keep the checkpoint's observation normalizer. Select the source by
straight walking and cliff approach performance, not by its name or iteration.

The next reward and reset-state changes should follow measured approach and
landing failure modes. The current midair reset is still the earlier synthetic
state; it is reported separately and should be replaced with states collected
from real ledge departures before a landing-recovery pilot.
