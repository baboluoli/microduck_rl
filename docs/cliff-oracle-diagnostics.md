# Oracle descent: diagnose before changing the policy

The first oracle experiment is frozen. The existing 72-episode GPU evaluation
found 65 completions at checkpoint 500, 37 at 600, and 14 at 700. This establishes
regression under that evaluation, not its cause. Successful episodes ended as
soon as the original 40 cm / 0.3 s goal was met, so those videos cannot establish
sustained walking after success.

## Diagnostic contract

`scripts/diagnose_cliff_oracle.py` loads saved weights and normalizers for
`Mjlab-CliffOracle-Flat-MicroDuck`. It never learns or changes the training task.

- Keep the real 2 cm cliff and the 63-input observation layout.
- Record the original success pulse and bonus, but do not terminate on success.
- Observe up to 12 seconds from reset. Record whether five clean seconds after
  edge crossing and after the original goal are actually observed.
- For these follow-through metrics, "clean" means no latched head/trunk impact
  violation and no tilt reaching 45 degrees in that observation window. This is
  a deliberately stricter diagnostic, not the original success definition.
- Preserve the evaluator's four-second fallen timeout and NaN termination.
- Censor departures from the finite terrain tile before a neighboring tile or
  platform boundary can be mistaken for a gait failure. A short, clean,
  boundary-censored trace is unknown, not a successful five-second observation.
- Capture the first episode before automatic reset changes its state. Save raw
  joint positions, velocities, torques and actions, plus body heading/tilt,
  yaw-rate command, support contacts, impacts, action changes and goal events.

The initial conditions are three seeds and three approach distances for the
main real-cliff condition. Hold the middle distance for these controls:

1. **Real cliff, zero heading-feedback gain.** Keep the same heading-command
   sampling and random-number consumption, but set its proportional gain to
   zero. This tests how the correction command affects the resulting behavior;
   it does not by itself establish whether a learned gait is intrinsically bad.
2. **Sham plane with the original oracle cue.** Remove the physical step but
   retain the trained distance/2 cm height input. This deliberately contradicts
   the cue and is only a counterfactual control. It is not a truthful flat-ground
   perception test or an estimate of hardware performance. Feeding a true zero
   height instead would change a constant training input and add another
   distribution shift.

## Run

On a CUDA machine holding all three original checkpoints:

```bash
uv run python scripts/diagnose_cliff_oracle.py \
  /path/to/model_500.pt /path/to/model_600.pt /path/to/model_700.pt \
  --device cuda:0 --num-envs 8 --seeds 0 1 2 \
  --approach-distances .35 .55 .75 --output logs/oracle-diagnostics-gpu
```

Each condition writes per-episode JSON and compressed `.npz` time series. The
trace column names and first-episode lengths are embedded in the NPZ. Repeated
runs should use separate output directories; a single example must not be
presented as the aggregate outcome.

## How findings should inform changes

- Compare pre-edge and post-edge failures before changing descent rewards.
- If normal heading feedback is worse than the matched zero-gain condition,
  inspect command/yaw timing before proposing a controller change. Correlation
  alone is not a causal explanation.
- If old successes regularly fail later, the current goal certifies only a
  short maneuver. Evaluate a longer recovery objective before claiming that
  the policy has learned robust walking.
- Use joint-level traces to distinguish high load while moving from repeated
  high load while nearly stalled. Peak torque alone was about 0.64 N m across
  all three checkpoints and does not distinguish their performance.
- Treat backend, cue/geometry mismatch, finite terrain, small samples and
  post-success states absent from training as explicit limitations.

No reward, policy architecture, or training change is justified solely by a
visually irregular gait. A proposed change should name the measured failure,
the evidence supporting it, and the held-out test expected to improve.
