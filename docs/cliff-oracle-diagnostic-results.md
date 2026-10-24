# Frozen oracle policy: initial diagnostic results

Date: 2026-10-24. No learning was performed. The policy is checkpoint 500 from
`cliff_oracle_2cm/2026-10-24_12-06-38_oracle-2cm-3000`, saved locally as
`logs/cliff-oracle-20261024/evaluations/selected.pt`. SHA-256:
`4b1eacacdab3773c8000a47416a30ed693d8d141915c258e2b0948b59702c651`.

> Subsequent user decision: further diagnosis of checkpoints 600 and 700 is
> not required before the next experiment. Follow the
> [next-run handoff plan](cliff-oracle-next-run-plan.md), which supersedes the
> earlier recommendation to make that comparison a prerequisite.

## What was tested

192 first episodes on CPU: 72 reproduction/control episodes on seeds 0, 1, 2,
then 120 episodes on held-out seeds 10, 11, 12. Main-condition held-out starts
were 35, 55 and 75 cm from the edge. The two controls held distance at 55 cm.
All comparisons in the table below use only the matched 55 cm starts (24
first episodes per condition and seed set).

The success calculation remains active, but no longer ends an episode. Each
policy can run for 12 seconds from reset. The existing four-second fallen
termination and NaN guard remain active. Departures from the finite tile are
censored before crossing its physical boundary.

A clean follow-through means no latched impact violation and no tilt reaching
45 degrees in the specified five-second window. Observed threshold breaches
are failures, even if the episode ends before five seconds; a short clean
trace is unobserved, never counted as success.

## Matched-distance results

| Measurement | Seeds 0–2 | Held-out seeds 10–12 |
|---|---:|---:|
| Real cliff, original goal reached | 23/24 | 22/24 |
| Real cliff, goal followed by later threshold breach within the rollout | 18/23 goal-reaching episodes | 18/22 goal-reaching episodes |
| Real cliff, five clean seconds after crossing edge | 16/24 | 18/24 |
| Real cliff, five clean seconds after original goal | 7/24 | 6/24 |
| Zero heading feedback, tile-boundary exits | 21/24 | 19/24 |
| Sham plane, five clean seconds after virtual edge | 19/24 | 21/24 |

The five-seconds-after-goal denominator includes episodes that never reached
that goal. Of the original-goal-reaching episodes, 16 showed a threshold
breach within the next five seconds in the first set, and 15 in the held-out
set. No goal-reaching episode was counted clean without the full five seconds
of observation. Other later breaches can occur after that five-second window.

On the sham plane, 11 episodes in each seed set crossed the virtual goal and
later breached a tilt/impact threshold. This control intentionally retains the
trained 2 cm oracle input although there is no physical cliff. It therefore
shows that a physical step is not necessary for every observed failure under
that cue; it does not prove failure on truthful flat-ground observations.

Zero-feedback trials retain identical heading-target sampling but use zero
proportional gain. Most leave the finite tile. Their low count of observed
impacts cannot be interpreted as improved safety: those episodes are largely
censored. Removing heading correction is not supported by this experiment.

## What this establishes

1. The old goal and early-ending success videos overstate how much sustained
   walking has been demonstrated. Repeated held-out failures follow that goal.
2. The heading correction is useful for staying in the test area. The measured
   irregular gait cannot be fixed simply by disabling that correction.
3. The physical step is not the only ingredient needed to produce late failures
   in this counterfactual test. The mismatch in the sham oracle cue remains a
   limitation.

## What remains unresolved

These CPU results do not establish why the GPU policies changed between
checkpoints 500, 600 and 700. They also do not establish an actuator, reward,
normalization, or PPO optimization cause. Peak torque alone was approximately
0.64 N m for all three checkpoints in the original GPU evaluations and does not
separate their outcomes.

The old pod could not restart because its host reported no free GPU. Its
volume still contains checkpoints 600 and 700. A causal comparison of those checkpoints would require the same frozen
diagnostic on a common CUDA backend. That comparison is not a prerequisite for
the next experiment. CPU results must not be pooled with GPU counts.

The executable cliff task code was compared with the uploaded training archive
and is unchanged; local differences in that task module are comments/docstrings.
Random external pushes are removed by the cliff recipe. No reward, policy
architecture, or training configuration was edited for this diagnostic.

## Next decision

Keep one policy. Adopt continued-walking measurements for evaluation, then
follow the fixed-55-cm sustained-walking experiment in the
[next-run plan](cliff-oracle-next-run-plan.md). Re-evaluate the frozen checkpoint
500 reference on the new experiment's CUDA backend. Preserve the measured
failures and traces, and use held-out seeds and longer observation after success
to judge the proposed objective change. Do not treat the underlying cause of
the old cross-checkpoint regression as established.

Raw records, manifests and compressed traces are in
`logs/cliff-oracle-20261024/diagnostic-cpu-v2/` and
`logs/cliff-oracle-20261024/diagnostic-cpu-heldout/`.
`logs/cliff-oracle-20261024/diagnostic-summary.json` contains matched-distance
summaries, and `logs/cliff-oracle-20261024/diagnostic-traces.png` plots one matched
seed/episode from each condition. It is an example, not the aggregate result.
The [protocol and command](cliff-oracle-diagnostics.md) reproduce the test.

Verification: 13 focused tests passed, covering truthful oracle geometry, replay
labeling, continued goal detection, failure versus censoring, JSON output, and
distinct trace filenames for different approach distances. A two-environment,
10-step CPU smoke captured finite traces before the longer batches.
