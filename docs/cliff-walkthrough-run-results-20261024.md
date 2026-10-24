# Fixed-start oracle walk-through: Runpod preflight, 2026-10-24

Follow-up: [offline clean-prefix analysis and partial-credit proposal](cliff-walking-block-proposal-20261024.md)
scores the same retained traces without training. The historical results below
remain unchanged.

**The training pilot was not launched.** Frozen checkpoint 500 scored **0/72**
under the proposed sustained-walking rule. The plan explicitly requires revising
that objective with Pablo before a long run when no frozen episode qualifies.
There were no PPO updates, smoke-derived initializations, replication runs or
held-out evaluations.

## Measured final CUDA reference

One RTX A5000, fixed 2 cm drop, 55 cm start, 12-second horizon, 24 first episodes
per development seed. Actor/critic/action dimensions verified as 63/78/14.
Reference and prospective candidates use the same physical setup, heading
controller and evaluation commands. Existing immediate impact termination and
0.5-second fall handling remain active; the diagnostic evaluator's longer
recovery allowance is not used for qualification.

| Seed | Sustained success | Old goal / milestone | Impact termination | Fall termination | Tile exit | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 0/24 | 20/24 | 19 | 0 | 0 | 5 |
| 21 | 0/24 | 20/24 | 17 | 1 | 1 | 5 |
| 22 | 0/24 | 22/24 | 18 | 0 | 0 | 6 |
| Total | **0/72** | **62/72** | **54** | **1** | **1** | **16** |

No nonfinite termination occurred. Of the 16 timeout episodes, 14 failed at
least one complete one-second progress block; the other two did not provide
the required milestone/five full walking blocks. Failure flags may overlap:
tilt and progress failures can occur before a later impact termination.

The independently implemented scalar trace classifier agreed with the
vectorized training state on every episode's success and walking-block count.
The original evaluator's `complete` or `success` fields denote only its old,
early goal; use `sustained.success` for this experiment.

## Implementation and checks

- One new task: `Mjlab-CliffOracleWalkThrough-Fixed-MicroDuck`, experiment
  directory `oracle_walkthrough_fixed`. Historical oracle configuration and
  existing user edits were preserved.
- New state machine remembers a genuine lower-floor landing milestone, checks
  each full one-second progress block, latches later invalidity and permits
  the unchanged 80-point bonus only at the scheduled timeout.
- Final GPU tests: **32 passed**. This includes actual training/playback reset
  positions and dimensions, synthetic timing/failure traces, staggered resets,
  independent classification and a manual-stepping late-payout regression.
- Local focused tests: 30 passed, 2 CUDA tests skipped locally. Diff whitespace
  checks passed. No CUDA training smoke or ONNX export ran because the frozen
  reference failed the preceding launch gate.
- Original source checkpoint remained untouched; both local source and retained
  pod copy match SHA256
  `4b1eacacdab3773c8000a47416a30ed693d8d141915c258e2b0948b59702c651`.

A first reference pass also scored 0/72. The final audit restored the original
landing hold's float32 time accumulation exactly instead of counting integer
steps, and made timeout equality explicit for synthetic stepping beyond the
horizon. The authoritative results above come from `reference-final`; the first
pass remains in `reference`. Repeated CUDA rollouts differed in some secondary
outcomes, so their counts are shown separately and are not pooled. This is not
evidence that the timing correction caused the secondary differences.

The final frozen reference was re-evaluated after these corrections. Neither
pass justified launching the pilot.

## Retained evidence

All paths below are relative to the RL project root:

- `logs/walkthrough-20261024/logs/walkthrough-session/reference-final/episodes.json`
  — authoritative per-episode results.
- Same directory: per-seed `.npz` joint/action/physical traces, `.json` episodes,
  and `manifest.json` with checkpoint, software, GPU and source fingerprints.
- `logs/walkthrough-20261024/logs/walkthrough-session/final-tests.log`
  and `final-result.json` — final checks and explicit no-pilot decision.
- `logs/walkthrough-20261024/logs/rsl_rl/oracle_walkthrough_fixed/source-oracle500/model_500.pt`
  — frozen baseline copy.
- `logs/walkthrough-20261024/artifacts.tar.gz`, `artifact-sha256.json`, and
  `retrieval-verified.json` — complete retrieved code/evidence snapshot and
  **273 successfully verified file checksums**.

An illustrative, separately rendered seed-20/environment-0 replay is saved as
`logs/walkthrough-20261024/logs/walkthrough-session/reference-seed20-env0.mp4`.
The file is 12.04 seconds, H.264, 640×480 at 25 fps, without frozen final frames.
It is a separate rendering replay, not the authoritative scored trace; its
outcome can differ from the batched evaluation. Its static camera loses the
robot later in the rollout, so it cannot visually validate sustained walking.
The old-goal success in its sidecar JSON must not be read as a new-rule success.
No finalist comparison videos exist because no policy was trained.

## Operations and next decision

Community RTX 4090, A6000 and A5000 requests were unavailable. One secure RTX
A5000 pod, `ei8dt1ga6cyqvg`, ran at the CLI-reported **$0.27/hour**, from about
14:25:24 to 14:44:10 UTC (18 minutes 47 seconds; approximately $0.085 of GPU
compute, excluding storage and any other billing adjustments). Missing EGL
libraries were installed before CUDA tests; locked Python dependencies synced
successfully. No account API key was copied to the pod.

Artifacts were downloaded and verified before stopping. The CLI subsequently
confirmed **desiredStatus=EXITED, runtimeStatus=stopped**, with reason
`stopped_by_user`. The local shutdown sentinel retires the session watchdog.
The stopped pod's volume and the historical pod/volume were retained.

The measured bottleneck is that the frozen policy frequently reaches the old
goal but never satisfies this complete sustained-walking objective. Training
could still learn from dense rewards, but this reference provides no observed
success-bonus signal; that is the sparse-reward risk the preflight gate was
intended to expose. These measurements do not identify an optimizer, actuator
or normalization defect.

The next decision is to revise the continued-walking objective or learning
stages with Pablo, then re-score the same frozen reference before authorizing a
pilot under the revised plan. No thresholds or reward weights were silently
relaxed, and the held-out seeds remain unused.
