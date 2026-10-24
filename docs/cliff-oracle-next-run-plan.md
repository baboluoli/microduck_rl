# Handoff: one policy, sustained walking, fixed 55 cm start

Current objective: [drop, land, walk one metre; revised scores and height progression](cliff-distance-goal-and-height-plan-20261024.md). Walking speed is unrestricted. The old 12-second completion requirement below is historical; the revised training task is not yet installed.

Execution update: the partial-credit revision has been implemented and its approved 250-update pilot completed. See [2026-10-24 results, artifacts and confirmed shutdown](cliff-walking-block-run-results-20261024.md). The original handoff below is retained as historical planning.

Follow-up proposal: [offline walking-prefix analysis, 2026-10-24](cliff-walking-block-proposal-20261024.md)
revises the proposed training payout and zero-success launch gate while preserving
the strict final test. That reward revision is not yet implemented; do not run
the existing session runner as though it already pays partial credit.

Status: implementation and experiment plan only. **No new training is authorized
by this document.** Prepare and validate the implementation, then ask Pablo to
approve the concrete GPU, current price, run count, and wall-clock cap before
starting training, as required by AGENTS.md. Prior approval was for the previous
run and its expired deadline.

## Decision and scope

Keep a single oracle policy and **fix the start at 0.55 m from the edge in
both training and evaluation**. Spawn randomization is deferred and is not part
of this handoff. The sole experimental change is the sustained-walking
success/reward/episode rule described below.

| Reference / experiment | Initialization | Start distance | Objective |
|---|---|---|---|
| Frozen reference | Original checkpoint 500, no updates | 0.55 m | Common sustained-walking evaluation |
| Fixed-start continuation | Untouched checkpoint 500 | 0.55 m | Revised sustained-walking objective |

Compare the continuation with frozen 500 under the same evaluation protocol.
This is not a full causal experiment on why the old PPO run regressed. **Do not
add a randomized-start arm, block this work on further testing checkpoints 600
and 700, or retrain the old recipe as an additional control.** The user wants
the change kept trackable and has explicitly chosen only the 55 cm start.

Do not change the optimizer, learning rate, robot, actuator model, observation
normalization, action scale, heading controller, smoothing weight, other reward
weights, or 2 cm drop in this experiment. Do not introduce multiple policies,
flat-terrain training, height curricula, or new perception inputs.

## Evidence and limitations

- Original GPU evaluation: checkpoint 500 completed 65/72 episodes; 600 completed
  37/72; 700 completed 14/72. Those successful episodes ended early.
- Extended CPU diagnostics of checkpoint 500: 192 first episodes, including
  held-out seeds and multiple approach distances.
- At the matched 55 cm start, held-out seeds 10–12 reached the original goal in
  22/24 episodes; 18 of those 22 subsequently breached a tilt/impact threshold.
  Only 6/24 demonstrated five clean seconds after the original goal.
- Held-out starts at 35, 55 and 75 cm demonstrated five clean seconds after the
  original goal in 7/24, 6/24 and 7/24 episodes respectively. Proximity alone did
  not remove the failure pattern.
- Removing heading feedback caused 19/24 held-out trials to leave the finite
  test tile. Keep it enabled.
- The sham-plane control deliberately contradicted the terrain cue. Do not
  present its results as truthful flat-ground walking performance.
- CPU findings are diagnostic evidence, not interchangeable with GPU counts.
  Re-evaluate the frozen reference on the GPU used for the new experiment.
- The evidence establishes a short success criterion and weak continued walking.
  It does not identify a particular PPO, actuator, or normalization defect.

Read `docs/cliff-oracle-diagnostic-results.md` and
`docs/cliff-oracle-diagnostics.md` for protocol details and raw-data locations.

## Source checkpoint and current workspace

Project directory: `tinyduck/lemming/rl`.

Local source weights:

```text
logs/cliff-oracle-20261024/evaluations/selected.pt
SHA256 4b1eacacdab3773c8000a47416a30ed693d8d141915c258e2b0948b59702c651
```

These are the original checkpoint-500 weights, normalizers and optimizer state.
Copy and checksum them before each approved run; never initialize from a smoke
run or an intermediate candidate.

The existing task is `Mjlab-CliffOracle-Flat-MicroDuck`: fixed 2 cm drop, steering
profile, 63 actor inputs, 78 critic inputs, 14 joint-target outputs. Preserve
these interfaces. Any ONNX export remains an experimental 63-input artifact;
do not publish it as compatible with the current 61-input hardware runtime.
Its two terrain inputs remain truthful distance to the edge
and positive drop height 0.02 m. No 61-input checkpoint is needed.

Historical pod: `dlqufekx623hkk`, stopped after video rendering. Its host later
reported no free GPU on three restart attempts. All required initialization
weights are local, so a newly approved pod can be used without recovering that
host. The old volume contains other checkpoints; do not delete it as part of
this task. Prior SSH ports and deadlines are stale; discover fresh details.

There are existing user edits in `src/mjlab_microduck/tasks/cliff/mdp.py` and
`src/mjlab_microduck/tasks/cliff/README.md`. Inspect current git status and preserve
those edits. New diagnostic files from this conversation may be uncommitted.
Do not revert or include unrelated changes accidentally.

## Implementation: one separate experiment task

Keep the historical oracle task unchanged so its checkpoints and evaluation
remain reproducible. Add one registration, suggested name:

```text
Mjlab-CliffOracleWalkThrough-Fixed-MicroDuck
```

Use the distinct experiment directory `oracle_walkthrough_fixed`. Build this
factory from the current oracle factory and preserve the existing reset:

- Reset X offset `(0.0, 0.0)`, meaning 0.55 m to the edge.
- Keep reset yaw, Y, initial joint/base-velocity distributions, head/body
  commands, and domain randomization unchanged.
- Do not randomize distance, spawn at the edge, or start in midair.
- Training, smoke tests, evaluation and comparison videos all use 0.55 m.

## Implementation: sustained walking and reward timing

Use this explicit first hypothesis: the existing early reward/reset gives too
little incentive and experience for continued walking. Correct that objective
without adding a collection of speculative gait penalties.

1. Use a fixed 12-second episode horizon from reset. Do not terminate or pay the
   success bonus when the original landing criterion first fires.
2. Preserve the original landing calculation as a **milestone**: robot root at
   least 0.40 m beyond the edge, the original upright/support/speed hold, and no
   disqualifying head/trunk impact. Additionally confirm a real foot contact on
   the lower floor using the existing position-reporting foot sensor and the
   actual tile geometry. A virtual marker or contact on the upper floor is not
   sufficient. Record the first qualifying milestone time/position once.
3. From that milestone, observe at least five further seconds of walking.
   Latch a failure if tilt reaches 45 degrees or a head/trunk impact exceeds the
   existing limits (15 N / 20 N). Do not slide the observation window forward to
   erase an earlier failure.
4. Prevent standing still from qualifying. For each complete one-second block
   after the milestone, require at least 0.05 m of net world-X progress. This is
   the existing 0.05 m/s minimum expressed over a gait-tolerant window, not a
   claim that this threshold is optimal. Five full blocks must be observed.
   Continue checking full blocks and tilt/impact validity until episode end;
   ignore an incomplete final block for the progress check only. Normal
   alternating support and brief aerial phases must not reset the timer merely
   because a foot is airborne.
5. Retain the existing 80-point success-bonus magnitude, but **pay it once at
   the scheduled 12-second timeout**, only if the milestone, five walking blocks,
   and subsequent validity checks all passed. This prevents collecting the
   bonus and deliberately falling to restart sooner. Never pay it for an early
   failure or tile-boundary exit. Respect RewardManager's dt scaling.
6. Keep the existing dense rewards, heading feedback and impact/fall/NaN
   handling otherwise unchanged. A fall or disqualifying impact must remain a
   failure; the four-second recovery window in the diagnostic evaluator is not
   an instruction to relax the training failure handling.
7. End/censor departures from the finite tile before neighboring terrain can
   create false failures or successes. Use the diagnostic bounds consistently
   (root-relative X outside [-0.75, 3.8] m or |Y| > 1.15 m). Do not pay a success
   bonus on a boundary exit. Report these exits separately from impact failures.

Implement the milestone, walking-block counters, validity flags and payout
latch as vectorized per-environment state. Update once per control step even
when reward and termination managers both call helpers. Reset only the rows
whose episodes reset. Do not mutate shared cfg dictionaries during stepping.

These are pre-registered experimental choices, not proven optimal settings.
Before any long run, score checkpoint 500 under this exact new rule. If no
reference episode can qualify, report that sparse-reward risk and revise the
plan with Pablo rather than silently changing thresholds or launching a long
run that receives no success signal.

## Common evaluation and selection

Use an independent evaluator for all policies, including frozen 500. Keep the
physical robot, oracle observations, action conventions and evaluation command
settings identical. The original oracle task plus the continued-rollout
configuration in `scripts/diagnose_cliff_oracle.py` can provide that common
physical setup; different training experiment names do not require different
inference networks. Verify compatibility explicitly.

Extend the evaluator to compute the exact new training qualification/payout
rule, including true lower-floor contact, progress blocks and full-horizon
validity. Cross-check its classification against the training state machine on
synthetic traces. Retain the existing old-goal and clean-follow-through metrics
for comparison. The field `original_evaluator_outcome` can still say `complete`
after a later fall because the original evaluator assumed success ended the
episode: **do not use that field as the new success metric.**

Evaluation protocol:

- CUDA backend for reference and candidates; log software, source snapshot,
  checkpoint SHA, device, seeds and conditions. Never pool CPU and CUDA counts.
- Development seeds 20, 21, 22, 24 environments per seed, all starting 0.55 m
  from the edge: 72 first episodes per evaluated checkpoint. Re-evaluate frozen
  500 with this exact layout; older mixed-distance counts are historical context.
- Final held-out seeds 30, 31, 32 with the same fixed-55-cm, 72-episode layout. Reserve these
  for finalists; do not use them to choose intermediate checkpoints.
- Fixed 12-second window, first episodes only, state captured before auto-reset.
- Primary: fraction passing the exact revised sustained-walking rule out of all
  attempted episodes. Show failures, nonfinite cases, boundary exits and
  insufficient observation separately; none count as success.
- Secondary: old goal rate; five clean seconds after edge and after old goal;
  impacts; path efficiency; corridor departures; heading error; joint/action
  traces before first instability. Show results by seed, not just pooled reward.
- Final videos use matched starts/cameras and show full rollouts after the old
  goal, with actual outcomes. Disclose selection of examples and frozen final
  frames. No early-cut success clips presented as durable walking.

## Tests and preflight

Run the existing oracle/diagnostic tests and add tests for the new state machine:

- Shapes 63/78/14 and fixed 0.02 m terrain in train and play.
- A configuration test and runtime reset test verify the fixed 0.55 m start in
  both train and play; no randomized-distance factory or arm is introduced.
- The old milestone cannot terminate an episode or pay the bonus.
- Genuine lower-floor contact is required.
- Standing, moving backward, falling, an impact, a boundary exit, fewer than
  five full walking blocks, or later invalidity cannot earn the bonus.
- Valid continued walking pays exactly once at scheduled timeout, not once per
  frame or twice because two managers called a helper.
- Staggered resets clear only the appropriate environments; one env's success
  or failure never changes another's counters.
- Partial observations and NaN cases cannot become successes. Trace filenames
  for different checkpoints and seeds remain distinct.

Useful existing command:

```bash
uv run --with pytest pytest tests/test_cliff_oracle.py \
  tests/test_cliff_eval_videos.py tests/test_cliff_oracle_diagnostic.py -q
```

The previous diagnostic implementation passed 13 focused tests and a short CPU
trace smoke. That is not a substitute for new task tests or the mandatory CUDA
training smoke: 64 environments, five updates, for the new task and the same
checkpoint initialization as its pilot. Validate finite observations, rewards,
losses and exports; confirm 63/78/14. Also use synthetic trajectories to exercise
12-second reward timing, because a five-update smoke need not reach timeout.

## Checkpoint loading: explicit warm start

New experiment names change the directory searched by `--agent.load-run`.
Stage an unchanged copy of `selected.pt` as `model_500.pt` under
`logs/rsl_rl/oracle_walkthrough_fixed/source-oracle500/`, and verify the SHA
above. Keep the original file untouched.

Use `MICRODUCK_WARM_START=1` together with explicit resume/load flags. This keeps
weights, normalizers and optimizer state while restarting iteration and
curriculum counters. Require both the actual checkpoint-load message and the
`WARM START` message confirming counters reset and state retained. The generic
`Patch 5 active` message alone does not establish a load.

Proposed smoke template, runnable only after implementing the new task:

```bash
WANDB_MODE=offline MICRODUCK_WARM_START=1 uv run train \
  Mjlab-CliffOracleWalkThrough-Fixed-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.seed 42 --agent.run-name walkthrough-fixed-smoke \
  --agent.resume True --agent.load-run source-oracle500 \
  --agent.load-checkpoint model_500.pt --enable-nan-guard True
```

**Restart the pilot from the original source checkpoint, not the smoke's final
checkpoint.**

## Proposed pilot, requiring fresh approval

1. Quote current GPU availability/pricing using `uv run runpodctl ...`. Prefer
   one suitable GPU rather than parallel paid pods. Prepare source, task tests,
   source checkpoint, runner, artifact retrieval and shutdown control before
   asking for approval. Keep account credentials local.
2. One fixed-start pilot: seed 42, 2048 environments, at most 250 new PPO
   updates, checkpoint every 50 updates. Compare against the frozen source on
   the same backend. Keep all candidates and the frozen source.
3. Check progress no more frequently than **every seven minutes**, as requested
   by Pablo. Evaluate the newest completely written checkpoint; evaluate the
   final one even if the pilot finishes between checks. Report behavior metrics,
   not total reward as the decision criterion.
4. A repeat of this same fixed-start experiment with seed 43 is an optional
   replication, not implicitly authorized extra work. Include it in the approval
   if desired. Run it only if the first pilot is finite and shows a useful
   signal; otherwise report results and stop. Do not add experiment arms.
5. A pilot command uses the smoke template with 2048 envs, 250 updates,
   `--agent.save-interval 50`, and a unique run/seed name. Every approved run
   starts from the same untouched checkpoint 500.
6. Pre-register early stop: nonfinite state/loss is an immediate failure. Stop
   the pilot after two completed evaluation checks showing at least a 15-point
   primary-success regression versus its retained best and at least a 10-point
   regression in two of three evaluation seeds. Retain best/source on stopping.
   Boundary exits or sharply worse heading must be reported even if the primary
   rate rises. At 250 updates, no clear gain means do not extend automatically.
7. Set a fresh approved hard wall-clock limit for the entire session, including
   setup/evaluation and any explicitly approved replication. A proposed ceiling
   is two hours, subject to approval and actual timing; do not automatically
   consume it or extend the 250-update pilot to fill it. Wall-clock cap wins over iteration targets. If shutdown depends on
   the local Mac, keep it awake/online and disclose that limitation. Never copy
   the Runpod account API key to the pod; prior auto-review rejected that.
8. Retrieve/checksum checkpoints, configs, logs, manifests, raw evaluation data
   and videos BEFORE stopping the pod. Use adequate transfer timeouts. The old
   20-second retrieval timeout interrupted downloads; don't repeat it. Retire
   obsolete watchdogs after verified shutdown so they cannot stop a later run.

More iterations are not the selection rule. If the continuation does not improve sustained
walking on reserved tests, preserve the evidence and propose the next targeted
experiment rather than tuning multiple reward/optimizer terms at once.

## Deliverables from the executing agent

- Reviewable diff, tests, one new task registration and exact state-machine semantics.
- Source checkpoint verification and untouched frozen baseline.
- Approval request with concrete GPU, current rate, run count and session cap.
- Per-run, per-seed learning/evaluation records and early-stop decisions.
- Matched full-horizon comparison videos and best checkpoints downloaded locally.
- Short report separating measured results, interpretations, uncertainties and
  the next decision; explicit confirmed pod shutdown.

Suggested message to the next agent:

> Follow `tinyduck/lemming/rl/docs/cliff-oracle-next-run-plan.md`. Implement and
> validate only the fixed-55-cm single-policy walk-through experiment. Do not
> randomize spawn distance or add another arm. Preserve existing user edits. Do not spend time diagnosing 600/700
> first. Prepare the complete launch and shutdown workflow, then ask me to approve
> the GPU training budget before starting any training.
