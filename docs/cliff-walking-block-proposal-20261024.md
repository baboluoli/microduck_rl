# Offline walking-prefix analysis and revised reward proposal

The untouched CUDA checkpoint 500 completes at least one clean walking second
in **57/72** attempts, but five clean seconds in only **2/72**. Strict 12-second
success remains **0/72**. This supports teaching continued walking with reachable
intermediate credit; it does not diagnose an optimizer or actuator problem.

Execution follow-up: [bounded pilot results](cliff-walking-block-run-results-20261024.md). The original proposal and offline measurements below are preserved.

Status at proposal time: offline analysis complete; reward change proposed, **not installed in
the training task**. No pod, smoke test or training run was started. Held-out
seeds 30–32 remain unused. The historical task and frozen reference are intact.

## Evidence and exact counting rule

Only the authoritative `reference-final` CUDA traces from
`logs/walkthrough-20261024/logs/walkthrough-session/` were used: development seeds
20–22, 24 first episodes each, fixed 55 cm start and 2 cm drop. Earlier reference
passes, CPU diagnostics and rendering replays are excluded.

Reconstruct the original milestone with its float32 0.30-second supported,
upright, forward-motion hold and real lower-floor foot contact. Begin one-second
blocks at that first milestone. Each completed block must advance world X by
at least 0.05 m. Tilt reaching 45 degrees, head force above 15 N, trunk force
above 20 N, nonfinite evidence or tile-boundary exit disqualifies the prefix.
Impact, nonfinite and boundary validity are retained from before the milestone,
as in the existing strict classifier. Tilt checks begin after the milestone.

Count only completed blocks before the **first** failure. A violation on a
block's ending sample vetoes that block. A failed progress block permanently
ends eligibility; later good blocks cannot restart it. A final incomplete
second earns no block. Normal alternating foot support does not reset walking.
Later failure preserves already completed clean blocks for this secondary metric.

The standalone NumPy analyzer reconstructs and checks the recorded milestone,
total observed blocks and latched invalidity **on every saved sample**. It also
checks strict timeout success against the saved pulse and the prior independent
episode classifications. All checks passed for all 72 episodes. The recorded
`walk_blocks` field alone is unsuitable: it counts observed blocks after failure.

| Development seed | Attempts | Landing milestone | ≥1 s | ≥2 s | ≥3 s | ≥4 s | ≥5 s | Strict success | Impact terminations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 24 | 20 | 19 | 16 | 13 | 3 | 0 | 0 | 19 |
| 21 | 24 | 20 | 19 | 17 | 11 | 5 | 0 | 0 | 17 |
| 22 | 24 | 22 | 19 | 16 | 14 | 8 | 2 | 0 | 18 |
| Total | 72 | 62 | **57** | **49** | **38** | **16** | **2** | **0** | **54** |

Pooled percentages of all attempts: **79.2%, 68.1%, 52.8%, 22.2%, 2.8%**
for at least one through five clean seconds. These are cumulative counts.
The disjoint histogram for 0, 1, 2, 3, 4 and ≥5 completed clean seconds is
**15, 8, 11, 22, 14, 2**. Of the 62 milestone episodes, five complete no clean
second. The other ten attempts never reach the milestone.

The two ≥5-second episodes are seed 22/env 6 (five clean blocks, then a failed
sixth progress block at episode time 8.70 s) and seed 22/env 23 (six clean blocks,
then a tilt failure at 9.70 s and later impact termination). Neither passes the
strict test. Across all attempts, the first disqualification is tilt in 33,
progress in 21, impact in 15 and boundary in one. Two attempts reach timeout
without a latched violation but lack the required milestone/five blocks.
These first-failure counts differ from final termination causes; an earlier tilt
or progress failure can precede a later impact.

Machine-readable per-episode results, input SHA256 fingerprints and summaries:
[`cliff-walking-block-analysis-20261024.json`](cliff-walking-block-analysis-20261024.json).
Reproduce from the RL project root, without a GPU:

```bash
.venv/bin/python scripts/analyze_walkthrough_blocks.py \
  logs/walkthrough-20261024/logs/walkthrough-session/reference-final \
  --output docs/cliff-walking-block-analysis-20261024.json
uv run --with pytest pytest tests/test_walkthrough_block_analysis.py -q
```

## Proposed training bonus: 40 intermediate + 40 final

Replace only the training task's all-or-nothing 80-point bonus with:

| Event | Once-only points | Maximum cumulative points |
|---|---:|---:|
| First clean block | 8 | 8 |
| Second clean block | 8 | 16 |
| Third clean block | 8 | 24 |
| Fourth clean block | 8 | 32 |
| Fifth clean block | 8 | 40 |
| Pass the unchanged full 12-second test | 40 | 80 |

Equal block payments keep the initial hypothesis simple and attach extra credit
to the observed three-to-five-second frontier. Half the budget remains reserved
for strict completion. These weights are a starting experiment, not an optimum
established by offline data or a comparison against the dense reward mass.

Pay a block immediately on its valid completion, once per episode. Permanently
stop further payments on any failed block or later invalidity. Do not claw back
earlier credit when a later failure occurs. Reaching five blocks neither ends
the episode nor stops validity checking: every subsequent full block must still
pass progress, and tilt/impact/boundary/nonfinite rules remain active through
timeout. Existing physical failure terminations remain active. If a block and
strict completion coincide at timeout, pay both only when valid; any same-step
failure vetoes both. Divide point pulses by control dt to cancel RewardManager's
dt multiplier. Keep paid-block state separate from total observed block counts
and from the strict success pulse. Cache calls per step and reset only episode
rows that reset.

Counterfactual replay on the frozen traces gives **1,296 bonus points total,
18.0 per attempted episode**, with nonzero bonus in **57/72** attempts.
Per-seed means are 17.0, 17.33 and 19.67. Strict-completion credit stays zero.
This measures availability of a training signal, not evidence of learning or
an estimate of PPO's discounted return. Existing dense rewards may already teach
something even when the old completion bonus pays zero.

The deliberate risk is collecting early credit and then falling to restart.
Bounded payments and the reserved final reward constrain this incentive but
cannot eliminate it. Select by actual walking and impacts, never total reward.

Before the pilot, audit the bonus against the existing dense reward scale using
the frozen reference on the approved session's CUDA backend. Report actual
per-episode weighted, dt-integrated contributions for each reward term, the
positive dense subtotal, the negative penalty subtotal, and the proposed block
and final bonuses separately. Show means and distributions across all first
episodes, including failures, so positive and negative terms cannot hide each
other through cancellation. Confirm all penalty contributions have the intended
sign and the implemented pulses pay the stated point amounts. A five-update
smoke may not reach timeout, so it cannot alone establish the final payout.

Keeping the maximum bonus at 80 does not preserve the learning incentives:
earlier, more frequent payments change both reward frequency and discounted
return. This audit is a scale check, not proof that 40/40 is optimal. Record the
comparison before launching the pilot; if it motivates different weights,
revise this proposal explicitly rather than silently tuning them during the run.

## One bounded continuation, pending implementation and GPU approval

For the next experiment this proposal replaces the training payout and
zero-strict-success launch gate in `cliff-oracle-next-run-plan.md`. The strict
test itself and all physical conditions stay fixed. Zero strict baseline success
is no longer alone a reason to block this partial-credit experiment: the offline
prefix distribution supplies its reachable intermediate signal.

Before requesting a run, implement and test the payout changes and adapt the
session runner/evaluator. In particular, the evaluator must keep strict success
separate from partial payments, and the runner must replace its old zero-success
gate explicitly. This offline change does not make that runner launch-ready.
Test once-only payments, failure-step vetoes, retained early credit, post-fifth
failure, final 80-point cap, manager caching and staggered resets. Complete the
RunPod EGL/CUDA preflight and required 64-env/five-update smoke after approval.
Check finite state/losses, checkpoint loading and normalizer-baked ONNX export.

Then one seed-42 continuation, **at most 250 updates**, 2048 environments,
checkpoints every 50. Start smoke and pilot independently from untouched source
500 (SHA256 `4b1eacacdab3773c8000a47416a30ed693d8d141915c258e2b0948b59702c651`),
using the existing explicit warm start. Keep the 55 cm start, 2 cm drop, heading
controller, physics, actuator, optimizer, normalization and other rewards fixed.
Retain the seven-minute minimum between behavior checks and always evaluate the
final checkpoint. Re-score source/candidates on the same CUDA backend with seeds
20–22; no held-out evaluation during checkpoint selection.

Report strict success, the full clean-prefix histogram and cumulative 1–5-second
rates, impact terminations, old landing rate, boundary exits and nonfinite cases
by seed. Compare against this frozen reference; report backend changes separately.

Select the retained best from the frozen source and fully evaluated candidates
using the following order: higher strict-success rate first, then higher mean
clean-prefix length capped at five completed seconds, then fewer impact
terminations, then higher old landing rate. Exact ties retain the earlier
checkpoint (the source precedes every candidate). Compute the capped mean over
all 72 attempts, assigning zero to episodes with no clean block; do not average
only successful landings. The offline baseline is 162/72 = **2.25 seconds**.
Use the newly scored same-backend source as the run's actual baseline. This
ordering selects a candidate for review; it does not authorize an extension or
override reporting of landing, impact or boundary tradeoffs.

Keep the existing immediate nonfinite stop and two-check strict-regression rule
(at least 15 percentage points pooled and at least 10 points in two of three
seeds versus the retained best). Add a partial-progress stop that also works
while strict success remains zero. At each completed behavior check, compare
the candidate with the fixed, same-session frozen-source baseline:

- Walking regression: the capped mean drops by at least **0.50 seconds** pooled
  and at least **0.25 seconds in two of three seeds**.
- Impact regression: the impact-termination rate rises by at least **15 percentage
  points** pooled and at least **10 points in two of three seeds**.

Stop if either condition holds at **two consecutive completed behavior checks**;
track each condition separately, so one walking regression followed by one
impact regression does not meet this rule. A check that does not breach a
condition clears that condition's streak. Apply this rule regardless of the
candidate's reward or strict-success rank, retain source/best artifacts, and
report the triggering measurements. These tolerances are pre-registered pilot
choices, not statistical significance thresholds. They guard against deterioration
relative to the source; failure to improve still ends the pilot at 250 updates.
Test ranking, tie handling, all-attempt denominators and consecutive-check
logic when adapting the runner. Preserve the seven-minute minimum between
behavior checks; a final evaluation counts as a completed check.

An extension requires a separate decision supported by longer clean walking
and fewer impacts, while reporting landing/boundary tradeoffs; higher training
reward alone never qualifies. Do not automatically extend or add a replication.
Keep held-out seeds 30–32 reserved for a finalist.

GPU approval must specify current device/rate, one run, and a hard session cap
(proposed maximum two hours including setup and evaluation). No current price or
availability is asserted by this offline report. Follow the existing retrieval,
checksum, shutdown and local-watchdog requirements when a session is approved.
