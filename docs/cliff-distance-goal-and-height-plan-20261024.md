# Drop-and-continue goal, offline re-score, and height progression

Date: 2026-10-24. This is the current proposed direction, reflecting Pablo’s decision to focus on handling a drop and walking a useful distance afterwards. Speed is unrestricted. The earlier 12-second sustained-walking requirement is superseded for the next experiment; its historical results remain unchanged.

Status: the saved CUDA traces have been re-scored offline. The new training goal and height curriculum are **not installed or launched**. No GPU, new rollout, training update, held-out evaluation, or hardware deployment was used for this analysis.

## Goal and exact success rule

The immediate goal is reliable handling of a 2 cm drop, followed by 1.0 metre of forward walking. The longer-term goal is the same behavior over progressively larger drops. One metre is approximately four robot heights: it provides evidence of continued movement beyond touchdown without asking for prolonged walking.

1. Start on the upper platform, 55 cm before the drop and centered sideways, as in the existing experiment.
2. Record the **first genuine foot contact on the lower floor**, using contact-found plus contact position against the actual lower platform. Store the robot root’s world-X position on that sample. The old 0.30-second continuous support hold is not required.
3. Complete when the root has advanced at least 1.0 metre in net world X from that stored position, while upright and valid. This is displacement, not accumulated path length: pacing backwards and forwards cannot earn it.
4. From that first lower-floor contact through completion, tilt reaching 45 degrees invalidates the attempt, including the contact sample. Head force above 15 N, trunk force above 20 N, or nonfinite evidence invalidate from reset. These are the existing measurement thresholds, not new hardware safety guarantees.
5. A failure on the goal-reaching sample vetoes success. Brief airborne or single-support gait phases do not restart the distance measurement. There is no speed target, upper speed limit, or requirement to remain continuously supported for 0.30 seconds.
6. Success is recorded on its first valid occurrence. A later fall or tile departure does not undo it: the proposed training/evaluation episode would already have ended. This defines completion of the maneuver; it does not establish indefinite walking stability.
7. Keep 12 seconds as an initial maximum attempt duration, not a required walking duration. A boundary departure before qualification is an unqualified/censored attempt, not automatically a physical fall. A departure after qualification is irrelevant to the primary result. Allow at least the full one-metre follow-through and margin after expected touchdown.

The saved lower-floor boolean was generated from actual foot contact positions checked against the 2 cm lower floor, including tile footprint and height tolerance. The offline analyzer uses this evidence directly, not the old landing milestone or its block counters. It preserves the original 45-degree numerical tolerance (1e-5 degrees) and permits only 1e-6 m tolerance on the distance subtraction.

## Re-scored 2 cm results

All results below use the same RTX A6000 session: development seeds 20–22, 24 first episodes per seed, 72 per checkpoint. These are retrospective scores under the newly agreed objective, not a pre-registered independent validation. Historical A5000 counts and separate rendering replays are excluded. Only three checkpoints have the required saved CUDA traces; saved weights alone for checkpoints 0/50/150/200 cannot be re-scored offline.

| Policy | Seed 20 | Seed 21 | Seed 22 | Total | Success rate |
|---|---:|---:|---:|---:|---:|
| Original checkpoint 500 | 21/24 | 21/24 | 24/24 | 66/72 | 91.7% |
| Continuation 100 | 22/24 | 21/24 | 21/24 | 64/72 | 88.9% |
| Continuation 249 | 22/24 | 21/24 | 22/24 | 65/72 | 90.3% |

**Interpretation:** the existing policies already solve this shorter 2 cm task in about nine out of ten saved attempts. The pilot has not demonstrated an improvement in the one-metre success rate: the original scores 66, continuation 100 scores 64, and continuation 249 scores 65. Differences of one or two episodes in this selected development sample do not justify a confident ranking. My earlier statement that the pilot itself made a large improvement specifically on the shorter 2 cm goal was too strong.

The continuation’s clearer benefit appears later in the rollout. This is useful secondary evidence, but the primary goal is completion of the landing plus one metre:

| Policy | Recorded lower-floor contact /72 | Head/trunk-impact episodes before new completion or old trial end | Full old-rollout impact terminations | Mean observed old rollout (s) | Newly successful episodes that later ended in impact |
|---|---:|---:|---:|---:|---:|
| Original checkpoint 500 | 71 | 5 | 60 | 7.23 | 55 |
| Continuation 100 | 68 | 5 | 13 | 8.44 | 8 |
| Continuation 249 | 68 | 3 | 5 | 9.30 | 2 |

The drop in observed impacts is real, and the continuations were observed for longer on average. This supports meaningful improvement in continued walking after the drop. The short one-metre criterion was already largely solved by the original; it does not capture all of that later improvement. Boundary censoring limits claims about what would happen beyond the observed floor, rather than erasing the observed improvement.

The impact counts before completion or trial end are diagnostic event counts, not disjoint outcome categories. An unsuccessful episode can breach tilt and subsequently impact. Boundary censoring shortens later exposure, so full-rollout impact counts alone do not establish equal-duration safety. In particular, 55 of the original’s 66 newly successful episodes later ended in impact; accepting the shorter objective deliberately stops judging the maneuver before those later failures.

One original and four episodes in each continuation contain no recorded qualifying lower-floor foot contact. They remain non-successes in the all-attempt denominator. The saved evidence cannot distinguish absence of the required foot contact from a missed contact measurement in every such case. One checkpoint-100 episode records its first qualifying contact only at root X=3.374 m, leaving insufficient room before the existing X=3.8 m evaluation limit. These are reasons to validate contact recording and usable floor length in the fresh confirmation, not reasons to invent contact evidence or count an exit as a fall.

All scored samples are finite. Failure histories, first contact and goal positions/times, per-seed summaries, checkpoint SHA256 values, and fingerprints of every input trace are retained in [cliff-distance-rescore-20261024.json](cliff-distance-rescore-20261024.json). The original traces and previous score files were not modified.

## How to proceed

### 1. Prepare the distance task and confirm the current 2 cm capability

Create a separate distance-goal task/experiment so the historical oracle and walk-through runs remain reproducible. Share one vectorized state machine between reward and termination, with once-per-step caching, per-row episode resets, the failure-step veto, genuine lower-floor contact, and the exact offline distance rule above. The evaluator must independently agree on synthetic traces.

Proposed training payout: the existing maximum 80-point completion budget, paid once at full distance completion, with division by control dt. Remove the old time-block payments and strict-timeout payout from this new task. Keep other dense reward weights, speed behavior, optimizer, robot, actuator model, normalizers, heading feedback, domain randomization and fixed 55 cm start unchanged. Audit actual integrated rewards before launching the continuation. Ending the training episode on success does not teach a physical stop command; the policy can keep producing walking actions in deployment.

Success should be a true terminal outcome, distinguished from time-limit truncation and from physical failure. Do not reuse the old blanket reward mask based on reset_terminated: the new success termination itself would otherwise suppress its own reward. Mask only competing failure conditions and test same-step success/failure ordering explicitly.

Preserve original 500, continuation 100 and continuation 249, plus their normalizers and optimizer state. Do not start from scratch. Do not assume that the old 12-second ranking makes checkpoint 100 best for the revised objective. Checkpoint 249 is the recommended lead candidate for fresh confirmation: its one-metre success rate is similar, its observed rollouts last longer, and it has the fewest full-rollout impacts. Retain checkpoint 100 as an alternative. The available scores do not prove checkpoint 249 is superior on the first one metre alone. Numerically, the original has the highest current one-metre score.

In the next approved GPU session, first validate the task, then compare all three unchanged checkpoints under the new rule at 2 cm. Use seeds 20–22 for matched development checks and an additional development confirmation batch (proposed seeds 40–42). The latter are confirmation development seeds, not held-out final tests. Save actual found/position/force evidence for each foot and inspect missing-contact cases, including contact geometry and possible between-control-step contacts, before relying on a newly trained goal reward.

Select the continuation base by the highest fresh confirmed distance-success rate, then fewer disqualifying head/trunk-impact episodes before completion, then fewer physical failures in the saved extended diagnostic rollouts. Exact remaining ties retain the earlier checkpoint. Keep the other checkpoints as controls/fallbacks. Later diagnostic behavior is a tie-breaker and reported context, not a reinstated 12-second success requirement.

### 2. Finish 2 cm only if necessary

Proposed advancement gate: at least 65/72 successes (90.3%) overall, at least 21/24 in every development seed, and at most 3/72 episodes with a disqualifying head/trunk impact before completion or trial end. Require this in the selection batch and the separate development confirmation batch. These are proposed practical pilot gates, not statistical confidence guarantees. Keep the one-metre distance fixed instead of shortening it to make a difficult height pass.

If the fresh 2 cm evaluation clears this gate, there is no reason to spend a long training run relearning 2 cm. Move to the next height. If it does not, use one bounded continuation at 2 cm under the distance objective, retaining the best measured checkpoint; do not automatically extend because training reward rises.

### 3. Increase the challenge in height, while retaining easier drops

Nominal progression: **2 → 4 → 6 → 8 → 10 cm**, using the same policy and the same one-metre success rule. This matches the existing terrain height buckets. Ten centimetres is a research target, not a claim of demonstrated or physically safe capability. If the 2-to-4 cm jump is too difficult, explicitly introduce and document a 3 cm bridge stage; do not silently change the success distance, impact limits, or walking speed.

At each stage, first evaluate the frozen entering policy on the new height before training. Then train on a mix of the new height and already-mastered heights so it does not forget easier landings. Proposed starting mixture: 75% frontier height, 25% mastered heights, with the latter sampled evenly. This mixture is a starting experiment choice. Advance only after the frontier clears the same development gate and each retained easier height still clears it; otherwise keep the frontier fixed and inspect actual failed landings.

Keep full-approach starts at 55 cm during this first height progression. Vary height deliberately; do not simultaneously add randomized start distances, change the optimizer, or impose a slower gait. Once a useful height range is mastered, separately test and then train approach-distance and heading variability if needed.

### 4. Make height observations and contact checks truthful

The present oracle factory and contact helper assume a fixed 0.02 m drop. They cannot be used unchanged at 4/6/8/10 cm. Parameterize the terrain, terrain observation, lower-floor contact-height test, evaluator, and video/manifest labels from the actual per-environment height. Mixed-height batches require per-environment truthful height values, not one constant height for every robot.

Preserve the existing 63 actor inputs / 78 critic inputs / 14 actions for this experimental policy family. Warm-start with the complete weights, normalizers and optimizer state, resetting training/curriculum counters explicitly. The height observation was nearly constant in the 2 cm source: inspect normalized observations and finite outputs when introducing a new height, rather than assuming the inherited normalizer handles it correctly.

This remains a simulation policy with oracle distance/height inputs. Larger-drop simulation results alone do not establish compatibility with the current 61-input hardware runtime; real deployment still needs the matched observation contract and terrain estimates.

### 5. Bound and evaluate each future training session

Proposed next training budget: one seed-42 continuation at 2048 environments, at most 1000 new PPO updates at the first unmastered height (or 2 cm if confirmation fails), checkpoints every 100, and a hard session cap of two hours including setup, evaluations and retrieval. This is a proposed longer pilot, not authorization to launch it or a promise that 1000 updates suffice. The session cap wins over the update target; no automatic extension, replication or extra frontier is included.

Follow the RunPod EGL/CUDA preflight, focused synthetic/config/CUDA tests, and a 64-environment/five-update smoke before the longer run. Smoke and pilot independently load the selected unchanged base checkpoint. Verify the actual load and warm-start messages, finite state/losses, exact reward payouts, and normalizer-baked ONNX export.

Behavior checks stay at least seven minutes apart and always include the final checkpoint. Log one-metre success, genuine contact rate, pre-completion impact/tilt/nonfinite cases, distance reached, boundary censoring, peak servo loads, and later diagnostic behavior separately by height and seed. Watch matched full approach/drop/follow-through videos; do not pick a winner by total reward or speed.

Stop immediately on nonfinite evidence. Stop after two consecutive completed checks in which a previously mastered height loses at least 15 percentage points of one-metre success overall and at least 10 points in two of its three development seeds versus the retained best for that height. Preserve source/best artifacts and investigate failure geometry and impacts before changing rewards. A lack of frontier improvement still ends the run at its budget; it does not authorize more updates.

Reserve seeds 30–32 for a finalist evaluated across the mastered height range. Do not use them to choose intermediate checkpoints or repeatedly tune stage transitions. Once used for final review, record that use and do not continue describing the same examples as unseen.

Quote the actual available GPU, current rate, one-run scope and hard cap for fresh approval before any paid session. Retrieve/checksum artifacts before shutdown and confirm actual stopped runtime. Previous RunPod approval and stopped-pod pricing do not authorize this new height experiment.

## Reproduce the offline result

From the RL project directory:

~~~bash
.venv/bin/python scripts/analyze_cliff_distance.py \
  logs/walking-blocks-20261024/artifacts/logs/walkthrough-session \
  --output docs/cliff-distance-rescore-20261024.json
uv run --with pytest pytest tests/test_cliff_distance_analysis.py -q
~~~

Validation: **17 passed**. An independent NumPy cumulative-invalidity reconstruction agreed on success and first completion sample for all **216 episodes**, and all **15 input-file SHA256 fingerprints** were reverified. Tests cover genuine-contact requirement, first-contact anchoring, net distance versus path length, unrestricted speed and airborne phases, incomplete/missing evidence, latched failures, goal-sample vetoes, staggered event timing, later physical failures versus later boundary exits, and all-attempt denominators.

No new pod or training run was started. The previously used pod remains stopped.
