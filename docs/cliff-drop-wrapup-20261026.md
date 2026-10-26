# Cliff drop: wrap-up of all attempts, 2026-09-26 → 2026-10-26

**Where it ended:** one policy steps off a ledge and walks 1 m after landing.
It passes the gate at every height from **2 to 18 cm**. At **24 cm** it missed
the gate only at the 65 cm start, mostly on timeouts. This is a simulation
result with the **oracle** interface: 63 inputs, given the exact edge distance
and drop height. It is not deployable on the 61-input robot runtime.

What finally worked: **train from scratch on a corrected walking task
(2000 updates), then resume (not warm-start) through a gated height ladder.**
Everything before that was fine-tuning old checkpoints whose problems came
from the reward and config, not from the cliff.

Gate per condition: ≥ 65/72 successes, no seed below 21/24, ≤ 3 head/trunk
impacts. It must pass on development seeds 20–22 and confirmation seeds 40–42.

## Timeline (one line per attempt)

| Date | Attempt | Result | Lesson |
|---|---|---|---|
| 09-26 | Scratch, 3000 updates, 2–10 cm terrain curriculum, 35% mid-air starts | Curriculum stuck at ~2 cm. Update 2998 worse than 999 (edge reached 3/64 vs 41/64) | More updates ≠ better. Training metrics lied |
| 10-03 | Fine-tune 999, frozen schedules, zero yaw | +100 best, +200 collapsed (approach 28%) | Robot walked *away* from the ledge: no heading signal |
| 10-03 | Steering repair: heading controller, world-X progress, no mid-air starts | 2 cm completion 24 → 89/144 | First real gain: reward the right *direction* |
| 10-22/24 | Oracle task (edge distance + height as inputs), scratch | ckpt 500: 65/72; 600: 37; 700: 14 | Same regression with more updates |
| 10-24 | Diagnostics on 500 without ending at success | Reaches old goal 22/24, but 18 of those later fall | Success gate (40 cm + 0.3 s, then stop) certified only a short manoeuvre |
| 10-24 | 12 s sustained-walking rule, all-or-nothing bonus | Baseline 0/72 → pilot not launched | Sparse reward with no reachable signal |
| 10-24 | Partial credit (8 pts per clean second + 40 final) | Strict 1/72. Robot sped up and ran off the tile | Progress reward had no overspeed cost → rushing |
| 10-24 | Re-score on "land + walk 1 m" | Existing policies ~90% already | The goal was easier than the rules being tested |
| 10-24 | Course task (finish line), 2–2.25 cm curriculum | Source passed 2 cm. +100 updates: 92 side exits, then NaN | Terrain bug and free sideways motion (below) |
| 10-24 | Boundary = failure + sideways-drift penalty (w 5) | 322 → 408/432, crabbing −0.132 → −0.031 m/s | Fixed the drift |
| 10-25 | Height ladder session 1 (old course task) | 2.25 cm passed; 2.5 cm collapsed at *all* heights | Walking degraded, not cliff skill. Action std 1.05 → 1.43 |
| 10-25 | Hold 2.25 cm, entropy 0.003, 1 m goal, pay-as-you-go | Std fell, but robot **loitered** 0.9 m past landing until timeout | Success termination ended per-step income → stopping short paid more |
| 10-25 | **Course-walk task from scratch**, 2000 updates | 2 cm 216/216, pace 0.26 m/s (≈ command), std 0.14 | Clean base policy |
| 10-25/26 | Ladder on Vast, `--resume-source`, height input zeroed | 2 → 18 cm passed, 24 cm close. 4 sessions, ~$2.70 | The approach that works |

## Mistakes and what to do instead

### Reward and task design (the main cause of failure)

1. **Body-frame velocity reward** paid walking in any direction the robot faced,
   including away from the ledge. → Reward world-frame progress toward the goal,
   with a heading controller.
2. **Progress counted world-X only**, so sideways crab-walking was free and grew
   with training. → Penalise sideways speed (`sideways_drift`).
3. **Capped progress with no overspeed cost.** Speed was 0.45–0.6 m/s against a
   0.18–0.24 command. A lump-sum bonus plus discounting also rewards arriving
   early. → Use the walking recipe's velocity tracking: it peaks at the command
   and charges overspeed.
4. **Ending the episode on success** stops all per-step "stay upright" income.
   The policy learned to stop just short of the goal (8 cm break-even, observed
   loitering at 0.73–0.99 m). → In training, run every episode to the timeout,
   with no success termination.
5. **Success gates that are too short** (40 cm + 0.3 s, then the episode ends).
   Videos cut at success hid falls that came later. → Keep watching after
   success, and use full-length videos.
6. **All-or-nothing sparse bonus** that the baseline never reached (0/72). →
   Score the frozen policy under a new rule *before* training on it.
7. **Boundary exit flagged as `time_out=True`.** PPO then bootstraps future value,
   so leaving the course was not a real failure. → Make exits terminal.
8. **35% synthetic mid-air starts** inflated the success metrics. **Demoting the
   curriculum on every non-success** kept the terrain at level ~0.
9. **Finish line 3 m away**, unreachable at the commanded pace in 12 s. →
   Size distances and time limits from command speed × the longest start.

### Config carried over from the walking policy

10. **Walking schedules frozen at stage 0.** Smoothing was `action_rate_l2` −0.1
    instead of −1.0, 10× weaker than the source was trained under. → When
    fine-tuning, collapse curricula to their **final** stage.
11. **Warm start resets `common_step_counter`**, which silently resets those
    schedules. → For the same task, **resume** (`--resume-source`) and check the
    first logged `action_rate_l2` values.
12. **Entropy 0.01 while fine-tuning** inflated action std by ~0.1 per 100
    updates, and the gait got janky. → Use 0.003 for fine-tuning. A from-scratch
    run with proper schedules was fine at 0.01.
13. **Dead input columns.** An input that never varied (yaw command, drop height)
    has variance 0 in the normalizer and untrained random weights. Once it
    starts to vary, it feeds noise that grows with the value. The yaw command
    became 60 normalized units at once. → Zero those first-layer columns before
    introducing the input (`prepare_cliff_steering_checkpoint.py`,
    `prepare_height_input.py`).
14. **Fine-tuning a policy with baked-in bad habits** (rushing, noisy actions)
    cost about six sessions. → When habits persist after a reward fix, retrain
    from scratch on the corrected task. It cost one session.

### Evaluation and measurement

15. **Trusting training logs.** Termination counts and mean reward rose while the
    behaviour got worse. → Use a per-episode evaluator with true lower-floor
    contact and fixed seeds and starts, and select checkpoints by behaviour.
16. **More updates regressed every time** (999→2998, +100→+200, 500→600→700).
    → Evaluate each saved checkpoint and keep the best, not the last.
17. **Terrain bug:** course boxes were centered on mjlab's grid corner, so the
    flat border covered half the lower floor. Robots walked on flat ground while
    being told the drop was 2 cm. → Raycast-test the real surface across the
    whole terrain bank.
18. **Video labels came from the sampled outcome, not the replay.** A video
    named "success" showed a fall. CPU and GPU replays also differ. → Label
    videos from the replay's own result, and never pool CPU and GPU counts.
19. **Finite tile:** fast policies left it forward, so they were censored rather
    than counted as failures. Lower impact counts partly reflected shorter
    observed episodes.
20. **Train 8 s vs eval 12 s.** At 24 cm the policy learned a careful 0.14 m/s
    approach and timed out at 0.95 m of 1 m. The gate was sized for a different
    pace. *Still open, see below.*
21. Small bugs: float32 reward sum 80.00016 > 80 crashed the audit; ImageIO
    needed NumPy arrays instead of PIL images; `original_evaluator_outcome`
    said "complete" for episodes that later fell.

### Process and operations

22. **Too much ceremony.** There were long pre-registered plans, multi-condition
    gates and monolithic session runners. Meanwhile the obvious reward bugs
    (1–4) went unfixed for many sessions. The user prefers step-by-step
    commands and watching videos. The user's eye ("it's not the height, it's
    the walking") found the cause faster than the statistics did.
23. **The NaN check only scanned the log after a subprocess finished**, so 13
    more updates ran after a NaN. → Monitor output continuously. NaNs were then
    tolerated with `--max-nan-resets 5`. They never clustered at tall drops.
    The suspected cause (duplicate foot-mesh contacts) is still unconfirmed.
24. Ops gotchas, all of which cost time:
    - Missing EGL libraries (follow the preflight).
    - Pods have no `.git`, so export `MICRODUCK_GIT_HEAD`.
    - `pkill -f` matches its own shell over ssh, so use `[t]rain`.
    - `ssh -n` swallows input in loops and breaks heredocs.
    - zsh doesn't word-split variables.
    - The launcher's EXIT trap stopped the pod.
    - Stopped RunPod pods often can't restart (no free GPU).
    - A 20 s retrieval timeout cut off downloads.
    - Retrieval needs `SHA256SUMS`.
    - Vast `destroy` needs `-y`, and stopped Vast instances still bill for disk.
    - The working Vast recipe is in `cliff-height-ladder-24cm-handover-20261026.md`.

## Open items

- **24 cm:** the user must choose the evaluation fix. Recommended: 12 → 16 s
  eval episodes (`env_cfg.py:347`), and lengthen training to 12 s only if
  timeouts persist. Then one short session from `model_3596.pt`.
- **Walking quality:** yaw rate crept from 0.18 to 0.40, and action std from
  0.14 to 0.24.
- **Hardware:** 24 cm is about the robot's height; landing speed is ~2.2 m/s,
  and gear stress isn't modelled. Test the 10–14 cm policy first. Deployment
  needs a policy without oracle inputs (61-input contract).
- **Billing:** stopped RunPod pods and volumes are still retained (e.g.
  `ytfxzw61zdoe5k`, `823jn94ii5ko8o`, `w16n6lirviqqfa`, `uipvsf3d1z8g1t`, plus
  older ones). Delete them once you're sure nothing else is needed.
