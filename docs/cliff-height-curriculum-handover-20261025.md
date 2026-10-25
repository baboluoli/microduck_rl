# Handover: height curriculum toward 10 cm — 2026-10-25

**Goal:** starting from the drift-penalty policy, keep training in 100-update
blocks and raise the cliff height whenever the policy passes a test at the
current height. The ladder now goes up to **10 cm** instead of stopping at 3 cm.
This needs code changes first (Part A), then a pod session (Part B). Reaching
10 cm will take several sessions, and each one continues from the curriculum
state the previous one saved.

The user is new to robotics: explain results in plain language. Per the root
`AGENTS.md`, **ask the user before creating a RunPod pod**, for every session.
One session costs at most about $1.20 (A40 at $0.59/h, hard 2-hour cap).

## Decisions the user approved on 2026-10-25

1. **Gated curriculum.** Height goes up only after the policy passes at the
   current top height on two separate sets of seeds. Training time alone never
   raises the height.
2. **Proportional steps up to 10 cm** (each step about 15–25% harder):
   `2.25 → 2.5 → 3 → 3.5 → 4 → 5 → 6 → 7 → 8 → 9 → 10 cm`.
   Fixed 0.25 cm steps would need 32 stages, and near 10 cm a 0.25 cm step is
   only a 2.5% change. 10 cm is a research target (40% of the robot's height),
   not a promise (see `cliff-distance-goal-and-height-plan-20261024.md`).
3. **A check of fixed cost after every block.** The old runner re-tested every
   mastered height at 3 distances × 3 seeds. That costs about 5 min per height,
   so near 10 cm one check would take over 2 hours. The new check costs about
   8 min no matter how high the curriculum is, plus about 5 min of confirmation
   when the top height passes.
4. **Stop on stall:** if a stage still fails after 5 blocks (500 updates),
   stop and report instead of spending time.
5. **The 2-hour cap per session stays.** Keep the `sideways_drift` penalty
   (weight 5) and the boundary-as-failure setting unchanged.

## Starting point

The drift-penalty run (`logs/drift-penalty-20261024/RESULTS.md`) produced:

| Policy | Successes /432 | Off the side | Falls | Sideways stepping |
|---|---|---|---|---|
| source | 426 | 1 | 3 | −0.056 m/s |
| old model_99 | 322 | 97 | 12 | −0.132 m/s |
| **drift-penalty model_99** | **408** | **5** | **19** | **−0.031 m/s** |

Against the curriculum gate (seeds 20–22, 24 episodes each, rule: ≥65/72
successes, no seed below 21, at most 3 head/trunk impacts), the new policy
passes all three 2 cm conditions. At 2.25 cm it passes at 45 and 55 cm starts
and fails at 65 cm only on impacts (5 > 3). So the first stage, 2.25 cm, is
nearly mastered.

Inputs (local, gitignored, from the RL directory):

```
logs/drift-penalty-20261024/artifacts/logs/rsl_rl/oracle_course/2026-10-24_22-01-47_drift-penalty-100/model_99.pt
  sha256 010a94116935924381dd16769bb4e79f1ee4e723491f6307b97eac097e78cdea
logs/course-session-20261024/artifacts/logs/course-session/curriculum.json   (upper_mm 22.5)
```

## Part A — code changes (local, before any pod)

Work in `tinyduck/lemming/rl`. Today the code caps heights at 3 cm in three
places. All three have to change, plus a new runner.

### A1. Height bank: `src/mjlab_microduck/tasks/cliff/course.py`

- `HEIGHTS` is the set of precompiled terrain rows (one row per height). Keep
  the existing 20–30 mm rows at 0.5 mm unchanged, so old checkpoints' row
  indices and `course_height_counts` stay meaningful. Append 31–100 mm at 1 mm.
  That is 21 + 70 = **91 rows**.
  *Changed 2026-10-25 (user decision):* above 3 cm the spacing is **2.5 mm**
  (32.5–100 mm), so **49 rows**. Every row is a real tile in every simulated
  world (×20 columns), and 91 rows risked slowing training. Build them from integer tenths of a mm, so
  `.0225` and friends compare exactly, as they do today.
- Add `STAGES = (.02, .0225, .025, .03, .035, .04, .05, .06, .07, .08, .09, .10)`.
- `reset(upper=…)`: accept any `upper` in `STAGES`, not the hard-coded
  2–3 cm tuple. Keep `fixed_height` restricted to `HEIGHTS`.
- **Sampler.** Today it assumes uniform 0.5 mm bins with a 2.5 mm frontier
  band (`top = round((upper-.02)*2000)`). Rewrite it in terms of stages:
  - *Frontier band:* rows with `previous_stage < h ≤ upper`, sampled 75% of
    the time.
  - *Easier band:* rows with `h ≤ previous_stage`, sampled 25% of the time.
  - At `upper = .02` everything samples 2 cm, as now.
  - At `upper = .0225` the bands must be exactly what they are today, so the
    current checkpoint trains on the same distribution. Add a test for this.
- The terrain platform is 0.3 m thick and `CliffDropTerrainCfg` already allows
  drops up to 0.10 m, so no terrain-geometry change should be needed. Verify it
  with the bank test (A6).
- `make_microduck_cliff_course_env_cfg` reads `len(course.HEIGHTS)`,
  `HEIGHTS[0]` and `HEIGHTS[-1]`, so it should pick up the new bank. Update its
  docstring ("immutable 0.5 mm height bank").
- `src/mjlab_microduck/tasks/__init__.py:356`: update the printed label
  "(20 to 30 mm, 0.5 mm bins)".

### A2. Curriculum: `src/mjlab_microduck/tasks/cliff/course_curriculum.py`

- Advance along `course.STAGES` instead of `upper_mm += increment_mm`. Replace
  the `upper_mm >= 30` stop with "last stage reached". Keep the
  `increment_mm` and `resolution_mm` fields so existing `curriculum.json` files
  still load, but mark them as unused.
- New fixed-cost check. `conditions()` becomes:
  - **frontier:** current top height × starts 45/55/65 cm (3 conditions);
  - **regression:** 2 cm at 55 cm, plus the previous stage at 55 cm (1–2
    conditions).
  - `advance()` requires the frontier on both development seeds (20/21/22) and
    confirmation seeds (40/41/42), using the existing `Result.passes` rule.
  - The regression conditions are checked on development seeds only.
- `regression()`: compare only the regression conditions. A condition is
  breached when it no longer passes `Result.passes`. Two breaches in a row
  ⇒ stop (keep the existing streak idea, applied to this subset).
- Add `blocks_at_stage` (incremented per block, reset on advance) and stop when
  it reaches 5 without advancing.

### A3. New runner: `scripts/run_height_curriculum_session.py`

Do not edit `run_boundary_ab_session.py` or `run_cliff_course_session.py`; both
are records of earlier experiments. Build the new runner from:
- `run_boundary_ab_session.py`: the `command()` wrapper, `wait_checked` with
  `--max-nan-resets`, NaN-dump copying, SHA256SUMS, the `stage()` and
  `train()` helpers;
- `run_cliff_course_session.py`: the block loop.

Arguments: `--source`, `--source-sha` (required, asserted), `--curriculum`,
`--output`, `--deadline-epoch`, `--max-nan-resets` (default 5), `--label`,
`--block-updates` (default 100), `--max-blocks-per-stage` (default 5).

Session flow:

1. GPU runtime tests (`tests/test_cliff_course.py tests/test_nan_forensics.py`).
2. 5-update smoke at 64 envs, plus ONNX export, with the curriculum staged at
   its current `upper`.
3. Loop:
   1. **Train one block** of 100 updates at 2048 envs, seed 42.
      - The first block is a warm start (`MICRODUCK_WARM_START=1`).
      - Later blocks resume from the previous block's checkpoint, staged with
        `iter` and `common_step_counter = iter*24`, as
        `run_cliff_course_session.train()` does.
      - The staged checkpoint carries the curriculum state, so after loading,
        `MicroduckCourseRunner.load` sets `upper`.
   2. **Development check** (seeds 20–22): the 3 frontier conditions plus the
      1–2 regression conditions.
   3. If the regression streak fails, stop.
   4. If the whole frontier passes, run **confirmation** (seeds 40–42, the 3
      frontier conditions). If that passes too, advance to the next stage and
      record this checkpoint as `best` for the stage it just mastered.
   5. Save `curriculum.json`, `selection.json` (`best`, `last`, `upper_mm`,
      `blocks_at_stage`) and append to `stages.jsonl` after every check.
4. **Time budget.** Before starting a block, require about 17 min
   (block + check + possible confirmation) before the internal deadline, which
   is the deadline minus 15 min reserved for retrieval. Otherwise write
   `provisional.json` and stop cleanly.
5. Stop at the last stage, on a regression stop, or on a stall
   (5 blocks without advancing).

Rough timings on an A40, measured on 2026-10-24: 100 updates ≈ 4.5 min, one
evaluation file (one height × one start × one seed, 24 episodes) ≈ 33 s.
- A development check is 5 conditions × 3 seeds = 15 files ≈ 8 min.
- A confirmation is 3 × 3 = 9 files ≈ 5 min.
- So a stage takes **at least ~17 min**.

About 85 min is usable per session, after bootstrap, tests, smoke and the
retrieval reserve. Expect **about 4 stages per session if every check passes
first time**, so roughly 3.5–4 cm in session 1. Fewer if some stages need more
than one block.

### A4. Evaluation script

`scripts/eval_cliff_course.py` needs no logic change: heights are passed with
`--heights`, must be members of `HEIGHTS`, and output files are named
`h{h:.4f}-…`. Check that floats like `0.035` round-trip through argparse to an
exact `HEIGHTS` member. If not, snap to the nearest member, or pass tenths of
a mm.

### A5. Analysis helper

Add `scripts/summarize_height_curriculum.py`. It reads `stages.jsonl` and every
`eval-*` folder, then prints one row per check: block, top height, frontier
successes per start, impacts, falls, side exits, regression results, and
whether the policy advanced. Reuse `analyze_course_drift.py` for crabbing per
stage.

### A6. Tests (CPU unless noted)

- Bank: `test_precompiled_bank_matches_declared_heights_and_footprint` and the
  raycast test parametrize over `HEIGHTS`, so they now cover all 91 rows.
  Construct the tile with `max_drop=HEIGHTS[-1]` instead of `.03`. If the
  raycast test becomes slow, keep every row in the footprint test and
  parametrize the raycast over `STAGES` plus every 5th row.
- `test_cuda_course_reset_geometry`: loop over `STAGES` instead of the
  2–3 cm tuple (GPU).
- Sampler: at each stage, every sampled row lies in the right band, the
  frontier share is ≈75%, and `.0225` matches today's distribution exactly.
- Curriculum: advances along `STAGES` in order; never past 10 cm; old
  `curriculum.json` (upper 22.5) loads; stall counter stops at 5; regression
  streak uses only the regression conditions.
- Runner: a CPU dry run with `train` and `evaluate` stubbed. It must cover
  advance → next stage, stall stop, regression stop, and the time-budget stop.
  Use fake time; no GPU.

Then run
`uv run --with pytest pytest tests/test_nan_forensics.py tests/test_cliff_course.py tests/test_cliff_course_process.py -q`
plus the new test file. Commit with `nihiluis` in the RL submodule, then bump
the submodule in the parent repo, as for earlier runs.

## Part B — run a session (after Part A is committed)

Follow `cliff-drift-penalty-run-handover-20261024.md` steps 1–9 exactly, with
these differences:

- **Session folder:** `S=logs/height-curriculum-YYYYMMDD`, pod name
  `height-curriculum`.
- **Inputs to upload into `/workspace/rl/sources/`:** the drift-penalty
  `model_99.pt` above (verify its sha on the pod), `curriculum.json`, and
  `COMMIT`.
- **Session command (step 7):**
  ```
  nohup bash -c 'uv run python scripts/run_height_curriculum_session.py --source sources/model_99.pt \
    --source-sha 010a94116935924381dd16769bb4e79f1ee4e723491f6307b97eac097e78cdea \
    --curriculum sources/curriculum.json --output logs/height-curriculum --deadline-epoch DEADLINE \
    --max-nan-resets 5 --label height-curriculum > /workspace/session.log 2>&1; \
    echo $? > /workspace/session-exit' >/dev/null 2>&1 &
  ```
- **Monitoring (step 8):** in addition to the earlier checks, follow
  `stages.jsonl`, and check that the `[course] Actual sampled height counts`
  line at each save matches the current stage's bands.
  - Reward signs: every penalty term must stay ≤ 0.
  - Do not read the per-episode `Episode_Reward/sideways_drift` sum on its
    own: it grows while episodes get longer after a reset. It fell from
    −0.25 to −0.10 after the first warm start.
  - Falls and NaNs may increase at taller drops. The session stops itself
    after 5 NaN resets.
- **Retrieve and stop (step 9):** use `--output-relative logs/height-curriculum`.
  Retrieve before anything else if the session stopped early. Ask the user
  before deleting the pod. The old pod `823jn94ii5ko8o` (drift-penalty) is
  stopped, and the user has not yet decided whether to delete it.

**Next sessions:** use that session's `selection.json` `best` (or `last` if it
was still improving at the current stage, which is the user's call) and its
`curriculum.json` as the new `--source`/`--curriculum`. Record the new sha.

## What to report

This uses one seed, so differences are evidence, not proof. Report:
- The highest height **passed** (development and confirmation), and how many
  blocks each stage took.
- The frontier results at the top height: successes, falls, side exits,
  timeouts. Say plainly if it stalled and where.
- Regression at 2 cm: did it forget the easy drop?
- Crabbing per stage (`analyze_course_drift.py`). Is it still at or below
  −0.056 m/s?
- NaN resets, and whether they cluster at taller drops.
- Whether the robot steps off or jumps. Compare the longest airborne stretch at
  the edge with the flat approach, as in the 2026-10-25 analysis: at 2 cm it
  was 80 ms against 60 ms, consistent with stepping off. Free fall from h takes
  √(2h/g): 64 ms at 2 cm, 143 ms at 10 cm.

## Risks to keep in mind

- At larger drops the landing impact grows. "Impact" in the gate means
  head/trunk contact force (head > 15 N, trunk > 20 N), not foot force, so
  hard but clean foot landings do not count against it. Falls might.
- The suspected NaN cause is duplicate foot-mesh contacts on the lower floor
  (unconfirmed hypothesis). It may get more frequent with harder landings. If
  NaN resets cluster at a stage, stop and investigate rather than raise
  `--max-nan-resets`.
- The terrain observation's normalizer has only seen 2–2.25 cm. It adapts
  during training (empirical normalization), but expect the first block at a
  new stage to look worse before it gets better.
- Gate numbers are experiment choices, not established optima. If a stage
  stalls, report the evidence and let the user decide: smaller step, more
  blocks, or reward changes. Do not relax the gate silently.
