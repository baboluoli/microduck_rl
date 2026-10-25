# Handover: height ladder from the course-walk policy — 2026-10-25

**Goal:** raise the cliff drop from 2 cm toward 10 cm, starting from the
from-scratch course-walk policy (`model_2000`), **without losing its clean
walking**. Use the existing gated ladder (`scripts/run_height_curriculum_session.py`,
stages `2.5 → 3 → 3.5 → 4 → 5 → 6 → 7 → 8 → 9 → 10 cm`), training on the
**course-walk task** rather than the old course task. **User decision
(2026-10-25): start directly at 2.5 cm**, skipping the 2.25 cm stage, because
the untrained policy already gets down 3 cm in 15–20 of 24 attempts (see the
probe below). A few small code changes are needed first (Part A), then one GPU
session on **Vast.ai** (Part B). Reaching 10 cm will take several sessions. Each
session continues from the `curriculum.json` and `selection.json` that the
previous one saved.

The user is new to robotics: explain results in plain language. **The session
runs on Vast.ai, not RunPod** (user decision 2026-10-25: RunPod had $0.31 left;
Vast had $10 credit). Per the root `AGENTS.md` "Vast.ai" section, **ask the user
before renting an instance**, for every session, and destroy it when done.
Stopped Vast instances still bill for disk. The runner caps a session at 2 h,
so one session costs about 2 × the offer's $/h.

## Starting point (user decision 2026-10-25: "the walk is fine for now")

Kept copy, local and gitignored:

```
logs/keep/course-walk-scratch-model_2000/model_2000.pt
  sha256 49f742155ebfdf74cbd7689cdf2f839c2b979b3d32177eabcee339d638df52d4
logs/keep/course-walk-scratch-model_2000/curriculum-start-25mm.json   (fresh Curriculum(upper_mm=25))
  sha256 eba91ecee02e3a896a6f245db1a31260540da346f62e2d920cee09ecbe39faae
```

At the 25 mm stage the sampler draws 75% of resets from the frontier rows
(2.3–2.5 cm) and 25% from the easier rows (2–2.25 cm). Each check covers:
- the frontier: 2.5 cm at starts 45/55/65 cm;
- regression: 2 cm and 2.25 cm at the 55 cm start.

So 2.25 cm stays watched even though its stage is skipped.

- Training: `Mjlab-CliffOracleCourseWalk-MicroDuck` from random weights, 2048
  envs, seed 42, RL commit `d06755a`. It stopped at update 2000 of a planned
  3000, at the user's request.
- Checkpoint contents: `iter` 2000, `common_step_counter` 48024, so **every
  walking schedule is at its final stage**. The last one, `head_pose_range`,
  ends at update 2000. Its own `course_curriculum` is the default (20 mm).
- Run folder and logs:
  `logs/course-walk-scratch-20261025/artifacts/` (train.log, eval-2000, all
  checkpoints 0–2000). Videos are in `logs/course-walk-scratch-20261025/videos/`.

Evaluation of `model_2000` (12 s course rule, seeds 20–22, 24 attempts each):

| | Old source (hold-height block 0) | `model_2000` |
|---|---|---|
| 2 cm | — | 216/216 |
| 2.25 cm | 214/216 | 211/216 (5 tilts, each with a head impact) |
| Speed before edge / after landing | 0.435 / 0.582 m/s | **0.256 / 0.258 m/s** |
| Yaw rate | 1.19 | **0.18** |
| Action std | 1.05 | **0.14** |
| Time to goal | 3.26 s | 6.3 s |

The command mix is 0.18–0.24 m/s (80%) and 0.3 m/s (20%), so the pace is about
right. Against the ladder gate, 2.25 cm already **passes on development seeds**
at all three starts (45: 71/72, 2 impacts; 55: 69/72, **3 impacts = the
limit**; 65: 72/72). Together with the 3 cm probe, that is why the ladder skips
ahead to 2.5 cm. The 2.25 cm stage was never confirmed on seeds 40–42. If 2.5 cm
stalls, falling back to 22.5 mm is the user's call.

Training curve, for context: speed tracking peaked around updates 820–1450. It
dipped after the final smoothness step at 1500 and only partly recovered.

## Two problems the existing runner would cause

### 1. The first block warm-starts and resets the walking schedules

`run_height_curriculum_session.py` stages the source at `iteration=0` and runs
block 1 (and the smoke run) with `MICRODUCK_WARM_START=1`. Both restart
`common_step_counter` at 0. On the walk task that drops `action_rate_l2` back
to −0.1 (10× weaker than the −1.0 `model_2000` was trained under). It also
shrinks `com_range`, `head_com_range` and `head_pose_range` back to stage 0.
This is exactly the "schedules frozen at their easiest stage" mistake that the
scratch run was made to fix (`cliff-course-walk-scratch-handover-20261025.md`,
cause 3). The warm start was right for crossing from another task. Here source
and target are the **same task**, so this is a **resume**: keep the source's
iteration and counter.

### 2. The drop-height input has never been trained

The oracle interface appends `[edge distance, drop height]` (actor index 62,
critic index 77). Since `model_2000` saw only 2 cm:
- The normalizer has mean 0.0200 and **variance 0.0** for that input.
  `count` is 9.8e7, so new samples barely move it, about 5% weight per 100
  updates.
- The normalized input is `(h − 0.02)/(0 + 0.01)`: about **1 unit per cm above
  2 cm**, so 0.25 at 2.25 cm and 8 at 10 cm.
- Its first-layer weights never received a gradient (the input was always 0).
  They are still at their random initial size (mean |w| 0.064, vs 0.067 for
  the trained columns). AGENTS.md: "a command input that is never non-zero has
  dead weights forever".

So at taller drops this input feeds **random noise** into every hidden unit, and
the noise grows with the height. At 2.25 cm it is negligible, which is why the
eval looks fine. Untrained local probe on 2026-10-25 (CPU, seed 20, 55 cm start,
24 attempts, one seed, so this is noisy evidence):

| Drop | as is | input columns zeroed |
|---|---|---|
| 2 cm | 24/24 | 24/24 |
| 3 cm | 15/24 | 20/24 |
| 4 cm | 16/24 | 14/24 |
| 5 cm | 11/24 | 16/24 |

The policy is effectively blind to the height but partly copes anyway. Zeroing
does no harm at 2 cm and gives the network a clean, zero starting point to
*learn* how to use the height. Recommendation, **the user's call**: zero both
columns and leave the normalizer alone. Its 1 unit/cm scale is reasonable up to
10 cm. The probe script is kept at
`logs/keep/course-walk-scratch-model_2000/blind_probe.py`; reuse its logic in A2.

## Part A — code (local, before any GPU instance)

Work in `tinyduck/lemming/rl`.

### A1. `--resume-source` in `scripts/run_height_curriculum_session.py`

- New flag `--resume-source`. With it, read `iter` from the source checkpoint
  (`torch.load(...)['iter']`, 2000 here) and:
  - start `previous = dict(checkpoint=copied, iteration=source_iter)`;
  - run block 1 **and** the smoke run with `warm=False`, staged at
    `source_iter`. `stage()` already writes `iter` and
    `common_step_counter = iter*24`.
  - record `source_iteration` and `resume_source` in `session.json`.
- Without the flag, nothing changes. Old sessions stay reproducible.
- After each training call, assert that the schedules really are at their final
  stage. Simplest: grep the training log's first iteration for
  `Episode_Reward/action_rate_l2` and assert it is in line with weight −1.0, or
  log the counter from the runner. Prefer a direct check over inference.
- Keep `run_blocks` pure. Factor the start iteration and warm choice into a
  small helper, e.g. `start_point(payload, resume)`, and test it on CPU:
  - resume → (source iter, warm False);
  - default → (0, warm True).

### A2. `scripts/prepare_height_input.py` (if the user approves the zeroing)

- Input: a checkpoint. Output: a new checkpoint plus its sha256.
- Find the drop-height column in actor and critic. **Assert** it is the last
  column (62 and 77), has normalizer mean ≈ 0.02 and variance ≈ 0. Fail
  loudly otherwise, rather than guessing.
- Set those first-layer weight columns (`mlp.0.weight[:, i]`) to 0 in both
  `actor_state_dict` and `critic_state_dict`. Leave the normalizer, the
  optimizer state, `iter` and `infos` untouched.
  - Adam's moments for those columns are 0 (no gradient ever reached them), so
    zeroing needs no optimizer surgery. Verify this in the script and print it.
- CPU test: on a tiny synthetic checkpoint, only those columns change, and a
  wrong layout raises.
- Record the source sha, output sha and indices in a JSON next to the output.

### A3. Vast.ai support in `scripts/cliff_course_pod_control.py`

`retrieve` only uses SSH/rsync and works on Vast as is. `watch` and `stop` call
`runpodctl`. Add `--provider {runpod,vast}` (default `runpod`, so old commands
are unchanged). For `vast`:
- stop runs `vastai stop instance ID`;
- confirm with `vastai show instance ID --raw` until `actual_status` reads
  stopped/exited. Check the exact field and value on the first real instance,
  and fail loudly if it never appears, as the RunPod path does.
- It must still **never destroy** anything. Destroying stays a separate,
  explicit step after retrieval.

CPU-test the provider dispatch with `subprocess.run` stubbed.

### A4. Tests and commit

```
uv run --with pytest pytest tests/test_nan_forensics.py tests/test_cliff_course.py \
  tests/test_cliff_course_process.py tests/test_height_curriculum.py -q
```

Expect 114 passed and 2 CUDA-skipped, plus the new tests. Commit in the RL
submodule with `nihiluis`, then bump the submodule in the parent repo, as for
earlier runs.

## Part B — the session on Vast.ai (after Part A is committed)

Follow the *structure* of `cliff-drift-penalty-run-handover-20261024.md` steps
1–9 and its **Gotchas**, with the Vast.ai commands from the root `AGENTS.md`.
Work from `tinyduck/lemming/rl` with `S=logs/height-ladder-walk-YYYYMMDD`.

1. **Local tests, clean tree, `COMMIT`:** as step 1 there.
2. **Ask the user**, then check credit (`vastai show user`: `credit`; $10 on
   2026-10-25) and pick an offer:
   ```
   vastai search offers 'gpu_name=A40 num_gpus=1 reliability>0.98 inet_down>200 cuda_max_good>=12.8 disk_space>=40' -o 'dph+'
   ```
   - An RTX 4090 or RTX A6000 also works. The timings below are A40 numbers;
     on another GPU, re-measure from the smoke run and block 1.
   - The runner's per-block time budget (`BLOCK_BUDGET_S`, 17 min) is
     conservative on faster GPUs.
3. **Rent:**
   ```
   vastai create instance OFFER --image nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 --disk 40 \
     --ssh --direct --label height-ladder-walk --env '-e NVIDIA_DRIVER_CAPABILITIES=all'
   ```
   `NVIDIA_DRIVER_CAPABILITIES=all` asks the container runtime to expose
   NVIDIA's graphics (EGL) libraries as well as CUDA. Without it, the EGL
   preflight is likely to fail even with Mesa installed. This is untested on
   Vast. If the preflight fails, destroy the instance and try another host;
   don't debug the image. Note the instance id.
4. **Watchdog first:**
   `D=$(( $(date +%s) + 7200 )); uv run --no-sync python scripts/cliff_course_pod_control.py watch --provider vast --pod-id ID --deadline-epoch $D`
   (A3), in the background.
5. **SSH:** `vastai ssh-url ID` prints `ssh://root@HOST:PORT`. Write the same
   `ssh_config`/`known_hosts` pattern as before. Use the private key that
   matches one of `vastai show ssh-keys`.
6. **Upload and bootstrap:**
   - `mkdir -p /workspace/rl/sources` (the CUDA image has no `/workspace`).
   - Upload the source tarball, the inputs below and `COMMIT`. Verify each sha
     on the instance.
   - Then `apt-get install -y curl ca-certificates` (not in the CUDA image)
     **plus** the AGENTS.md EGL packages, install uv, `uv sync --frozen`, and
     run the CUDA+EGL preflight. Stop if it fails.
   - Inputs:
     - the prepared checkpoint from A2 (or `model_2000.pt` itself, if the
       user declines the zeroing);
     - `curriculum-start-25mm.json`, renamed to `curriculum.json`;
     - `COMMIT`.
7. **Session command:**
   ```
   cd /workspace/rl; export PATH=/root/.local/bin:$PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
     WANDB_MODE=offline PYTHONUNBUFFERED=1 MICRODUCK_GIT_HEAD=$(cat sources/COMMIT)
   nohup bash -c 'uv run python scripts/run_height_curriculum_session.py --source sources/model.pt \
     --source-sha SHA --curriculum sources/curriculum.json --output logs/height-ladder-walk \
     --deadline-epoch DEADLINE --max-nan-resets 5 --label height-ladder-walk \
     --task walk --resume-source > /workspace/session.log 2>&1; \
     echo $? > /workspace/session-exit' >/dev/null 2>&1 &
   ```
   This runs, in order:
   - GPU tests (~3 min);
   - smoke run and ONNX export;
   - **baseline check of the source at 2.5 cm** (~8 min). Nobody has measured
     2.5 cm on all seeds yet, so this is the first real number;
   - 100-update blocks, each followed by a development check: 2.5 cm ×
     45/55/65 plus 2 cm and 2.25 cm at 55. When the frontier passes, a
     confirmation on seeds 40–42 follows, then the advance.
8. **Timing (A40, 2048 envs):**
   - Real pace is about **2.5 s/update** (the log's "Iteration time" says
     ~2.3), so a 100-update block takes about 4.3 min.
   - One evaluation file (24 episodes) takes about 35 s, so a development
     check (15 files) takes about 9 min and a confirmation about 5 min.
   - A stage takes at least about 18 min, and about 80 min are usable. Expect
     **at most ~4 stages** (2.5 → 3 → 3.5 → 4 cm, maybe 5 cm) if every check
     passes first time.
9. **Monitoring** (every ~2 min):
   - `session-exit`, `Traceback|RuntimeError`, NaN resets (the runner stops
     after 5 per block), and `stages.jsonl`.
   - The `[course] Actual sampled height counts` line must fit the current
     stage.
   - Every penalty `Episode_Reward/*` must be ≤ 0.
   - **The first log lines of block 1 must show the final-stage smoothing**:
     `action_rate_l2` sums of about −0.5 to −0.9 per episode, as at the end of
     the scratch run. Something near −0.1 means the counter reset. Stop at
     once.
   - **Walking quality per block** (`quality` in `stages.jsonl`):
     - action std should stay near 0.14. The old course sessions crept up
       +0.1 per block, so report any creep.
     - speed before the edge and after landing should stay near 0.25 m/s.
     - yaw rate should stay near 0.18.
10. **Afterwards:**
    - The runner writes `SHA256SUMS`, so `cliff_course_pod_control.py
      retrieve` works here, unlike for plain `train` runs. Pass `--ssh-host`,
      `--ssh-config`, `--remote-root /workspace/rl` and
      `--output-relative logs/height-ladder-walk`.
    - Then copy the checkpoints under `logs/rsl_rl/oracle_course_walk` with
      `tar` over SSH and diff the sha256 lists.
    - Stop the instance (`... stop --provider vast --pod-id ID`, or
      `vastai stop instance ID`) and kill the watchdog.
    - Then ask the user and **destroy** the instance with `vastai destroy
      instance ID`. Stopped Vast instances still bill for disk.
    - Record videos locally on the Mac CPU for the best checkpoint at its
      highest passed height, plus one failure:
      `MUJOCO_GL=cgl uv run --no-sync python scripts/record_course_videos.py CKPT --label ladder-N --height H --output $S/videos --max-failures 2 --with-success`.

**Gotchas seen on 2026-10-25:**
- `ssh host 'pkill -f "train Mjlab…"'` also kills its own remote shell. Use
  `pkill -f "[t]rain Mjlab…"`.
- `ssh -n` breaks a heredoc fed on stdin. Use plain `ssh` for `bash -s <<EOF`.
- In zsh, `echo ====` fails (`=word` expansion). Quote it.

**Next sessions:** continue from that session's `curriculum.json` and its
`selection.json` `last` checkpoint (or `best`, which is the user's call), again
with `--resume-source`. The resume picks up the cumulative iteration, so the
schedules stay final. Do **not** re-zero the height input once it has been
trained.

## What to report

This is one seed, so differences are evidence, not proof.
- Highest height **passed** (development + confirmation), and blocks per stage.
- Frontier results at the top height: successes, impacts, falls, side exits,
  timeouts. Say plainly if it stalled and where.
- Regression at 2 cm: did it forget the easy drop?
- **Did the walking stay clean?** Action std, speed vs command (before and
  after landing), yaw rate and air time per block, compared with `model_2000`
  above. The past ladders failed by degrading the walk, not on height, so this
  is the main question.
- Stepping off or jumping: compare the longest airborne stretch at the edge with
  the flat approach. Free fall from height h takes √(2h/g): 64 ms at 2 cm,
  143 ms at 10 cm.
- NaN resets, and whether they cluster at taller drops.
- Videos: does it look clean to the user?

## Risks

- **Episodes are 8 s in training and 12 s in the evaluation.** That is fine for
  the 65 cm start (edge at about 2.6 s), but the policy never trains on seconds
  8–12. The scratch eval showed no timeouts, so it walks on regardless. Watch
  timeouts anyway.
- **Velocity tracking vs. landing.** The walk task charges speed above the
  command. Taller drops may need a slower, more careful step, which the reward
  allows. A forced burst of speed on landing would cost a little. If landings
  fail with good walking, report that before touching rewards.
- **Skipping 2.25 cm.** 2.25 cm passed on development seeds only, with impacts
  at the limit (3 at the 55 cm start). It is now a regression condition, so a
  failure there twice in a row stops the session. If that happens early, it
  says the jump was too big, not that walking broke. Report it, and let the
  user choose between restarting at 22.5 mm or more blocks.
- **Vast hosts vary** (driver, disk speed, network). If the smoke run or block
  1 is much slower than the A40 numbers, re-plan the stage count before
  relying on the time budget.
- **Height-input scale.** With the normalizer frozen at std 0, 10 cm reads as
  8 normalized units. That is fine for a learned weight. If training at the
  higher stages shows instability right at stage changes, consider rescaling
  that normalizer dimension, as a user decision.
- NaN hypothesis (duplicate foot-mesh contacts on the lower floor, unconfirmed):
  harder landings may make it more frequent. If resets cluster at a stage, stop
  and investigate rather than raise `--max-nan-resets`.
- Gate numbers (≥ 65/72, no seed below 21, ≤ 3 head/trunk impacts) are
  experiment choices. Do not relax them silently.

RunPod pods: `uipvsf3d1z8g1t` (course-walk-scratch) is **stopped**, and
everything from it is local. The user has not decided whether to delete it, nor
the older `ytfxzw61zdoe5k` and `823jn94ii5ko8o`.
