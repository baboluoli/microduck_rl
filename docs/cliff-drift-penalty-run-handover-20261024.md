# Handover: sideways-drift penalty run — 2026-10-24

**Goal:** continue the cliff-course *source* policy for 100 PPO updates with a new
anti-crab-walking penalty, then evaluate it on the standard 432-episode battery and
compare against the source and the old `model_99`.

The user is new to robotics: explain results in plain language. Per the root
`AGENTS.md`, **ask the user before creating a RunPod pod**. Expected cost is about
$0.40–0.60 on one A40 at $0.59/h, with a hard 2-hour cap.

## Background, in short

- **Source** (`model_source.pt`, sha `ef70ae03…`) is a good 2 cm cliff walker:
  426/432 successes.
- **`model_99`** is the source after 100 more updates on the height curriculum (up
  to 2.25 cm). It got worse, with 322/432 successes and 97 exits off the side of
  the course.
- Telemetry shows `model_99` **crab-walks**: it faces forward (slightly left) but
  steps right. Its sideways stepping speed is −0.132 m/s, against −0.056 m/s for
  the source. It is not turning.
- **Cause:** the course reward pays world-X progress only
  (`cliff_forward_progress`), so sideways travel was free. Until commit `ab4a15e`,
  leaving the course also counted as a timeout rather than a failure.
- **Fixes now in the code:**
  - The boundary exit is a terminal failure (from `ab4a15e`).
  - New: `course.sideways_drift`, weight 5. It penalizes the ~1 s moving average of
    heading-frame sideways speed. It returns values ≤ 0, so it needs a positive
    weight.
- **NaNs.** A NaN stopped the previous attempt at update 14: robot 50, right foot
  mesh contacts on the lower floor (duplicate contacts, solver hit 30 iterations).
  The user chose to **set this aside for now**. The session therefore runs with
  `--max-nan-resets 5`: `nan_state` resets the broken robot and training continues,
  and the session stops if more than 5 NaN events occur. A NaN in losses or rewards
  always stops it. Forensics are still saved to `nan-dumps/`. The earlier dump is
  in `logs/boundary-ab-20261024/artifacts/logs/boundary-ab/nan-dumps/`. Its
  pre-step forces were normal, and the state went nonfinite within one 5 ms
  step, while the right foot had 3 near-coincident contacts with `terrain_1`.
  Each foot has 4 overlapping collision meshes. This is a hypothesis, not
  confirmed.

## Inputs (local, gitignored, from the RL directory)

```
logs/course-session-20261024/artifacts/logs/rsl_rl/oracle_course/course-session-source/model_source.pt
logs/course-session-20261024/artifacts/logs/course-session/curriculum.json
```

Baseline evaluations with telemetry, for comparison:
`logs/boundary-ab-20261024/artifacts/logs/boundary-ab/eval-source` and `.../eval-old-model99`.

## Run it step by step (do NOT use `scripts/launch_boundary_ab.sh`)

The launcher is hard-wired for the boundary A/B. It re-runs the baselines, uses
boundary-ab names, and stops the pod on *any* exit, including being killed. The
user prefers running the steps as individual commands, so failures can be
handled as they happen. Work from `tinyduck/lemming/rl`. Pick a session folder,
e.g. `S=logs/drift-penalty-YYYYMMDD`.

1. **Local tests and a clean tree:**
   `uv run --with pytest pytest tests/test_nan_forensics.py tests/test_cliff_course.py tests/test_cliff_course_process.py -q`.
   Then `git status --porcelain -- src scripts tests` must be empty. Write
   `git rev-parse HEAD > $S/COMMIT`.
2. **Create the pod:**
   ```
   uv run runpodctl pod create --name drift-penalty --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
     --gpu-id "NVIDIA A40" --cloud-type SECURE --container-disk-in-gb 40 --volume-in-gb 20 --ports '22/tcp' \
     --wait --wait-timeout 15m
   ```
   Note the pod id, IP and port from the output.
3. **Watchdog first.** Run this in the background; it stops the pod at the deadline:
   ```
   D=$(( $(date +%s) + 7200 )); echo $D > $S/deadline_epoch
   uv run --no-sync python scripts/cliff_course_pod_control.py watch --pod-id POD --deadline-epoch $D
   ```
4. **SSH config:** copy the `ssh_config`/`known_hosts` pattern from
   `scripts/launch_boundary_ab.sh`: host alias, key `~/.runpod/ssh/runpodctl-ssh-key`,
   `ssh-keyscan` for host keys.
5. **Upload.** Run
   `git ls-files -co --exclude-standard . | grep -v '^logs/' | tar -czf $S/source.tar.gz -T -`.
   Then `scp` the tarball to `/workspace/`, and copy `model_source.pt`,
   `curriculum.json` and `COMMIT` to `/workspace/rl/sources/`. Verify the sha256 on
   the pod.
6. **Bootstrap and preflight on the pod:** extract the tarball into `/workspace/rl`
   (`tar --no-same-owner -xzf`), then follow the **RunPod headless graphics
   preflight** in `AGENTS.md`: apt EGL libraries, install uv, `uv sync --frozen`,
   then the CUDA+EGL check. Stop if the preflight fails.
7. **Start the session under nohup:**
   ```
   cd /workspace/rl; export PATH=/root/.local/bin:$PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
     WANDB_MODE=offline PYTHONUNBUFFERED=1 MICRODUCK_GIT_HEAD=$(cat sources/COMMIT)
   nohup bash -c 'uv run python scripts/run_boundary_ab_session.py --source sources/model_source.pt \
     --curriculum sources/curriculum.json --output logs/drift-penalty --deadline-epoch DEADLINE \
     --skip-baselines --max-nan-resets 5 --label drift-penalty > /workspace/session.log 2>&1; \
     echo $? > /workspace/session-exit' >/dev/null 2>&1 &
   ```
   The session runs these stages in order: GPU tests (~2.5 min), 5-update smoke at
   64 envs plus ONNX export, then 100 updates at 2048 envs with seed 42 and warm
   start (~5 min), then `eval-new-model99` (~10 min).
8. **Monitor** roughly every 60 s. Check `/workspace/session-exit`, the newest
   `logs/drift-penalty/*.log`, `/tmp/mjlab/nan_dumps`, and grep for
   `Traceback|RuntimeError|\[wait_checked\] NaN event`. In the training log, check
   that `Episode_Reward/sideways_drift` is **≤ 0** and its magnitude drops over
   training. Check that every other penalty term is ≤ 0 too.
9. **Retrieve, then stop:**
   ```
   uv run --no-sync python scripts/cliff_course_pod_control.py retrieve --pod-id POD --ssh-host ALIAS \
     --ssh-config $S/ssh_config --remote-root /workspace/rl --output-relative logs/drift-penalty --destination $S/artifacts
   uv run --no-sync python scripts/cliff_course_pod_control.py stop --pod-id POD   # must print "Confirmed stopped"
   ```
   Then kill the watchdog. Ask the user before deleting the pod.

## Analysis

```
A=logs/boundary-ab-20261024/artifacts/logs/boundary-ab
uv run python scripts/analyze_course_drift.py $A/eval-source $A/eval-old-model99 $S/artifacts/logs/drift-penalty/eval-new-model99
```

| Baseline | Successes | Boundary exits | World vy | from turning | from crabbing |
|---|---|---|---|---|---|
| source | 426/432 | 1 | −0.024 | +0.032 | −0.056 |
| model_99 | 322/432 | 97 | −0.088 | +0.045 | −0.132 |

Report the following, without overclaiming. This is one seed, so a difference is
evidence, not proof.
- **Did crabbing shrink?** Is it at or below the source's −0.056 m/s?
- **Did boundary exits drop?**
- **Did successes stay near the source?** The penalty must not make the robot timid,
  so also check timeouts and falls.
- **How many NaN resets occurred?**

## Gotchas hit on 2026-10-24

- **Pods have no `.git`.** Evaluation reads `MICRODUCK_GIT_HEAD` instead; export it,
  as in step 7. Forgetting it crashes `eval_cliff_course.py`.
- **`ssh` inside a heredoc or loop swallows stdin.** Use `ssh -n` for commands that
  need no input.
- **zsh does not word-split `$SSH`-style variables.** Write the full `ssh -F … host`
  command each time, or use bash arrays.
- **`scripts/launch_boundary_ab.sh` stops the pod in its EXIT trap.** If you ever run
  it and need to take over by hand, `kill -9` it. A normal kill stops the pod.
- **Harmless noise:** `tar` warnings about `LIBARCHIVE.xattr` (macOS metadata) and
  uv "failed to hardlink" warnings.
- **Restarting a stopped pod can fail** with "not enough free GPUs on the host". If
  so, create a fresh pod instead.
