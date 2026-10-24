# Course implementation and validation

Implemented locally on 2026-10-24. No GPU session or training was started.
The original oracle and walk-through tasks retain their behavior and identifiers.
New task: `Mjlab-CliffOracleCourse-MicroDuck`, experiment `oracle_course`.

## Geometry and episode rules

Coordinates below are relative to each tile's nominal spawn, **not** the robot's
random starting position. The compiled tile is 5 × 3 m, with supported width
2.7 m. The actual cliff is X=0.55 m; lower floor extends through X=4.05 m.
The finish is X=3.55 m, leaving **0.50 m** to the physical end. Side truncation
is |Y|>1.15 m, rear truncation X<−0.75 m, unexpected forward truncation X>3.8 m.
Success at the finish terminates normally and pays 80 points once, including
RewardManager's dt integration. Boundary exits remain in the attempt denominator.

Both feet's contact positions are tested against the lower box footprint and
actual row height, with the existing 8 mm vertical tolerance. First contact is
remembered; missing evidence remains missing. Impact and nonfinite invalidity
start at reset; tilt ≥45° invalidates from first contact, including that sample.
Inherited physical failures veto success on the same sample. No walking block
or timeout payment is present. Ending the episode does not train a physical stop.

The line provides 2.6 m after touchdown at the old nominal landing milestone
X=0.95 m. Contact later than X=2.55 m is explicitly flagged as late, rather than
inventing a metre of contact evidence. Foot geom centres and root crossing
overshoot are recorded for GPU clearance review. **The 0.50 m margin is verified
against compiled terrain, but actual foot mesh clearance at crossing remains a
GPU validation gate.** It is not yet a measured rollout result.

Initial attempt timeout remains 12 s. Historical measured upright post-edge
speeds were 0.356/0.445/0.409 m/s for source/100/249 (walking-block run report).
At the slowest of those averages, the longest 3.65 m course takes about 10.3 s,
before accounting for approach/transient differences. This is preliminary
support, not validation. Review valid timeout traces before the pilot; revise
and document duration if healthy walkers are routinely cut short. No speed
objective, restriction, command or dense reward weight was changed.

## Terrain and curriculum

The simulator exposes precompiled terrain origins/levels and immutable collision
geometry. This implementation selects those tiles at reset instead of mutating
live collision geometry and risking stale collision data. The finite bank has
**21 heights at 0.5 mm spacing from 20 to 30 mm**, inclusive. It is not continuous.
All planned 2.5 mm curriculum endpoints are included. Rows reset to a selected
tile and the complete robot pose is translated consistently; velocity and joint
state keep the ordinary reset initialization. Upper-floor spawn Z stays fixed.
Signed edge distance and height come from the actual tile and root position.

Training starts remain uniform over 45–65 cm; play defaults to fixed 55 cm.
At 20 mm, all samples use 20 mm. Above that, 75% sample uniformly over bank
levels strictly above the previous endpoint through the current endpoint;
25% sample uniformly over all admitted easier levels, including the previous
endpoint. At the first expansion the easier share is exactly 20 mm. Nothing
advances from iteration count. The upper limit starts at 20 mm and only explicit
complete development and confirmation gates admit another 2.5 mm, capped at 30.

The curriculum object stores stage evidence, frozen development-seed baselines
and per-condition consecutive regression streaks. Checkpoints embed that object
alongside the ordinary full optimizer and observation normalizer state. The
runner restores live reset-event parameters through the manager. First episode
ages are not randomized in this task, so the initial batch gets full attempts.
Source checkpoint hashes and exact local paths are in
`cliff-course-sources-20261024.json`.

## Local results and remaining runtime checks

Focused local suite: **74 passed, 4 skipped** (CUDA unavailable). Covers compiled
terrain heights/footprint, multiple-height contact, same-step contact/tilt/goal,
failure veto and latches, cached pulses, staggered reset rows, dt payout,
randomized start geometry and sampler proportions, independent scalar scoring,
gates, ranking, regression streaks and saved curriculum restoration. Historical
cliff/oracle/walk-through tests also pass. All new Python files compile and
entry-point `--help` imports succeed.

The installed project environment lacks pytest. `uv --with pytest` encountered
a macOS system-configuration panic even offline. Tests ran with `.venv/bin/python`
and pytest from the existing cached tool environment. No project dependency
was added or silently installed. The normal documented GPU test command remains
`uv run --with pytest pytest ...` after a fresh dependency sync.

Unrun: actual CUDA reset/step tests, checkpoint normalizer behavior at new
heights, terminal contact/clearance traces, timeout adequacy, matched videos,
64-env/five-update smoke, frozen-policy reward audit, ONNX runtime/export check,
and end-to-end session/watchdog/retrieval. Synthetic evidence does not establish
course-completion capability. The scripts below are prepared, not GPU-certified.

## Prepared commands for an approved session

Run the RL AGENTS.md OS-library installation and EGL/CUDA context preflight on
the approved pod **before** collecting tests or importing simulation workloads.
Export `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` in the actual remote runner session.

```bash
uv run --with pytest pytest tests/test_cliff_course.py -q

# Independent frozen-policy evaluation, fixed anchors; preselected video env 0.
uv run python scripts/eval_cliff_course.py CHECKPOINT.pt \
  --output logs/course-confirmation --heights .02 \
  --distances .45 .55 .65 --seeds 20 21 22 --videos

# One seed-42 continuation: three sources in original/100/249 order.
# SESSION_DEADLINE is an absolute Unix timestamp, <= session start + 7200 s.
uv run python scripts/run_cliff_course_session.py \
  --sources ORIGINAL500.pt CONTINUATION100.pt CONTINUATION249.pt \
  --output logs/course-session --deadline-epoch "$SESSION_DEADLINE"
```

The session helper compares sources at 55 cm, confirms selection separately,
checks all three approaches before admitting height, runs smoke and export from
the selected source, then starts main weights afresh from that unchanged source.
Main training uses at most ten blocks of 100 updates, seed 42, 2048 environments,
and full checkpoint/optimizer/normalizer continuation between blocks. Block
restarts reset simulator episodes; this differs from one uninterrupted process.
Behavior checks are >=7 min apart; incomplete final selection is provisional.
Held-out seeds 30–32 are used only for final review. No second arm is launched.

The helper reserves 15 min inside the two-hour cap for retrieval/shutdown, kills
its subprocess group at its work deadline, snapshots source, and writes hashes.
It does **not** provision or stop the pod itself. Before remote work, start the
independent local watchdog under `caffeinate` and preserve its process/log:

```bash
caffeinate -i uv run python scripts/cliff_course_pod_control.py watch \
  --pod-id "$POD_ID" --deadline-epoch "$SESSION_DEADLINE"

uv run python scripts/cliff_course_pod_control.py retrieve \
  --pod-id "$POD_ID" --ssh-host "$POD_SSH_ALIAS" \
  --remote-root "$REMOTE_PROJECT" --output-relative logs/course-session \
  --destination logs/course-retrieved

uv run python scripts/cliff_course_pod_control.py stop --pod-id "$POD_ID"
```

Retrieval validates SHA256SUMS locally. Confirm actual stopped status in CLI
output, then retire the watchdog. This watchdog depends on the Mac staying awake
and online; retained historical volumes are never deleted. The installed RunPod CLI's `stop pod` / `get pod` syntax was checked locally;
end-to-end stop/status verification remains unrun.

## Approval boundary

No paid execution is authorized by these files. Proposed scope remains one
available GPU, one seed-42 continuation, ≤1000 main updates, height ≤3 cm and
≤2 hours total. The current CLI `uv run --no-sync runpodctl gpu list` reported an available
48 GB NVIDIA A40 in EU-SE-1 (high stock), secure-cloud on-demand rate **$0.59/h**
on 2026-10-24. Proposed two-hour compute ceiling: **$1.18**, with a **$2 total
session spending cap** including storage contingency. Recheck availability and
price immediately before provisioning; do not substitute a higher rate without
approval. Selected source candidates and hashes are in the linked source manifest.

Concrete approval scope: one A40, one bounded session including EGL/CUDA tests,
fixed-policy confirmation, smoke/export, and conditional seed-42 continuation
up to 1000 main updates at 2048 environments, never above 3 cm, with retrieval
and verified stop inside two hours. If setup/confirmation consumes the budget,
return evidence and stop without training. The Mac must remain awake/online for
the independent shutdown watchdog. This request does not include deployment.
