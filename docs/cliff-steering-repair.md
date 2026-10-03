# Cliff steering repair: prepared experiment

Status: completed; [results](cliff-steering-pilot-report.md). The approved GPU
pilot ran on replacement A40 pod `kswm9yejyj537g`, now stopped.
The original stopped pod could not restart because its host had no free GPU.
The replacement's actual quote is $0.49/hour. Its process deadline is October
3, 2026 at 22:26:06 UTC, reserving two minutes for download/shutdown inside the
two-hour pod budget. Runtime dependencies are on local disk `/opt/cliff-venv`.

Final paired measurements: source 24/144 completions, selected +75 checkpoint
89/144. The +100 checkpoint regressed and was excluded from selection. Flat
upright survival improved from 17/24 at baseline to 22/24 at +75. The GPU
stopped October 3 at 21:44:07 UTC, about $0.62 GPU cost. No height advancement
is authorized by these results.

## Measured failure

On October 3, 2026, `scripts/diagnose_cliff_transfer.py` compared the original
checkpoint and its +100/+200 continuations on a plane and a 2 cm ledge. This
uses the cliff robot, BAM actuators and full collision model throughout.
CPU, seed 0, eight episodes per checkpoint and terrain, ten-second episodes.
Flat-ground success here means no disqualifying impact, tilt below 45°
throughout, and surviving to timeout. It does not imply straight walking.

| Checkpoint | Flat upright / 8 | Median flat forward travel | Median flat heading error | 2 cm complete / 8 |
| --- | ---: | ---: | ---: | ---: |
| Original model_999 | 7 | 0.618 m | 9.2° | 1 |
| +100 model_99 | 6 | 0.134 m | 77.2° | 1 |
| +200 model_198 | 7 | -0.074 m | 93.4° | 1 |

Evidence: `logs/cliff-transfer-diagnostic-seed0-v2/{summary.json,episodes.csv}`.
This small CPU sample is diagnostic, not a replacement for the larger GPU
batteries in `cliff-model999-pilot-report.md`. Terrain construction changes
random draws, so identical seeds do not establish identical initial physical
states across the plane and ledge. It supports testing steering and training
incentives before changing the collision model. Peak torque was recorded, but
these measurements do not establish voltage saturation or hardware validity.

The old body-frame velocity reward pays for walking in the direction the
robot faces, including away from the ledge. At zero yaw command, the actor has
no heading-error input. A head/trunk impact invalidates completion but the
old training episode can continue collecting other rewards. Synthetic
midair resets consume 35% of starts despite the poor upper-platform approach.

## Implemented changes

`MICRODUCK_CLIFF_PROFILE=steering` starts from the frozen model999 conditions:

- Enable the existing yaw-rate command as a gentle world-heading controller,
  gain 0.5, capped at ±0.3 rad/s. The 61D actor and 14D action contracts stay intact.
- Replace body-frame linear tracking with signed world-X progress, weight 4.
  Standing and sideways motion pay zero; backward motion costs; forward
  credit is capped at commanded speed and requires tilt below 40°.
- Use upper-platform starts only. End head/trunk-disqualified attempts
  immediately and terminate sustained excessive tilt after 0.5 seconds.
- Keep BAM, collision geometry, observation noise/delay, domain randomization,
  posture commands, and source-stage smoothing unchanged.
- Use fixed PPO learning rate 1e-5, clip 0.1 and zero entropy bonus. This is a
  conservative experiment, not evidence that learning rate caused the old
  regression; the +100 checkpoint's saved adaptive optimizer LR was already 1e-5.

`scripts/prepare_cliff_steering_checkpoint.py` copies the original model_999.
It zeros only the unused yaw column in the actor/critic first layers and gives
that input a 0.3 normalizer standard deviation. It refuses sources whose yaw
statistics are nonzero or whose input widths differ. The actor yaw slot is 50;
the critic slot is 53 because privileged linear velocity precedes the shared
proprioception. All other network weights and normalizer statistics are retained.
Adam moments and counters are reset explicitly, with optimizer LR 1e-5.
The source file is preserved and its SHA-256 is recorded.

The original source is selected for this experiment because its flat-ground
direction retention was stronger in this diagnostic. The previous pilot's
best cliff checkpoint (+100) remains saved as a separate candidate.

## Verification so far

- 30 focused tests passed: cliff configuration, steering preparation,
  reward cases, actual replay labels, observation NaN guards and posture reward.
- Actual source versus prepared full actor and critic: 1,000 observation
  probes each, new yaw swept across ±0.3; maximum output difference exactly 0.
- Local prepared-checkpoint inference: 4/4 flat episodes stayed upright,
  0/4 cliff episodes completed. **This is initialization validation, not an
  improved trained policy.** Artifacts: `logs/cliff-steering-preflight/`.
- The 64-environment, five-update training smoke passed with all NaN
  termination rates zero; the required ONNX export passed before the first
  2048-environment training block.

## Bounded Runpod run

The root AGENTS.md requires asking before starting a Runpod training run.
The retained A40 pod `ky2i3y3b6vjn5d` is stopped. Its previous quote was
$0.49/GPU-hour; the two-hour experiment cap is approximately $0.98 GPU cost,
excluding retained storage and provisioning/teardown time. Confirm availability
and pricing when starting. No W&B credentials are needed; logging is offline.

Copy the updated source and original checkpoint to the pod. Prepare the
checkpoint there, or copy the already prepared local directory:

```bash
uv run python scripts/prepare_cliff_steering_checkpoint.py \
  logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt \
  logs/rsl_rl/cliff_drop/cliff-steering-seed/model_0.pt
uv run python scripts/run_cliff_steering_pilot.py
```

The runner evaluates the prepared baseline, performs and exports the smoke,
then starts afresh from the prepared source at 2048 environments. It caps
training at 200 updates with paired cliff and flat evaluations every 25.
Each cliff battery has 144 upper-platform episodes across three seeds and
three approach distances, plus a separately reported midair diagnostic.
Each flat battery has 24 episodes across three seeds. Stop on >15 percentage
points regression or no measured completion gain by 100 updates. Retain the
source on ties and exclude regressed candidates from selection. These are
sampling-based safeguards, not statistical proof of improvement.

A two-hour process deadline terminates the entire active uv/python process
group. **The operator must still stop the Runpod pod** after completion or
failure; the script does not provision or shut down cloud resources. Download
all checkpoints, reports and labelled videos first. Results go to
`logs/cliff-steering-pilot/`. Height stays 2 cm; evaluate the selected candidate
again and inspect the actual videos before calling the experiment successful.
