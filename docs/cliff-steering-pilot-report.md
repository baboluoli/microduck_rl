# Cliff steering pilot: 2 cm descent improved; consistency remains limited

The final paired battery improved from **24/144 completions (16.7%)** for
the starting policy to **89/144 (61.8%)** for the retained +75 checkpoint.
Recorded successful rollouts show walking to the edge, stepping down, and
continued upright walking. Training stopped at +100 because approach accuracy
crossed the regression threshold. The +75 weights were retained, exported
and downloaded. The Runpod GPU is stopped.

This establishes progress at **2 cm in simulation**. It does not establish
reliable descent or passing the height-advancement gates. Larger drops were
not trained or evaluated in this pilot.

## Initialization and training

Source: `logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt`,
the original 1,000-update scratch run. SHA-256:
`526907c9727e4c4084451e7bf5c6b5e17e7cc1ee4f904e3e60bdadef5a885af1`.

`scripts/prepare_cliff_steering_checkpoint.py` kept the source actor and
critic, all other normalizer statistics, and action standard deviations.
It zeroed the previously unused yaw input columns and initialized their
normalizer standard deviation to 0.3, preserving the original network
response at initialization. Adam moments and counters were reset. The copied
source is `logs/rsl_rl/cliff_drop/cliff-steering-seed/model_0.pt`, SHA-256
`794c014f39644a3c07690d6a119f390943566787327ba89236538ace9e772a7e`.
No W&B download or login was needed; training logs were offline.

The `steering` profile enabled gentle heading correction, replaced body-frame
velocity tracking with signed world-X progress, removed synthetic midair
training starts, and ended disqualified landing attempts promptly. BAM,
collision geometry, observation layout, noise/delay and domain randomization
were preserved. PPO used fixed LR 1e-5, clip 0.1, entropy bonus 0. See
[the recipe](cliff-steering-repair.md) for the exact changes and diagnosis.

Five smoke updates ran at 64 environments, followed by four 25-update blocks
at 2048 environments. The pilot always started again from the prepared source
after smoke; smoke weights were not used for the main run. Every block loaded
the explicitly named preceding checkpoint. There were 100 main updates plus
5 smoke updates, with 75 main updates in the selected policy. All 105 logged
NaN termination rates were zero.

RSL-RL repeats the saved iteration label on resume. End labels were 24, 48,
72 and 96 for actual update counts 25, 50, 75 and 100. The selected checkpoint
stores `common_step_counter=1800` (75 × 24 steps):
`logs/rsl_rl/cliff_drop/2026-10-03_21-06-36_cliff-steering-update75/model_72.pt`.

## Measurements

Each upper-platform battery has 144 episodes: 16 environments × three seeds
× approach distances 0.35, 0.55 and 0.75 m. Separate 48-episode midair batteries
never determine selection. Flat-ground batteries have 24 ten-second episodes
across the same three seeds. All checkpoints use the same evaluation success
criterion and observation window, including a four-second sustained-fall
termination; training's earlier failed-attempt termination is removed for eval.

First measured rates used for selection:

| Added updates | Upper completions / 144 | Ledge reached / 144 | Corridor crossings / 144 | Flat upright / 24 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 26 | 95 | 61 | 17 |
| 25 | 58 | 119 | 60 | 20 |
| 50 | 76 | 125 | 53 | 21 |
| 75 | 80 | 130 | 75 | 22 |
| 100 | 71 | 128 | 56 | 22 |

At the +100 gate, its paired +75 reference completed 89/144, compared with
71/144 for +100. The actual stop trigger was seed 2 corridor accuracy: 14/48
for +100 versus the initial source's 22/48, a 16.7-point drop exceeding the
15-point safeguard. The flat-ground guard passed. These sampling-based gates
are conservative safeguards, not statistical proof of a causal regression.

Final paired battery:

| Metric | Prepared source | Retained +75 |
| --- | ---: | ---: |
| Complete descent and continued walking | 24/144 (16.7%) | 89/144 (61.8%) |
| Reach ledge | 90/144 (62.5%) | 130/144 (90.3%) |
| Cross within ±15 cm corridor | 54/144 (37.5%) | 72/144 (50.0%) |
| Seed 0 completion | 8/48 | 33/48 |
| Seed 1 completion | 9/48 | 34/48 |
| Seed 2 completion | 7/48 | 22/48 |
| 35 cm approach completion | 11/48 | 35/48 |
| 55 cm approach completion | 9/48 | 30/48 |
| 75 cm approach completion | 4/48 | 24/48 |
| Separate midair completion | 28/48 | 38/48 |

The selected policy still failed 55 upper-platform episodes: 26 impact
disqualifications, 14 before the edge, 11 contact without completion, and 4
without lower-floor contact. Twenty-seven failures exceeded the 15 N head
impact threshold; none exceeded the 20 N trunk threshold. Longer approaches
and seed 2 remain weak. Median heading error among successful final trials
was 16.2°, so accurate straight approaches remain an unfinished part of the task.

GPU physics replays vary despite matching seeds. The selected policy's first
battery completed 80/144; the next paired battery and final battery each
completed 89/144. Video filenames and JSON record the actual replay outcome.
All four recorded selected-policy episodes completed, including two chosen
from failed earlier rollouts; these clips do not represent the 38% failure
rate in the full battery. All video exports completed without errors.

## Visual and export checks

Inspected sampled frames of
`logs/cliff-steering-pilot/final-model_72-upper-complete-2cm-seed0-env0.mp4`
(3.84 seconds): the robot advances across the edge and continues upright on
the lower platform. The companion `env1` upper video also records an actual
completion. Source-policy impact failures are retained alongside successes.

`scripts/export.py` produced `logs/cliff-steering-pilot/selected.onnx`.
Local ONNX Runtime validation confirmed input `[1,61]`, output `[1,14]`, baked
normalization, and finite outputs. Across 34 probes, maximum absolute error
against the selected PyTorch actor was `6.5565109e-7`. Evidence:
`logs/cliff-steering-pilot/export-validation.json`. The selected optimizer LR
was verified as 1e-5. Thirty focused local tests passed before launching.

A separate CPU diagnostic evaluated the local deployed
`BEST_alpha_walking.onnx` directly, without a W&B login, on the same cliff
robot. Eight plane trials stayed upright; two of eight 2 cm cliff trials
completed. This small control supports feasible walking under this collision
model, but does not establish a superior cliff initialization. Evidence:
`logs/cliff-local-walking-control/episodes.json`.

## Artifacts and resources

All checkpoints, exact runtime source, CSV/JSON, logs, eight videos and two
ONNX exports are downloaded under `logs/`. The archive
`logs/cliff-steering-pilot-artifacts.tar.gz` matches the remote SHA-256:
`e66005aaf0e3e3491137c763f41ae4252976f44e68d5db449b8fa98ef4986fa1`.
Local post-download export validation and shutdown metadata are additional
files in `logs/cliff-steering-pilot/`; they were created after that archive.

The retained old pod could not restart because its host lacked a free GPU.
Replacement A40 pod `kswm9yejyj537g` was created October 3, 2026 at
20:28:06.292 UTC and stopped at 21:44:07 UTC. Runpod reports `EXITED`, with no
active runtime. At $0.49/hour, approximately 76 minutes cost **$0.62 GPU**,
excluding retained storage. Dependencies were moved to `/opt/cliff-venv`
after slow network-volume imports; the interrupted startup trained no updates
and its logs are preserved. The two-hour shutdown watchdog was cancelled
after supervised shutdown. Resource metadata is in
`logs/cliff-steering-pilot/runpod-session.json`.
