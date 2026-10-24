# Cliff course curriculum session — 2026-10-24

The approved session stopped after a nonfinite simulation state during the second training block. **Keep unchanged continuation checkpoint 249. No trained checkpoint was promoted.** The source passed the new full-course gate at 2 cm across 45/55/65 cm starts. Training sampled 2–2.25 cm, but did not establish mastery at 2.25 cm; 2.5–3 cm was not attempted.

The user explicitly selected 249, so source ranking against original 500 and continuation 100 was skipped. Its SHA-256 is `ef70ae031f61b64686aef2d428338de5d2562adf0d6847b3d133092770223a06`. The [retained full source checkpoint](../logs/course-session-20261024/artifacts/logs/rsl_rl/oracle_course/course-session-source/model_source.pt) matches that hash. Weights, normalizers and optimizer state remain together in the `.pt` file.

One seed-42 continuation used 2,048 environments. There were **192 completed main PPO updates**, plus the separately scoped 64-environment/five-update smoke. The only fully evaluated trained checkpoint was [model_99.pt after 100 new updates](../logs/course-session-20261024/artifacts/logs/rsl_rl/oracle_course/2026-10-24_18-58-57_course-100/model_99.pt). Its evaluation failed every required condition. The second block stopped before its final save and evaluation; its durable `model_100.pt` represents 101 new updates, not 200. No final held-out evaluation was run, and seeds 30–32 remain unused.

RunPod A40 48 GB pod `fpyiaev8rp057l`, EU-SE-1, used image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` at $0.59/hour. Creation was 18:19:15 UTC; the exit event was 19:19:04 UTC, approximately 59 minutes 49 seconds and **$0.588 GPU compute**. This is a runtime estimate, not a billing invoice; storage/billing adjustments are excluded. Actual runtime was independently confirmed `stopped`, desired status `EXITED`, at 19:20:14 UTC. The local shutdown watchdog was then retired. The two-hour/$2 session cap was respected; no historical volume was deleted.

Two runtime defects were fixed before valid confirmation. ImageIO needed a NumPy array rather than a PIL image. More importantly, terrain boxes were centered on mjlab's grid corner in Y: the flat outer border covered half of the single-column course's lower floor. Some robots therefore walked on level ground while receiving a 2 cm terrain observation. Those runs are retained under `course-session-border-invalid` and excluded from all tables below. The course now centers its boxes and spawn inside the tile; historical task layouts retain their existing behavior. A new raycast test measures the actual surface across the complete bank, including borders, on both sides and at all 21 heights. **39 course tests passed on the A40**, including CUDA reset/step checks. Required EGL libraries and a real CUDA/EGL context were verified before those tests.

The camera was also changed to isolate and follow the preselected robot throughout the course. The [source video](../logs/course-session-20261024/artifacts/logs/course-session/eval-001/h0.0200-d0.55-s20.mp4) is seed 20, environment 0, 2 cm height and 55 cm start, chosen before seeing the outcome. The source video batch scored 72/72 at 55 cm; the separate gate battery below is the frozen regression baseline. GPU/rendered replays are not assumed to reproduce identical scores. Scoring uses pre-reset traces; the final rendered frame may show autoreset. There was no accepted trained finalist for a matched source/finalist video.

The source's development gate used seeds 20/21/22, 24 first episodes each. All three conditions passed. “One metre” remains secondary to reaching the finish. Maximum tilt includes failed attempts; actuator loads are simulated actuator-force magnitudes. All boundary events in these reported conditions were lateral exits, verified from terminal coordinates.

| Height | Start | Successes by seed | Course | One metre | Impacts | Side exits | Timeouts | Mean seconds | Max tilt | Max servo load |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 cm | 45 cm | 23 / 22 / 24 | 69/72 | 72/72 | 2 | 0 | 1 | 8.59 | 86.6° | 0.641 |
| 2 cm | 55 cm | 24 / 23 / 22 | 69/72 | 71/72 | 1 | 0 | 2 | 8.79 | 81.9° | 0.641 |
| 2 cm | 65 cm | 24 / 23 / 23 | 70/72 | 72/72 | 0 | 1 | 1 | 8.99 | 39.0° | 0.641 |

Separate confirmation used seeds 40/41/42. Every condition passed; these are development confirmation seeds, not held-out final evidence.

| Height | Start | Successes by seed | Course | One metre | Impacts | Side exits | Timeouts | Mean seconds | Max tilt | Max servo load |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 cm | 45 cm | 24 / 23 / 24 | 71/72 | 72/72 | 0 | 0 | 1 | 8.50 | 17.4° | 0.641 |
| 2 cm | 55 cm | 24 / 22 / 24 | 70/72 | 72/72 | 0 | 0 | 2 | 8.70 | 24.1° | 0.641 |
| 2 cm | 65 cm | 24 / 22 / 24 | 70/72 | 72/72 | 0 | 0 | 2 | 8.90 | 30.6° | 0.641 |

At 2.25 cm, unchanged source 249 was measured before training. The initial exact-height battery scored 71/72, 71/72 and 66/72 across 45/55/65 cm. The runner's complete entering-policy battery then scored 70/72, 71/72 and 70/72 at that height. No confirmation at 2.25 cm established mastery there. Occasional healthy timeouts were retained as failures; the 12-second duration was unchanged.

All 432 source gate attempts and 216 exact-frontier attempts recorded valid lower-floor contact. The conservative bound from compiled foot geometry gave at least **0.402 m forward clearance** at successful crossings. The per-episode, dt-integrated reward audit averaged **+69.84 dense reward and -7.22 penalties**. Successful episodes paid exactly 80 completion points once; failed episodes paid zero. Synthetic and actual-episode assertions passed. No reward weight, speed requirement, approach range, finish rule or impact threshold was changed. Full term totals are in [pretraining-review.json](../logs/course-session-20261024/artifacts/logs/course-session/pretraining-review.json).

The smoke loaded the intended checkpoint, reset iteration/curriculum counters while preserving weights/normalizers/optimizer, and exported a working normalizer-baked 63-input/14-output ONNX. The critic remained 78-dimensional. The smoke ONNX is a validation artifact. Main training restarted independently from unchanged 249. After 100 main updates, normalizer count was 41,828,352 = 36,913,152 + 100 × 2,048 × 24, excluding smoke samples. Optimizer steps increased from 15,020 to 17,020, confirming continuation. The first main block logged 20,154 actual reset samples: `[5009, 2988, 3024, 3069, 2962, 3102]` in 20/20.5/21/21.5/22/22.5 mm bins, and zero above 22.5 mm; 75.15% were in the newest band.

After 100 new updates, the development check on seeds 20/21/22 showed substantial lateral drift. All six conditions failed:

| Height | Start | Successes by seed | Course | One metre | Impacts | Side exits | Timeouts | Mean seconds | Max tilt | Max servo load |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 cm | 45 cm | 17 / 19 / 16 | 52/72 | 71/72 | 2 | 18 | 0 | 6.76 | 85.9° | 0.641 |
| 2 cm | 55 cm | 19 / 19 / 16 | 54/72 | 72/72 | 1 | 17 | 0 | 7.05 | 28.2° | 0.641 |
| 2 cm | 65 cm | 19 / 18 / 17 | 54/72 | 70/72 | 2 | 16 | 0 | 7.26 | 82.3° | 0.641 |
| 2.25 cm | 45 cm | 20 / 21 / 19 | 60/72 | 72/72 | 1 | 11 | 0 | 6.90 | 85.6° | 0.641 |
| 2.25 cm | 55 cm | 18 / 19 / 16 | 53/72 | 72/72 | 2 | 16 | 1 | 7.11 | 73.5° | 0.641 |
| 2.25 cm | 65 cm | 19 / 21 / 19 | 59/72 | 72/72 | 0 | 12 | 1 | 7.31 | 39.6° | 0.641 |

At 2 cm the course-success losses versus frozen baselines were 23.6, 20.8 and 22.2 percentage points. Each condition also lost at least 10 points in at least two seeds. This established **one completed regression strike on every mastered condition**. The curriculum stayed at 2–2.25 cm and the next continuation block began. High one-metre success did not override course failure. The [failure-geometry record](../logs/course-session-20261024/artifacts/logs/course-session/failure-geometry.json) confirms lateral boundary crossings and records post-edge forward speeds and failure endpoints.

The session stopped for **nonfinite state**, before a second completed regression check. NaN guard detected environment 1232 at physics step 7544 during zero-based iteration 178; the dump timestamp is 19:14:17 UTC. The existing helper scanned its log only after a subprocess block finished, so it did not abort immediately. The external monitor found the event and terminated training after iteration 191. About 13 additional updates were logged after the affected iteration; this is an operational deviation from the plan's immediate-stop rule. Losses being finite did not justify continuing. The buffered dump contains the 100 preceding finite states (steps 7444–7543) and the compiled MJB model; it does not by itself isolate the cause of the offending state.

The helper now monitors output continuously, recognizes NaN guard events, nonfinite losses/rewards and nonzero `nan_state` termination, and uses unbuffered Python output. Five subprocess tests passed locally, including interrupting a still-running child. This correction was made after the stop; no GPU session was resumed to certify it end to end. The RunPod stop helper was also corrected to use current JSON CLI commands and actual `runtimeStatus`, and retrieval accepts a session SSH config compatible with the Mac's rsync.

[The compressed artifact archive](../logs/course-session-20261024/course-session-artifacts.tar.gz) is 317,624,921 bytes with SHA-256 `6bdc28c98adbfe895bc1f4169c6bac5f8d862071b8766b46d979e8ab2d971c65`. Its archive checksum and **all 163 manifest file checksums were verified locally before shutdown**. The immutable [artifact directory](../logs/course-session-20261024/artifacts/logs/course-session) contains checkpoints, normalizers/optimizer states, configs, evaluation and training source snapshots, curriculum history, raw traces, source videos, failed-validation evidence and NaN diagnostics. Additional analysis lives outside that verified tree: [per-seed CSV](../logs/course-session-20261024/analysis/results-by-seed.csv), [checkpoint state audit](../logs/course-session-20261024/analysis/checkpoint-audit.json), and [dump field summary](../logs/course-session-20261024/analysis/nan-dump-summary.json).

The measured result is a source policy passing this bounded 2 cm course on development and confirmation batches. This continuation did not improve it or validate a higher frontier. It establishes neither indefinite walking nor hardware readiness. An unchanged continuation is not warranted on this evidence; inspect the saved nonfinite-state history and lateral-drift behavior before proposing another bounded session. No further session is authorized.

## Follow-up forensic audit

[Drift comparison, boundary fix, height-normalization audit, pre-NaN inspection, and four diagnostic videos](cliff-course-failure-audit-20261024.md). The additional negative-Y drift is visible before the drop. Boundary exits were incorrectly marked as timeouts and are corrected locally; the NaN trigger remains unresolved. No new training run was started.

