# Course drift and NaN audit — 2026-10-24

The continuation develops additional negative-Y drift **before reaching the drop**. Boundary failure semantics were inconsistent between scoring and training and are now corrected locally. Height normalization remains finite and changes modestly. The NaN trigger is still unresolved: the saved finite history has a late foot-contact penetration, without a preceding velocity explosion.

No training or GPU pod was started. The verified artifact tree was read only. New outputs are in [analysis/failure-audit](../logs/course-session-20261024/analysis/failure-audit).

## Videos

These three videos animate the actual saved positions and world-Y velocity, computed from successive positions. The dot represents the robot root; they are **trajectory animations, not rendered robot rollouts**, because the evaluation files did not save joint poses. A terminated trajectory stops and is explicitly marked ENDED while its counterpart continues.

- [45 cm start: source versus trained failure](../logs/course-session-20261024/analysis/failure-audit/failure-d0.45-s20-env23.mp4) — seed 20, env 23; trained exit at 6.98 s.
- [55 cm start: source versus trained failure](../logs/course-session-20261024/analysis/failure-audit/failure-d0.55-s21-env22.mp4) — seed 21, env 22; trained exit at 6.90 s.
- [65 cm start: source versus trained failure](../logs/course-session-20261024/analysis/failure-audit/failure-d0.65-s21-env22.mp4) — seed 21, env 22; trained exit at 7.42 s.
- [Rendered pre-NaN robot history, 8× slow motion](../logs/course-session-20261024/analysis/failure-audit/pre-nan-finite-history.mp4) — all 100 saved physics states of training env 1232, with heading and world velocities. This is a pose replay of the preceding 0.5 s, not a new simulation or a reproduction of the NaN. It contains no post-detection state.

Failure clips were deliberately selected: median-duration negative-Y boundary failure at each start distance, at 2 cm, requiring a successful source counterpart. They are illustrations, not an unbiased success-rate sample. Matching seed/environment labels does not guarantee identical stochastic histories after trajectories diverge.

## Where drift develops

Compared `eval-005` (unchanged source 249) with `eval-006` (`model_99`, 100 new updates), using all 432 first episodes per policy: heights 2/2.25 cm × starts 45/55/65 cm × seeds 20/21/22 × 24 environments. This is the immediately preceding source battery, **not** the earlier frozen `eval-003` gate; different replay scores should not be mixed.

| Quantity | Source | After 100 updates |
|---|---:|---:|
| Course successes | 424/432 | 332/432 |
| Negative-Y boundary exits | 0 | 90 |
| Positive-Y boundary exits | 0 | 0 |
| Mean Y at first root crossing of drop edge | −3.19 cm | −12.37 cm |
| Mean world sideways speed on approach | −0.021 m/s | −0.080 m/s |
| Mean world sideways speed, X=0.55–1.55 m | −0.003 m/s | −0.108 m/s |
| Mean world sideways speed, X≥1.55 m | −0.0004 m/s | −0.143 m/s |

Velocity summaries are the mean of each episode's phase mean, not pooled time samples. Finite differences use recorded episode ages at 0.02 s per control step. The initial velocity is unknown and excluded; phase-boundary intervals can straddle a crossing. Short/failed episodes have less late-course coverage.

| Time from reset | Source mean Y | Trained mean Y |
|---|---:|---:|
| 0.10 s | +0.19 cm | +0.30 cm |
| 0.20 s | −0.24 cm | −0.31 cm |
| 0.50 s | −1.07 cm | −2.75 cm |
| 1.00 s | −2.20 cm | −6.87 cm |

Every episode is still before the drop at 0.5 s. Both policies have an early negative bias; training amplifies it and fails to recover afterward. The additional bias is evident by 0.5 s, so touchdown cannot be its first cause. The trained policy crosses the edge at negative Y in 406/432 episodes, versus 297/432 for source. At 2 cm alone there are 51 trained boundary exits; at 2.25 cm there are 39.

**Heading versus sideways walking is not identifiable from these evaluation files.** They contain neither root orientation nor angular velocity nor commands. Travel direction is not body heading: a robot can walk sideways while facing forward. We therefore do not substitute `atan2(vy,vx)` for heading or label world-Y speed as body-sideways speed.

The archived training configuration requests body-forward speed 0.18–0.24 m/s, body-sideways speed zero, and world heading zero. Its heading controller requests `clip(-0.5 × wrapped_heading, -0.3, +0.3)` rad/s. The actual requested and achieved turning rates cannot be compared retrospectively. The evaluator now records heading, full command, body/world angular velocities and body/world linear velocities before autoreset. This addition has been syntax checked; it has not been exercised in a fresh GPU rollout.

[Per-episode CSV](../logs/course-session-20261024/analysis/failure-audit/episodes.csv) and [summary/selection JSON](../logs/course-session-20261024/analysis/failure-audit/summary.json) provide the underlying numbers.

## Boundary termination semantics

Course limits are strict: X < −0.75 m, X > 3.8 m, or |Y| > 1.15 m. A breach is latched and vetoes completion even when finish and boundary crossing coincide. Every reported boundary failure in this comparison first breaches a limit on the final pre-reset sample and receives no completion pulse; assertions check this directly.

However, the archived configuration set `course_boundary.time_out=True`, while `is_finite_horizon=False`. The mjlab wrapper forwards truncations as `extras['time_outs']`; RSL-RL PPO adds `gamma × V(previous observation)` to their transition rewards. Thus scoring calls a side exit failure, while learning treats it as an artificial interruption with estimated future value. This does not create a positive completion reward, but it weakens the intended failure signal whenever that value estimate is positive.

Changed **only the course task** to make boundary exits ordinary terminations (`time_out=False`). The real 12-second time limit remains a timeout. Added checks for this distinction, strict ±Y thresholds, and the latched same-step completion veto. This is a semantic correction, not proof it caused or will cure the drift; existing checkpoints and archived results are unchanged.

## Height normalization transition

The actor has 63 inputs; the critic has 78. Height is the last input in each. Their source height standard deviation is zero because source training always used 0.02 m. RSL-RL's actual formula is `(height − mean) / (std + 0.01)`, **not division by standard deviation alone**.

| Actor height input | Source | First saved new update (`model_0`) | 100 updates (`model_99`) |
|---|---:|---:|---:|
| Normalized 2 cm | ≈0 | −0.000151 | −0.012631 |
| Normalized 2.25 cm | +0.250000 | +0.248547 | +0.226045 |

At 100 updates the actor mean is 0.02013230 m, std 0.00047446 m, count 41,828,352. Critic values agree closely. There is no normalization singularity or huge height-input jump. The appended height itself is constant throughout an episode, including touchdown; the other appended input is continuous signed edge distance.

Nevertheless, the height input was effectively constant zero during source training, so its input weights had essentially no task gradient from height variation. New heights activate an unpracticed dependency. That is a plausible transfer concern, not an established cause. The large inherited count also makes normalization adapt slowly. No normalizer or policy weights were modified. `model_staged.pt` contains the later 100-update state and must not be mistaken for the original source.

[Full actor/critic normalization audit](../logs/course-session-20261024/analysis/failure-audit/height-normalization.json).

## Pre-NaN finite history

The dump metadata records env 1232, detection step 7544, and 100 preceding physics states at steps 7444–7543. Physics dt is 0.005 s. State specification 30 is `mjSTATE_PHYSICS`: in this model the 41 values are 21 positions and 20 velocities. There are **no saved controls, actuator forces, solver warm starts, contact forces, or offending post-step state**.

| Maximum in each consecutive 0.125 s quarter | First | Second | Third | Last |
|---|---:|---:|---:|---:|
| Root speed (m/s) | 0.638 | 0.568 | 0.774 | 0.448 |
| Root angular speed (rad/s) | 5.409 | 5.410 | 5.429 | 3.625 |
| Absolute joint speed (rad/s) | 10.899 | 12.744 | 11.229 | 10.536 |

There is no monotonic velocity growth toward detection. Heading changes from −0.51° to +36.39° over the saved interval. These training states are a different episode from the negative-Y evaluation failures and cannot establish their heading behavior.

Restoring each pose into the compiled MJB and running **position/geometry calculations only**, without stepping dynamics, reconstructs a final left-foot/`terrain_1` contact with penetration 3.23, 4.93 and 5.48 mm in the last three samples. This is the deepest reconstructed contact in the window. It is a useful contact-onset lead, not proof of numerical instability: soft contact can penetrate, and reconstruction does not restore original GPU solver history or per-environment randomization fields. The replay likewise shows the compiled model rather than all per-environment physical settings.

Extreme controls cannot be ruled in or out. Before any next simulation intended to isolate the trigger, capture applied controls, policy actions, actuator force, acceleration, original contact/solver diagnostics, per-environment randomized model fields, and both sides of the offending physics step. Preserve the existing immediate-stop monitor. Do not infer safety from finite losses or from this finite prehistory.

[Complete per-step forensic JSON](../logs/course-session-20261024/analysis/failure-audit/nan-forensics.json).

## Reproduction and validation

From the RL directory:

```bash
.venv/bin/python scripts/analyze_cliff_course_failures.py --videos
.venv/bin/python scripts/render_cliff_nan_history.py
uv run --with pytest pytest tests/test_cliff_course.py tests/test_cliff_course_process.py -q
```

The renderer needs access to a working local graphics context. Offline analysis uses no GPU and never advances physics. Video frames were visually inspected. Course/process tests: **44 passed, 2 CUDA-only tests skipped** on this Mac. All four MP4s decode fully (212/261/262/100 frames); report artifact links resolve. The boundary fix and extra evaluator telemetry still need a GPU evaluation before relying on them in another training session. No new session is authorized or launched by this audit.
