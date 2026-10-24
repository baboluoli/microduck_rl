# Partial-credit continuation implementation, 2026-10-24

Implements the [approved block proposal](cliff-walking-block-proposal-20261024.md).
The historical oracle task remains separate. The walk-through task now uses
weight 1 for its landing reward, whose function returns point pulses divided by
control dt: 8 for each of the first five valid complete one-second blocks and
40 for unchanged strict completion at exactly the 12-second timeout.

Observed blocks, clean-prefix blocks, paid blocks, and strict completion are
separate state. Failure latches before block eligibility, so a violation on the
ending sample vetoes that block. Already paid points survive later failure.
Checks continue after block five. All helpers cache the common control-step
counter and clear only resetting episode rows.

The evaluator independently reconstructs strict completion and uninterrupted
walking prefixes from pre-reset traces. It verifies observed counters, clean
counters, paid counters, and actual RewardManager bonus totals. It records
weighted, dt-integrated contributions for every reward term on every first
episode, including failed attempts. Positive and negative dense contributions
are accumulated separately at each step, including signed shaping. Summary
distributions contain mean, minimum, quartiles, median and maximum. Penalty
terms with negative weights must contribute no positive episode totals.

The runner requires reachable partial credit rather than nonzero strict
completion. It pauses after the source audit until the controller records a
review, before either smoke or pilot PPO updates. Smoke and pilot each load an
unchanged source-500 copy with explicit warm start. Candidate ranking uses
strict success, all-attempt capped clean-prefix mean, fewer impact
terminations, then old goal rate, preserving earlier checkpoints on ties.
Separate strict, walking and impact regression streaks reset independently.
Walking/impact compare with the fixed same-session source; strict compares with
retained best. Behavior checks are at least seven minutes apart; the final
checkpoint is checked even when training ends between checks. Held-out seeds
30–32 remain reserved.

Local validation: 52 passed, two CUDA tests skipped locally. GPU tests,
graphics preflight, reward audit and run results are recorded in the session
artifacts under logs/walking-blocks-20261024/. The current session is one
secure RTX A6000 pod at $0.59/hour, explicitly approved for a two-hour hard cap
including setup/evaluation, with a local shutdown watchdog. No automatic
extension or replication is authorized.

The local watchdog requires this Mac to remain awake and online. Caffeinate
keeps it awake for this session; the controller must retrieve and checksum
artifacts and verify pod shutdown before retiring the watchdog. Account API
credentials remain on the Mac.

Completed session: [measured results and shutdown](cliff-walking-block-run-results-20261024.md). The final local runner also starts the independent numerical sidecar automatically; this session launched that monitor separately during evaluation. Four local session/monitor tests passed after that integration.
