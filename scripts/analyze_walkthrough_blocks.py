"""Score uninterrupted walking prefixes from saved CUDA traces, without simulation.

Unlike walk_blocks (all observed blocks), clean_prefix_blocks stops permanently
at the first invalid sample or failed progress block. No Torch/MuJoCo import or
GPU is needed. The source evidence is read only; write results elsewhere.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def classify_prefix(trace, columns, dt=.02):
    """Reconstruct the original milestone and audit the recorded state machine.

    A block includes its ending sample: an impact/tilt on that sample vetoes
    credit. Later failure preserves earlier earned blocks, but never restarts
    the prefix. Progress in an incomplete final second earns no block.
    """
    c = dict(zip(columns, trace.T))
    hold = np.float32(0)
    milestone = None
    block_x = None
    observed_blocks = clean_blocks = 0
    failure = None
    bad = False
    block_steps = round(1 / dt)
    if abs(block_steps * dt - 1) > 1e-6:
        raise ValueError('One second must contain whole control steps')
    previous_age = 0
    for k, sample in enumerate(trace):
        age = round(float(c['t'][k]) / dt)
        if age != previous_age + 1:
            raise ValueError('Expected contiguous first-episode control steps')
        previous_age = age
        x, y = c['x'][k], c['y'][k]
        finite = bool(np.isfinite(sample).all())
        impact = (c['bad_impact'][k] > .5 or c['head_force'][k] > 15
                  or c['trunk_force'][k] > 20)
        reasons = []
        if not finite:
            reasons.append('nonfinite')
        if impact:
            reasons.append('impact')
        if x < -.75 or x > 3.8 or abs(y) > 1.15:
            reasons.append('boundary')
        if milestone is not None and c['tilt_deg'][k] >= 45 - 1e-5:
            reasons.append('tilt')
        stable = (x >= .95 and (c['left_support'][k] or c['right_support'][k])
                  and c['tilt_deg'][k] <= 30 + 1e-5 and abs(c['vz'][k]) <= .25
                  and c['vx'][k] >= .05 and not impact and finite)
        # Preserve float32 hold accumulation: the existing 0.30 s hold fires
        # on step 16, rather than silently changing the landing milestone.
        hold = np.float32(hold + np.float32(dt)) if stable else np.float32(0)
        if milestone is None and hold >= np.float32(.30) and c['lower_floor'][k] > .5:
            milestone, block_x = age, x
        complete = (milestone is not None and age > milestone
                    and (age - milestone) % block_steps == 0)
        if complete:
            observed_blocks += 1
            if x - block_x < .05 - 1e-6:
                reasons.append('progress')
            block_x = x
        if reasons:
            bad = True
            if failure is None:
                failure = dict(time_s=age * dt, reasons=reasons)
        if complete and not bad:
            clean_blocks += 1
        # Audit every saved sample, including those after disqualification.
        assert int(c['walk_milestone_step'][k]) == (-1 if milestone is None else milestone)
        assert int(c['walk_blocks'][k]) == observed_blocks
        assert bool(c['walk_bad'][k]) == bad
    full = previous_age == round(12 / dt)
    strict = full and milestone is not None and observed_blocks >= 5 and not bad
    assert bool(c['walk_pulse'].any()) == strict
    return dict(milestone_time_s=None if milestone is None else milestone * dt,
                clean_prefix_blocks=clean_blocks, observed_blocks=observed_blocks,
                first_failure=failure, full_horizon=full, strict_success=bool(strict))


def summarize(rows):
    counts = [sum(r['clean_prefix_blocks'] >= b for r in rows) for b in range(1, 6)]
    histogram = {str(b): sum(min(r['clean_prefix_blocks'], 5) == b for r in rows)
                 for b in range(6)}
    # Candidate budget: five once-only 8-point blocks plus 40 at strict timeout.
    credits = [8 * min(r['clean_prefix_blocks'], 5) + 40 * r['strict_success'] for r in rows]
    return dict(episodes=len(rows), milestones=sum(r['milestone_time_s'] is not None for r in rows),
                at_least_1_to_5_clean_blocks=counts, capped_prefix_histogram=histogram,
                strict_successes=sum(r['strict_success'] for r in rows),
                impact_terminations=sum(r['termination'] == 'cliff_failed_landing' for r in rows),
                proposed_bonus_total=sum(credits), proposed_bonus_mean=float(np.mean(credits)),
                proposed_bonus_nonzero_episodes=sum(v > 0 for v in credits))


def analyze(source):
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    episodes_path = source / 'episodes.json'
    episodes = json.loads(episodes_path.read_text())
    fingerprints = {}
    def fingerprint(path):
        fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint(manifest_path)
    fingerprint(episodes_path)
    rows = []
    for seed in manifest['seeds']:
        matches = list(source.glob(f'*-s{seed}-d0p55.npz'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one trace for seed {seed}: {matches}')
        fingerprint(matches[0])
        with np.load(matches[0], allow_pickle=False) as z:
            seed_rows = [r for r in episodes if r['seed'] == seed]
            assert len(seed_rows) == manifest['num_envs']
            assert sorted(r['env'] for r in seed_rows) == list(range(manifest['num_envs']))
            for r in seed_rows:
                assert 0 < r['steps'] <= z['values'].shape[0]
                trace = z['values'][:r['steps'], r['env']]
                result = classify_prefix(trace, manifest['columns'])
                # The prior independent scalar evaluator must still agree on
                # strict success, milestone and total blocks (not clean prefix).
                old = r['sustained']
                assert result['strict_success'] == old['success']
                assert result['milestone_time_s'] == old['milestone_time_s']
                assert result['observed_blocks'] == old['walking_blocks']
                rows.append(dict(seed=seed, env=r['env'], termination=r['termination'], **result))
    return dict(source=str(source), checkpoint_sha256=manifest['sha256'],
                input_sha256=fingerprints, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                bonus_proposal=dict(block_points=[8]*5, strict_timeout_points=40),
                pooled=summarize(rows),
                by_seed={str(s): summarize([r for r in rows if r['seed'] == s]) for s in manifest['seeds']},
                episodes=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('Write analysis outside the retained source directory')
    result = analyze(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('pooled', 'by_seed')}, indent=2))
