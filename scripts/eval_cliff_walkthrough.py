"""Common CUDA evaluation of frozen source and continuation, first episodes only.

The offline scalar classifier is intentionally independent of the vectorized
training state. Both consume the same pre-reset evidence and must agree.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
import torch
import diagnose_cliff_oracle as diagnostic
import eval_cliff_drop as baseline
from mjlab_microduck.tasks.cliff import walkthrough
from analyze_walkthrough_blocks import classify_prefix

EXTRA = ['lower_floor', 'walk_milestone_step', 'walk_blocks', 'walk_bad', 'walk_pulse',
         'walk_clean_blocks', 'walk_paid_blocks']


def classify(trace, dt=.02):
    c = dict(zip(diagnostic.COLUMNS, trace.T))
    hold = np.float32(0)
    milestone = None
    blocks = 0
    bad = False
    block_x = None
    reasons = set()
    # Use time-to-step conversion to avoid float32 timestamp boundary errors.
    for k in range(len(trace)):
        age = round(float(c['t'][k]) / dt)
        x, y = c['x'][k], c['y'][k]
        finite = bool(np.isfinite(trace[k]).all())
        impact = c['bad_impact'][k] > .5
        boundary = x < -.75 or x > 3.8 or abs(y) > 1.15
        tilt = milestone is not None and c['tilt_deg'][k] >= 45 - 1e-5
        for name, flag in [('nonfinite', not finite), ('impact', impact), ('boundary', boundary), ('tilt', tilt)]:
            if flag:
                reasons.add(name)
                bad = True
        stable = (x >= .95 and (c['left_support'][k] or c['right_support'][k])
                  and c['tilt_deg'][k] <= 30 + 1e-5 and abs(c['vz'][k]) <= .25
                  and c['vx'][k] >= .05 and not impact and finite)
        hold = np.float32(hold + np.float32(dt)) if stable else np.float32(0)
        if milestone is None and hold >= np.float32(.30) and c['lower_floor'][k] > .5:
            milestone = age
            block_x = x
        if milestone is not None and age > milestone and (age-milestone) % round(1/dt) == 0:
            blocks += 1
            if x - block_x < .05 - 1e-6:
                bad = True
                reasons.add('progress')
            block_x = x
    full = len(trace) > 0 and round(float(c['t'][-1]) / dt) >= round(12/dt)
    success = full and milestone is not None and blocks >= 5 and not bad
    if not full or milestone is None or blocks < 5:
        reasons.add('insufficient_observation')
    return dict(success=bool(success), milestone_time_s=None if milestone is None else milestone*dt,
                walking_blocks=blocks, full_horizon=bool(full), reasons=sorted(reasons))


def run(checkpoint, output, seeds, videos=False):
    output.mkdir(parents=True, exist_ok=False)
    # Save the exact classifier column order. Original diagnostic fields remain
    # available for the historical goal and clean-follow-through comparisons.
    original_probe = baseline._probe
    def probe(env):
        if not hasattr(env, '_walk_interface_checked'):
            assert env.observation_manager.group_obs_dim['actor'] == (63,)
            assert env.observation_manager.group_obs_dim['critic'] == (78,)
            assert env.action_manager.total_action_dim == 14
            env._walk_interface_checked = True
            print('Verified common evaluation interface 63/78/14', flush=True)
        zero = original_probe(env)
        s = walkthrough.update(env)
        env._walk_eval_extra = torch.stack((
            walkthrough.lower_floor_contact(env.scene.sensors['cliff_foot_contact'].data.found,
                env.scene.sensors['cliff_foot_contact'].data.pos, env.scene.env_origins).float(),
            s.milestone_step.float(), s.blocks.float(), s.bad.float(), s.pulse.float(),
            s.clean_blocks.float(), s.paid_blocks.float()), -1).cpu().numpy()
        return zero
    # Diagnostic.run_condition builds its frames by concatenating torch tensors.
    # Add the extra evidence directly through its optional probe extension.
    baseline._probe = probe
    diagnostic.EXTRA_PROBE = lambda env: env._walk_eval_extra
    def strict_failures(cfg):
        # Continuing after the old goal must not grant diagnostic recovery
        # privileges that the pilot does not have.
        from mjlab.managers import TerminationTermCfg
        from mjlab_microduck.tasks.cliff import mdp
        cfg.terminations['fallen_too_long'].params['max_duration_s'] = .5
        cfg.terminations['cliff_failed_landing'] = TerminationTermCfg(
            func=mdp.cliff_failed_landing, params={'edge_distance':.55, 'landing_distance':.4})
        from mjlab.managers import RewardTermCfg
        cfg.rewards['cliff_landing'] = RewardTermCfg(func=walkthrough.reward, weight=1.)
        # Reinsert the probe last so it sees all termination evidence pre-reset.
        probe_term = cfg.terminations.pop('cliff_eval_probe')
        cfg.terminations['cliff_eval_probe'] = probe_term
        return cfg
    diagnostic.EXPERIMENT_CFG_TRANSFORM = strict_failures
    # Capture real RewardManager contributions after weight and dt scaling.
    # The probe runs before reward computation, so audit compute itself.
    from mjlab.managers.reward_manager import RewardManager
    original_compute = RewardManager.compute
    reward_frames = []
    reward_names = []
    def audited_compute(manager, dt):
        result = original_compute(manager, dt)
        reward_names[:] = manager._term_names
        reward_frames.append((manager._step_reward * dt).cpu().numpy().copy())
        return result
    RewardManager.compute = audited_compute
    all_rows = []
    try:
        for seed in seeds:
            label = diagnostic.case_label(checkpoint, 'cliff-heading', seed, .55)
            reward_frames.clear()
            rows = diagnostic.run_condition(checkpoint, condition='cliff-heading', seed=seed,
                approach_m=.55, num_envs=24, horizon_s=12, device='cuda:0', output=output/label)
            z = np.load((output/label).with_suffix('.npz'))
            for i, row in enumerate(rows):
                trace = z['values'][:row['steps'], i]
                result = classify(trace)
                cols = dict(zip(diagnostic.COLUMNS, trace.T))
                # Training's state machine must agree with independent scoring.
                assert result['success'] == bool(cols['walk_pulse'].any()), (seed, i, result)
                assert result['walking_blocks'] == int(cols['walk_blocks'][-1]), (seed, i)
                prefix = classify_prefix(trace, diagnostic.COLUMNS)
                assert prefix['clean_prefix_blocks'] == int(cols['walk_clean_blocks'][-1])
                assert min(prefix['clean_prefix_blocks'], 5) == int(cols['walk_paid_blocks'][-1])
                row['sustained'] = result
                row['prefix'] = prefix
                step_contributions = np.stack(reward_frames)[:row['steps'], i]
                contributions = step_contributions.sum(0)
                row['reward_contributions'] = dict(zip(reward_names, map(float, contributions)))
                blocks = 8 * min(prefix['clean_prefix_blocks'], 5)
                final = 40 * result['success']
                assert abs(row['reward_contributions']['cliff_landing'] - blocks - final) < 1e-4
                row['bonus'] = dict(block=blocks, final=final)
                dense = step_contributions[:, [j for j,k in enumerate(reward_names) if k != 'cliff_landing']]
                row['positive_dense'] = float(np.maximum(dense, 0).sum())
                row['negative_dense'] = float(np.minimum(dense, 0).sum())
            all_rows.extend(rows)
            (output/'episodes.json').write_text(json.dumps(all_rows, indent=2))
            print(json.dumps({'seed':seed, 'success':sum(r['sustained']['success'] for r in rows),
                              'attempts':len(rows)}), flush=True)
    finally:
        RewardManager.compute = original_compute
        baseline._probe = original_probe
        diagnostic.EXTRA_PROBE = None
        diagnostic.EXPERIMENT_CFG_TRANSFORM = None
    def distribution(values):
        a = np.asarray(values)
        return dict(mean=float(a.mean()), min=float(a.min()), p25=float(np.quantile(a,.25)),
                    median=float(np.median(a)), p75=float(np.quantile(a,.75)), max=float(a.max()))
    audit = {name:distribution([r['reward_contributions'][name] for r in all_rows]) for name in reward_names}
    for name in ('positive_dense','negative_dense'):
        audit[name] = distribution([r[name] for r in all_rows])
    for name in ('block','final'):
        audit[name+'_bonus'] = distribution([r['bonus'][name] for r in all_rows])
    # All costs in this recipe return nonnegative values with negative weights.
    # Upright progress is signed shaping and is reported separately.
    from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_walkthrough_env_cfg
    cfg = make_microduck_cliff_walkthrough_env_cfg(False)
    penalties = [name for name,c in cfg.rewards.items() if c.weight < 0]
    for name in penalties:
        assert audit[name]['max'] <= 1e-6, (name, audit[name])
    audit['penalties_checked'] = penalties
    (output/'reward-audit.json').write_text(json.dumps(audit, indent=2))
    manifest = {'checkpoint':str(checkpoint), 'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'device':torch.cuda.get_device_name(), 'torch':torch.__version__, 'python':platform.python_version(),
                'seeds':seeds, 'num_envs':24, 'approach_m':.55, 'horizon_s':12,
                'columns':diagnostic.COLUMNS,
                'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(diagnostic.__file__), Path(walkthrough.__file__)]}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return all_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20,21,22])
    args = parser.parse_args()
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    import mjlab.tasks
    diagnostic.COLUMNS += EXTRA
    run(args.checkpoint, args.output, args.seeds)
