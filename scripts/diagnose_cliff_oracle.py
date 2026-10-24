"""Frozen-policy diagnostics for the 63-input, fixed-2-cm oracle experiment.

Success is recorded but DOES NOT end an episode. This exposes delayed falls
hidden by the ordinary evaluator. Each trace is captured before auto-reset.
The sham-plane condition deliberately preserves the trained oracle cue while
removing the physical step: it is a counterfactual control, not truthful terrain
perception and not a hardware/deployment evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import eval_cliff_drop as baseline
from mjlab.managers import TerminationTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks.cliff import mdp as cliff_mdp
from mjlab_microduck.tasks import mdp as robot_mdp

TASK = 'Mjlab-CliffOracle-Flat-MicroDuck'
EXPERIMENT_CFG_TRANSFORM = None
EXTRA_PROBE = None  # Optional evidence columns for the independent walk-through evaluator.
CONDITIONS = ('cliff-heading', 'cliff-zero-yaw', 'sham-plane-heading')
COLUMNS = ['t', 'x', 'y', 'vx', 'vz', 'yaw_deg', 'tilt_deg', 'roll_deg',
           'pitch_deg', 'yaw_rate', 'yaw_command', 'left_support', 'right_support',
           'head_force', 'trunk_force', 'goal_pulse', 'bad_impact', 'peak_torque',
           'stall_count', 'action_delta_l2']
# Preserve joint-level evidence so a later hypothesis can be checked without
# rerunning the experiment. These are servo indices, not passive-joint indices.
for prefix in ('q', 'qd', 'torque', 'action'):
    COLUMNS += [f'{prefix}_{i}' for i in range(14)]


def record_goal_without_ending(env, **kwargs):
    """Keep the original completion calculation and reward, but continue walking."""
    cliff_mdp.cliff_landing_complete(env, **kwargs)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def tile_boundary(env):
    """Censor departures from the finite test tile; do not call these falls."""
    p = env.scene['robot'].data.root_link_pos_w - env.scene.env_origins
    return (p[:, 0] > 3.8) | (p[:, 0] < -.75) | (p[:, 1].abs() > 1.15)


def diagnostic_cfg(num_envs, seed, approach_m, condition, horizon_s):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    cfg = baseline._configure(num_envs, seed, .02, approach_m, 'upper', task=TASK)
    cfg.episode_length_s = horizon_s
    cfg.terminations['cliff_landing'].func = record_goal_without_ending
    cfg.terminations['diagnostic_tile_boundary'] = TerminationTermCfg(func=tile_boundary, time_out=True)
    if condition == 'cliff-zero-yaw':
        # Only change steering feedback; keep all 63 observation slots intact.
        twist = cfg.commands['twist']
        # Keep identical command sampling/RNG consumption; only zero the
        # heading-feedback gain. All environments still sample target heading 0.
        twist.heading_control_stiffness = 0.0
    if condition == 'sham-plane-heading':
        cfg.scene.terrain = TerrainEntityCfg(terrain_type='plane', env_spacing=4.0)
    return cfg


def describe_trace(trace: np.ndarray, *, termination: str, physical_cliff: bool):
    """Summarize one first episode, explicitly distinguishing failure/censoring."""
    c = {name: trace[:, i] for i, name in enumerate(COLUMNS)}
    first = lambda mask: float(c['t'][np.flatnonzero(mask)[0]]) if np.any(mask) else None
    edge = first(c['x'] >= baseline.EDGE_DISTANCE)
    goal = first(c['goal_pulse'] > .5)
    impact = first(c['bad_impact'] > .5)
    tilt = first(c['tilt_deg'] >= 45)
    end = float(c['t'][-1])
    result = {'edge_time_s': edge, 'original_goal_time_s': goal,
              'first_bad_impact_s': impact, 'first_tilt45_s': tilt,
              'observed_duration_s': end, 'boundary_censored': 'diagnostic_tile_boundary' in termination,
              'nonfinite_termination': 'nan_state' in termination,
              'physical_cliff': physical_cliff}
    for anchor, t in [('edge', edge), ('goal', goal)]:
        # None means unobserved, never a successful five-second follow-through.
        if t is None:
            result[f'{anchor}_plus_5s_clean'] = None
            continue
        window = (c['t'] >= t) & (c['t'] <= t + 5 + 1e-5)
        observed_failure = np.any(c['bad_impact'][window] > .5) or np.any(c['tilt_deg'][window] >= 45)
        if observed_failure:
            result[f'{anchor}_plus_5s_clean'] = False
        elif end + 1e-5 < t + 5:
            result[f'{anchor}_plus_5s_clean'] = None
        else:
            result[f'{anchor}_plus_5s_clean'] = True
    result['original_goal_then_failure'] = bool(goal is not None and (
        (impact is not None and impact > goal) or np.any((c['t'] > goal) & (c['tilt_deg'] >= 45))))
    for name, mask in [('approach', c['x'] < baseline.EDGE_DISTANCE),
                       ('after_edge', c['x'] >= baseline.EDGE_DISTANCE)]:
        result[f'{name}_samples'] = int(mask.sum())
        if np.any(mask):
            result[f'{name}_mean_abs_yaw_deg'] = float(np.mean(np.abs(c['yaw_deg'][mask])))
            result[f'{name}_rms_yaw_rate'] = float(np.sqrt(np.mean(c['yaw_rate'][mask] ** 2)))
            result[f'{name}_mean_stall_count'] = float(np.mean(c['stall_count'][mask]))
            result[f'{name}_mean_action_delta_l2'] = float(np.mean(c['action_delta_l2'][mask]))
    return result


@torch.inference_mode()
def run_condition(checkpoint, *, condition, seed, approach_m, num_envs, horizon_s, device, output):
    frames = []
    original_configure = baseline._configure
    original_probe = baseline._probe

    def probe(env):
        zero = original_probe(env)
        snapshot = env._cliff_eval_state
        robot = env.scene['robot']
        data = robot.data
        q = data.root_link_quat_w
        roll = torch.rad2deg(torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]), 1-2*(q[:,1]**2+q[:,2]**2)))
        pitch = torch.rad2deg(torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1)))
        support = env.scene.sensors['feet_ground_contact'].data.found.reshape(env.num_envs, -1)
        action = env.action_manager.action
        previous = env.action_manager.prev_action
        scalars = [env.episode_length_buf.float()*env.step_dt,
            snapshot['x'], snapshot['y'], snapshot['vx'], snapshot['vz'],
            torch.rad2deg(snapshot['yaw']), snapshot['tilt_deg'], roll, pitch,
            data.root_link_ang_vel_w[:,2], env.command_manager.get_command('twist')[:,2],
            support[:,0].float(), support[:,1].float(), snapshot['head_force'],
            snapshot['trunk_force'], snapshot['success'].float(), snapshot['bad_impact'].float(),
            snapshot['peak_torque_nm'], snapshot['stall_count'], torch.sum((action-previous)**2,dim=1)]
        tensor = torch.cat([torch.stack(scalars, dim=1), robot_mdp._servo_joint_pos(env,robot),
                            robot_mdp._servo_joint_vel(env,robot), data.actuator_force, action], dim=1)
        frame = tensor.cpu().numpy().copy()
        if EXTRA_PROBE is not None:
            frame = np.concatenate((frame, EXTRA_PROBE(env)), axis=1)
        frames.append(frame)
        return zero

    def configure(*args, **kwargs):
        # Avoid recursion: diagnostic_cfg calls the original configuration helper.
        baseline._configure = original_configure
        try:
            cfg = diagnostic_cfg(num_envs, seed, approach_m, condition, horizon_s)
        finally:
            baseline._configure = configure
        if EXPERIMENT_CFG_TRANSFORM is not None:
            cfg = EXPERIMENT_CFG_TRANSFORM(cfg)
        cfg.terminations['cliff_eval_probe'].func = probe
        return cfg

    baseline._configure = configure
    try:
        rows = baseline.evaluate(checkpoint, num_envs=num_envs, seed=seed, device=device,
            drop_m=.02, approach_m=approach_m, task=TASK)
    finally:
        baseline._configure = original_configure
    array = np.stack(frames)
    assert array.shape[2] == len(COLUMNS), array.shape
    # A nonfinite trace must be reported as a diagnostic failure, not silently averaged.
    assert np.isfinite(array).all(), 'Nonfinite diagnostic trace'
    lengths = [row['steps'] for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), values=array, columns=np.array(COLUMNS), lengths=np.array(lengths))
    for i, row in enumerate(rows):
        trace = array[:row['steps'], i]
        # In a sham plane there is no physical lower contact or descent success.
        row['diagnostic'] = describe_trace(trace, termination=row['termination'], physical_cliff=condition!='sham-plane-heading')
        row['condition'] = condition
        row['original_evaluator_outcome'] = row.pop('outcome')
    output.with_suffix('.json').write_text(json.dumps(rows, indent=2))
    return rows


def case_label(checkpoint, condition, seed, distance):
    # A decimal point would be interpreted as a suffix by Path.with_suffix,
    # overwriting traces for different approach distances.
    distance_label = f'{distance:.2f}'.replace('.', 'p')
    return f'{checkpoint.stem}-{condition}-s{seed}-d{distance_label}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--approach-distances', nargs='+', type=float, default=[.35,.55,.75])
    parser.add_argument('--horizon-s', type=float, default=12.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure_torch_backends()
    import mjlab.tasks  # noqa: F401
    args.output.mkdir(parents=True,exist_ok=False)
    manifest = vars(args).copy()
    manifest['checkpoints'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.checkpoints}
    manifest['output'] = str(args.output)
    manifest['frozen_policy'] = True
    manifest['torch_version'] = torch.__version__
    manifest['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    all_rows=[]
    for checkpoint in args.checkpoints:
        for condition in args.conditions:
            # Vary approaches in the main condition; hold the middle distance
            # for the two controls, avoiding needless combinations initially.
            distances = args.approach_distances if condition=='cliff-heading' else [.55]
            for seed in args.seeds:
                for distance in distances:
                    label=case_label(checkpoint,condition,seed,distance)
                    rows=run_condition(checkpoint, condition=condition, seed=seed, approach_m=distance,
                        num_envs=args.num_envs,horizon_s=args.horizon_s,device=args.device,output=args.output/label)
                    all_rows.extend(rows)
                    print(json.dumps({'case':label,'episodes':len(rows),
                        'original_goals':sum(r['diagnostic']['original_goal_time_s'] is not None for r in rows),
                        'goal_then_failure':sum(r['diagnostic']['original_goal_then_failure'] for r in rows),
                        'clean_5s_after_edge':sum(r['diagnostic']['edge_plus_5s_clean'] is True for r in rows)}),flush=True)
                    snapshot=args.output/'episodes.tmp'
                    snapshot.write_text(json.dumps(all_rows,indent=2))
                    snapshot.replace(args.output/'episodes.json')

if __name__=='__main__':
    main()
