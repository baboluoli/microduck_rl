"""Matched seed-20, environment-0 full episodes; no cherry-picked successes."""
import argparse
from pathlib import Path
import torch
import diagnose_cliff_oracle as diagnostic
import eval_cliff_drop as baseline
from mjlab.managers import TerminationTermCfg
from mjlab.utils.torch import configure_torch_backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure_torch_backends()
    import mjlab.tasks
    original = baseline._configure
    def configure(*a, **kw):
        cfg = original(*a, **kw)
        cfg.episode_length_s = 12.
        from mjlab_microduck.tasks.cliff import mdp
        cfg.terminations['fallen_too_long'].params['max_duration_s'] = .5
        cfg.terminations['cliff_failed_landing'] = TerminationTermCfg(
            func=mdp.cliff_failed_landing, params={'edge_distance':.55, 'landing_distance':.4})
        cfg.terminations['cliff_landing'].func = diagnostic.record_goal_without_ending
        cfg.terminations['diagnostic_tile_boundary'] = TerminationTermCfg(func=diagnostic.tile_boundary, time_out=True)
        return cfg
    # Follow the preselected robot throughout the episode; a fixed camera
    # previously lost it after landing and could not validate continued walking.
    original_probe = baseline._probe
    def follow_probe(env):
        if getattr(env, '_cliff_eval_video_active', False):
            pos = env.scene['robot'].data.root_link_pos_w[0].cpu().numpy()
            env._offline_renderer._cam.lookat[:] = pos
        return original_probe(env)
    baseline._probe = follow_probe
    baseline._configure = configure
    rows = baseline.evaluate(args.checkpoint, num_envs=24, seed=20, device='cuda:0',
        drop_m=.02, approach_m=.55, task=diagnostic.TASK, video_env=0, video_path=args.output)
    import json
    rows[0]['original_evaluator_outcome'] = rows[0].pop('outcome')
    args.output.with_suffix('.json').write_text(json.dumps({'selection':'seed 20, environment 0 chosen in advance; first episode, full rollout until termination; no frozen final frames', 'episode':rows[0]},indent=2))


if __name__ == '__main__':
    main()
