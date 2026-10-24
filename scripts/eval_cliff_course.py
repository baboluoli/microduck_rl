"""Frozen-policy course evaluation. Run only after GPU/EGL preflight.

Every row is one attempted first episode; evidence is captured before autoreset.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import subprocess
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_course_env_cfg
from mjlab_microduck.tasks.cliff import course
from mjlab_microduck.tasks.cliff.course_evaluation import classify

TASK = 'Mjlab-CliffOracleCourse-MicroDuck'


def evaluate(checkpoint, *, height, distance, seed, output, device='cuda:0', video=None):
    cfg = make_microduck_cliff_course_env_cfg(play=True)
    cfg.scene.num_envs = 24
    cfg.seed = seed
    cfg.events['course_reset'].params.update(fixed_height=height, start_range=(distance,distance))
    if video:
        # Isolate and follow the preselected robot for the complete course.
        cfg.viewer.env_idx = 0
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.max_extra_envs = 0
        cfg.viewer.width, cfg.viewer.height = 960, 640
        cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 1.4, 90., -15.
    frames = []
    def probe(env):
        s = course.update(env)
        d = env.scene['robot'].data
        p = d.root_link_pos_w-env.scene.env_origins
        q = d.root_link_quat_w
        foot = env.scene.sensors['cliff_foot_contact'].data
        pos = foot.pos.reshape(24,-1,3)-env.scene.env_origins[:,None,:]
        force = foot.force.reshape(24,-1,3)
        found = foot.found.reshape(24,-1)
        contact_forces = [env.scene.sensors[n].data.force.sum(1).norm(dim=-1) for n in
                          ('head_ground_contact','trunk_ground_contact')]
        failure = torch.zeros(24,dtype=torch.bool,device=device)
        for name in env.termination_manager.active_terms:
            if name.startswith('course_') or env.termination_manager.get_term_cfg(name).time_out:
                continue
            failure |= env.termination_manager.get_term(name)
        foot_ids, _ = env.scene['robot'].find_geoms('^(left_foot_collision|right_foot_collision)$')
        foot_centres = d.geom_pose_w[:,foot_ids,:3]-env.scene.env_origins[:,None,:]
        rows = []
        for i in range(24):
            rows.append(dict(x=p[i,0].item(), y=p[i,1].item(), age=env.episode_length_buf[i].item(),
                # Capture before autoreset. World Y drift and body-sideways
                # motion differ when the robot turns; retain both frames.
                heading=d.heading_w[i].item(),
                vy=d.root_link_lin_vel_w[i,1].item(),
                velocity_body=d.root_link_lin_vel_b[i].tolist(),
                angular_velocity_body=d.root_link_ang_vel_b[i].tolist(),
                angular_velocity_world=d.root_link_ang_vel_w[i].tolist(),
                command=env.command_manager.get_command('twist')[i].tolist(),
                cos_tilt=(1-2*(q[i,1]**2+q[i,2]**2)).item(), head=contact_forces[0][i].item(),
                trunk=contact_forces[1][i].item(), finite=not s.nonfinite[i].item(),
                failure=failure[i].item(), pulse=s.pulse[i].item(),
                foot_geom_centres=foot_centres[i].tolist(),
                vx=d.root_link_lin_vel_w[i,0].item(),
                feet=[dict(found=found[i,j].item(), position=pos[i,j].tolist(),force=force[i,j].tolist())
                      for j in range(found.shape[1])], servo_loads=d.actuator_force[i].tolist()))
        frames.append(rows)
        return torch.zeros(24,dtype=torch.bool,device=device)
    cfg.terminations['course_probe'] = TerminationTermCfg(func=probe)
    env = ManagerBasedRlEnv(cfg, device=device, render_mode='rgb_array' if video else None)
    writer = None
    rewards = [[] for _ in range(24)]
    traces = [[] for _ in range(24)]
    active = [True]*24
    try:
        agent_cfg = load_rl_cfg(TASK)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), map_location=device)
        # A source course checkpoint can restore training ranges. Evaluation
        # anchors remain fixed, independent of that checkpoint's sampler.
        env.event_manager.get_term_cfg('course_reset').params.update(fixed_height=height,start_range=(distance,distance))
        env.reset(seed=seed)
        obs = wrapped.get_observations()
        assert env.observation_manager.group_obs_dim['actor'] == (63,)
        assert env.observation_manager.group_obs_dim['critic'] == (78,)
        assert env.action_manager.total_action_dim == 14
        policy = runner.get_inference_policy(device=device)
        if video:
            import imageio.v2 as imageio
            video.parent.mkdir(parents=True,exist_ok=True)
            writer = imageio.get_writer(video,fps=25,codec='libx264',quality=7)
        with torch.inference_mode():
            for _ in range(env.max_episode_length+1):
                actions = policy(obs)
                assert torch.isfinite(actions).all(), 'Nonfinite normalized-policy output'
                obs, _, done, _ = wrapped.step(actions)
                if writer and active[0] and len(traces[0]) % 2 == 0:
                    from PIL import Image, ImageDraw
                    env._offline_renderer._cam.lookat[:] = env.scene['robot'].data.root_link_pos_w[0].cpu().numpy()
                    frame = Image.fromarray(env.render())
                    ImageDraw.Draw(frame).text((10,10),f'{checkpoint.name} h={height*100:.2f}cm start={distance:.2f}m seed={seed} env=0',fill='white')
                    writer.append_data(np.asarray(frame))
                terms = env.reward_manager._term_names
                weighted = env.reward_manager._step_reward * env.step_dt
                for i in range(24):
                    if not active[i]: continue
                    traces[i].append(frames[-1][i])
                    rewards[i].append(weighted[i].cpu().tolist())
                    if done[i]: active[i] = False
                if not any(active): break
        results = []
        for i,trace in enumerate(traces):
            result = classify(trace,height=height,dt=env.step_dt)
            assert result['success'] == any(s['pulse'] for s in trace)
            contributions = torch.tensor(rewards[i]).sum(0).tolist()
            audit = dict(zip(terms,contributions))
            assert abs(audit['cliff_landing']-80*result['success']) < 1e-4
            for name,term in cfg.rewards.items():
                if term.weight < 0:
                    assert audit[name] <= 1e-5, (name,audit[name])
            if result['success']:
                result['crossing_foot_centres'] = trace[-1]['foot_geom_centres']
                result['crossing_root_x'] = trace[-1]['x']
                result['crossing_overshoot'] = trace[-1]['x']-course.FINISH
            result.update(env=i, seed=seed, height=height, distance=distance,
                          trace=trace, reward_contributions=audit)
            results.append(result)
    finally:
        if writer: writer.close()
        env.close()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2,allow_nan=True)+'\n')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heights',nargs='+',type=float,default=[.02])
    p.add_argument('--distances',nargs='+',type=float,default=[.45,.55,.65])
    p.add_argument('--videos',action='store_true',help='Record preselected env 0 for each condition')
    p.add_argument('--seeds',nargs='+',type=int,default=[20,21,22])
    args = p.parse_args()
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    args.output.mkdir(parents=True,exist_ok=False)
    sources = [Path(__file__), Path('src/mjlab_microduck/tasks/__init__.py'), Path('uv.lock'), *Path('src/mjlab_microduck/tasks/cliff').glob('*.py')]
    manifest = dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        heights=args.heights, distances=args.distances, seeds=args.seeds, attempts_per_seed=24,
        height_bank=course.HEIGHTS, resolution_m=.0005, finish_x=course.FINISH, forward_edge=course.FORWARD_EDGE,
        python=platform.python_version(),torch=torch.__version__,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for height in args.heights:
        for distance in args.distances:
            for seed in args.seeds:
                evaluate(args.checkpoint,height=height,distance=distance,seed=seed,
                         output=args.output/f'h{height:.4f}-d{distance:.2f}-s{seed}.json',
                         video=args.output/f'h{height:.4f}-d{distance:.2f}-s{seed}.mp4' if args.videos else None)
