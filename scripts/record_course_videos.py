"""Record videos of cliff-course episodes on the Mac CPU (no pod needed).

Pass 1 runs eval_cliff_course.evaluate (24 envs, no video) to find failures.
Pass 2 re-runs the same seed, following one chosen env per video. CPU runs are
deterministic for a fixed seed, so the replayed outcome is checked against pass 1
("matches" / "DIFFERS" in the output). Different failure kinds are recorded
first; ``--with-success`` adds one successful episode for comparison.

Videos play at half speed (25 fps for 50 Hz control) and say so in the caption.

Usage (from the RL directory):
  MUJOCO_GL=cgl uv run --no-sync python scripts/record_course_videos.py CHECKPOINT --label block2 \
    --height .0225 --distance .55 --seed 20 --output DIR --max-failures 3 --with-success
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_cliff_course as ev  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls  # noqa: E402
from dataclasses import asdict  # noqa: E402
from mjlab_microduck.tasks.cliff.course_evaluation import classify  # noqa: E402


def record(checkpoint, *, height, distance, seed, env_idx, path, label):
    cfg = ev.make_microduck_cliff_course_env_cfg(play=True)
    cfg.scene.num_envs = 24
    cfg.seed = seed
    cfg.events['course_reset'].params.update(fixed_height=height, start_range=(distance, distance))
    cfg.viewer.env_idx = env_idx
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width, cfg.viewer.height = 960, 640
    cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = .9, 60., -20.
    # Same probe as the evaluator, so the replay is classified identically.
    trace = []

    def probe(env):
        from mjlab_microduck.tasks.cliff import course
        s = course.update(env)
        d = env.scene['robot'].data
        p = d.root_link_pos_w - env.scene.env_origins
        q = d.root_link_quat_w
        foot = env.scene.sensors['cliff_foot_contact'].data
        pos = foot.pos.reshape(24, -1, 3) - env.scene.env_origins[:, None, :]
        found = foot.found.reshape(24, -1)
        force = foot.force.reshape(24, -1, 3)
        cf = [env.scene.sensors[n].data.force.sum(1).norm(dim=-1) for n in ('head_ground_contact', 'trunk_ground_contact')]
        failure = torch.zeros(24, dtype=torch.bool)
        for name in env.termination_manager.active_terms:
            if name.startswith('course_') or env.termination_manager.get_term_cfg(name).time_out:
                continue
            failure |= env.termination_manager.get_term(name)
        i = env_idx
        trace.append(dict(x=p[i, 0].item(), y=p[i, 1].item(), age=env.episode_length_buf[i].item(),
            cos_tilt=(1 - 2 * (q[i, 1]**2 + q[i, 2]**2)).item(), head=cf[0][i].item(), trunk=cf[1][i].item(),
            finite=not s.nonfinite[i].item(), failure=failure[i].item(), pulse=s.pulse[i].item(),
            feet=[dict(found=found[i, j].item(), position=pos[i, j].tolist(), force=force[i, j].tolist())
                  for j in range(found.shape[1])]))
        return torch.zeros(24, dtype=torch.bool)

    cfg.terminations['course_probe'] = TerminationTermCfg(func=probe)
    env = ManagerBasedRlEnv(cfg, device='cpu', render_mode='rgb_array')
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    writer = imageio.get_writer(path, fps=25, codec='libx264', quality=8)
    try:
        agent_cfg = load_rl_cfg(ev.TASK)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls(ev.TASK)(wrapped, asdict(agent_cfg), device='cpu')
        runner.load(str(checkpoint), map_location='cpu')
        env.event_manager.get_term_cfg('course_reset').params.update(fixed_height=height, start_range=(distance, distance))
        env.reset(seed=seed)
        obs = wrapped.get_observations()
        policy = runner.get_inference_policy(device='cpu')
        steps = []
        with torch.inference_mode():
            for _ in range(env.max_episode_length + 1):
                obs, _, done, _ = wrapped.step(policy(obs))
                if done[env_idx]:
                    # The env has already auto-reset; rendering now would show
                    # the next episode. Hold the last real frame ~1 s instead.
                    for _ in range(25):
                        writer.append_data(last)
                    break
                env._offline_renderer._cam.lookat[:] = env.scene['robot'].data.root_link_pos_w[env_idx].cpu().numpy()
                frame = Image.fromarray(env.render())
                s = trace[-1]
                ImageDraw.Draw(frame).text((10, 10),
                    f'{label}  h={height*100:.2f}cm start={distance*100:.0f}cm seed={seed} env={env_idx}  '
                    f't={s["age"]*.02:.2f}s x={s["x"]:+.2f}m head={s["head"]:.0f}N trunk={s["trunk"]:.0f}N  (half speed)',
                    fill='white')
                last = np.asarray(frame)
                writer.append_data(last)
    finally:
        writer.close()
        env.close()
    return classify(trace, height=height)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--label', required=True, help='Caption and file-name prefix, e.g. block2')
    p.add_argument('--height', type=float, required=True, help='Cliff height in metres, e.g. .0225')
    p.add_argument('--distance', type=float, default=.55, help='Start distance from the edge in metres')
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-failures', type=int, default=3, help='Failure videos to record (0 for none)')
    p.add_argument('--with-success', action='store_true', help='Also record the first success, for comparison')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    stem = f'{a.label}-h{a.height*100:.2f}cm-d{a.distance*100:.0f}-s{a.seed}'
    rows = ev.evaluate(a.checkpoint, height=a.height, distance=a.distance, seed=a.seed,
                       output=a.output / f'{stem}.json', device='cpu')
    fails = [r for r in rows if not r['success']]
    print(f'{stem}: {24 - len(fails)}/24 successes; fails:',
          [(r['env'], r['outcome'], r['events']) for r in fails], flush=True)
    # Prefer different failure kinds first.
    first = {}
    for r in fails:
        first.setdefault(tuple(r['events']), r)
    chosen = list(first.values()) + [r for r in fails if r not in first.values()]
    chosen = chosen[:a.max_failures] + ([next(r for r in rows if r['success'])]
                                        if a.with_success and len(fails) < 24 else [])
    for r in chosen:
        name = f"{stem}-env{r['env']:02}-{r['outcome']}-{'+'.join(r['events']) or 'none'}.mp4"
        replay = record(a.checkpoint, height=a.height, distance=a.distance, seed=a.seed, env_idx=r['env'],
                        path=a.output / name, label=a.label)
        same = replay['success'] == r['success'] and replay['events'] == r['events']
        print(f'  {name}: replay {replay["outcome"]} {replay["events"]} {"(matches)" if same else "(DIFFERS from pass 1)"}', flush=True)


if __name__ == '__main__':
    main()
