"""Headless, paired evaluation of two cliff-drop checkpoints.

Run from this repository with a working mjlab installation:

    uv run python scripts/eval_cliff_drop.py \
      logs/rsl_rl/cliff_drop/2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt \
      logs/rsl_rl/cliff_drop/2026-09-26_20-40-00_cliff-drop-to-3000/model_2998.pt

Each environment contributes its first episode only. A termination probe
captures the last physics state before mjlab automatically resets done envs.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks.microduck_cliff_drop_env_cfg import EDGE_DISTANCE


TASK = "Mjlab-CliffDrop-Flat-MicroDuck"
DROP_M = 0.02


def _probe(env):
    """Cache stage evidence before automatic reset changes the robot state."""
    robot = env.scene["robot"].data
    sensors = env.scene.sensors
    quat = robot.root_link_quat_w
    tilt = torch.rad2deg(torch.acos(torch.clamp(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1)))
    foot = sensors["feet_ground_contact"].data.found.reshape(env.num_envs, -1).bool().any(dim=1)
    def force(name):
        value = sensors[name].data.force
        return torch.linalg.vector_norm(torch.nan_to_num(value).sum(dim=1), dim=1)

    env._cliff_eval_state = {
        "x": (robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]).clone(),
        "z": (robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).clone(),
        "tilt_deg": tilt.clone(),
        "vx": robot.root_link_lin_vel_w[:, 0].clone(),
        "vz": robot.root_link_lin_vel_w[:, 2].clone(),
        "foot": foot.clone(),
        "head_force": force("head_ground_contact").clone(),
        "trunk_force": force("trunk_ground_contact").clone(),
        "bad_impact": env._cliff_bad_impact.clone(),
        "stable_s": env._cliff_stable_s.clone(),
        "success": env._cliff_completed_this_step.clone(),
    }
    if getattr(env, "_cliff_eval_video_active", False) and env.common_step_counter % 2 == 0:
        env._cliff_eval_video_writer.append_data(env.render())
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _configure(num_envs, seed, video_env=None):
    cfg = load_env_cfg(TASK, play=True)
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    if video_env is not None:
        cfg.viewer.env_idx = video_env
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.max_extra_envs = 0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
        cfg.viewer.distance = 2.0
        cfg.viewer.azimuth = 90.0
        cfg.viewer.elevation = -12.0
    cfg.scene.terrain.max_init_terrain_level = 0
    terrain = cfg.scene.terrain.terrain_generator
    terrain.curriculum = False
    terrain.sub_terrains["cliff_drop"].min_drop = DROP_M
    terrain.sub_terrains["cliff_drop"].max_drop = DROP_M
    cfg.curriculum.clear()
    # The inherited bound uses grid-centered world coordinates and truncates
    # valid outer tiles at step one. The cliff tile and 20 m border provide the
    # actual collision geometry for this short, fixed-height evaluation.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.events["cliff_midair_reset"].params["probability"] = 0.0
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.21, 0.21)
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6
    cfg.terminations["cliff_eval_probe"] = TerminationTermCfg(func=_probe, time_out=False)
    return cfg


@torch.inference_mode()
def evaluate(checkpoint: Path, *, num_envs: int, seed: int, device: str,
             video_env: int | None = None, video_path: Path | None = None):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = _configure(num_envs, seed, video_env)
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=cfg, device=device,
                            render_mode="rgb_array" if video_env is not None else None)
    writer = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK) or OnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        env.reset(seed=seed)
        obs = wrapped.get_observations()
        if video_env is not None:
            assert video_path is not None
            video_path.parent.mkdir(parents=True, exist_ok=True)
            origin = env.scene.env_origins[video_env].cpu().numpy()
            env._offline_renderer._cam.lookat[:] = origin + (0.65, 0.0, 0.07)
            writer = imageio.get_writer(video_path, fps=25, codec="libx264", quality=7)
            writer.append_data(env.render())
            env._cliff_eval_video_writer = writer
            env._cliff_eval_video_active = True
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        rows = [{"env": i, "checkpoint": str(checkpoint), "seed": seed,
                 "reached_edge": False, "lower_contact": False,
                 "bad_impact": False, "success": False,
                 "contact_tilt_deg": None, "contact_vx_m_s": None,
                 "contact_vz_m_s": None,
                 "max_x_m": -math.inf, "max_tilt_deg": 0.0,
                 "max_head_force_n": 0.0, "max_trunk_force_n": 0.0,
                 "max_stable_s": 0.0, "steps": 0, "done": False,
                 "termination": ""}
                for i in range(num_envs)]
        # Timeouts must not silently masquerade as complete first episodes.
        limit = math.ceil(cfg.episode_length_s / env.step_dt) + 2
        for _ in range(limit):
            obs, _, done, _ = wrapped.step(policy(obs))
            snapshot = env._cliff_eval_state
            finished = done.reshape(-1).bool()
            for i in torch.where(active)[0].tolist():
                row = rows[i]
                x = float(snapshot["x"][i])
                tilt = float(snapshot["tilt_deg"][i])
                supported = bool(snapshot["foot"][i])
                row["steps"] += 1
                row["max_x_m"] = max(row["max_x_m"], x)
                row["max_tilt_deg"] = max(row["max_tilt_deg"], tilt)
                row["max_head_force_n"] = max(row["max_head_force_n"], float(snapshot["head_force"][i]))
                row["max_trunk_force_n"] = max(row["max_trunk_force_n"], float(snapshot["trunk_force"][i]))
                row["max_stable_s"] = max(row["max_stable_s"], float(snapshot["stable_s"][i]))
                row["reached_edge"] |= x >= EDGE_DISTANCE
                # Contact well past the edge is a useful landing proxy; the
                # task's success gate remains the authoritative completion test.
                if x >= EDGE_DISTANCE + 0.15 and supported and not row["lower_contact"]:
                    row["lower_contact"] = True
                    row["contact_tilt_deg"] = tilt
                    row["contact_vx_m_s"] = float(snapshot["vx"][i])
                    row["contact_vz_m_s"] = float(snapshot["vz"][i])
                row["bad_impact"] |= bool(snapshot["bad_impact"][i])
                row["success"] |= bool(snapshot["success"][i])
                if bool(finished[i]):
                    row["done"] = True
                    row["termination"] = ",".join(
                        name for name in env.termination_manager.active_terms
                        if bool(env.termination_manager.get_term(name)[i])
                    )
                    active[i] = False
                    if i == video_env:
                        env._cliff_eval_video_active = False
            if not bool(active.any()):
                break
        for row in rows:
            row["duration_s"] = round(row["steps"] * env.step_dt, 4)
            if not row["done"]:
                row["outcome"] = "incomplete_evaluation"
            elif row["success"]:
                row["outcome"] = "complete"
            elif not row["reached_edge"]:
                row["outcome"] = "before_edge"
            elif not row["lower_contact"]:
                row["outcome"] = "no_lower_contact"
            elif row["bad_impact"]:
                row["outcome"] = "impact_disqualified"
            elif row["contact_tilt_deg"] > 30.0:
                row["outcome"] = "contact_tilted"
            else:
                row["outcome"] = "contact_no_completion"
        return rows
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs=2, type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("cliff_eval"))
    parser.add_argument("--video-env", type=int, default=None,
                        help="Render this environment's first episode for both checkpoints")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.video_env is not None and not 0 <= args.video_env < args.num_envs:
        parser.error("--video-env must be within --num-envs")
    configure_torch_backends()
    import mjlab.tasks  # noqa: F401 - load task plugins

    results = [evaluate(path, num_envs=args.num_envs, seed=args.seed, device=args.device,
                        video_env=args.video_env,
                        video_path=args.output.with_name(
                            f"{args.output.name}-{path.stem}-env{args.video_env}.mp4"
                        ) if args.video_env is not None else None)
               for path in args.checkpoints]
    rows = [row for group in results for row in group]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    outcomes = ("complete", "before_edge", "no_lower_contact",
                "impact_disqualified", "contact_tilted", "contact_no_completion",
                "incomplete_evaluation")
    summary = {str(path): {outcome: sum(row["outcome"] == outcome for row in group)
                           for outcome in outcomes}
               for path, group in zip(args.checkpoints, results)}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Per-episode records: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
