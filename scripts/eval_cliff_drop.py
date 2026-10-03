"""Evaluate full cliff-drop sequences by height, seed, and approach distance.

Example::

  uv run python scripts/eval_cliff_drop.py model_999.pt walking_model.pt \
    --num-envs 16 --seeds 0 1 2 --approach-distances 0.35 0.55 0.75

The first episode of every environment is scored. Selected success and failure
rollouts are rendered automatically unless --no-videos is passed.
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks.microduck_cliff_drop_env_cfg import EDGE_DISTANCE, HEIGHT_BUCKETS_CM
from mjlab_microduck.tasks import mdp as microduck_mdp

TASK = "Mjlab-CliffDrop-Flat-MicroDuck"
EDGE_CORRIDOR_M = 0.15
CONTACT_HEIGHT_TOLERANCE_M = 0.008


def _probe(env):
    """Cache last-step evidence before automatic reset changes the state."""
    robot = env.scene["robot"].data
    quat = robot.root_link_quat_w
    tilt = torch.rad2deg(torch.acos(torch.clamp(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1)))
    contact = env.scene.sensors["cliff_foot_contact"].data

    def force(name):
        value = env.scene.sensors[name].data.force
        return torch.linalg.vector_norm(torch.nan_to_num(value).sum(dim=1), dim=1)

    env._cliff_eval_state = {
        "x": (robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]).clone(),
        "y": (robot.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]).clone(),
        "yaw": robot.heading_w.clone(),
        "tilt_deg": tilt.clone(),
        "vx": robot.root_link_lin_vel_w[:, 0].clone(),
        "vz": robot.root_link_lin_vel_w[:, 2].clone(),
        "contact_found": contact.found.clone(),
        "contact_pos": contact.pos.clone(),
        "head_force": force("head_ground_contact").clone(),
        "trunk_force": force("trunk_ground_contact").clone(),
        "bad_impact": env._cliff_bad_impact.clone(),
        "stable_s": env._cliff_stable_s.clone(),
        "success": env._cliff_completed_this_step.clone(),
        "peak_torque_nm": robot.actuator_force.abs().amax(dim=1).clone(),
        "stall_count": microduck_mdp.servo_stall_penalty(env).clone(),
    }
    if getattr(env, "_cliff_eval_video_active", False) and env.common_step_counter % 2 == 0:
        env._cliff_eval_video_writer.append_data(env.render())
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _configure(num_envs, seed, drop_m, approach_m, start, video_env=None,
               legacy_command=False, flat_ground=False):
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
    terrain.sub_terrains["cliff_drop"].min_drop = drop_m
    terrain.sub_terrains["cliff_drop"].max_drop = drop_m
    cfg.curriculum.clear()
    # Grid-centered world bounds truncate valid outer tiles at step one.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # Evaluate all profiles with the same observation window and success
    # criterion, even when training ends disqualified attempts earlier.
    cfg.terminations.pop("cliff_failed_landing", None)
    cfg.terminations["fallen_too_long"].params["max_duration_s"] = 4.0
    cfg.events["cliff_midair_reset"].params["probability"] = 1.0 if start == "midair" else 0.0
    offset = EDGE_DISTANCE - approach_m
    cfg.events["reset_base"].params["pose_range"]["x"] = (offset, offset)
    cfg.commands["twist"].ranges.lin_vel_x = (0.21, 0.21)
    if legacy_command:
        twist = cfg.commands["twist"]
        twist.heading_command = False
        twist.rel_heading_envs = 0.0
        twist.ranges.heading = None
        twist.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6
    if flat_ground:
        if start != "upper":
            raise ValueError("Flat-ground control requires an upper-platform start")
        cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane", env_spacing=4.0)
        # Keep the task-state probe initialized, but observe a full ten seconds
        # of walking instead of terminating at the virtual landing position.
        cfg.terminations["cliff_landing"].params["min_stable_s"] = 1e9
        cfg.rewards["cliff_landing"].params["min_stable_s"] = 1e9
    cfg.terminations["cliff_eval_probe"] = TerminationTermCfg(func=_probe, time_out=False)
    return cfg


def _is_lower_contact(snapshot, i, origin, drop_m):
    """Identify a foot contact on the lower platform's top face and footprint."""
    found = snapshot["contact_found"][i].reshape(-1).bool()
    pos = snapshot["contact_pos"][i].reshape(-1, 3)
    relative = pos - origin
    valid = (
        found & (relative[:, 0] >= EDGE_DISTANCE - 0.005)
        & (relative[:, 0] <= 5.0 - 0.005)
        & (relative[:, 1].abs() <= 1.35)
        & ((relative[:, 2] + drop_m).abs() <= CONTACT_HEIGHT_TOLERANCE_M)
    )
    return bool(valid.any())


@torch.inference_mode()
def evaluate(checkpoint: Path, *, num_envs: int, seed: int, device: str, drop_m: float,
             approach_m: float, start: str = "upper", video_env: int | None = None,
             video_path: Path | None = None, legacy_command: bool = False,
             flat_ground: bool = False):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = _configure(num_envs, seed, drop_m, approach_m, start, video_env, legacy_command,
                     flat_ground)
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
        origins = env.scene.env_origins.clone()
        rows = [{
            "env": i, "checkpoint": str(checkpoint), "seed": seed,
            "height_cm": round(drop_m * 100), "approach_m": approach_m, "start": start,
            "terrain_mode": "flat_control" if flat_ground else "cliff",
            "command_mode": "world_heading" if cfg.commands["twist"].heading_command else "legacy_zero_yaw",
            "reached_edge": False, "edge_in_corridor": False, "time_to_edge_s": None,
            "lower_contact": False, "lower_contact_count": 0,
            "post_landing_travel_m": 0.0, "bad_impact": False, "success": False,
            "contact_tilt_deg": None, "contact_vx_m_s": None, "contact_vz_m_s": None,
            "start_x_m": None, "end_x_m": None, "forward_progress_m": 0.0,
            "distance_traveled_m": 0.0, "path_efficiency": 0.0,
            "lateral_drift_m": 0.0, "max_lateral_drift_m": 0.0,
            "mean_heading_error_deg": 0.0, "max_heading_error_deg": 0.0,
            "max_x_m": -math.inf, "max_tilt_deg": 0.0,
            "max_head_force_n": 0.0, "max_trunk_force_n": 0.0,
            "max_stable_s": 0.0, "steps": 0, "done": False, "termination": "",
            "peak_torque_nm": 0.0, "high_torque_low_speed_steps": 0,
        } for i in range(num_envs)]
        previous = [None] * num_envs
        first_contact_x = [None] * num_envs
        prior_contact = [False] * num_envs
        limit = math.ceil(cfg.episode_length_s / env.step_dt) + 2
        for _ in range(limit):
            obs, _, done, _ = wrapped.step(policy(obs))
            snapshot = env._cliff_eval_state
            finished = done.reshape(-1).bool()
            for i in torch.where(active)[0].tolist():
                row = rows[i]
                x, y = float(snapshot["x"][i]), float(snapshot["y"][i])
                tilt = float(snapshot["tilt_deg"][i])
                heading = abs(math.degrees(math.atan2(math.sin(float(snapshot["yaw"][i])),
                                                       math.cos(float(snapshot["yaw"][i])))))
                row["steps"] += 1
                if previous[i] is None:
                    previous[i] = (x, y)
                    row["start_x_m"] = x
                    row["start_y_m"] = y
                else:
                    row["distance_traveled_m"] += math.hypot(x - previous[i][0], y - previous[i][1])
                    previous[i] = (x, y)
                row["end_x_m"] = x
                row["forward_progress_m"] = x - row["start_x_m"]
                row["lateral_drift_m"] = y - row["start_y_m"]
                row["max_lateral_drift_m"] = max(row["max_lateral_drift_m"], abs(row["lateral_drift_m"]))
                row["mean_heading_error_deg"] += heading
                row["max_heading_error_deg"] = max(row["max_heading_error_deg"], heading)
                row["max_x_m"] = max(row["max_x_m"], x)
                row["max_tilt_deg"] = max(row["max_tilt_deg"], tilt)
                row["max_head_force_n"] = max(row["max_head_force_n"], float(snapshot["head_force"][i]))
                row["max_trunk_force_n"] = max(row["max_trunk_force_n"], float(snapshot["trunk_force"][i]))
                row["max_stable_s"] = max(row["max_stable_s"], float(snapshot["stable_s"][i]))
                row["peak_torque_nm"] = max(row["peak_torque_nm"], float(snapshot["peak_torque_nm"][i]))
                row["high_torque_low_speed_steps"] += int(snapshot["stall_count"][i] > 0)
                if start == "upper" and not row["reached_edge"] and x >= EDGE_DISTANCE:
                    row["reached_edge"] = True
                    row["time_to_edge_s"] = row["steps"] * env.step_dt
                    row["edge_in_corridor"] = abs(y - row["start_y_m"]) <= EDGE_CORRIDOR_M
                contact = _is_lower_contact(snapshot, i, origins[i], drop_m)
                if contact and not prior_contact[i]:
                    row["lower_contact_count"] += 1
                prior_contact[i] = contact
                if contact and not row["lower_contact"]:
                    row["lower_contact"] = True
                    first_contact_x[i] = x
                    row["contact_tilt_deg"] = tilt
                    row["contact_vx_m_s"] = float(snapshot["vx"][i])
                    row["contact_vz_m_s"] = float(snapshot["vz"][i])
                if first_contact_x[i] is not None:
                    row["post_landing_travel_m"] = max(row["post_landing_travel_m"], x - first_contact_x[i])
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
            row["mean_heading_error_deg"] /= max(row["steps"], 1)
            row["path_efficiency"] = max(row["forward_progress_m"], 0.0) / max(row["distance_traveled_m"], 1e-9)
            if not row["done"]:
                row["outcome"] = "incomplete_evaluation"
            elif flat_ground:
                row["outcome"] = (
                    "flat_upright" if row["max_tilt_deg"] < 45 and not row["bad_impact"]
                    and row["termination"] == "time_out" else "flat_fall"
                )
            elif row["success"] and row["lower_contact"]:
                row["outcome"] = "complete"
            elif start == "upper" and not row["reached_edge"]:
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


def summarize(rows, checkpoints):
    summary = {}
    for checkpoint in checkpoints:
        group = [r for r in rows if r["checkpoint"] == str(checkpoint)]
        by_start = {}
        for start in ("upper", "midair"):
            by_height = {}
            for cm in HEIGHT_BUCKETS_CM:
                sample = [r for r in group if r["start"] == start and r["height_cm"] == cm]
                if sample:
                    by_height[str(cm)] = {
                        "episodes": len(sample),
                        "complete": sum(r["outcome"] == "complete" for r in sample),
                        "edge_in_corridor": sum(r["edge_in_corridor"] for r in sample),
                        "lower_contact": sum(r["lower_contact"] for r in sample),
                        "outcomes": {k: sum(r["outcome"] == k for r in sample)
                                     for k in sorted({r["outcome"] for r in sample})},
                    }
            if by_height:
                by_start[start] = by_height
        summary[str(checkpoint)] = by_start
    return summary


def advancement_gates(rows, checkpoints, seeds):
    """Require every seed at each of the last two ordered checkpoints to pass."""
    pair = checkpoints[-2:]

    def passes(checkpoint, seed, height, field, threshold):
        sample = [r for r in rows if r["checkpoint"] == str(checkpoint)
                  and r["start"] == "upper" and r["seed"] == seed
                  and r["height_cm"] == height]
        if not sample:
            return False
        return sum((r["outcome"] == "complete") if field == "complete" else bool(r[field])
                   for r in sample) / len(sample) >= threshold

    gate = {"checkpoints": [str(p) for p in pair], "seeds": seeds}
    for height in HEIGHT_BUCKETS_CM:
        if height == 2:
            gate["straight_approach"] = len(pair) == 2 and all(
                passes(p, seed, 2, "edge_in_corridor", 0.95)
                for p in pair for seed in seeds
            )
            gate["2_cm"] = len(pair) == 2 and all(
                passes(p, seed, 2, "complete", 0.90)
                for p in pair for seed in seeds
            )
        else:
            gate[f"{height}_cm"] = gate["2_cm"] and all(
                passes(p, seed, height, "complete", 0.85)
                for p in pair for seed in seeds
            )
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--approach-distances", type=float, nargs="+", default=[0.35, 0.55, 0.75])
    parser.add_argument("--heights-cm", type=int, nargs="+", default=list(HEIGHT_BUCKETS_CM))
    parser.add_argument("--include-midair", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("cliff_eval"))
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--legacy-command", action="store_true",
                        help="Diagnostic baseline: use the original zero-yaw command")
    args = parser.parse_args()
    if args.num_envs < 1 or any(d <= 0 or d >= EDGE_DISTANCE + 0.9 for d in args.approach_distances):
        parser.error("num-envs must be positive and approach distances must start on the upper platform")
    if any(h not in HEIGHT_BUCKETS_CM for h in args.heights_cm):
        parser.error(f"heights must be among {HEIGHT_BUCKETS_CM}")
    configure_torch_backends()
    import mjlab.tasks  # noqa: F401 - load task plugins

    rows = []
    representatives = {}
    for checkpoint in args.checkpoints:
        for start in (["upper", "midair"] if args.include_midair else ["upper"]):
            for cm in args.heights_cm:
                for seed in args.seeds:
                    for distance in args.approach_distances if start == "upper" else [args.approach_distances[0]]:
                        group = evaluate(checkpoint, num_envs=args.num_envs, seed=seed,
                                         device=args.device, drop_m=cm / 100,
                                         approach_m=distance, start=start,
                                         legacy_command=args.legacy_command)
                        rows.extend(group)
                        if args.no_videos:
                            continue
                        for label, candidate in (("success", next((r for r in group if r["outcome"] == "complete"), None)),
                                                 ("failure", next((r for r in group if r["outcome"] != "complete"), None))):
                            key = (str(checkpoint), start, label)
                            if candidate is not None and key not in representatives:
                                representatives[key] = (cm, seed, distance, candidate["env"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.checkpoints)
    videos = []
    video_results = []
    video_errors = []
    for (checkpoint, start, label), (cm, seed, distance, env_idx) in representatives.items():
        path = args.output.with_name(f"{args.output.name}-{Path(checkpoint).stem}-{start}-{label}-{cm}cm-seed{seed}-env{env_idx}.mp4")
        try:
            replay = evaluate(Path(checkpoint), num_envs=args.num_envs, seed=seed, device=args.device,
                              drop_m=cm / 100, approach_m=distance, start=start,
                              video_env=env_idx, video_path=path,
                              legacy_command=args.legacy_command)
        except Exception as exc:
            video_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        else:
            # GPU physics/rendering replays need not reproduce the sampled
            # episode. Name and report the actual recorded outcome.
            recorded = replay[env_idx]
            actual_path = args.output.with_name(
                f"{args.output.name}-{Path(checkpoint).stem}-{start}-"
                f"{recorded['outcome']}-{cm}cm-seed{seed}-env{env_idx}.mp4"
            )
            path.rename(actual_path)
            path = actual_path
            videos.append(str(path))
            video_results.append({"path": str(path), "selected_from": label,
                                  "episode": recorded})
    result = {"summary": summary, "advancement_gates": advancement_gates(rows, args.checkpoints, args.seeds),
              "representative_videos": videos, "video_results": video_results,
              "video_errors": video_errors}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Per-episode records: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
