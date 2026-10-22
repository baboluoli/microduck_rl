"""Cliff observations, reset events, rewards, and success conditions."""

import math

import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ── Cliff drop: reach the lower floor, land on the feet, keep walking ────────
def cliff_terrain_observation(
    env: ManagerBasedRlEnv,
    edge_distance: float,
    drop_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Exact fixed-cliff geometry in metres, independent of robot height/yaw.

    Distance is positive before the edge and negative after crossing it.
    env_origins includes the terrain tile's spawn offset. Drop height is the
    known fixed terrain parameter, not the robot's changing elevation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    local_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    distance = edge_distance - local_x
    return torch.stack((distance, torch.full_like(distance, drop_height)), dim=-1)


def cliff_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Signed world-X speed: standing/sideways travel earns zero, retreat costs.

    Cap forward credit at commanded speed, and suppress positive credit while
    tipped over. RewardManager multiplies by dt, making this a distance reward
    below the speed cap; backwards travel cannot refill a progress jackpot.
    """
    robot = env.scene[asset_cfg.name].data
    target = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.01)
    speed = torch.nan_to_num(robot.root_link_lin_vel_w[:, 0])
    quat = robot.root_link_quat_w
    upright = (1 - 2 * (quat[:, 1].square() + quat[:, 2].square())) > math.cos(math.radians(40))
    positive = torch.minimum(speed.clamp_min(0), target) * upright
    return (positive + speed.clamp_max(0)) / target


def cliff_failed_landing(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """End attempts already disqualified by the same impacts as success."""
    cliff_landing_complete(env, **kwargs)
    return env._cliff_bad_impact.clone()


def cliff_midair_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    edge_distance: float,
    probability: float = 0.35,
    beyond_edge: float = 0.22,
    spawn_height: tuple[float, float] = (0.12, 0.15),
    forward_speed: tuple[float, float] = (0.18, 0.26),
    downward_speed: tuple[float, float] = (0.10, 0.30),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Start some episodes in flight so landing gets on-policy data immediately.

    This reset runs after reset_base. Its z is above the upper platform, so it
    remains valid for every sampled lower-floor height without reading hidden
    terrain metadata into the actor observation.
    """
    if len(env_ids) == 0 or probability <= 0.0:
        return
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    if len(chosen) == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    q_adr = asset.data.indexing.free_joint_q_adr
    pose = env.sim.data.qpos[chosen][:, q_adr].clone()
    pose[:, 0] = env.scene.env_origins[chosen, 0] + edge_distance + beyond_edge
    pose[:, 2] = env.scene.env_origins[chosen, 2] + spawn_height[0] + (
        spawn_height[1] - spawn_height[0]
    ) * torch.rand(len(chosen), device=env.device)
    asset.write_root_link_pose_to_sim(pose, env_ids=chosen)
    velocity = torch.zeros(len(chosen), 6, device=env.device)
    velocity[:, 0] = forward_speed[0] + (forward_speed[1] - forward_speed[0]) * torch.rand(
        len(chosen), device=env.device
    )
    velocity[:, 2] = -(downward_speed[0] + (downward_speed[1] - downward_speed[0]) * torch.rand(
        len(chosen), device=env.device
    ))
    asset.write_root_link_velocity_to_sim(velocity, env_ids=chosen)


def cliff_landing_complete(
    env: ManagerBasedRlEnv,
    edge_distance: float,
    landing_distance: float = 0.40,
    min_stable_s: float = 0.30,
    max_tilt_deg: float = 30.0,
    max_vertical_speed: float = 0.25,
    min_forward_speed: float = 0.05,
    foot_sensor_name: str = "feet_ground_contact",
    head_sensor_name: str = "head_ground_contact",
    trunk_sensor_name: str = "trunk_ground_contact",
    max_head_force: float = 15.0,
    max_trunk_force: float = 20.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Success only after a feet-first drop and continued lower-floor travel.

    A head/trunk impact above the configured thresholds invalidates this
    episode's success even if the robot eventually gets up. The edge is fixed
    relative to the spawn, while its height is randomized by the terrain.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_cliff_bad_impact"):
        env._cliff_bad_impact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cliff_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cliff_stable_s = torch.zeros(env.num_envs, device=env.device)
        env._cliff_completed_this_step = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cliff_last_update_step = -1

    # Both reward and termination call this function. Update episode state
    # once per control step so the 0.30 s hold cannot be counted twice.
    step = int(env.common_step_counter)
    if step != env._cliff_last_update_step:
        fresh = env.episode_length_buf <= 1
        env._cliff_bad_impact[fresh] = False
        env._cliff_success[fresh] = False
        env._cliff_stable_s[fresh] = 0.0

        for sensor_name, threshold in (
            (head_sensor_name, max_head_force),
            (trunk_sensor_name, max_trunk_force),
        ):
            force = env.scene.sensors[sensor_name].data.force
            magnitude = torch.linalg.vector_norm(torch.nan_to_num(force).sum(dim=1), dim=1)
            env._cliff_bad_impact |= magnitude > threshold

        found = env.scene.sensors[foot_sensor_name].data.found
        feet_supported = found.reshape(env.num_envs, -1).bool().any(dim=1)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        velocity = torch.nan_to_num(asset.data.root_link_lin_vel_w)
        x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        stable = (
            (x >= edge_distance + landing_distance)
            & feet_supported
            & (cos_tilt >= math.cos(math.radians(max_tilt_deg)))
            & (velocity[:, 2].abs() <= max_vertical_speed)
            & (velocity[:, 0] >= min_forward_speed)
            & ~env._cliff_bad_impact
        )
        env._cliff_stable_s = torch.where(
            stable, env._cliff_stable_s + env.step_dt, torch.zeros_like(env._cliff_stable_s)
        )
        complete = (env._cliff_stable_s >= min_stable_s) & ~env._cliff_success
        env._cliff_completed_this_step = complete
        env._cliff_success |= complete
        env._cliff_last_update_step = step
    return env._cliff_completed_this_step


def cliff_landing_reward(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """One terminal success bonus; compensate for RewardManager's dt scaling."""
    return cliff_landing_complete(env, **kwargs).float() / env.step_dt


