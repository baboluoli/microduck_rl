"""Cliff observations, reset events, rewards, and success conditions.

MDP means Markov decision process: the rules describing what the robot observes,
how it is rewarded, and when an attempt ends. The sibling ``env_cfg.py`` plugs
these functions into mjlab's managers; the functions do not start training.
Observation/reward/termination functions are evaluated during environment steps,
whereas reset events run when episodes restart.

Most calculations are batched: each tensor row belongs to one independent
simulated robot (``env.num_envs`` robots run together). ``*_w`` data is expressed
in world coordinates; subtracting ``env_origins`` gives a position relative to
that robot's terrain spawn. Distances are metres, speeds metres/second, contact
forces newtons, and times simulated seconds.
"""

import math

import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Select the scene entity named "robot" unless a caller explicitly chooses another.
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ── Cliff drop: reach the lower floor, land on the feet, keep walking ────────
def cliff_terrain_observation(
    env: ManagerBasedRlEnv,
    edge_distance: float,
    drop_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return two terrain inputs per robot: [distance to edge, drop height].

    Called by the observation manager through the ``cliff_terrain`` term in
    both actor and critic groups of ``make_microduck_cliff_oracle_env_cfg``
    (env_cfg.py). The actor chooses actions; the critic estimates future reward.
    These exact simulator inputs extend that experiment's 61 inputs to 63;
    the ordinary cliff-drop task does not use this observation.

    Distance is positive before the edge and negative after crossing it.
    env_origins includes the terrain tile's spawn offset. Drop height is the
    known fixed terrain parameter, not the robot's changing elevation.
    This separates knowledge of the cliff from the robot's motion: crouching
    must not look like a shorter drop, and turning must not move the edge.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Parallel environments occupy different tiles; remove each tile's spawn X.
    local_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    distance = edge_distance - local_x
    # Shape (num_envs, 2); every tile in this fixed-height experiment has the same drop.
    return torch.stack((distance, torch.full_like(distance, drop_height)), dim=-1)


def cliff_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward travel toward and beyond the cliff, rather than idling.

    Called by the reward manager as ``track_linear_velocity`` in the steering
    profile of ``make_microduck_cliff_drop_env_cfg`` (also used by the oracle).
    The heading profile instead keeps the walking task's velocity tracker.

    Signed world-X speed: standing/sideways travel earns zero, retreat costs.
    Cap forward credit at commanded speed, and suppress positive credit while
    tipped over. RewardManager multiplies by dt, making this a distance reward
    below the speed cap; backwards travel cannot refill a progress jackpot.
    Returns one dimensionless score per robot: 1 at/above the forward target
    while upright, 0 at rest, and negative values for retreat. The configured
    reward weight controls its contribution relative to other rewards.
    """
    robot = env.scene[asset_cfg.name].data
    # The twist command stores [forward, sideways, yaw rate]. The floor on the
    # forward target prevents division by zero if a caller supplies a stop command.
    target = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.01)
    speed = torch.nan_to_num(robot.root_link_lin_vel_w[:, 0])
    quat = robot.root_link_quat_w
    # For a unit quaternion ordered (w, x, y, z), this is cos(body tilt from
    # vertical). Yaw alone does not change it; no expensive angle conversion is needed.
    upright = (1 - 2 * (quat[:, 1].square() + quat[:, 2].square())) > math.cos(math.radians(40))
    positive = torch.minimum(speed.clamp_min(0), target) * upright
    # Only positive credit is tilt-gated: falling over must not erase retreat costs.
    return (positive + speed.clamp_max(0)) / target


def cliff_failed_landing(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return a per-robot failure flag after a disqualifying head/trunk impact.

    Called by the termination manager as ``cliff_failed_landing`` in the
    steering profile (including the oracle). The heading profile allows recovery
    instead of installing this immediate failure termination.

    Delegate to the shared landing check so failure and success use identical
    impact thresholds, even if this termination is evaluated first. ``kwargs``
    forwards its landing-check parameters from env_cfg.py.
    """
    cliff_landing_complete(env, **kwargs)
    # Give the manager a snapshot, rather than an alias of our mutable episode state.
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
    """Replace a random subset of reset poses with an already-airborne start.

    Called by the event manager as the reset event ``cliff_midair_reset`` in
    ``make_microduck_cliff_drop_env_cfg``. The heading profile uses probability
    0.35 during training and 0 during playback; steering/oracle use 0. The eval
    script ``scripts/eval_cliff_drop.py`` can override it for midair-only tests.

    This is a learning shortcut: a new policy may rarely reach the edge, so
    full-approach episodes alone provide little landing practice. Starting some
    episodes in flight gives on-policy data (experience from the current policy)
    for the final part of the task right away. Only ``env_ids`` being reset are
    eligible; other robots' ongoing episodes are left alone.

    This reset runs after reset_base. Its z is above the upper platform, so it
    remains valid for every sampled lower-floor height without reading hidden
    terrain metadata into the actor observation.
    Mutates simulator pose/velocity and returns nothing.
    """
    if len(env_ids) == 0 or probability <= 0.0:
        return
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    if len(chosen) == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    # A free joint is the floating base's 7-value pose: xyz + quaternion.
    # Copy the reset pose so Y and orientation survive; only X and Z are replaced.
    q_adr = asset.data.indexing.free_joint_q_adr
    pose = env.sim.data.qpos[chosen][:, q_adr].clone()
    pose[:, 0] = env.scene.env_origins[chosen, 0] + edge_distance + beyond_edge
    pose[:, 2] = env.scene.env_origins[chosen, 2] + spawn_height[0] + (
        spawn_height[1] - spawn_height[0]
    ) * torch.rand(len(chosen), device=env.device)
    asset.write_root_link_pose_to_sim(pose, env_ids=chosen)
    # Root velocity has 6 values: linear xyz followed by angular xyz. Start
    # without sideways motion or spin, then add forward motion and downward fall.
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
    """Return a one-step success pulse after sustained supported forward travel.

    Called directly by the termination manager as ``cliff_landing`` in both
    cliff profiles, and indirectly by ``cliff_landing_reward`` and (in steering)
    ``cliff_failed_landing``. env_cfg.py shares completion parameters between
    these callers so the success bonus and episode end agree.

    A head/trunk impact above the configured thresholds invalidates this
    episode's success even if the robot eventually gets up. The edge is fixed
    relative to the spawn, while its height is randomized by the terrain.

    The implemented test requires enough post-edge X travel, at least one foot
    in contact, limited tilt/vertical speed, and continued forward motion for
    ``min_stable_s`` continuously. This rejects a brief upright bounce or a
    landing followed by stopping. It does not explicitly track a flight phase
    or which body part touched first; "safe landing" is inferred from these
    conditions and the absence of a disqualifying head/trunk impact.

    State lives on ``env`` as one value per robot. Bad impacts and success are
    latched for the episode; stability time resets whenever a condition fails.
    Repeated calls in one control step return the same cached Boolean pulse.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Allocate lazily: managers can call this without a separate startup event.
    if not hasattr(env, "_cliff_bad_impact"):
        env._cliff_bad_impact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cliff_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cliff_stable_s = torch.zeros(env.num_envs, device=env.device)
        env._cliff_completed_this_step = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cliff_last_update_step = -1

    # Both reward and termination call this function. Update episode state
    # once per control step so the 0.30 s hold cannot be counted twice.
    # It can get counted twice because there are multiple calls to this method during the same step.
    # Both reward and termination managers call this method.
    step = int(env.common_step_counter)
    if step != env._cliff_last_update_step:
        # Episodes reset independently in a batch. Clear only newly started rows,
        # using episode age rather than clearing every robot when any one resets.
        fresh = env.episode_length_buf <= 1
        env._cliff_bad_impact[fresh] = False
        env._cliff_success[fresh] = False
        env._cliff_stable_s[fresh] = 0.0

        for sensor_name, threshold in (
            (head_sensor_name, max_head_force),
            (trunk_sensor_name, max_trunk_force),
        ):
            force = env.scene.sensors[sensor_name].data.force
            # Sum contact-slot force vectors, then take the net force's length.
            # This checks the current sensor reading, not an impulse integrated over time.
            magnitude = torch.linalg.vector_norm(torch.nan_to_num(force).sum(dim=1), dim=1)
            # |= latches a violation: later gentle contact cannot undo an earlier hit.
            env._cliff_bad_impact |= magnitude > threshold

        found = env.scene.sensors[foot_sensor_name].data.found
        # At least one foot is enough: requiring both would reject normal walking
        # single-support phases. The sensor is inherited from the walking task.
        feet_supported = found.reshape(env.num_envs, -1).bool().any(dim=1)
        # quaternion = root-link orientation, four-number representation of rotation, (w, x, y, z)
        quat = asset.data.root_link_quat_w
        # how upright is the robot, is it leaning forward/backward or to the left/right
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
        # Accumulate continuous simulated time, not a count of calls or wall time.
        env._cliff_stable_s = torch.where(
            stable, env._cliff_stable_s + env.step_dt, torch.zeros_like(env._cliff_stable_s)
        )
        # Pay/terminate once on crossing the hold threshold. The success latch
        # prevents another bonus even if a caller keeps stepping this episode.
        # ~ means logical NOT for a boolean tensor. It flips each value: True becomes False, and False becomes True
        # These are all tensors btw so while they look like booleans they track the robots that have (not) succeeded.
        complete = (env._cliff_stable_s >= min_stable_s) & ~env._cliff_success
        env._cliff_completed_this_step = complete
        # |= means logical OR, then store the result back
        # add newly completed robos
        env._cliff_success |= complete
        env._cliff_last_update_step = step
    return env._cliff_completed_this_step


def cliff_landing_reward(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Turn the shared success pulse into a single, timestep-independent bonus.

    Called by the reward manager as ``cliff_landing`` in env_cfg.py, alongside
    the termination using ``cliff_landing_complete``. ``kwargs`` forwards the
    same completion parameters to that shared check.

    The reward manager multiplies each term by its weight and ``env.step_dt``.
    Returning 1/dt on success therefore pays exactly the configured weight
    (currently 80), once, rather than a smaller bonus dependent on control rate.
    All other steps return zero. The bonus compensates for the walking rewards
    forfeited when a successful episode ends early.
    """
    return cliff_landing_complete(env, **kwargs).float() / env.step_dt


